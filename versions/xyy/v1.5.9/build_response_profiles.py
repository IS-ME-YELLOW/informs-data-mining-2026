"""构建并验证 v1.9 前 72 小时县域响应画像和完整特征数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BASE_FEATURE_VERSION,
    CACHE_DIR,
    FEATURE_DEFINITIONS_FILE,
    FEATURE_NAMES_FILE,
    FEATURE_VALIDATION_FILE,
    FEATURES_TEST_FILE,
    FEATURES_TRAIN_FILE,
    GUST_THRESHOLD,
    MATERIAL_THRESHOLD,
    PROFILE_TEST_FILE,
    PROFILE_TRAIN_FILE,
    RAW_TEST_FILE,
    RAW_TRAIN_FILE,
    RESPONSE_FEATURES,
    ensure_dirs,
)

EPS = 1e-12
OBS_END = 72
LAST24_START = 48


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_fips(values: pd.Series) -> pd.Series:
    result = values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    if not result.str.fullmatch(r"\d{5}").all():
        raise ValueError("FIPS 格式异常")
    return result


def slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(len(y), dtype=float)[valid]
    y = y[valid]
    if np.std(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def first_peak_hour(values: np.ndarray) -> int | None:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return None
    return int(np.nanargmax(values))


def first_material_hour(values: np.ndarray) -> int | None:
    hit = np.flatnonzero(np.asarray(values, dtype=float) >= MATERIAL_THRESHOLD)
    return int(hit[0]) if len(hit) else None


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 4:
        return np.nan
    x, y = left[valid], right[valid]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def prepare_raw(path: Path, expected_counties: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"fipsCode": str})
    required = {"timestamp_et", "fipsCode", "P_t", "N_t", "D_t", "R_t", "gust"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} 缺列: {sorted(missing)}")
    frame["fipsCode"] = normalize_fips(frame["fipsCode"])
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    frame = frame.sort_values(["fipsCode", "timestamp_et"]).reset_index(drop=True)
    if frame[["fipsCode", "timestamp_et"]].duplicated().any():
        raise ValueError(f"{path.name} 存在重复县时键")
    frame["_hour_idx"] = frame.groupby("fipsCode", sort=False).cumcount()
    sizes = frame.groupby("fipsCode").size()
    if len(sizes) != expected_counties or not sizes.eq(216).all():
        raise ValueError(f"{path.name} 县数或逐县小时数错误")
    return frame


def compute_county_profile(group: pd.DataFrame) -> dict[str, float | str]:
    obs = group.loc[group["_hour_idx"] < OBS_END].sort_values("_hour_idx")
    if not np.array_equal(obs["_hour_idx"].to_numpy(), np.arange(OBS_END)):
        raise ValueError(f"县 {group['fipsCode'].iloc[0]} 观测窗口不完整")

    p = obs["P_t"].to_numpy(dtype=float)
    n = obs["N_t"].to_numpy(dtype=float)
    d = obs["D_t"].to_numpy(dtype=float)
    r = obs["R_t"].to_numpy(dtype=float)
    gust = obs["gust"].to_numpy(dtype=float)
    last = slice(LAST24_START, OBS_END)

    has_p = bool(np.any(p >= MATERIAL_THRESHOLD))
    has_n = bool(np.any(n >= MATERIAL_THRESHOLD))
    has_r = bool(np.any(r >= MATERIAL_THRESHOLD))
    p_peak = first_peak_hour(p)
    n_peak = first_peak_hour(n) if has_n else None
    r_peak = first_peak_hour(r) if has_r else None
    first_n = first_material_hour(n)
    first_r = first_material_hour(r)
    excess = np.maximum(gust - GUST_THRESHOLD, 0.0)
    excess_sum = float(np.nansum(excess))
    n_sum = float(np.nansum(n))
    r_sum = float(np.nansum(r))

    lag_corrs: list[tuple[int, float]] = []
    for lag in range(4):
        if lag == 0:
            corr = safe_corr(gust, n)
        else:
            corr = safe_corr(gust[:-lag], n[lag:])
        if np.isfinite(corr):
            lag_corrs.append((lag, corr))
    if lag_corrs:
        best_lag, best_corr = max(lag_corrs, key=lambda item: item[1])
    else:
        best_lag, best_corr = np.nan, np.nan

    post_peak_slope = np.nan
    if p_peak is not None and OBS_END - p_peak >= 2:
        post_peak_slope = slope(p[p_peak:])

    result: dict[str, float | str] = {
        "fipsCode": str(group["fipsCode"].iloc[0]),
        "N_nonzero_frac_obs": float(np.mean(n > 0)),
        "R_nonzero_frac_obs": float(np.mean(r > 0)),
        "P_mean_last24_obs": float(np.nanmean(p[last])),
        "D_mean_last24_obs": float(np.nanmean(d[last])),
        "N_mean_last24_obs": float(np.nanmean(n[last])),
        "R_mean_last24_obs": float(np.nanmean(r[last])),
        "flow_balance_last24_obs": float(np.nanmean(n[last]) - np.nanmean(r[last])),
        "restoration_ratio_obs": r_sum / n_sum if n_sum > EPS else np.nan,
        "P_D_gap_last_obs": float(p[-1] - d[-1]),
        "hours_since_P_peak_obs": float(71 - p_peak) if has_p and p_peak is not None else np.nan,
        "hours_since_N_peak_obs": float(71 - n_peak) if n_peak is not None else np.nan,
        "hours_since_R_peak_obs": float(71 - r_peak) if r_peak is not None else np.nan,
        "first_material_N_hour_obs": float(first_n) if first_n is not None else np.nan,
        "first_material_R_hour_obs": float(first_r) if first_r is not None else np.nan,
        "P_trend_last24_obs": slope(p[last]),
        "D_trend_last24_obs": slope(d[last]),
        "P_post_peak_slope_obs": post_peak_slope,
        "P_active_frac_obs": float(np.mean(p >= MATERIAL_THRESHOLD)),
        "has_material_P_obs": float(has_p),
        "has_material_N_obs": float(has_n),
        "has_material_R_obs": float(has_r),
        "gust_excess30_sum_obs": excess_sum,
        "N_per_gust_excess_obs": n_sum / excess_sum if excess_sum > EPS else np.nan,
        "P_change_per_gust_excess_obs": (
            float(p[-1] - np.nanmean(p[:LAST24_START])) / excess_sum
            if excess_sum > EPS else np.nan
        ),
        "gust_to_N_best_corr_obs": float(best_corr),
        "gust_to_N_best_lag_obs": float(best_lag),
        "gust_at_first_material_N_obs": float(gust[first_n]) if first_n is not None else np.nan,
    }
    if list(result.keys()) != ["fipsCode", *RESPONSE_FEATURES]:
        raise AssertionError("响应画像输出顺序与固定清单不一致")
    return result


def build_profiles(raw: pd.DataFrame) -> pd.DataFrame:
    records = [compute_county_profile(group) for _, group in raw.groupby("fipsCode", sort=True)]
    profiles = pd.DataFrame(records)
    profiles["fipsCode"] = normalize_fips(profiles["fipsCode"])
    if profiles["fipsCode"].duplicated().any() or list(profiles.columns) != ["fipsCode", *RESPONSE_FEATURES]:
        raise ValueError("县域画像键或列异常")
    values = profiles[list(RESPONSE_FEATURES)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("县域画像包含无穷值")
    return profiles


def attach_profiles(
    base: pd.DataFrame, meta: pd.DataFrame, profiles: pd.DataFrame, expected_counties: int
) -> pd.DataFrame:
    keys = normalize_fips(meta["fipsCode"])
    lookup = profiles.set_index("fipsCode")
    additions = lookup.reindex(keys).reset_index(drop=True)
    if additions.shape != (len(base), len(RESPONSE_FEATURES)):
        raise ValueError("画像附加形状异常")
    result = pd.concat([base.reset_index(drop=True), additions], axis=1)
    if len(result) != expected_counties * 144 or len(result.columns) != len(base.columns) + len(RESPONSE_FEATURES):
        raise ValueError("v1.9 特征形状异常")
    for fips, idx in meta.groupby(keys, sort=False).groups.items():
        block = result.loc[list(idx), list(RESPONSE_FEATURES)]
        first = block.iloc[[0]].reset_index(drop=True)
        repeated = pd.concat([first] * len(block), ignore_index=True)
        repeated.index = block.index
        pd.testing.assert_frame_equal(block, repeated, check_dtype=False, check_exact=True)
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("v1.9 特征包含无穷值")
    return result


def definitions() -> pd.DataFrame:
    descriptions = {
        "N_nonzero_frac_obs": "前72小时 N_t>0 的比例",
        "R_nonzero_frac_obs": "前72小时 R_t>0 的比例",
        "P_mean_last24_obs": "观测窗口最后24小时 P_t 均值",
        "D_mean_last24_obs": "观测窗口最后24小时 D_t 均值",
        "N_mean_last24_obs": "观测窗口最后24小时 N_t 均值",
        "R_mean_last24_obs": "观测窗口最后24小时 R_t 均值",
        "flow_balance_last24_obs": "最后24小时 mean(N_t)-mean(R_t)",
        "restoration_ratio_obs": "前72小时 sum(R_t)/sum(N_t)",
        "P_D_gap_last_obs": "第71小时 P_t-D_t",
        "hours_since_P_peak_obs": "距前72小时 P_t 最早峰值的小时数",
        "hours_since_N_peak_obs": "距实质 N_t 最早峰值的小时数",
        "hours_since_R_peak_obs": "距实质 R_t 最早峰值的小时数",
        "first_material_N_hour_obs": "首次 N_t>=0.001 的小时",
        "first_material_R_hour_obs": "首次 R_t>=0.001 的小时",
        "P_trend_last24_obs": "最后24小时 P_t 线性斜率",
        "D_trend_last24_obs": "最后24小时 D_t 线性斜率",
        "P_post_peak_slope_obs": "P_t 最早峰值至观测结束的线性斜率",
        "P_active_frac_obs": "前72小时 P_t>=0.001 的比例",
        "has_material_P_obs": "是否存在 P_t>=0.001",
        "has_material_N_obs": "是否存在 N_t>=0.001",
        "has_material_R_obs": "是否存在 R_t>=0.001",
        "gust_excess30_sum_obs": "前72小时 sum(max(gust-30,0))",
        "N_per_gust_excess_obs": "前72小时 sum(N_t)/强风超额暴露",
        "P_change_per_gust_excess_obs": "P末值相对前48小时均值变化/强风超额暴露",
        "gust_to_N_best_corr_obs": "gust与随后0至3小时N_t的最大相关",
        "gust_to_N_best_lag_obs": "最大gust-N相关对应滞后",
        "gust_at_first_material_N_obs": "首次实质N_t时的gust",
    }
    return pd.DataFrame([
        {"feature": name, "description_zh": descriptions[name], "source_hours": "0-71"}
        for name in RESPONSE_FEATURES
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    outputs = [
        PROFILE_TRAIN_FILE, PROFILE_TEST_FILE, FEATURES_TRAIN_FILE, FEATURES_TEST_FILE,
        FEATURE_NAMES_FILE, FEATURE_DEFINITIONS_FILE, FEATURE_VALIDATION_FILE,
    ]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("v1.9 画像或特征已存在；使用 --overwrite 明确重建")

    raw_train = prepare_raw(RAW_TRAIN_FILE, 239)
    raw_test = prepare_raw(RAW_TEST_FILE, 63)
    train_profiles = build_profiles(raw_train)
    test_profiles = build_profiles(raw_test)

    base_names = json.loads((CACHE_DIR / f"feature_names_{BASE_FEATURE_VERSION}.json").read_text(encoding="utf-8"))
    base_train = pd.read_parquet(CACHE_DIR / f"features_train_{BASE_FEATURE_VERSION}.parquet")
    base_test = pd.read_parquet(CACHE_DIR / f"features_test_{BASE_FEATURE_VERSION}.parquet")
    meta_train = pd.read_parquet(CACHE_DIR / f"meta_train_{BASE_FEATURE_VERSION}.parquet")
    meta_test = pd.read_parquet(CACHE_DIR / f"meta_test_{BASE_FEATURE_VERSION}.parquet")
    if list(base_train.columns) != base_names or list(base_test.columns) != base_names:
        raise ValueError("v1.5.6 特征清单不一致")

    train_features = attach_profiles(base_train, meta_train, train_profiles, 239)
    test_features = attach_profiles(base_test, meta_test, test_profiles, 63)
    feature_names = [*base_names, *RESPONSE_FEATURES]
    if list(train_features.columns) != feature_names or list(test_features.columns) != feature_names:
        raise ValueError("v1.9 特征列顺序错误")
    pd.testing.assert_frame_equal(train_features[base_names], base_train, check_exact=True)
    pd.testing.assert_frame_equal(test_features[base_names], base_test, check_exact=True)

    train_profiles.to_parquet(PROFILE_TRAIN_FILE, index=False)
    test_profiles.to_parquet(PROFILE_TEST_FILE, index=False)
    train_features.to_parquet(FEATURES_TRAIN_FILE, index=False)
    test_features.to_parquet(FEATURES_TEST_FILE, index=False)
    FEATURE_NAMES_FILE.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    definitions().to_csv(FEATURE_DEFINITIONS_FILE, index=False)

    validation = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed",
        "base_feature_version": BASE_FEATURE_VERSION,
        "base_feature_count": len(base_names),
        "response_feature_count": len(RESPONSE_FEATURES),
        "total_feature_count": len(feature_names),
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "train_counties": int(train_profiles["fipsCode"].nunique()),
        "test_counties": int(test_profiles["fipsCode"].nunique()),
        "material_threshold": MATERIAL_THRESHOLD,
        "gust_threshold": GUST_THRESHOLD,
        "profile_missing_counts_train": train_profiles.isna().sum().to_dict(),
        "profile_missing_counts_test": test_profiles.isna().sum().to_dict(),
        "base_columns_unchanged_exact": True,
        "inputs_sha256": {
            str(path): sha256_file(path) for path in [
                RAW_TRAIN_FILE, RAW_TEST_FILE,
                CACHE_DIR / f"features_train_{BASE_FEATURE_VERSION}.parquet",
                CACHE_DIR / f"features_test_{BASE_FEATURE_VERSION}.parquet",
                CACHE_DIR / f"meta_train_{BASE_FEATURE_VERSION}.parquet",
                CACHE_DIR / f"meta_test_{BASE_FEATURE_VERSION}.parquet",
                CACHE_DIR / f"feature_names_{BASE_FEATURE_VERSION}.json",
            ]
        },
    }
    validation["outputs_sha256"] = {
        path.name: sha256_file(path) for path in [
            PROFILE_TRAIN_FILE, PROFILE_TEST_FILE, FEATURES_TRAIN_FILE, FEATURES_TEST_FILE,
            FEATURE_NAMES_FILE, FEATURE_DEFINITIONS_FILE,
        ]
    }
    FEATURE_VALIDATION_FILE.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
