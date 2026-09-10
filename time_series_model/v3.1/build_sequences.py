"""构建并验证 v3.1 县级序列缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd

from config import (
    ARTIFACT_DIR,
    FEATURE_NAMES,
    FEATURE_TEST,
    FEATURE_TRAIN,
    HORIZONS,
    LAST_STATE_COLUMNS,
    META_TEST,
    META_TRAIN,
    OBSERVED_CHANNELS,
    OBSERVED_HOURS,
    OUTAGE_CHANNELS,
    SEQUENCE_MANIFEST,
    STATIC_CATEGORICAL_COLUMN,
    STATIC_NUMERIC_COLUMNS,
    TARGET_TRAIN,
    TEST_RAW,
    TEST_SEQUENCE_CACHE,
    TOTAL_HOURS,
    TRAIN_RAW,
    TRAIN_SEQUENCE_CACHE,
    WEATHER_CHANNELS,
    WEATHER_SOURCE_CHANNELS,
    ensure_artifact_dirs,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_fips(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def weather_matrix(group: pd.DataFrame) -> np.ndarray:
    direction_rad = np.deg2rad(group["wind_dir_10m"].to_numpy(dtype=float))
    columns = [
        group["gust"].to_numpy(dtype=float),
        group["wind_speed_10m"].to_numpy(dtype=float),
        np.sin(direction_rad),
        np.cos(direction_rad),
        group["t2m"].to_numpy(dtype=float),
        group["tp"].to_numpy(dtype=float),
        group["sp"].to_numpy(dtype=float),
        group["r2"].to_numpy(dtype=float),
        group["csnow"].to_numpy(dtype=float),
        group["soil_moist"].to_numpy(dtype=float),
    ]
    return np.column_stack(columns)


def validate_raw_frame(raw: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"timestamp_et", "fipsCode", *OUTAGE_CHANNELS, *WEATHER_SOURCE_CHANNELS}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{label} 原始数据缺少列: {missing}")

    raw = raw.copy()
    raw["fipsCode"] = normalize_fips(raw["fipsCode"])
    raw["_timestamp"] = pd.to_datetime(raw["timestamp_et"], errors="raise")
    counts = raw.groupby("fipsCode").size()
    if not (counts == TOTAL_HOURS).all():
        bad = counts[counts != TOTAL_HOURS].to_dict()
        raise ValueError(f"{label} 每县必须有 {TOTAL_HOURS} 行: {bad}")

    ordered = raw.sort_values(["fipsCode", "_timestamp"]).reset_index(drop=True)
    for fips, group in ordered.groupby("fipsCode", sort=False):
        times = group["_timestamp"].to_numpy(dtype="datetime64[ns]")
        if len(times) != TOTAL_HOURS:
            raise ValueError(f"{label} 县 {fips} 行数异常")
        gaps = np.diff(times).astype("timedelta64[h]").astype(int)
        if not np.all(gaps == 1):
            raise ValueError(f"{label} 县 {fips} 时间不连续")
    return ordered


def build_bundle(
    raw_path: Path,
    feature_path: Path,
    meta_path: Path,
    output_path: Path,
    target_path: Path | None,
    label: str,
) -> dict:
    raw = validate_raw_frame(
        pd.read_csv(raw_path, dtype={"fipsCode": str}), label
    )
    features = pd.read_parquet(feature_path).reset_index(drop=True)
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    meta["fipsCode"] = normalize_fips(meta["fipsCode"])
    meta["hour_idx"] = pd.to_numeric(meta["hour_idx"], errors="raise").astype(int)

    if len(features) != len(meta):
        raise ValueError(f"{label} 特征和元信息行数不一致")
    required_features = set(STATIC_NUMERIC_COLUMNS + [STATIC_CATEGORICAL_COLUMN] + LAST_STATE_COLUMNS)
    missing_features = sorted(required_features - set(features.columns))
    if missing_features:
        raise ValueError(f"{label} v1.5.6 特征缺少列: {missing_features}")

    targets = None
    if target_path is not None:
        targets = pd.read_parquet(target_path).reset_index(drop=True)
        if list(targets.columns) != HORIZONS or len(targets) != len(meta):
            raise ValueError(f"{label} 目标结构不符合 v1.5.6 契约")

    raw_groups = {fips: group.reset_index(drop=True) for fips, group in raw.groupby("fipsCode", sort=True)}
    counties = sorted(meta["fipsCode"].unique().tolist())
    if set(counties) != set(raw_groups):
        raise ValueError(f"{label} 原始数据与 meta 县集合不一致")

    n_counties = len(counties)
    n_origins = TOTAL_HOURS - OBSERVED_HOURS
    observed = np.empty((n_counties, OBSERVED_HOURS, len(OBSERVED_CHANNELS)), dtype=np.float32)
    future = np.empty((n_counties, n_origins, len(WEATHER_CHANNELS)), dtype=np.float32)
    static_numeric = np.empty((n_counties, len(STATIC_NUMERIC_COLUMNS)), dtype=np.float32)
    static_category = np.empty(n_counties, dtype=np.float32)
    last_state = np.empty((n_counties, len(LAST_STATE_COLUMNS)), dtype=np.float32)
    row_index = np.empty((n_counties, n_origins), dtype=np.int64)
    target_values = np.full((n_counties, n_origins, len(HORIZONS)), np.nan, dtype=np.float32)

    reference_grid = None
    for county_idx, fips in enumerate(counties):
        group = raw_groups[fips]
        timestamp_grid = group["_timestamp"].to_numpy(dtype="datetime64[ns]")
        if reference_grid is None:
            reference_grid = timestamp_grid
        elif not np.array_equal(reference_grid, timestamp_grid):
            raise ValueError(f"{label} 县 {fips} 的时间网格与其他县不一致")

        weather = weather_matrix(group)
        outage = group[OUTAGE_CHANNELS].to_numpy(dtype=float)
        if np.isnan(outage[:OBSERVED_HOURS]).any():
            raise ValueError(f"{label} 县 {fips} 前72小时停电分量存在缺失")
        observed[county_idx] = np.column_stack([outage[:OBSERVED_HOURS], weather[:OBSERVED_HOURS]])
        future[county_idx] = weather[OBSERVED_HOURS:]

        county_meta = meta.loc[meta["fipsCode"] == fips]
        expected_hours = np.arange(OBSERVED_HOURS, TOTAL_HOURS)
        if len(county_meta) != n_origins or not np.array_equal(
            np.sort(county_meta["hour_idx"].to_numpy()), expected_hours
        ):
            raise ValueError(f"{label} 县 {fips} 的预测原点不完整")

        position_by_hour = dict(zip(county_meta["hour_idx"], county_meta.index))
        meta_positions = np.array([position_by_hour[h] for h in expected_hours], dtype=np.int64)
        row_index[county_idx] = meta_positions

        county_features = features.iloc[meta_positions]
        for column in STATIC_NUMERIC_COLUMNS + [STATIC_CATEGORICAL_COLUMN] + LAST_STATE_COLUMNS:
            values = pd.to_numeric(county_features[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size and not np.allclose(finite, finite[0], rtol=0, atol=1e-10):
                raise ValueError(f"{label} 县 {fips} 的静态列 {column} 随时间变化")

        static_numeric[county_idx] = county_features.iloc[0][STATIC_NUMERIC_COLUMNS].to_numpy(dtype=float)
        static_category[county_idx] = float(county_features.iloc[0][STATIC_CATEGORICAL_COLUMN])
        last_state[county_idx] = county_features.iloc[0][LAST_STATE_COLUMNS].to_numpy(dtype=float)
        if targets is not None:
            target_values[county_idx] = targets.iloc[meta_positions][HORIZONS].to_numpy(dtype=float)

    if label == "train" and n_counties != 239:
        raise ValueError(f"训练县数量应为239，实际为 {n_counties}")
    if label == "test" and n_counties != 63:
        raise ValueError(f"测试县数量应为63，实际为 {n_counties}")

    timestamps = np.array([str(pd.Timestamp(value)) for value in reference_grid], dtype="U32")
    np.savez_compressed(
        output_path,
        county_fips=np.asarray(counties, dtype="U5"),
        timestamps=timestamps,
        observed=observed,
        future_weather=future,
        static_numeric=static_numeric,
        static_category=static_category,
        last_state=last_state,
        row_index=row_index,
        targets=target_values,
    )
    return {
        "counties": n_counties,
        "observed_shape": list(observed.shape),
        "future_weather_shape": list(future.shape),
        "static_numeric_shape": list(static_numeric.shape),
        "last_state_shape": list(last_state.shape),
        "row_index_shape": list(row_index.shape),
        "targets_shape": list(target_values.shape),
        "observed_nan_cells": int(np.isnan(observed).sum()),
        "future_weather_nan_cells": int(np.isnan(future).sum()),
        "static_nan_cells": int(np.isnan(static_numeric).sum()),
        "valid_targets": {
            horizon: int(np.isfinite(target_values[:, :, idx]).sum())
            for idx, horizon in enumerate(HORIZONS)
        } if targets is not None else None,
    }


def verify_cache(path: Path, expected_counties: int, has_targets: bool) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "county_fips", "timestamps", "observed", "future_weather",
            "static_numeric", "static_category", "last_state", "row_index", "targets",
        }
        if set(data.files) != required:
            raise ValueError(f"{path.name} 数组键不一致: {data.files}")
        if data["observed"].shape != (expected_counties, 72, len(OBSERVED_CHANNELS)):
            raise ValueError(f"{path.name} observed 形状错误")
        if data["future_weather"].shape != (expected_counties, 144, len(WEATHER_CHANNELS)):
            raise ValueError(f"{path.name} future_weather 形状错误")
        if data["row_index"].shape != (expected_counties, 144):
            raise ValueError(f"{path.name} row_index 形状错误")
        if len(np.unique(data["row_index"])) != expected_counties * 144:
            raise ValueError(f"{path.name} row_index 不是一一映射")
        if has_targets:
            expected = [expected_counties * (144 - h) for h in [1, 6, 24, 48]]
            actual = [int(np.isfinite(data["targets"][:, :, k]).sum()) for k in range(4)]
            if actual != expected:
                raise ValueError(f"{path.name} 目标有效数错误: {actual} != {expected}")
        return {
            "sha256": sha256_file(path),
            "arrays": {key: list(data[key].shape) for key in data.files},
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    ensure_artifact_dirs()

    inputs = [
        TRAIN_RAW, TEST_RAW, FEATURE_TRAIN, FEATURE_TEST, TARGET_TRAIN,
        META_TRAIN, META_TEST, FEATURE_NAMES,
    ]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    build_needed = args.overwrite or not (TRAIN_SEQUENCE_CACHE.exists() and TEST_SEQUENCE_CACHE.exists())
    summaries = {}
    if build_needed:
        summaries["train"] = build_bundle(
            TRAIN_RAW, FEATURE_TRAIN, META_TRAIN, TRAIN_SEQUENCE_CACHE, TARGET_TRAIN, "train"
        )
        summaries["test"] = build_bundle(
            TEST_RAW, FEATURE_TEST, META_TEST, TEST_SEQUENCE_CACHE, None, "test"
        )

    verification = {
        "train": verify_cache(TRAIN_SEQUENCE_CACHE, 239, True),
        "test": verify_cache(TEST_SEQUENCE_CACHE, 63, False),
    }
    manifest = {
        "version": "v3.1",
        "evaluation_status": "diagnostic_only",
        "baseline_version": "v1.5.6",
        "observed_hours": [0, 71],
        "future_weather_hours": [72, 215],
        "observed_channels": OBSERVED_CHANNELS,
        "future_weather_channels": WEATHER_CHANNELS,
        "static_numeric_columns": STATIC_NUMERIC_COLUMNS,
        "static_categorical_column": STATIC_CATEGORICAL_COLUMN,
        "last_state_columns": LAST_STATE_COLUMNS,
        "horizons": HORIZONS,
        "input_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in inputs},
        "build_summary": summaries,
        "cache_verification": verification,
    }
    SEQUENCE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(SEQUENCE_MANIFEST), "verification": verification}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
