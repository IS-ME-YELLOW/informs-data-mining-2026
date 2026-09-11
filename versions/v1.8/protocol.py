"""v1.8 数据、标签、折、组合和提交协议。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CACHE_DIR,
    COMPONENT_MAX,
    COMPONENT_MIN,
    COMPONENT_WEIGHTS,
    COMPONENTS,
    CV_FILE,
    FEATURE_VERSION,
    HORIZON_HOURS,
    HORIZONS,
    N_FOLDS,
    OSI_MAX,
    RAW_TRAIN_FILE,
    SUBMISSION_TEMPLATE,
    TARGET_VALIDATION_FILE,
    TARGETS_FILE,
    ZERO_THRESHOLD,
)


@dataclass
class ExperimentData:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    meta_train: pd.DataFrame
    X_test: pd.DataFrame
    meta_test: pd.DataFrame
    folds: list[tuple[np.ndarray, np.ndarray]]
    row_folds: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_fips(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    if not normalized.str.fullmatch(r"\d{5}").all():
        raise ValueError("FIPS 格式异常")
    return normalized


def component_target_name(component: str, horizon: str) -> str:
    return f"{component}_target_{horizon.rsplit('_', 1)[-1]}"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not valid.any():
        raise ValueError("没有可评估的有限值")
    error = pred[valid] - truth[valid]
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "n": int(valid.sum()),
    }


def post_process(predictions: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(predictions, dtype=float), 0.0, OSI_MAX)
    return np.where(values < ZERO_THRESHOLD, 0.0, values)


def clip_component(predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(predictions, dtype=float), COMPONENT_MIN, COMPONENT_MAX)


def compose_osi(predictions: dict[str, np.ndarray]) -> np.ndarray:
    if set(predictions) != set(COMPONENTS):
        raise ValueError(f"分量集合错误: {sorted(predictions)}")
    shape = np.asarray(predictions[COMPONENTS[0]]).shape
    result = np.zeros(shape, dtype=float)
    finite = np.ones(shape, dtype=bool)
    for component in COMPONENTS:
        values = np.asarray(predictions[component], dtype=float)
        if values.shape != shape:
            raise ValueError("分量预测形状不一致")
        finite &= np.isfinite(values)
        result += COMPONENT_WEIGHTS[component] * values
    result[~finite] = np.nan
    return np.maximum(result, 0.0)


def _load_folds(meta_train: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    assignment = pd.read_csv(CV_FILE, dtype={"fipsCode": str, "stateAbbr": str})
    required = {"fipsCode", "stateAbbr", "severity_tier", "fold"}
    if set(assignment.columns) != required:
        raise ValueError("固定 CV 清单列错误")
    assignment["fipsCode"] = normalize_fips(assignment["fipsCode"])
    assignment["fold"] = pd.to_numeric(assignment["fold"], errors="raise").astype(int)
    assignment["severity_tier"] = pd.to_numeric(
        assignment["severity_tier"], errors="raise"
    ).astype(int)
    if assignment["fipsCode"].duplicated().any() or set(assignment["fold"]) != set(range(N_FOLDS)):
        raise ValueError("固定 CV 清单重复或折号不完整")

    counties = meta_train[["fipsCode", "stateAbbr", "severity_tier"]].copy()
    counties["fipsCode"] = normalize_fips(counties["fipsCode"])
    counties["stateAbbr"] = counties["stateAbbr"].astype(str)
    counties["severity_tier"] = pd.to_numeric(counties["severity_tier"], errors="raise").astype(int)
    counties = counties.drop_duplicates().set_index("fipsCode").sort_index()
    actual = assignment.set_index("fipsCode").sort_index()
    if not counties.index.equals(actual.index):
        raise ValueError("v1.5.6 县集合与固定 CV 清单不一致")
    if not counties[["stateAbbr", "severity_tier"]].equals(actual[["stateAbbr", "severity_tier"]]):
        raise ValueError("v1.5.6 CV 分层字段与固定清单不一致")
    counts = actual["fold"].value_counts()
    if counts.max() - counts.min() > 1:
        raise ValueError("固定 CV 县数不平衡")

    row_folds = normalize_fips(meta_train["fipsCode"]).map(actual["fold"])
    if row_folds.isna().any():
        raise ValueError("存在未分配折的训练行")
    row_folds_array = row_folds.to_numpy(dtype=int)
    folds = [
        (np.flatnonzero(row_folds_array != fold), np.flatnonzero(row_folds_array == fold))
        for fold in range(N_FOLDS)
    ]
    return folds, row_folds_array


def load_data() -> ExperimentData:
    names_path = CACHE_DIR / f"feature_names_{FEATURE_VERSION}.json"
    feature_names = json.loads(names_path.read_text(encoding="utf-8"))
    X_train = pd.read_parquet(CACHE_DIR / f"features_train_{FEATURE_VERSION}.parquet")
    y_train = pd.read_parquet(CACHE_DIR / f"targets_train_{FEATURE_VERSION}.parquet")
    meta_train = pd.read_parquet(CACHE_DIR / f"meta_train_{FEATURE_VERSION}.parquet")
    X_test = pd.read_parquet(CACHE_DIR / f"features_test_{FEATURE_VERSION}.parquet")
    meta_test = pd.read_parquet(CACHE_DIR / f"meta_test_{FEATURE_VERSION}.parquet")
    if list(X_train.columns) != feature_names or list(X_test.columns) != feature_names:
        raise ValueError("特征列与 v1.5.6 固定清单不一致")
    if list(y_train.columns) != list(HORIZONS):
        raise ValueError("官方目标列错误")
    if len(X_train) != len(y_train) or len(X_train) != len(meta_train) or len(X_test) != len(meta_test):
        raise ValueError("特征、目标和元信息行数不一致")
    for frame, label in [(X_train, "训练"), (X_test, "测试")]:
        if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in frame):
            raise ValueError(f"{label}特征包含非数值列")
        if np.isinf(frame.to_numpy(dtype=float, copy=False)).any():
            raise ValueError(f"{label}特征包含无穷值")
    for meta in (meta_train, meta_test):
        meta["fipsCode"] = normalize_fips(meta["fipsCode"])
        meta["timestamp_et"] = pd.to_datetime(meta["timestamp_et"])
        if meta[["fipsCode", "timestamp_et"]].duplicated().any():
            raise ValueError("元信息存在重复县时键")
    folds, row_folds = _load_folds(meta_train)
    return ExperimentData(
        X_train, y_train, meta_train, X_test, meta_test, folds, row_folds,
    )


def build_component_targets(data: ExperimentData, force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    if TARGETS_FILE.exists() and TARGET_VALIDATION_FILE.exists() and not force:
        targets = pd.read_parquet(TARGETS_FILE)
        validation = json.loads(TARGET_VALIDATION_FILE.read_text(encoding="utf-8"))
        if validation.get("raw_train_sha256") != sha256_file(RAW_TRAIN_FILE):
            raise ValueError("已保存分量标签相对原始训练数据过期")
        if validation.get("component_targets_sha256") != sha256_file(TARGETS_FILE):
            raise ValueError("已保存分量标签哈希不一致")
        validate_component_alignment(targets, data)
        return targets, validation

    raw = pd.read_csv(RAW_TRAIN_FILE, dtype={"fipsCode": str})
    raw["fipsCode"] = normalize_fips(raw["fipsCode"])
    raw["timestamp_et"] = pd.to_datetime(raw["timestamp_et"])
    if raw[["fipsCode", "timestamp_et"]].duplicated().any():
        raise ValueError("原始训练数据存在重复县时键")
    raw_components = raw.set_index(["fipsCode", "timestamp_et"])[list(COMPONENTS)]
    targets = data.meta_train[["fipsCode", "timestamp_et", "hour_idx"]].copy()
    validation: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "data/DM_Train.csv，县内按 horizon 平移",
        "formula": "max(0, 0.40*P_t + 0.35*N_t + 0.25*D_t - 0.10*R_t)",
        "raw_train_sha256": sha256_file(RAW_TRAIN_FILE),
        "horizons": {},
    }
    for horizon in HORIZONS:
        hours = HORIZON_HOURS[horizon]
        keys = pd.MultiIndex.from_arrays(
            [targets["fipsCode"], targets["timestamp_et"] + pd.to_timedelta(hours, unit="h")],
            names=["fipsCode", "timestamp_et"],
        )
        shifted = raw_components.reindex(keys).reset_index(drop=True)
        true_components: dict[str, np.ndarray] = {}
        for component in COMPONENTS:
            name = component_target_name(component, horizon)
            targets[name] = shifted[component].to_numpy(dtype=float)
            true_components[component] = targets[name].to_numpy(dtype=float)
        official = data.y_train[horizon].to_numpy(dtype=float)
        composed = compose_osi(true_components)
        official_valid = np.isfinite(official)
        component_valid = np.column_stack(
            [np.isfinite(true_components[component]) for component in COMPONENTS]
        ).all(axis=1)
        if not np.array_equal(official_valid, component_valid):
            raise ValueError(f"{horizon} 分量标签与官方标签缺失掩码不同")
        error = composed[official_valid] - official[official_valid]
        max_abs = float(np.max(np.abs(error)))
        if max_abs > 5.1e-5:
            raise ValueError(f"{horizon} 真实分量重组误差过大: {max_abs}")
        validation["horizons"][horizon] = {
            "valid_rows": int(official_valid.sum()),
            "formula_rmse_vs_official": float(np.sqrt(np.mean(error ** 2))),
            "formula_mae_vs_official": float(np.mean(np.abs(error))),
            "formula_max_abs_error_vs_official": max_abs,
            "component_ranges": {
                component: {
                    "min": float(np.nanmin(true_components[component])),
                    "max": float(np.nanmax(true_components[component])),
                    "mean": float(np.nanmean(true_components[component])),
                    "zero_pct": float(
                        100 * np.mean(true_components[component][official_valid] == 0)
                    ),
                }
                for component in COMPONENTS
            },
        }
    validate_component_alignment(targets, data)
    targets.to_parquet(TARGETS_FILE, index=False)
    validation["component_targets_sha256"] = sha256_file(TARGETS_FILE)
    TARGET_VALIDATION_FILE.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return targets, validation


def validate_component_alignment(targets: pd.DataFrame, data: ExperimentData) -> None:
    expected = ["fipsCode", "timestamp_et", "hour_idx"] + [
        component_target_name(component, horizon)
        for horizon in HORIZONS for component in COMPONENTS
    ]
    if list(targets.columns) != expected or len(targets) != len(data.meta_train):
        raise ValueError("v1.8 分量标签列或行数错误")
    if not normalize_fips(targets["fipsCode"]).equals(data.meta_train["fipsCode"]):
        raise ValueError("分量标签 FIPS 顺序与 v1.5.6 不一致")
    if not pd.to_datetime(targets["timestamp_et"]).equals(data.meta_train["timestamp_et"]):
        raise ValueError("分量标签时间顺序与 v1.5.6 不一致")


def align_same_target(meta: pd.DataFrame, predictions: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """将四个 horizon 对同一目标时刻的预测等权平均，再映射回各 horizon。"""
    records = []
    for horizon in HORIZONS:
        values = np.asarray(predictions[horizon], dtype=float)
        valid = np.isfinite(values) & (meta["hour_idx"].to_numpy() + HORIZON_HOURS[horizon] <= 215)
        records.append(pd.DataFrame({
            "fipsCode": meta.loc[valid, "fipsCode"].to_numpy(),
            "target_timestamp": meta.loc[valid, "timestamp_et"].to_numpy()
                + pd.to_timedelta(HORIZON_HOURS[horizon], unit="h"),
            "prediction": values[valid],
        }))
    long = pd.concat(records, ignore_index=True)
    unique = long.groupby(["fipsCode", "target_timestamp"], as_index=False, sort=True).agg(
        prediction=("prediction", "mean"), candidate_count=("prediction", "size")
    )
    expected_counts = {1: 5, 2: 18, 3: 24, 4: 96}
    actual_counts = unique["candidate_count"].value_counts().to_dict()
    county_count = meta["fipsCode"].nunique()
    if actual_counts != {k: v * county_count for k, v in expected_counts.items()}:
        raise ValueError(f"目标小时候选数量结构错误: {actual_counts}")
    lookup = unique.set_index(["fipsCode", "target_timestamp"])["prediction"]
    mapped: dict[str, np.ndarray] = {}
    for horizon in HORIZONS:
        target_time = meta["timestamp_et"] + pd.to_timedelta(HORIZON_HOURS[horizon], unit="h")
        keys = pd.MultiIndex.from_arrays([meta["fipsCode"], target_time])
        values = lookup.reindex(keys).to_numpy(dtype=float).copy()
        invalid = meta["hour_idx"].to_numpy() + HORIZON_HOURS[horizon] > 215
        values[invalid] = np.nan
        mapped[horizon] = values
    return mapped, unique


def fill_submission(meta_test: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE, dtype={"fipsCode": str})
    keys = pd.DataFrame({
        "fipsCode": normalize_fips(submission["fipsCode"]),
        "timestamp_et": pd.to_datetime(submission["timestamp_et"]),
    })
    frame = meta_test[["fipsCode", "timestamp_et", "hour_idx"]].copy()
    for horizon in HORIZONS:
        values = np.asarray(predictions[horizon], dtype=float).copy()
        if len(values) != len(frame):
            raise ValueError("测试预测长度错误")
        invalid = frame["hour_idx"].to_numpy() + HORIZON_HOURS[horizon] > 215
        values[invalid] = np.nan
        frame[horizon] = values
    aligned = frame.set_index(["fipsCode", "timestamp_et"]).reindex(pd.MultiIndex.from_frame(keys))
    if aligned["hour_idx"].isna().any():
        raise ValueError("提交模板存在无法匹配的键")
    for horizon in HORIZONS:
        submission[horizon] = aligned[horizon].to_numpy()
    counties = submission["fipsCode"].nunique()
    expected = {h: counties * HORIZON_HOURS[h] for h in HORIZONS}
    actual = {h: int(submission[h].isna().sum()) for h in HORIZONS}
    if actual != expected:
        raise ValueError(f"提交尾部 NaN 数量错误: {actual}")
    return submission


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
