"""独立重载 v1.9 全部模型，复算候选预测、汇总指标和提交。"""

from __future__ import annotations

import json
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    BASE_FEATURE_VERSION,
    CACHE_DIR,
    HORIZONS,
    PROJECT_ROOT,
    RESPONSE_FEATURES,
    V18_DIR,
    V19_DIR,
)
from protocol import (
    build_component_targets,
    clip_component,
    component_target_name,
    fill_submission,
    load_data,
    metrics,
    sha256_file,
)
from train_v19 import COMPONENTS, REGRESSION_CANDIDATES, candidate_key

ATOL = 1e-10


def max_diff(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise AssertionError("数组形状或 NaN 掩码不一致")
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0


def manifest_row(
    manifest: pd.DataFrame,
    role: str,
    feature_set: str,
    component: str,
    candidate: str,
    horizon: str,
    stage: str,
    fold: int | None = None,
) -> pd.Series:
    hit = (
        (manifest["role"] == role)
        & (manifest["feature_set"] == feature_set)
        & (manifest["component"] == component)
        & (manifest["candidate"] == candidate)
        & (manifest["horizon"] == horizon)
        & (manifest["stage"] == stage)
    )
    if role == "cv":
        hit &= manifest["fold"].eq(float(fold))
    rows = manifest[hit]
    if len(rows) != 1:
        raise AssertionError(f"模型清单匹配异常: {role}/{feature_set}/{component}/{candidate}/{horizon}/{stage}/{fold}")
    return rows.iloc[0]


def load_booster(row: pd.Series) -> lgb.Booster:
    path = PROJECT_ROOT / str(row["path"])
    if not path.exists() or sha256_file(path) != row["sha256"]:
        raise AssertionError(f"模型文件缺失或哈希错误: {path}")
    booster = lgb.Booster(model_file=str(path))
    if booster.num_trees() != int(row["rounds"]):
        raise AssertionError(f"模型轮数错误: {path}")
    return booster


def main() -> None:
    started = time.perf_counter()
    data = load_data()
    targets, _ = build_component_targets(data)
    manifest = pd.read_csv(V19_DIR / "model_manifest.csv")
    if len(manifest) != 576:
        raise AssertionError(f"预期576个模型清单项，实际{len(manifest)}")
    for _, row in manifest.iterrows():
        path = PROJECT_ROOT / str(row["path"])
        if not path.exists() or sha256_file(path) != row["sha256"]:
            raise AssertionError(f"模型哈希错误: {path}")

    base_names = json.loads(
        (CACHE_DIR / f"feature_names_{BASE_FEATURE_VERSION}.json").read_text(encoding="utf-8")
    )
    feature_sets = {
        "base": (data.X_train[base_names], data.X_test[base_names]),
        "profile": (data.X_train, data.X_test),
    }
    saved_oof = pd.read_parquet(V19_DIR / "oof_component_candidates.parquet")
    saved_test = pd.read_parquet(V19_DIR / "test_component_candidates.parquet")
    saved_v18_oof = pd.read_parquet(V18_DIR / "oof_component_predictions.parquet")
    saved_v18_test = pd.read_parquet(V18_DIR / "test_component_predictions.parquet")
    discrepancies: list[float] = []
    checked_candidates = 0

    for feature_set in ("base", "profile"):
        X_train, X_test = feature_sets[feature_set]
        for component in COMPONENTS:
            for candidate in REGRESSION_CANDIDATES[component]:
                for horizon in HORIZONS:
                    key = candidate_key(feature_set, component, candidate, horizon)
                    saved_oof_values = saved_oof[f"pred__{key}"].to_numpy(dtype=float)
                    saved_test_values = saved_test[f"pred__{key}"].to_numpy(dtype=float)
                    target = component_target_name(component, horizon)
                    y = targets[target].to_numpy(dtype=float)
                    valid = np.isfinite(y)

                    if feature_set == "base" and candidate == "huber":
                        discrepancies.append(max_diff(
                            saved_oof_values, saved_v18_oof[f"pred_{target}"].to_numpy(dtype=float)
                        ))
                        discrepancies.append(max_diff(
                            saved_test_values, saved_v18_test[f"pred_{target}"].to_numpy(dtype=float)
                        ))
                        checked_candidates += 1
                        continue

                    recomputed_oof = np.full(len(y), np.nan, dtype=float)
                    for fold in range(5):
                        idx = np.flatnonzero((data.row_folds == fold) & valid)
                        if candidate == "hurdle":
                            cls = load_booster(manifest_row(
                                manifest, "cv", feature_set, component, candidate, horizon,
                                "classification", fold,
                            ))
                            mag = load_booster(manifest_row(
                                manifest, "cv", feature_set, component, candidate, horizon,
                                "magnitude", fold,
                            ))
                            pred = np.clip(cls.predict(X_train.iloc[idx]), 0, 1) * np.maximum(
                                mag.predict(X_train.iloc[idx]), 0
                            )
                        else:
                            booster = load_booster(manifest_row(
                                manifest, "cv", feature_set, component, candidate, horizon,
                                "regression", fold,
                            ))
                            pred = booster.predict(X_train.iloc[idx])
                        recomputed_oof[idx] = clip_component(pred)

                    if candidate == "hurdle":
                        cls = load_booster(manifest_row(
                            manifest, "final", feature_set, component, candidate, horizon,
                            "classification",
                        ))
                        mag = load_booster(manifest_row(
                            manifest, "final", feature_set, component, candidate, horizon,
                            "magnitude",
                        ))
                        recomputed_test = np.clip(cls.predict(X_test), 0, 1) * np.maximum(
                            mag.predict(X_test), 0
                        )
                    else:
                        booster = load_booster(manifest_row(
                            manifest, "final", feature_set, component, candidate, horizon,
                            "regression",
                        ))
                        recomputed_test = booster.predict(X_test)
                    recomputed_test = clip_component(recomputed_test)
                    discrepancies.append(max_diff(recomputed_oof, saved_oof_values))
                    discrepancies.append(max_diff(recomputed_test, saved_test_values))
                    checked_candidates += 1

    max_prediction_discrepancy = max(discrepancies)
    if max_prediction_discrepancy > ATOL:
        raise AssertionError(f"模型重载预测差异过大: {max_prediction_discrepancy}")

    # 复算命名 OOF 的汇总指标。
    named_oof = pd.read_parquet(V19_DIR / "oof_predictions.parquet")
    summary = pd.read_csv(V19_DIR / "summary_metrics.csv")
    summary_diffs = []
    for _, row in summary.iterrows():
        pred = named_oof[f"pred__{row['model']}__{row['horizon']}"]
        truth = named_oof[f"actual__{row['horizon']}"]
        score = metrics(truth, pred)
        summary_diffs.extend([
            abs(score["rmse"] - float(row["rmse"])),
            abs(score["mae"] - float(row["mae"])),
        ])
    max_summary_discrepancy = float(max(summary_diffs))
    if max_summary_discrepancy > ATOL:
        raise AssertionError(f"汇总指标复算差异过大: {max_summary_discrepancy}")

    # 复算全部候选提交。
    named_test = pd.read_parquet(V19_DIR / "test_predictions.parquet")
    submission_models = [
        "E1_best_subset", "E2_profile_huber_all", "E3_best_subset",
        "E1_aligned", "E3_aligned", "V19_best_by_horizon",
    ]
    submission_discrepancies = []
    nan_counts = {}
    for model in submission_models:
        predictions = {
            horizon: named_test[f"pred__{model}__{horizon}"].to_numpy(dtype=float)
            for horizon in HORIZONS
        }
        recomputed = fill_submission(data.meta_test, predictions)
        saved = pd.read_csv(
            V19_DIR / f"diagnostic_submission_v1.9_{model}_balanced_v1.csv",
            dtype={"fipsCode": str},
        )
        for key in ("fipsCode", "countyName", "stateAbbr", "timestamp_et"):
            if not saved[key].astype(str).equals(recomputed[key].astype(str)):
                raise AssertionError(f"{model} 提交标识列改变: {key}")
        nan_counts[model] = {}
        for horizon in HORIZONS:
            submission_discrepancies.append(max_diff(saved[horizon], recomputed[horizon]))
            nan_counts[model][horizon] = int(saved[horizon].isna().sum())
    max_submission_discrepancy = max(submission_discrepancies)
    if max_submission_discrepancy > ATOL:
        raise AssertionError(f"提交复算差异过大: {max_submission_discrepancy}")

    feature_validation = json.loads((V19_DIR / "feature_validation.json").read_text(encoding="utf-8"))
    for name, expected in feature_validation["outputs_sha256"].items():
        path = V19_DIR / name
        if sha256_file(path) != expected:
            raise AssertionError(f"特征资产哈希错误: {path}")

    result = {
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed",
        "models_hashed": int(len(manifest)),
        "candidate_sets_recomputed": checked_candidates,
        "max_prediction_discrepancy": max_prediction_discrepancy,
        "max_summary_discrepancy": max_summary_discrepancy,
        "max_submission_discrepancy": max_submission_discrepancy,
        "feature_outputs_hashed": len(feature_validation["outputs_sha256"]),
        "base_feature_count": len(base_names),
        "response_feature_count": len(RESPONSE_FEATURES),
        "submission_nan_counts": nan_counts,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (V19_DIR / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
