"""Leakage-safe grouped-CV runner for direct-OSI sklearn forest baselines."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "versions"))
sys.path.insert(0, str(PROJECT_ROOT / "versions/v1.11_tree_ensemble"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from gbdt_experiment import HORIZONS, load_experiment_data, post_process  # noqa: E402
from ensemble_utils import (  # noqa: E402
    hash_tree,
    metric_rows,
    metric_values,
    paired_county_bootstrap,
    sha256_file,
    write_json,
)

FEATURE_VERSION = "v1.5.6"
CV_VERSION = "balanced_v1"
SEED = 42
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_REPLICATES = 2000
DEFAULT_HORIZONS = HORIZONS[2:]
REFERENCE_OOF = PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/oof_predictions.parquet"
REFERENCE_TEST = PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/test_predictions.csv"
REFERENCE_SUBMISSION = PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/submission_v1.12_tail_protected_balanced_v1.csv"


def _sha(path: Path) -> str:
    return sha256_file(path)


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["fipsCode"] = out["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    out["timestamp_et"] = pd.to_datetime(out["timestamp_et"])
    out["hour_idx"] = pd.to_numeric(out["hour_idx"], errors="raise").astype(int)
    out["stateAbbr"] = out["stateAbbr"].astype(str)
    return out


def _assert_keys(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    cols = ["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]
    a, b = _normalize_keys(left[cols]), _normalize_keys(right[cols])
    if len(a) != len(b) or any(not np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in cols):
        raise ValueError(f"{label} is not exactly row/key aligned")


def configurations(family: str) -> dict[str, dict[str, Any]]:
    common = dict(n_estimators=600, criterion="squared_error", random_state=SEED, n_jobs=-1)
    if family == "extratrees":
        return {
            "ET-A": {**common, "max_features": 0.5, "min_samples_leaf": 5, "bootstrap": False},
            "ET-B": {**common, "max_features": 0.7, "min_samples_leaf": 20, "bootstrap": False},
        }
    if family == "random_forest":
        return {
            "RF": {**common, "max_features": 0.5, "min_samples_leaf": 5, "bootstrap": True, "max_samples": 0.8}
        }
    raise ValueError(f"Unknown family: {family}")


def build_pipeline(family: str, params: dict[str, Any]) -> Pipeline:
    estimator = ExtraTreesRegressor(**params) if family == "extratrees" else RandomForestRegressor(**params)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def model_path(models_dir: Path, config: str, horizon: str, fold: int | str) -> Path:
    return models_dir / f"{config.lower().replace('-', '_')}_{horizon}_{fold}.joblib"


def compare_reference(data: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ref = pd.read_parquet(REFERENCE_OOF)
    _assert_keys(data.meta_train, ref, "v1.12 OOF")
    expected = np.full(len(ref), -1, dtype=int)
    for fold, (_, valid) in enumerate(data.folds):
        expected[valid] = fold
    if not np.array_equal(ref["fold"].to_numpy(int), expected):
        raise ValueError("v1.12 fold labels do not match balanced_v1")
    test_ref = pd.read_csv(REFERENCE_TEST, dtype={"fipsCode": str})
    _assert_keys(data.meta_test, test_ref, "v1.12 test predictions")
    return ref, test_ref, {
        "train_rows": len(ref), "test_rows": len(test_ref), "feature_version": FEATURE_VERSION,
        "feature_count": data.X_train.shape[1], "cv_version": CV_VERSION,
        "keys_exact": True, "folds_exact": True,
    }


def run(version_dir: Path, family: str, horizons: tuple[str, ...] = DEFAULT_HORIZONS) -> None:
    started = time.perf_counter()
    artifacts = version_dir / "artifacts"
    models = version_dir / "models"
    results = version_dir / "results"
    for directory in (artifacts, models, results):
        directory.mkdir(parents=True, exist_ok=True)
    data = load_experiment_data(FEATURE_VERSION)
    reference, reference_test, alignment = compare_reference(data)
    configs = configurations(family)
    out = _normalize_keys(data.meta_train[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]])
    fold_ids = np.full(len(out), -1, dtype=int)
    for fold, (_, valid) in enumerate(data.folds):
        fold_ids[valid] = fold
    out["fold"] = fold_ids
    test_out = _normalize_keys(data.meta_test[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]])
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    county_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    for hi, horizon in enumerate(horizons):
        y = data.y_train[horizon].to_numpy(float)
        valid_target = np.isfinite(y)
        baseline = reference[f"pred_v1.8_rule_{horizon}"].to_numpy(float)
        v112 = reference[f"pred_tail_protected_{horizon}"].to_numpy(float)
        out[f"actual_{horizon}"] = y
        out[f"pred_v1.8_rule_{horizon}"] = baseline
        out[f"pred_v1.12_{horizon}"] = v112
        horizon_predictions: dict[str, np.ndarray] = {"v1.8_rule": baseline, "v1.12": v112}
        for config_name, params in configs.items():
            oof = np.full(len(y), np.nan)
            for fold, (train_full, valid_full) in enumerate(data.folds):
                train = train_full[valid_target[train_full]]
                valid = valid_full[valid_target[valid_full]]
                pipeline = build_pipeline(family, params)
                fit_start = time.perf_counter()
                pipeline.fit(data.X_train.iloc[train], y[train])
                pred = post_process(pipeline.predict(data.X_train.iloc[valid]))
                oof[valid] = pred
                path = model_path(models, config_name, horizon, f"fold{fold}")
                joblib.dump(pipeline, path, compress=3)
                score = metric_values(y[valid], pred)
                fit_rows.append({"config": config_name, "horizon": horizon, "fold": fold, "train_rows": len(train), "valid_rows": len(valid), "fit_seconds": time.perf_counter() - fit_start, "model_bytes": path.stat().st_size, **score})
                print(f"{config_name} {horizon} fold={fold} RMSE={score['rmse']:.8f} seconds={fit_rows[-1]['fit_seconds']:.1f}", flush=True)
                del pipeline
                gc.collect()
            if np.isnan(oof[valid_target]).any() or np.isfinite(oof[~valid_target]).any():
                raise ValueError(f"Invalid OOF mask: {config_name}/{horizon}")
            horizon_predictions[config_name] = oof
            out[f"pred_{config_name}_{horizon}"] = oof
            final = build_pipeline(family, params)
            fit_start = time.perf_counter()
            final.fit(data.X_train.loc[valid_target], y[valid_target])
            final_path = model_path(models, config_name, horizon, "final")
            joblib.dump(final, final_path, compress=3)
            test_pred = post_process(final.predict(data.X_test))
            test_out[f"pred_{config_name}_{horizon}"] = test_pred
            fit_rows.append({"config": config_name, "horizon": horizon, "fold": "final", "train_rows": int(valid_target.sum()), "valid_rows": len(data.X_test), "fit_seconds": time.perf_counter() - fit_start, "model_bytes": final_path.stat().st_size})
            values = final.named_steps["model"].feature_importances_
            for rank, idx in enumerate(np.argsort(-values), 1):
                importance_rows.append({"config": config_name, "horizon": horizon, "rank": rank, "feature": data.X_train.columns[idx], "importance": float(values[idx])})
            del final
            gc.collect()

        base_metric = metric_values(y, baseline)
        threshold = float(np.nanquantile(y, 0.95))
        for name, pred in horizon_predictions.items():
            score = metric_values(y, pred)
            summary_rows.append({"horizon": horizon, "model": name, **score, "rmse_change_pct_vs_v1.8": 100 * (score["rmse"] / base_metric["rmse"] - 1), "mae_change_pct_vs_v1.8": 100 * (score["mae"] / base_metric["mae"] - 1)})
            tail_rows.append({"horizon": horizon, "model": name, "subset": "actual_top5", "actual_threshold": threshold, **metric_values(y[y >= threshold], pred[y >= threshold])})
        fold_rows.extend(metric_rows(out, y, horizon_predictions, horizon, "fold"))
        county_rows.extend(metric_rows(out, y, horizon_predictions, horizon, "fipsCode"))
        for ci, config_name in enumerate(configs):
            pred = horizon_predictions[config_name]
            bootstrap_rows.append({"horizon": horizon, "model": config_name, **paired_county_bootstrap(out, y, baseline, pred, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + hi * 10 + ci)})
            valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(baseline)
            corr_rows.append({"horizon": horizon, "model_a": config_name, "model_b": "v1.8_rule", "residual_correlation": float(np.corrcoef(y[valid] - pred[valid], y[valid] - baseline[valid])[0, 1])})
            corr_rows.append({"horizon": horizon, "model_a": config_name, "model_b": "v1.12", "residual_correlation": float(np.corrcoef(y[valid] - pred[valid], y[valid] - v112[valid])[0, 1])})
            disagreement_rows.append({"horizon": horizon, "model_a": config_name, "model_b": "v1.12", "mean_abs_prediction_difference": float(np.mean(np.abs(pred[valid] - v112[valid]))), "rmse_prediction_difference": float(np.sqrt(np.mean((pred[valid] - v112[valid]) ** 2)))})
        names = list(configs)
        if len(names) == 2:
            valid = np.isfinite(y)
            corr_rows.append({"horizon": horizon, "model_a": names[0], "model_b": names[1], "residual_correlation": float(np.corrcoef(y[valid] - horizon_predictions[names[0]][valid], y[valid] - horizon_predictions[names[1]][valid])[0, 1])})

    pd.DataFrame(summary_rows).to_csv(results / "summary_metrics.csv", index=False)
    folds_df = pd.DataFrame(fold_rows)
    base_folds = folds_df[folds_df.model == "v1.8_rule"][["horizon", "fold", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    folds_df = folds_df.merge(base_folds, on=["horizon", "fold"], how="left")
    folds_df["rmse_change_pct_vs_v1.8"] = 100 * (folds_df.rmse / folds_df.baseline_rmse - 1)
    folds_df["improved_vs_v1.8"] = folds_df.rmse < folds_df.baseline_rmse
    folds_df.to_csv(results / "fold_metrics.csv", index=False)
    pd.DataFrame(county_rows).to_csv(results / "county_metrics.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(results / "high_osi_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(results / "paired_county_bootstrap.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(results / "residual_correlations.csv", index=False)
    pd.DataFrame(disagreement_rows).to_csv(results / "prediction_disagreement.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(results / "feature_importance.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(results / "fit_runtime.csv", index=False)
    out.to_parquet(artifacts / "oof_predictions.parquet", index=False)
    test_out.to_parquet(artifacts / "test_predictions.parquet", index=False)
    for config_name in configs:
        submission = pd.read_csv(REFERENCE_SUBMISSION, dtype={"fipsCode": str})
        for horizon in horizons:
            mask = submission[horizon].notna().to_numpy()
            values = test_out[f"pred_{config_name}_{horizon}"].to_numpy(float)
            submission.loc[mask, horizon] = values[mask]
        submission.to_csv(artifacts / f"submission_{config_name}_{CV_VERSION}.csv", index=False)
    write_json(artifacts / "alignment_report.json", alignment)
    generated = list(models.glob("*.joblib")) + list(results.glob("*")) + [artifacts / "oof_predictions.parquet", artifacts / "test_predictions.parquet", artifacts / "alignment_report.json"] + list(artifacts.glob("submission_*.csv"))
    inputs = [REFERENCE_OOF, REFERENCE_TEST, REFERENCE_SUBMISSION, PROJECT_ROOT / f"cache/features_train_{FEATURE_VERSION}.parquet", PROJECT_ROOT / f"cache/targets_train_{FEATURE_VERSION}.parquet", PROJECT_ROOT / f"cache/meta_train_{FEATURE_VERSION}.parquet", PROJECT_ROOT / f"cache/features_test_{FEATURE_VERSION}.parquet", PROJECT_ROOT / f"cache/meta_test_{FEATURE_VERSION}.parquet", PROJECT_ROOT / f"cache/feature_names_{FEATURE_VERSION}.json", PROJECT_ROOT / "cv/cv_assignments_balanced_v1_seed42.csv"]
    metadata = {"experiment_version": version_dir.name, "family": family, "feature_version": FEATURE_VERSION, "feature_count": data.X_train.shape[1], "cv_version": CV_VERSION, "seed": SEED, "horizons": list(horizons), "configs": configs, "pipeline": ["SimpleImputer(strategy=median)", family], "imputer_fit_scope": "each outer training fold only; final imputer uses all finite-target training rows for test inference", "runtime_seconds": time.perf_counter() - started, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"), "python": sys.version, "platform": platform.platform(), "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}, "input_hashes": {str(p.relative_to(PROJECT_ROOT)).replace('\\', '/'): _sha(p) for p in inputs}, "artifact_hashes": hash_tree(generated, PROJECT_ROOT)}
    write_json(artifacts / "run_metadata.json", metadata)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


def cli(version_dir: Path, family: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", choices=HORIZONS, default=list(DEFAULT_HORIZONS))
    args = parser.parse_args()
    run(version_dir, family, tuple(args.horizons))
