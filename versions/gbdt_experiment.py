"""Shared, leakage-safe runner for the v1.6/v1.7 GBDT model-family trials.

The runner deliberately loads the frozen v1.5.2 feature cache. It never calls
the historical feature builder, whose current source no longer reproduces that
cache. Model-specific behavior is supplied by the small adapters in v1.6 and
v1.7; data validation, fixed county folds, scoring, post-processing, and output
validation stay identical across both experiments.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = PROJECT_ROOT / ".python_packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


CACHE_DIR = PROJECT_ROOT / "cache"
CV_FILE = PROJECT_ROOT / "cv" / "cv_assignments_balanced_v1_seed42.csv"
SUBMISSION_TEMPLATE = PROJECT_ROOT / "data" / "sample_submission.csv"
FEATURE_VERSION = "v1.5.2"
CV_VERSION = "balanced_v1"
SEED = 42
N_FOLDS = 5
OSI_MAX_OBSERVED = 0.65
ZERO_THRESHOLD = 0.001
MAX_BOOST_ROUNDS = 2_000
EARLY_STOPPING_ROUNDS = 100
HORIZONS = (
    "osi_target_t01h",
    "osi_target_t06h",
    "osi_target_t24h",
    "osi_target_t48h",
)
HORIZON_HOURS = {
    "osi_target_t01h": 1,
    "osi_target_t06h": 6,
    "osi_target_t24h": 24,
    "osi_target_t48h": 48,
}


class ModelAdapter(Protocol):
    family: str
    slug: str
    version: str
    model_extension: str

    def config(self, max_rounds: int, early_stopping_rounds: int) -> dict[str, Any]: ...
    def library_versions(self) -> dict[str, str]: ...
    def build_cv_model(self, max_rounds: int, early_stopping_rounds: int) -> Any: ...
    def fit_cv(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
               X_valid: pd.DataFrame, y_valid: pd.Series) -> None: ...
    def best_iteration(self, model: Any) -> int: ...
    def build_final_model(self, n_rounds: int) -> Any: ...
    def fit_final(self, model: Any, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray: ...
    def feature_importance(self, model: Any, feature_names: list[str]) -> np.ndarray: ...
    def save_model(self, model: Any, path: Path) -> None: ...


@dataclass
class ExperimentData:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    meta_train: pd.DataFrame
    X_test: pd.DataFrame
    meta_test: pd.DataFrame
    folds: list[tuple[np.ndarray, np.ndarray]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_fips(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    if normalized.str.fullmatch(r"\d{5}").eq(False).any():
        bad = normalized[normalized.str.fullmatch(r"\d{5}").eq(False)].head().tolist()
        raise ValueError(f"Invalid FIPS values: {bad}")
    return normalized


def _validate_numeric_features(X: pd.DataFrame, label: str) -> None:
    non_numeric = [name for name in X if not pd.api.types.is_numeric_dtype(X[name])]
    if non_numeric:
        raise ValueError(f"{label} contains non-numeric model features: {non_numeric}")
    values = X.to_numpy(dtype=float, copy=False)
    if np.isinf(values).any():
        raise ValueError(f"{label} contains infinite values.")


def _load_folds(meta_train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    assignment = pd.read_csv(CV_FILE, dtype={"fipsCode": str, "stateAbbr": str})
    expected_columns = {"fipsCode", "stateAbbr", "severity_tier", "fold"}
    if set(assignment.columns) != expected_columns:
        raise ValueError(f"Unexpected CV columns: {assignment.columns.tolist()}")
    assignment["fipsCode"] = _normalize_fips(assignment["fipsCode"])
    assignment["fold"] = pd.to_numeric(assignment["fold"], errors="raise").astype(int)
    assignment["severity_tier"] = pd.to_numeric(
        assignment["severity_tier"], errors="raise"
    ).astype(int)
    if assignment["fipsCode"].duplicated().any():
        raise ValueError("The fixed CV manifest contains duplicate FIPS codes.")
    if set(assignment["fold"]) != set(range(N_FOLDS)):
        raise ValueError("The fixed CV manifest does not contain folds 0..4.")

    meta_counties = meta_train.loc[
        :, ["fipsCode", "stateAbbr", "severity_tier"]
    ].copy()
    meta_counties["fipsCode"] = _normalize_fips(meta_counties["fipsCode"])
    meta_counties["stateAbbr"] = meta_counties["stateAbbr"].astype(str)
    meta_counties["severity_tier"] = pd.to_numeric(
        meta_counties["severity_tier"], errors="raise"
    ).astype(int)
    meta_counties = meta_counties.drop_duplicates()
    if meta_counties["fipsCode"].duplicated().any():
        raise ValueError("A training FIPS code maps to multiple CV strata.")

    expected = meta_counties.set_index("fipsCode").sort_index()
    actual = assignment.set_index("fipsCode").sort_index()
    if not expected.index.equals(actual.index):
        raise ValueError("CV manifest and v1.5.2 training metadata have different FIPS sets.")
    if not expected[["stateAbbr", "severity_tier"]].equals(
        actual[["stateAbbr", "severity_tier"]]
    ):
        raise ValueError("CV strata disagree with v1.5.2 training metadata.")

    fold_counts = assignment["fold"].value_counts().sort_index()
    if fold_counts.max() - fold_counts.min() > 1:
        raise ValueError(f"CV county counts are imbalanced: {fold_counts.to_dict()}")
    fold_by_fips = assignment.set_index("fipsCode")["fold"]
    row_folds = _normalize_fips(meta_train["fipsCode"]).map(fold_by_fips)
    if row_folds.isna().any():
        raise ValueError("At least one training row has no fixed CV assignment.")
    row_folds_array = row_folds.to_numpy(dtype=int)
    return [
        (
            np.flatnonzero(row_folds_array != fold),
            np.flatnonzero(row_folds_array == fold),
        )
        for fold in range(N_FOLDS)
    ]


def load_experiment_data() -> ExperimentData:
    names_path = CACHE_DIR / f"feature_names_{FEATURE_VERSION}.json"
    with names_path.open("r", encoding="utf-8") as handle:
        feature_names = json.load(handle)

    X_train = pd.read_parquet(CACHE_DIR / f"features_train_{FEATURE_VERSION}.parquet")
    y_train = pd.read_parquet(CACHE_DIR / f"targets_train_{FEATURE_VERSION}.parquet")
    meta_train = pd.read_parquet(CACHE_DIR / f"meta_train_{FEATURE_VERSION}.parquet")
    X_test = pd.read_parquet(CACHE_DIR / f"features_test_{FEATURE_VERSION}.parquet")
    meta_test = pd.read_parquet(CACHE_DIR / f"meta_test_{FEATURE_VERSION}.parquet")

    if list(X_train.columns) != feature_names or list(X_test.columns) != feature_names:
        raise ValueError("Feature columns do not exactly match the frozen v1.5.2 name manifest.")
    if list(y_train.columns) != list(HORIZONS):
        raise ValueError(f"Unexpected target columns: {y_train.columns.tolist()}")
    if len(X_train) != len(y_train) or len(X_train) != len(meta_train):
        raise ValueError("Training feature, target, and metadata row counts differ.")
    if len(X_test) != len(meta_test):
        raise ValueError("Test feature and metadata row counts differ.")
    if meta_train[["fipsCode", "timestamp_et"]].duplicated().any():
        raise ValueError("Training metadata contains duplicate county-time keys.")
    if meta_test[["fipsCode", "timestamp_et"]].duplicated().any():
        raise ValueError("Test metadata contains duplicate county-time keys.")
    _validate_numeric_features(X_train, "Training matrix")
    _validate_numeric_features(X_test, "Test matrix")

    # Both libraries internally train histogram trees on float32 values. Making
    # that conversion once reduces memory without changing the protocol.
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    folds = _load_folds(meta_train)
    return ExperimentData(X_train, y_train, meta_train, X_test, meta_test, folds)


def post_process(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    values = np.clip(values, 0.0, OSI_MAX_OBSERVED)
    return np.where(values < ZERO_THRESHOLD, 0.0, values)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(pred)
    if not mask.any():
        raise ValueError("No finite rows are available for metric computation.")
    errors = truth[mask] - pred[mask]
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
    }


def compute_baselines(data: ExperimentData) -> dict[str, dict[str, dict[str, float]]]:
    last_osi = data.X_train["last_osi"].to_numpy(dtype=float)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for horizon in HORIZONS:
        y = data.y_train[horizon].to_numpy(dtype=float)
        horizon_results: dict[str, dict[str, float]] = {}
        for baseline in ("Zero", "Mean", "Persistence"):
            oof = np.full(len(y), np.nan)
            fold_scores = []
            for train_idx, valid_idx in data.folds:
                if baseline == "Zero":
                    pred = np.zeros(len(valid_idx))
                elif baseline == "Mean":
                    pred = np.full(len(valid_idx), np.nanmean(y[train_idx]))
                else:
                    pred = last_osi[valid_idx]
                oof[valid_idx] = pred
                fold_scores.append(compute_metrics(y[valid_idx], pred))
            pooled = compute_metrics(y, oof)
            horizon_results[baseline] = {
                **pooled,
                "rmse_std": float(np.std([score["rmse"] for score in fold_scores])),
                "mae_std": float(np.std([score["mae"] for score in fold_scores])),
            }
        results[horizon] = horizon_results
    return results


def _fill_submission(meta_test: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE, dtype={"fipsCode": str})
    required = {"fipsCode", "timestamp_et", *HORIZONS}
    if not required.issubset(submission.columns):
        raise ValueError(f"Submission template is missing: {sorted(required - set(submission))}")

    pred_frame = meta_test.loc[:, ["fipsCode", "timestamp_et", "hour_idx"]].copy()
    pred_frame["fipsCode"] = _normalize_fips(pred_frame["fipsCode"])
    pred_frame["timestamp_et"] = pd.to_datetime(pred_frame["timestamp_et"])
    for horizon in HORIZONS:
        values = np.asarray(predictions[horizon], dtype=float)
        if len(values) != len(pred_frame) or not np.isfinite(values).all():
            raise ValueError(f"Invalid test predictions for {horizon}.")
        pred_frame[horizon] = values
        invalid_tail = pred_frame["hour_idx"].to_numpy() + HORIZON_HOURS[horizon] > 215
        pred_frame.loc[invalid_tail, horizon] = np.nan

    # Build normalized keys separately: the template's identifier columns and
    # their original string formatting must remain byte-for-byte unchanged in
    # the returned submission frame.
    submission_keys = pd.DataFrame({
        "fipsCode": _normalize_fips(submission["fipsCode"]),
        "timestamp_et": pd.to_datetime(submission["timestamp_et"]),
    })
    key_columns = ["fipsCode", "timestamp_et"]
    if submission_keys.duplicated().any() or pred_frame[key_columns].duplicated().any():
        raise ValueError("Duplicate county-time key found while filling submission.")
    aligned = pred_frame.set_index(key_columns).reindex(
        pd.MultiIndex.from_frame(submission_keys)
    )
    if aligned["hour_idx"].isna().any():
        raise ValueError("At least one submission key is absent from test metadata.")
    for horizon in HORIZONS:
        submission[horizon] = aligned[horizon].to_numpy()

    county_count = submission["fipsCode"].nunique()
    expected_nan = {h: county_count * HORIZON_HOURS[h] for h in HORIZONS}
    actual_nan = {h: int(submission[h].isna().sum()) for h in HORIZONS}
    if actual_nan != expected_nan:
        raise ValueError(f"Submission tail NaN counts are wrong: {actual_nan} != {expected_nan}")
    return submission


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def run_experiment(
    adapter: ModelAdapter,
    output_dir: Path,
    max_rounds: int = MAX_BOOST_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    validate_only: bool = False,
) -> dict[str, Any] | None:
    if max_rounds < 1 or early_stopping_rounds < 1:
        raise ValueError("Boost rounds and early-stopping rounds must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {adapter.family} {adapter.version} on frozen {FEATURE_VERSION} features ===")
    print("[1/6] Loading and validating frozen features and fixed folds...")
    data = load_experiment_data()
    print(
        f"  train={data.X_train.shape}, test={data.X_test.shape}, "
        f"fold rows={[len(valid) for _, valid in data.folds]}"
    )
    if validate_only:
        print("Validation complete; model training was skipped.")
        return None

    print("[2/6] Computing leakage-safe reference baselines...")
    baselines = compute_baselines(data)
    summaries: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}
    fold_records: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []
    oof_frame = data.meta_train.loc[
        :, ["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]
    ].copy()

    print("[3/6] Training four horizon-specific models with fixed 5-fold CV...")
    for horizon in HORIZONS:
        y_full = data.y_train[horizon]
        valid_target = y_full.notna().to_numpy()
        oof = np.full(len(y_full), np.nan)
        best_rounds: list[int] = []
        fold_scores: list[dict[str, float]] = []
        print(f"  [{horizon}] valid rows={int(valid_target.sum())}")

        for fold_number, (train_idx_full, valid_idx_full) in enumerate(data.folds):
            train_idx = train_idx_full[valid_target[train_idx_full]]
            valid_idx = valid_idx_full[valid_target[valid_idx_full]]
            model = adapter.build_cv_model(max_rounds, early_stopping_rounds)
            adapter.fit_cv(
                model,
                data.X_train.iloc[train_idx],
                y_full.iloc[train_idx],
                data.X_train.iloc[valid_idx],
                y_full.iloc[valid_idx],
            )
            best_round = adapter.best_iteration(model)
            pred = post_process(adapter.predict(model, data.X_train.iloc[valid_idx]))
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite OOF predictions for {horizon}, fold {fold_number}.")
            oof[valid_idx] = pred
            score = compute_metrics(y_full.iloc[valid_idx].to_numpy(), pred)
            best_rounds.append(best_round)
            fold_scores.append(score)
            fold_records.append({
                "horizon": horizon,
                "fold": fold_number,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "best_round": best_round,
                **score,
            })
            print(
                f"    fold={fold_number} RMSE={score['rmse']:.6f} "
                f"MAE={score['mae']:.6f} best_round={best_round}"
            )

        if np.isnan(oof[valid_target]).any() or np.isfinite(oof[~valid_target]).any():
            raise ValueError(f"OOF coverage is invalid for {horizon}.")
        pooled = compute_metrics(y_full.to_numpy(), oof)
        final_rounds = max(1, int(round(float(np.mean(best_rounds)))))
        summary = {
            **pooled,
            "rmse_std": float(np.std([score["rmse"] for score in fold_scores])),
            "mae_std": float(np.std([score["mae"] for score in fold_scores])),
            "best_rounds": best_rounds,
            "final_rounds": final_rounds,
            "valid_rows": int(valid_target.sum()),
        }
        summaries[horizon] = summary
        oof_frame[f"actual_{horizon}"] = y_full
        oof_frame[f"pred_{horizon}"] = oof

        final_model = adapter.build_final_model(final_rounds)
        adapter.fit_final(
            final_model,
            data.X_train.loc[valid_target],
            y_full.loc[valid_target],
        )
        model_path = output_dir / (
            f"{adapter.slug}_{horizon}_{adapter.version}_{CV_VERSION}{adapter.model_extension}"
        )
        adapter.save_model(final_model, model_path)
        test_predictions[horizon] = post_process(adapter.predict(final_model, data.X_test))
        importances = adapter.feature_importance(final_model, list(data.X_train.columns))
        if len(importances) != data.X_train.shape[1]:
            raise ValueError(f"Feature-importance length mismatch for {horizon}.")
        total_importance = float(np.sum(importances))
        for rank, feature_idx in enumerate(np.argsort(-importances), start=1):
            importance_records.append({
                "horizon": horizon,
                "rank": rank,
                "feature": data.X_train.columns[feature_idx],
                "importance": float(importances[feature_idx]),
                "importance_fraction": (
                    float(importances[feature_idx] / total_importance)
                    if total_importance > 0 else 0.0
                ),
            })
        print(
            f"    pooled OOF RMSE={pooled['rmse']:.6f}, MAE={pooled['mae']:.6f}; "
            f"final_rounds={final_rounds}"
        )

    print("[4/6] Writing OOF diagnostics, metrics, and model artifacts...")
    pd.DataFrame(fold_records).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(importance_records).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    summary_rows = [
        {"horizon": horizon, **{k: v for k, v in summaries[horizon].items()
                                if k != "best_rounds"}}
        for horizon in HORIZONS
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)

    print("[5/6] Filling and validating the competition submission...")
    submission = _fill_submission(data.meta_test, test_predictions)
    submission_path = output_dir / f"submission_{adapter.version}_{CV_VERSION}.csv"
    submission.to_csv(submission_path, index=False, date_format="%m/%d/%Y %H:%M")

    input_paths = [
        CACHE_DIR / f"features_train_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"targets_train_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"meta_train_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"features_test_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"meta_test_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"feature_names_{FEATURE_VERSION}.json",
        CV_FILE,
        SUBMISSION_TEMPLATE,
    ]
    metadata = {
        "experiment_version": adapter.version,
        "model_family": adapter.family,
        "feature_version": FEATURE_VERSION,
        "cv_version": CV_VERSION,
        "seed": SEED,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            **adapter.library_versions(),
        },
        "train_shape": list(data.X_train.shape),
        "test_shape": list(data.X_test.shape),
        "target_non_null_rows": {
            horizon: int(data.y_train[horizon].notna().sum()) for horizon in HORIZONS
        },
        "post_process": {
            "clip_min": 0.0,
            "clip_max": OSI_MAX_OBSERVED,
            "zero_threshold": ZERO_THRESHOLD,
        },
        "config": adapter.config(max_rounds, early_stopping_rounds),
        "metrics": summaries,
        "baselines": baselines,
        "prediction_stats": {
            horizon: {
                "min": float(np.min(test_predictions[horizon])),
                "max": float(np.max(test_predictions[horizon])),
                "mean": float(np.mean(test_predictions[horizon])),
                "zero_pct": float(100 * np.mean(test_predictions[horizon] == 0)),
            }
            for horizon in HORIZONS
        },
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
            for path in input_paths
        },
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metadata), handle, ensure_ascii=False, indent=2)

    print("[6/6] Experiment complete.")
    for horizon in HORIZONS:
        summary = summaries[horizon]
        print(
            f"  {horizon}: RMSE={summary['rmse']:.6f}, "
            f"MAE={summary['mae']:.6f}, fold RMSE SD={summary['rmse_std']:.6f}"
        )
    print(f"  submission: {submission_path}")
    return metadata
