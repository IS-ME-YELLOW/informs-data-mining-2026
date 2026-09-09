"""Shared protocol for v2: predict P/N/D/R first, then compose OSI."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


V2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V2_ROOT.parents[1]
VERSIONS_ROOT = PROJECT_ROOT / "versions"
LOCAL_PACKAGES = PROJECT_ROOT / ".python_packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))
if str(VERSIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSIONS_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gbdt_experiment as direct_protocol  # noqa: E402


COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
WEIGHTS = {"P_t": 0.40, "N_t": 0.35, "D_t": 0.25, "R_t": -0.10}
COMPONENT_MIN = 0.0
COMPONENT_MAX = 1.0
TARGETS_FILE = V2_ROOT / "component_targets_v2.parquet"
TARGET_VALIDATION_FILE = V2_ROOT / "component_target_validation.json"
RAW_TRAIN_FILE = PROJECT_ROOT / "data" / "DM_Train.csv"
MAX_BOOST_ROUNDS = 2_000
EARLY_STOPPING_ROUNDS = 100

DIRECT_OSI_REFERENCE = {
    "LightGBM": {
        "osi_target_t01h": {"rmse": 0.012232, "mae": 0.003654},
        "osi_target_t06h": {"rmse": 0.010677, "mae": 0.003538},
        "osi_target_t24h": {"rmse": 0.008637, "mae": 0.002934},
        "osi_target_t48h": {"rmse": 0.007687, "mae": 0.002293},
    },
    "XGBoost": {
        "osi_target_t01h": {"rmse": 0.011900408812564942, "mae": 0.003488166127072245},
        "osi_target_t06h": {"rmse": 0.010655281768276821, "mae": 0.003297989657138615},
        "osi_target_t24h": {"rmse": 0.008731686091476362, "mae": 0.0027337240770088918},
        "osi_target_t48h": {"rmse": 0.007942428231321573, "mae": 0.002103938576671536},
    },
    "CatBoost": {
        "osi_target_t01h": {"rmse": 0.012179441878483195, "mae": 0.003533583868609779},
        "osi_target_t06h": {"rmse": 0.010936939206295635, "mae": 0.003266998432346011},
        "osi_target_t24h": {"rmse": 0.008668708121215557, "mae": 0.0026494241071311543},
        "osi_target_t48h": {"rmse": 0.007786785421239254, "mae": 0.0020983768939278635},
    },
}


def component_target_name(component: str, horizon: str) -> str:
    return f"{component}_target_{horizon.rsplit('_', 1)[-1]}"


def _normalize_fips(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compose_osi(component_predictions: dict[str, np.ndarray]) -> np.ndarray:
    missing = set(COMPONENTS) - set(component_predictions)
    if missing:
        raise ValueError(f"Missing components for OSI composition: {sorted(missing)}")
    length = len(np.asarray(component_predictions[COMPONENTS[0]]))
    result = np.zeros(length, dtype=float)
    finite = np.ones(length, dtype=bool)
    for component in COMPONENTS:
        values = np.asarray(component_predictions[component], dtype=float)
        if len(values) != length:
            raise ValueError("Component predictions have inconsistent lengths.")
        finite &= np.isfinite(values)
        result += WEIGHTS[component] * values
    result[~finite] = np.nan
    return np.maximum(result, 0.0)


def clip_component(predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(predictions, dtype=float), COMPONENT_MIN, COMPONENT_MAX)


def prepare_component_targets(force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = direct_protocol.load_experiment_data()
    if TARGETS_FILE.exists() and TARGET_VALIDATION_FILE.exists() and not force:
        targets = pd.read_parquet(TARGETS_FILE)
        with TARGET_VALIDATION_FILE.open("r", encoding="utf-8") as handle:
            validation = json.load(handle)
        if validation.get("raw_train_sha256") != _sha256(RAW_TRAIN_FILE):
            raise ValueError(
                "The saved v2 component targets are stale relative to DM_Train.csv; "
                "rerun build_component_targets.py."
            )
        if validation.get("component_targets_sha256") != _sha256(TARGETS_FILE):
            raise ValueError("The saved v2 component-target file hash is invalid.")
        _validate_component_target_alignment(targets, data)
        return targets, validation

    raw = pd.read_csv(RAW_TRAIN_FILE, dtype={"fipsCode": str})
    raw["fipsCode"] = _normalize_fips(raw["fipsCode"])
    raw["timestamp_et"] = pd.to_datetime(raw["timestamp_et"])
    if raw[["fipsCode", "timestamp_et"]].duplicated().any():
        raise ValueError("Raw training data contains duplicate county-time keys.")
    raw_components = raw.set_index(["fipsCode", "timestamp_et"])[list(COMPONENTS)]

    targets = data.meta_train.loc[:, ["fipsCode", "timestamp_et", "hour_idx"]].copy()
    targets["fipsCode"] = _normalize_fips(targets["fipsCode"])
    targets["timestamp_et"] = pd.to_datetime(targets["timestamp_et"])
    validation: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "data/DM_Train.csv shifted within county by each forecast horizon",
        "formula": "max(0, 0.40*P_t + 0.35*N_t + 0.25*D_t - 0.10*R_t)",
        "horizons": {},
        "raw_train_sha256": _sha256(RAW_TRAIN_FILE),
    }

    for horizon in direct_protocol.HORIZONS:
        hours = direct_protocol.HORIZON_HOURS[horizon]
        target_keys = pd.MultiIndex.from_arrays(
            [
                targets["fipsCode"],
                targets["timestamp_et"] + pd.to_timedelta(hours, unit="h"),
            ],
            names=["fipsCode", "timestamp_et"],
        )
        shifted = raw_components.reindex(target_keys).reset_index(drop=True)
        component_values: dict[str, np.ndarray] = {}
        for component in COMPONENTS:
            name = component_target_name(component, horizon)
            targets[name] = shifted[component].to_numpy(dtype=float)
            component_values[component] = targets[name].to_numpy(dtype=float)

        official = data.y_train[horizon].to_numpy(dtype=float)
        composed = compose_osi(component_values)
        official_valid = np.isfinite(official)
        components_valid = np.column_stack(
            [np.isfinite(component_values[c]) for c in COMPONENTS]
        ).all(axis=1)
        if not np.array_equal(official_valid, components_valid):
            raise ValueError(f"Component/official target missing masks differ for {horizon}.")
        errors = composed[official_valid] - official[official_valid]
        max_abs_error = float(np.max(np.abs(errors)))
        if max_abs_error > 5.1e-5:
            raise ValueError(
                f"Composed targets disagree with official {horizon}: max error={max_abs_error}"
            )
        ranges = {
            component: {
                "min": float(np.min(component_values[component][official_valid])),
                "max": float(np.max(component_values[component][official_valid])),
                "mean": float(np.mean(component_values[component][official_valid])),
                "zero_pct": float(
                    100 * np.mean(component_values[component][official_valid] == 0)
                ),
            }
            for component in COMPONENTS
        }
        validation["horizons"][horizon] = {
            "valid_rows": int(official_valid.sum()),
            "formula_rmse_vs_official": float(np.sqrt(np.mean(errors ** 2))),
            "formula_mae_vs_official": float(np.mean(np.abs(errors))),
            "formula_max_abs_error_vs_official": max_abs_error,
            "component_ranges": ranges,
        }

    _validate_component_target_alignment(targets, data)
    targets.to_parquet(TARGETS_FILE, index=False)
    validation["component_targets_sha256"] = _sha256(TARGETS_FILE)
    with TARGET_VALIDATION_FILE.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)
    return targets, validation


def _validate_component_target_alignment(
    targets: pd.DataFrame, data: direct_protocol.ExperimentData
) -> None:
    expected_columns = ["fipsCode", "timestamp_et", "hour_idx"] + [
        component_target_name(component, horizon)
        for horizon in direct_protocol.HORIZONS
        for component in COMPONENTS
    ]
    if list(targets.columns) != expected_columns:
        raise ValueError("Unexpected component-target columns or order.")
    if len(targets) != len(data.meta_train):
        raise ValueError("Component targets and training metadata have different row counts.")
    actual_fips = _normalize_fips(targets["fipsCode"])
    expected_fips = _normalize_fips(data.meta_train["fipsCode"])
    actual_time = pd.to_datetime(targets["timestamp_et"])
    expected_time = pd.to_datetime(data.meta_train["timestamp_et"])
    if not actual_fips.equals(expected_fips) or not actual_time.equals(expected_time):
        raise ValueError("Component targets do not align with frozen v1.5.2 feature rows.")


def _fill_submission(
    meta_test: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    return direct_protocol._fill_submission(meta_test, predictions)


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


def run_component_experiment(
    adapter,
    experiment_version: str,
    output_dir: Path,
    max_rounds: int = MAX_BOOST_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    validate_only: bool = False,
) -> dict[str, Any] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"=== {experiment_version} {adapter.family}: P/N/D/R -> composed OSI "
        f"on frozen v1.5.2 features ===",
        flush=True,
    )
    print("[1/7] Loading features, folds, and component targets...", flush=True)
    data = direct_protocol.load_experiment_data()
    targets, target_validation = prepare_component_targets()
    _validate_component_target_alignment(targets, data)
    print(
        f"  train={data.X_train.shape}, test={data.X_test.shape}, "
        f"component_targets={targets.shape}",
        flush=True,
    )
    if validate_only:
        print("Validation complete; model training was skipped.", flush=True)
        return None

    component_fold_records: list[dict[str, Any]] = []
    component_summaries: list[dict[str, Any]] = []
    osi_fold_records: list[dict[str, Any]] = []
    osi_summaries: dict[str, dict[str, Any]] = {}
    importance_records: list[dict[str, Any]] = []
    component_oof: dict[str, dict[str, np.ndarray]] = {}
    component_test: dict[str, dict[str, np.ndarray]] = {}
    oof_component_frame = data.meta_train.loc[
        :, ["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]
    ].copy()
    oof_osi_frame = oof_component_frame.copy()
    test_component_frame = data.meta_test.loc[
        :, ["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]
    ].copy()

    print("[2/7] Training 16 component models with fixed 5-fold CV...", flush=True)
    for horizon in direct_protocol.HORIZONS:
        component_oof[horizon] = {}
        component_test[horizon] = {}
        for component in COMPONENTS:
            target_column = component_target_name(component, horizon)
            y = targets[target_column]
            valid_target = y.notna().to_numpy()
            oof = np.full(len(y), np.nan)
            fold_scores: list[dict[str, float]] = []
            best_rounds: list[int] = []
            print(f"  [{horizon} / {component}]", flush=True)

            for fold_number, (train_full, valid_full) in enumerate(data.folds):
                train_idx = train_full[valid_target[train_full]]
                valid_idx = valid_full[valid_target[valid_full]]
                model = adapter.build_cv_model(max_rounds, early_stopping_rounds)
                adapter.fit_cv(
                    model,
                    data.X_train.iloc[train_idx],
                    y.iloc[train_idx],
                    data.X_train.iloc[valid_idx],
                    y.iloc[valid_idx],
                    early_stopping_rounds,
                )
                best_round = adapter.best_iteration(model)
                pred = clip_component(adapter.predict(model, data.X_train.iloc[valid_idx]))
                oof[valid_idx] = pred
                score = direct_protocol.compute_metrics(y.iloc[valid_idx], pred)
                best_rounds.append(best_round)
                fold_scores.append(score)
                component_fold_records.append({
                    "horizon": horizon,
                    "component": component,
                    "fold": fold_number,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    "best_round": best_round,
                    **score,
                })
                print(
                    f"    fold={fold_number} RMSE={score['rmse']:.6f} "
                    f"MAE={score['mae']:.6f} best_round={best_round}",
                    flush=True,
                )

            if np.isnan(oof[valid_target]).any() or np.isfinite(oof[~valid_target]).any():
                raise ValueError(f"Invalid OOF coverage for {horizon}/{component}.")
            pooled = direct_protocol.compute_metrics(y, oof)
            final_rounds = max(1, int(round(float(np.mean(best_rounds)))))
            component_summaries.append({
                "horizon": horizon,
                "component": component,
                **pooled,
                "rmse_std": float(np.std([score["rmse"] for score in fold_scores])),
                "mae_std": float(np.std([score["mae"] for score in fold_scores])),
                "best_rounds": json.dumps(best_rounds),
                "final_rounds": final_rounds,
                "valid_rows": int(valid_target.sum()),
            })
            component_oof[horizon][component] = oof
            oof_component_frame[f"actual_{target_column}"] = y
            oof_component_frame[f"pred_{target_column}"] = oof

            final_model = adapter.build_final_model(final_rounds)
            adapter.fit_final(
                final_model,
                data.X_train.loc[valid_target],
                y.loc[valid_target],
            )
            model_path = output_dir / (
                f"{adapter.slug}_{component}_{horizon}_{experiment_version}_"
                f"{direct_protocol.CV_VERSION}{adapter.model_extension}"
            )
            adapter.save_model(final_model, model_path)
            test_pred = clip_component(adapter.predict(final_model, data.X_test))
            component_test[horizon][component] = test_pred
            test_component_frame[f"pred_{target_column}"] = test_pred

            importances = np.asarray(
                adapter.feature_importance(final_model, list(data.X_train.columns)),
                dtype=float,
            )
            total_importance = float(importances.sum())
            for rank, feature_idx in enumerate(np.argsort(-importances), start=1):
                importance_records.append({
                    "horizon": horizon,
                    "component": component,
                    "rank": rank,
                    "feature": data.X_train.columns[feature_idx],
                    "importance": float(importances[feature_idx]),
                    "importance_fraction": (
                        float(importances[feature_idx] / total_importance)
                        if total_importance > 0 else 0.0
                    ),
                })
            print(
                f"    pooled component RMSE={pooled['rmse']:.6f}, "
                f"MAE={pooled['mae']:.6f}; final_rounds={final_rounds}",
                flush=True,
            )

    print("[3/7] Composing OOF OSI and computing primary metrics...", flush=True)
    test_osi_predictions: dict[str, np.ndarray] = {}
    for horizon in direct_protocol.HORIZONS:
        composed_oof = direct_protocol.post_process(compose_osi(component_oof[horizon]))
        official = data.y_train[horizon].to_numpy(dtype=float)
        valid_target = np.isfinite(official)
        if np.isnan(composed_oof[valid_target]).any():
            raise ValueError(f"Composed OOF contains NaN on valid rows for {horizon}.")
        pooled = direct_protocol.compute_metrics(official, composed_oof)
        fold_scores = []
        for fold_number, (_, valid_full) in enumerate(data.folds):
            valid_idx = valid_full[valid_target[valid_full]]
            score = direct_protocol.compute_metrics(
                official[valid_idx], composed_oof[valid_idx]
            )
            fold_scores.append(score)
            osi_fold_records.append({
                "horizon": horizon,
                "fold": fold_number,
                "valid_rows": len(valid_idx),
                **score,
            })
        direct_reference = DIRECT_OSI_REFERENCE[adapter.family][horizon]
        osi_summaries[horizon] = {
            **pooled,
            "rmse_std": float(np.std([score["rmse"] for score in fold_scores])),
            "mae_std": float(np.std([score["mae"] for score in fold_scores])),
            "valid_rows": int(valid_target.sum()),
            "direct_rmse": direct_reference["rmse"],
            "direct_mae": direct_reference["mae"],
            "rmse_change_pct_vs_direct": float(
                100 * (pooled["rmse"] / direct_reference["rmse"] - 1)
            ),
            "mae_change_pct_vs_direct": float(
                100 * (pooled["mae"] / direct_reference["mae"] - 1)
            ),
        }
        oof_osi_frame[f"actual_{horizon}"] = official
        oof_osi_frame[f"pred_{horizon}"] = composed_oof
        test_osi_predictions[horizon] = direct_protocol.post_process(
            compose_osi(component_test[horizon])
        )
        print(
            f"  {horizon}: composed OOF RMSE={pooled['rmse']:.6f}, "
            f"MAE={pooled['mae']:.6f}",
            flush=True,
        )

    print("[4/7] Writing component and composed-OSI diagnostics...", flush=True)
    pd.DataFrame(component_fold_records).to_csv(
        output_dir / "component_fold_metrics.csv", index=False
    )
    pd.DataFrame(component_summaries).to_csv(
        output_dir / "component_summary_metrics.csv", index=False
    )
    pd.DataFrame(osi_fold_records).to_csv(
        output_dir / "osi_fold_metrics.csv", index=False
    )
    pd.DataFrame([
        {"horizon": horizon, **osi_summaries[horizon]}
        for horizon in direct_protocol.HORIZONS
    ]).to_csv(output_dir / "osi_summary_metrics.csv", index=False)
    pd.DataFrame(importance_records).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    oof_component_frame.to_csv(output_dir / "oof_component_predictions.csv", index=False)
    oof_osi_frame.to_csv(output_dir / "oof_osi_predictions.csv", index=False)
    test_component_frame.to_csv(output_dir / "test_component_predictions.csv", index=False)

    print("[5/7] Filling and validating competition submission...", flush=True)
    submission = _fill_submission(data.meta_test, test_osi_predictions)
    submission_path = output_dir / (
        f"submission_{experiment_version}_{direct_protocol.CV_VERSION}.csv"
    )
    submission.to_csv(submission_path, index=False)

    print("[6/7] Writing reproducibility metadata...", flush=True)
    input_paths = [
        PROJECT_ROOT / "cache" / "features_train_v1.5.2.parquet",
        PROJECT_ROOT / "cache" / "features_test_v1.5.2.parquet",
        PROJECT_ROOT / "cache" / "meta_train_v1.5.2.parquet",
        PROJECT_ROOT / "cache" / "meta_test_v1.5.2.parquet",
        PROJECT_ROOT / "cache" / "targets_train_v1.5.2.parquet",
        PROJECT_ROOT / "cv" / "cv_assignments_balanced_v1_seed42.csv",
        TARGETS_FILE,
    ]
    metadata = {
        "experiment_version": experiment_version,
        "strategy": "predict P_t/N_t/D_t/R_t independently, then compose OSI",
        "model_family": adapter.family,
        "feature_version": "v1.5.2",
        "cv_version": direct_protocol.CV_VERSION,
        "seed": direct_protocol.SEED,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            **adapter.library_versions(),
        },
        "component_formula": WEIGHTS,
        "component_prediction_clip": [COMPONENT_MIN, COMPONENT_MAX],
        "osi_post_process": {
            "clip": [0.0, direct_protocol.OSI_MAX_OBSERVED],
            "zero_threshold": direct_protocol.ZERO_THRESHOLD,
        },
        "model_count": len(COMPONENTS) * len(direct_protocol.HORIZONS),
        "config": adapter.config(max_rounds, early_stopping_rounds),
        "component_target_validation": target_validation,
        "osi_metrics": osi_summaries,
        "direct_osi_reference": DIRECT_OSI_REFERENCE[adapter.family],
        "baselines": direct_protocol.compute_baselines(data),
        "prediction_stats": {
            horizon: {
                "min": float(np.min(test_osi_predictions[horizon])),
                "max": float(np.max(test_osi_predictions[horizon])),
                "mean": float(np.mean(test_osi_predictions[horizon])),
                "zero_pct": float(100 * np.mean(test_osi_predictions[horizon] == 0)),
            }
            for horizon in direct_protocol.HORIZONS
        },
        "known_constraint": (
            "Independent component models do not enforce N/R consistency with P changes "
            "or D as a six-hour rolling mean of P."
        ),
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
            for path in input_paths
        },
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metadata), handle, ensure_ascii=False, indent=2)

    print("[7/7] Experiment complete.", flush=True)
    for horizon in direct_protocol.HORIZONS:
        score = osi_summaries[horizon]
        print(
            f"  {horizon}: RMSE={score['rmse']:.6f}, MAE={score['mae']:.6f}, "
            f"RMSE change vs direct={score['rmse_change_pct_vs_direct']:+.2f}%",
            flush=True,
        )
    print(f"  submission: {submission_path}", flush=True)
    return metadata
