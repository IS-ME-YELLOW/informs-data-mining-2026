"""Strictly nested OOF ensemble of existing LightGBM/XGBoost/CatBoost predictions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_utils import (
    HORIZONS,
    KEYS,
    SEED,
    assert_aligned,
    assert_values_equal,
    final_model,
    hash_tree,
    load_row_folds,
    metric_rows,
    metric_values,
    nested_predictions,
    normalize_frame,
    paired_county_bootstrap,
    post_process,
    read_table,
    runtime_metadata,
    sha256_file,
    write_json,
)

VERSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERSION_DIR.parents[1]
ARTIFACTS_DIR = VERSION_DIR / "artifacts"
MODELS_DIR = VERSION_DIR / "models"
RESULTS_DIR = VERSION_DIR / "results"

INPUTS = {
    "lgbm_oof": PROJECT_ROOT / "versions/v1.8/oof_predictions.parquet",
    "lgbm_test": PROJECT_ROOT / "versions/v1.8/test_control_predictions.parquet",
    "lgbm_metadata": PROJECT_ROOT / "versions/v1.8/run_metadata.json",
    "xgb_direct_oof": PROJECT_ROOT / "versions/v1.9_xgboost/oof_predictions.csv",
    "xgb_direct_test": PROJECT_ROOT / "versions/v1.9_xgboost/submission_v1.9_balanced_v1.csv",
    "xgb_direct_metadata": PROJECT_ROOT / "versions/v1.9_xgboost/run_metadata.json",
    "cat_direct_oof": PROJECT_ROOT / "versions/v1.10/oof_predictions.csv",
    "cat_direct_test": PROJECT_ROOT / "versions/v1.10/submission_v1.10_balanced_v1.csv",
    "cat_direct_metadata": PROJECT_ROOT / "versions/v1.10/run_metadata.json",
    "xgb_component_oof": PROJECT_ROOT / "versions/v2/v2.4/oof_osi_predictions.csv",
    "xgb_component_test": PROJECT_ROOT / "versions/v2/v2.4/submission_v2.4_balanced_v1.csv",
    "xgb_component_metadata": PROJECT_ROOT / "versions/v2/v2.4/run_metadata.json",
    "cat_component_oof": PROJECT_ROOT / "versions/v2/v2.5/oof_osi_predictions.csv",
    "cat_component_test": PROJECT_ROOT / "versions/v2/v2.5/submission_v2.5_balanced_v1.csv",
    "cat_component_metadata": PROJECT_ROOT / "versions/v2/v2.5/run_metadata.json",
    "cv": PROJECT_ROOT / "cv/cv_assignments_balanced_v1_seed42.csv",
    "submission_template": PROJECT_ROOT / "data/sample_submission.csv",
}

CANDIDATES = (
    "lgbm_direct",
    "lgbm_component",
    "xgboost_direct",
    "catboost_direct",
    "xgboost_component",
    "catboost_component",
)
METHODS = ("simple_average", "static_convex", "safe_convex")


def _validate_metadata() -> dict[str, object]:
    report: dict[str, object] = {"expected": {"feature_version": "v1.5.6", "cv_version": "balanced_v1", "seed": SEED}, "sources": {}}
    for name in ("lgbm", "xgb_direct", "cat_direct", "xgb_component", "cat_component"):
        path = INPUTS[f"{name}_metadata"]
        metadata = json.loads(path.read_text(encoding="utf-8"))
        observed = {key: metadata.get(key) for key in ("feature_version", "cv_version", "seed")}
        if observed != report["expected"]:
            raise ValueError(f"Metadata mismatch for {name}: {observed}")
        report["sources"][name] = {"path": str(path.relative_to(PROJECT_ROOT)), **observed}
    return report


def load_oof_inputs() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, object]]:
    frames = {
        "lgbm": read_table(INPUTS["lgbm_oof"]),
        "xgb_direct": read_table(INPUTS["xgb_direct_oof"]),
        "cat_direct": read_table(INPUTS["cat_direct_oof"]),
        "xgb_component": read_table(INPUTS["xgb_component_oof"]),
        "cat_component": read_table(INPUTS["cat_component_oof"]),
    }
    reference = frames["lgbm"]
    report = _validate_metadata()
    report["row_count"] = len(reference)
    report["key_columns"] = KEYS
    report["alignment"] = {}
    for name, frame in frames.items():
        assert_aligned(reference, frame, name)
        report["alignment"][name] = "exact"

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for horizon in HORIZONS:
        actual = reference[f"actual_{horizon}"].to_numpy(dtype=float)
        for name, frame in frames.items():
            assert_values_equal(actual, frame[f"actual_{horizon}"].to_numpy(dtype=float), f"{name}/{horizon}/target")
        lgbm_component_column = (
            f"pred_C3_aligned_component_{horizon}"
            if horizon in HORIZONS[:2]
            else f"pred_C1_component_osi_{horizon}"
        )
        predictions[horizon] = {
            "actual": actual,
            "lgbm_direct": reference[f"pred_C0_direct_osi_{horizon}"].to_numpy(dtype=float),
            "lgbm_component": reference[lgbm_component_column].to_numpy(dtype=float),
            "xgboost_direct": frames["xgb_direct"][f"pred_{horizon}"].to_numpy(dtype=float),
            "catboost_direct": frames["cat_direct"][f"pred_{horizon}"].to_numpy(dtype=float),
            "xgboost_component": frames["xgb_component"][f"pred_{horizon}"].to_numpy(dtype=float),
            "catboost_component": frames["cat_component"][f"pred_{horizon}"].to_numpy(dtype=float),
        }
        finite = np.isfinite(actual)
        for candidate in CANDIDATES:
            if not np.array_equal(finite, np.isfinite(predictions[horizon][candidate])):
                raise ValueError(f"Prediction mask differs from target: {candidate}/{horizon}")
    report["target_and_prediction_masks"] = "exact for every source and horizon"
    report["lgbm_component_rule"] = {"t01h": "C3", "t06h": "C3", "t24h": "C1", "t48h": "C1"}
    return reference[KEYS].copy(), predictions, report


def _submission_frame(path: Path, reference: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"fipsCode": str})
    frame["fipsCode"] = frame["fipsCode"].astype(str).str.zfill(5)
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    frame["stateAbbr"] = frame["stateAbbr"].astype(str)
    for column in ("fipsCode", "timestamp_et", "stateAbbr"):
        if column == "timestamp_et":
            equal = np.array_equal(frame[column].to_numpy(dtype="datetime64[ns]"), reference[column].to_numpy(dtype="datetime64[ns]"))
        else:
            equal = np.array_equal(frame[column].to_numpy(), reference[column].to_numpy())
        if not equal:
            raise ValueError(f"Test submission {path} is not aligned on {column}.")
    return frame


def load_test_inputs() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    reference = read_table(INPUTS["lgbm_test"])
    submissions = {
        name: _submission_frame(INPUTS[f"{name}_test"], reference)
        for name in ("xgb_direct", "cat_direct", "xgb_component", "cat_component")
    }
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for horizon in HORIZONS:
        lgbm_component_column = (
            f"pred_C3_aligned_component_{horizon}"
            if horizon in HORIZONS[:2]
            else f"pred_C1_component_osi_{horizon}"
        )
        predictions[horizon] = {
            "lgbm_direct": reference[f"pred_C0_direct_osi_{horizon}"].to_numpy(dtype=float),
            "lgbm_component": reference[lgbm_component_column].to_numpy(dtype=float),
            "xgboost_direct": submissions["xgb_direct"][horizon].to_numpy(dtype=float),
            "catboost_direct": submissions["cat_direct"][horizon].to_numpy(dtype=float),
            "xgboost_component": submissions["xgb_component"][horizon].to_numpy(dtype=float),
            "catboost_component": submissions["cat_component"][horizon].to_numpy(dtype=float),
        }
    return reference[KEYS].copy(), predictions


def _diagnostics(
    frame: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    evaluated_predictions: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    correlations: list[dict[str, object]] = []
    disagreements: list[dict[str, object]] = []
    high_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        y = predictions[horizon]["actual"]
        valid = np.isfinite(y)
        matrix = np.column_stack([predictions[horizon][name] for name in CANDIDATES])
        for first_index, first in enumerate(CANDIDATES):
            for second_index in range(first_index + 1, len(CANDIDATES)):
                second = CANDIDATES[second_index]
                residual_first = predictions[horizon][first][valid] - y[valid]
                residual_second = predictions[horizon][second][valid] - y[valid]
                delta = predictions[horizon][first][valid] - predictions[horizon][second][valid]
                correlations.append({"horizon": horizon, "model_a": first, "model_b": second, "residual_correlation": float(np.corrcoef(residual_first, residual_second)[0, 1])})
                disagreements.append({"horizon": horizon, "model_a": first, "model_b": second, "prediction_correlation": float(np.corrcoef(predictions[horizon][first][valid], predictions[horizon][second][valid])[0, 1]), "mean_absolute_disagreement": float(np.mean(np.abs(delta))), "rmse_disagreement": float(np.sqrt(np.mean(delta**2))), "max_absolute_disagreement": float(np.max(np.abs(delta)))})
        threshold = float(np.quantile(y[valid], 0.95))
        high = valid & (y >= threshold)
        for model, prediction in evaluated_predictions[horizon].items():
            high_rows.append({"horizon": horizon, "model": model, "threshold": threshold, **metric_values(y[high], prediction[high])})
        low = np.min(matrix[valid], axis=1)
        high_bound = np.max(matrix[valid], axis=1)
        oracle = np.clip(y[valid], low, high_bound)
        oracle_rows.append({"horizon": horizon, "model": "convex_hull_oracle_diagnostic_only", "coverage_pct": float(100 * np.mean((y[valid] >= low) & (y[valid] <= high_bound))), **metric_values(y[valid], oracle)})
    return pd.DataFrame(correlations), pd.DataFrame(disagreements), pd.DataFrame(high_rows), pd.DataFrame(oracle_rows)


def run() -> None:
    started = time.perf_counter()
    for directory in (ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    frame, source_predictions, alignment = load_oof_inputs()
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame["fold"].to_numpy(dtype=int)
    alignment["fold_alignment"] = "balanced_v1 county mapping applied identically to every aligned OOF source"
    alignment["fold_row_counts"] = {str(fold): int(np.sum(folds == fold)) for fold in range(5)}
    alignment["county_count"] = int(frame["fipsCode"].nunique())
    test_frame, test_sources = load_test_inputs()
    template = pd.read_csv(INPUTS["submission_template"], dtype={"fipsCode": str})

    summary: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    county_metrics: list[dict[str, object]] = []
    bootstrap: list[dict[str, object]] = []
    weights_rows: list[dict[str, object]] = []
    oof = frame.copy()
    test_output = test_frame.copy()
    per_horizon_predictions: dict[str, dict[str, np.ndarray]] = {}

    for horizon_index, horizon in enumerate(HORIZONS):
        y = source_predictions[horizon]["actual"]
        matrix = np.column_stack([source_predictions[horizon][name] for name in CANDIDATES])
        baseline = source_predictions[horizon]["lgbm_component"]
        nested, outer_models = nested_predictions(y, matrix, baseline, folds)
        all_predictions = {"v1.8_rule": baseline, **{name: source_predictions[horizon][name] for name in CANDIDATES if name != "lgbm_component"}, **nested}
        per_horizon_predictions[horizon] = all_predictions
        for model, prediction in all_predictions.items():
            metrics = metric_values(y, prediction)
            baseline_metrics = metric_values(y, baseline)
            summary.append({"horizon": horizon, "model": model, **metrics, "rmse_change_pct_vs_v1.8": 100 * (metrics["rmse"] / baseline_metrics["rmse"] - 1), "mae_change_pct_vs_v1.8": 100 * (metrics["mae"] / baseline_metrics["mae"] - 1)})
            oof[f"actual_{horizon}"] = y
            oof[f"pred_{model}_{horizon}"] = prediction
        fold_frame = frame.copy()
        fold_metrics.extend(metric_rows(fold_frame, y, all_predictions, horizon, "fold"))
        county_metrics.extend(metric_rows(frame, y, all_predictions, horizon, "fipsCode"))
        for method in METHODS:
            bootstrap.append({"horizon": horizon, "model": method, **paired_county_bootstrap(frame, y, baseline, nested[method], seed=20260913 + horizon_index * 10 + METHODS.index(method))})

        for model in outer_models:
            model.update({"horizon": horizon, "candidate_names": list(CANDIDATES), "baseline": "lgbm_component", "post_process": {"clip": [0.0, 0.65], "zero_threshold": 0.001}})
            path = MODELS_DIR / f"outer_fold{model['fold']}_{horizon}.json"
            write_json(path, model)
            for candidate, weight in zip(CANDIDATES, model["weights"]):
                weights_rows.append({"scope": "outer", "fold": model["fold"], "horizon": horizon, "candidate": candidate, "weight": weight, "safe_alpha": model["safe_alpha"]})

        fitted_final = final_model(y, matrix, baseline)
        fitted_final.update({"horizon": horizon, "candidate_names": list(CANDIDATES), "baseline": "lgbm_component", "post_process": {"clip": [0.0, 0.65], "zero_threshold": 0.001}, "evaluation_use": "test inference only; full-OOF fit is not used for reported OOF scores"})
        write_json(MODELS_DIR / f"final_{horizon}.json", fitted_final)
        for candidate, weight in zip(CANDIDATES, fitted_final["weights"]):
            weights_rows.append({"scope": "final", "fold": "all", "horizon": horizon, "candidate": candidate, "weight": weight, "safe_alpha": fitted_final["safe_alpha"]})
        test_matrix = np.column_stack([test_sources[horizon][name] for name in CANDIDATES])
        test_static = post_process(test_matrix @ np.asarray(fitted_final["weights"]))
        test_predictions = {
            "simple_average": post_process(np.mean(test_matrix, axis=1)),
            "static_convex": test_static,
            "safe_convex": post_process(test_sources[horizon]["lgbm_component"] + fitted_final["safe_alpha"] * (test_static - test_sources[horizon]["lgbm_component"])),
        }
        expected_mask = np.isfinite(template[horizon].to_numpy(dtype=float))
        for method, prediction in test_predictions.items():
            test_output[f"pred_{method}_{horizon}"] = np.where(expected_mask, prediction, np.nan)

    correlations, disagreements, high_metrics, oracle = _diagnostics(frame, source_predictions, per_horizon_predictions)
    summary_df = pd.DataFrame(summary)
    fold_df = pd.DataFrame(fold_metrics)
    baseline_fold = fold_df[fold_df["model"] == "v1.8_rule"][["horizon", "fold", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    fold_df = fold_df.merge(baseline_fold, on=["horizon", "fold"], how="left")
    fold_df["rmse_change_pct_vs_v1.8"] = 100 * (fold_df["rmse"] / fold_df["baseline_rmse"] - 1)
    fold_df["improved_vs_v1.8"] = fold_df["rmse"] < fold_df["baseline_rmse"]
    pd.DataFrame(county_metrics).to_csv(RESULTS_DIR / "county_metrics.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    fold_df.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(RESULTS_DIR / "paired_county_bootstrap.csv", index=False)
    pd.DataFrame(weights_rows).to_csv(RESULTS_DIR / "weights.csv", index=False)
    correlations.to_csv(RESULTS_DIR / "residual_correlations.csv", index=False)
    disagreements.to_csv(RESULTS_DIR / "prediction_disagreement.csv", index=False)
    high_metrics.to_csv(RESULTS_DIR / "high_osi_metrics.csv", index=False)
    oracle.to_csv(RESULTS_DIR / "convex_hull_oracle.csv", index=False)
    oof.to_parquet(ARTIFACTS_DIR / "oof_predictions.parquet", index=False)
    test_output.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

    for horizon in HORIZONS:
        template[horizon] = test_output[f"pred_safe_convex_{horizon}"].to_numpy()
    template.to_csv(ARTIFACTS_DIR / "submission_v1.11_safe_convex_balanced_v1.csv", index=False)

    decision_horizons: dict[str, object] = {}
    for horizon in HORIZONS:
        row = summary_df[(summary_df.horizon == horizon) & (summary_df.model == "safe_convex")].iloc[0]
        boot = next(item for item in bootstrap if item["horizon"] == horizon and item["model"] == "safe_convex")
        fold_count = int(fold_df[(fold_df.horizon == horizon) & (fold_df.model == "safe_convex")]["improved_vs_v1.8"].sum())
        rmse_change = float(row["rmse_change_pct_vs_v1.8"])
        decision_horizons[horizon] = {"rmse_change_pct": rmse_change, "improved_folds": fold_count, "bootstrap_rmse_ci": [boot["rmse_ci_low"], boot["rmse_ci_high"]], "passes_0_5_pct": bool(rmse_change <= -0.5), "passes_4_of_5_folds": fold_count >= 4, "passes_bootstrap": boot["rmse_ci_high"] < 0.0}
    promotion = {"primary_candidate": "safe_convex", "criteria": {"rmse_improvement_pct": 0.5, "minimum_improved_folds": 4, "bootstrap_ci_should_be_below_zero": True, "no_material_horizon_regression": True}, "horizons": decision_horizons}
    promotion["promote"] = bool(all(value["passes_0_5_pct"] and value["passes_4_of_5_folds"] and value["passes_bootstrap"] for value in decision_horizons.values()))
    write_json(RESULTS_DIR / "promotion_decision.json", promotion)
    write_json(ARTIFACTS_DIR / "alignment_report.json", alignment)

    output_paths = list(MODELS_DIR.glob("*.json")) + list(RESULTS_DIR.glob("*")) + [ARTIFACTS_DIR / "oof_predictions.parquet", ARTIFACTS_DIR / "test_predictions.csv", ARTIFACTS_DIR / "submission_v1.11_safe_convex_balanced_v1.csv", ARTIFACTS_DIR / "alignment_report.json"]
    metadata = {**runtime_metadata(), "experiment_version": "v1.11_tree_ensemble", "feature_version": "v1.5.6", "feature_count": 163, "cv_version": "balanced_v1", "seed": SEED, "bootstrap_seed": 20260913, "bootstrap_replicates": 2000, "candidate_names": list(CANDIDATES), "solver": "deterministic enumeration of every non-empty active set for simplex-constrained least squares", "post_process": {"clip": [0.0, 0.65], "zero_threshold": 0.001}, "strict_nested_cv": "For each outer fold, weights and safe alpha use only the other four OOF folds.", "final_model_usage": "Full-OOF weights are used only for test inference.", "runtime_seconds": time.perf_counter() - started, "input_hashes": {name: sha256_file(path) for name, path in INPUTS.items()}, "artifact_hashes": hash_tree(output_paths, PROJECT_ROOT)}
    write_json(ARTIFACTS_DIR / "run_metadata.json", metadata)
    print(summary_df[summary_df.model.isin(["v1.8_rule", *METHODS])].to_string(index=False))
    print(f"Completed v1.11 in {metadata['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
