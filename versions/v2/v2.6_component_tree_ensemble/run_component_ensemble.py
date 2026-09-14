"""Strict nested P/N/D/R tree-family ensemble followed by official OSI composition."""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

VERSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERSION_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "versions/v1.11_tree_ensemble"))

from ensemble_utils import (  # noqa: E402
    COMPONENTS,
    HORIZONS,
    KEYS,
    SEED,
    assert_aligned,
    assert_values_equal,
    clip_component,
    compose_osi,
    final_model,
    hash_tree,
    load_row_folds,
    metric_rows,
    metric_values,
    nested_predictions,
    paired_county_bootstrap,
    post_process,
    read_table,
    runtime_metadata,
    sha256_file,
    write_json,
)

ARTIFACTS_DIR = VERSION_DIR / "artifacts"
MODELS_DIR = VERSION_DIR / "models"
RESULTS_DIR = VERSION_DIR / "results"
FAMILIES = ("lightgbm", "xgboost", "catboost")
METHODS = ("simple_average", "static_convex", "safe_convex")

INPUTS = {
    "lightgbm_oof": PROJECT_ROOT / "versions/v1.8/oof_component_predictions.parquet",
    "lightgbm_test": PROJECT_ROOT / "versions/v1.8/test_component_predictions.parquet",
    "lightgbm_metadata": PROJECT_ROOT / "versions/v1.8/run_metadata.json",
    "xgboost_oof": PROJECT_ROOT / "versions/v2/v2.4/oof_component_predictions.csv",
    "xgboost_test": PROJECT_ROOT / "versions/v2/v2.4/test_component_predictions.csv",
    "xgboost_metadata": PROJECT_ROOT / "versions/v2/v2.4/run_metadata.json",
    "catboost_oof": PROJECT_ROOT / "versions/v2/v2.5/oof_component_predictions.csv",
    "catboost_test": PROJECT_ROOT / "versions/v2/v2.5/test_component_predictions.csv",
    "catboost_metadata": PROJECT_ROOT / "versions/v2/v2.5/run_metadata.json",
    "v18_oof": PROJECT_ROOT / "versions/v1.8/oof_predictions.parquet",
    "v18_test": PROJECT_ROOT / "versions/v1.8/test_control_predictions.parquet",
    "cv": PROJECT_ROOT / "cv/cv_assignments_balanced_v1_seed42.csv",
    "submission_template": PROJECT_ROOT / "data/sample_submission.csv",
}


def component_target(component: str, horizon: str) -> str:
    return f"{component}_target_{horizon.rsplit('_', 1)[-1]}"


def load_inputs(test: bool = False) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    suffix = "test" if test else "oof"
    frames = {family: read_table(INPUTS[f"{family}_{suffix}"]) for family in FAMILIES}
    reference = frames["lightgbm"]
    for family, frame in frames.items():
        assert_aligned(reference, frame, f"{suffix}/{family}")
    return reference[KEYS].copy(), frames


def validate_sources(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    expected = {"feature_version": "v1.5.6", "cv_version": "balanced_v1", "seed": SEED}
    report: dict[str, object] = {"expected": expected, "sources": {}, "alignment": {family: "exact" for family in FAMILIES}}
    for family in FAMILIES:
        metadata = json.loads(INPUTS[f"{family}_metadata"].read_text(encoding="utf-8"))
        observed = {key: metadata.get(key) for key in expected}
        if observed != expected:
            raise ValueError(f"Metadata mismatch for {family}: {observed}")
        report["sources"][family] = observed
    for horizon in HORIZONS:
        for component in COMPONENTS:
            target = component_target(component, horizon)
            actual = frames["lightgbm"][f"actual_{target}"].to_numpy(dtype=float)
            for family in FAMILIES[1:]:
                assert_values_equal(actual, frames[family][f"actual_{target}"].to_numpy(dtype=float), f"{family}/{target}")
            for family in FAMILIES:
                pred = frames[family][f"pred_{target}"].to_numpy(dtype=float)
                if not np.array_equal(np.isfinite(actual), np.isfinite(pred)):
                    raise ValueError(f"Prediction mask differs: {family}/{target}")
    report["target_and_prediction_masks"] = "exact for all 16 component targets"
    return report


def run() -> None:
    started = time.perf_counter()
    for directory in (ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    frame, oof_sources = load_inputs(test=False)
    test_frame, test_sources = load_inputs(test=True)
    alignment = validate_sources(oof_sources)
    alignment["row_count"] = len(frame)
    alignment["test_row_count"] = len(test_frame)
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame["fold"].to_numpy(dtype=int)
    alignment["fold_alignment"] = "balanced_v1 county mapping applied identically to all three component OOF sources"
    alignment["fold_row_counts"] = {str(fold): int(np.sum(folds == fold)) for fold in range(5)}
    alignment["county_count"] = int(frame["fipsCode"].nunique())
    v18_oof = read_table(INPUTS["v18_oof"])
    v18_test = read_table(INPUTS["v18_test"])
    assert_aligned(frame, v18_oof, "v1.8 OOF rule")
    assert_aligned(test_frame, v18_test, "v1.8 test rule")

    oof_output = frame.copy()
    test_output = test_frame.copy()
    component_predictions: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    component_test_predictions: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    component_summary: list[dict[str, object]] = []
    component_folds: list[dict[str, object]] = []
    component_counties: list[dict[str, object]] = []
    distribution_metrics: list[dict[str, object]] = []
    weights_rows: list[dict[str, object]] = []
    residual_correlations: list[dict[str, object]] = []

    for horizon in HORIZONS:
        for component in COMPONENTS:
            target = component_target(component, horizon)
            y = oof_sources["lightgbm"][f"actual_{target}"].to_numpy(dtype=float)
            source = {family: oof_sources[family][f"pred_{target}"].to_numpy(dtype=float) for family in FAMILIES}
            matrix = np.column_stack([source[family] for family in FAMILIES])
            nested, outer_models = nested_predictions(y, matrix, source["lightgbm"], folds, processor=clip_component)
            all_predictions = {**source, **nested}
            component_predictions[(component, horizon)] = all_predictions
            oof_output[f"actual_{target}"] = y
            for model, prediction in all_predictions.items():
                oof_output[f"pred_{model}_{target}"] = prediction
                component_summary.append({"component": component, "horizon": horizon, "model": model, **metric_values(y, prediction)})
            for row in metric_rows(frame, y, all_predictions, horizon, "fold"):
                row["component"] = component
                component_folds.append(row)
            for row in metric_rows(frame, y, all_predictions, horizon, "fipsCode"):
                row["component"] = component
                component_counties.append(row)

            finite = np.isfinite(y)
            positive = finite & (y > 0.0)
            tail_threshold = float(np.quantile(y[positive], 0.95)) if positive.any() else np.nan
            masks = {"zero": finite & (y == 0.0), "positive": positive, "positive_tail_95": positive & (y >= tail_threshold)}
            for subset, mask in masks.items():
                for model, prediction in all_predictions.items():
                    distribution_metrics.append({"component": component, "horizon": horizon, "model": model, "subset": subset, "threshold": 0.0 if subset != "positive_tail_95" else tail_threshold, "actual_zero_pct": float(100 * np.mean(y[finite] == 0.0)), **metric_values(y[mask], prediction[mask])})
            for first_index, first in enumerate(FAMILIES):
                for second in FAMILIES[first_index + 1:]:
                    a = source[first][finite] - y[finite]
                    b = source[second][finite] - y[finite]
                    residual_correlations.append({"component": component, "horizon": horizon, "model_a": first, "model_b": second, "residual_correlation": float(np.corrcoef(a, b)[0, 1]), "mean_absolute_prediction_disagreement": float(np.mean(np.abs(source[first][finite] - source[second][finite])))})

            for model in outer_models:
                model.update({"component": component, "horizon": horizon, "candidate_names": list(FAMILIES), "baseline": "lightgbm", "component_clip": [0.0, 1.0]})
                write_json(MODELS_DIR / f"outer_fold{model['fold']}_{component}_{horizon}.json", model)
                for family, weight in zip(FAMILIES, model["weights"]):
                    weights_rows.append({"scope": "outer", "fold": model["fold"], "component": component, "horizon": horizon, "candidate": family, "weight": weight, "safe_alpha": model["safe_alpha"]})
            fitted_final = final_model(y, matrix, source["lightgbm"], processor=clip_component)
            fitted_final.update({"component": component, "horizon": horizon, "candidate_names": list(FAMILIES), "baseline": "lightgbm", "component_clip": [0.0, 1.0], "evaluation_use": "test inference only"})
            write_json(MODELS_DIR / f"final_{component}_{horizon}.json", fitted_final)
            for family, weight in zip(FAMILIES, fitted_final["weights"]):
                weights_rows.append({"scope": "final", "fold": "all", "component": component, "horizon": horizon, "candidate": family, "weight": weight, "safe_alpha": fitted_final["safe_alpha"]})

            test_source = {family: test_sources[family][f"pred_{target}"].to_numpy(dtype=float) for family in FAMILIES}
            test_matrix = np.column_stack([test_source[family] for family in FAMILIES])
            test_static = np.clip(test_matrix @ np.asarray(fitted_final["weights"]), 0.0, 1.0)
            test_nested = {
                "simple_average": np.clip(np.mean(test_matrix, axis=1), 0.0, 1.0),
                "static_convex": test_static,
                "safe_convex": np.clip(test_source["lightgbm"] + fitted_final["safe_alpha"] * (test_static - test_source["lightgbm"]), 0.0, 1.0),
            }
            component_test_predictions[(component, horizon)] = {**test_source, **test_nested}
            for model, prediction in component_test_predictions[(component, horizon)].items():
                test_output[f"pred_{model}_{target}"] = prediction

    component_summary_df = pd.DataFrame(component_summary)
    component_folds_df = pd.DataFrame(component_folds)
    component_baseline = component_folds_df[component_folds_df.model == "lightgbm"][["component", "horizon", "fold", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    component_folds_df = component_folds_df.merge(component_baseline, on=["component", "horizon", "fold"], how="left")
    component_folds_df["rmse_change_pct_vs_lightgbm"] = 100 * (component_folds_df.rmse / component_folds_df.baseline_rmse - 1)
    component_folds_df["improved_vs_lightgbm"] = component_folds_df.rmse < component_folds_df.baseline_rmse

    osi_summary: list[dict[str, object]] = []
    osi_folds: list[dict[str, object]] = []
    osi_counties: list[dict[str, object]] = []
    osi_bootstrap: list[dict[str, object]] = []
    osi_high: list[dict[str, object]] = []
    osi_models = (*FAMILIES, *METHODS)
    for horizon_index, horizon in enumerate(HORIZONS):
        y = v18_oof[f"actual_{horizon}"].to_numpy(dtype=float)
        component_actual = oof_sources["lightgbm"][f"actual_{component_target('P_t', horizon)}"].to_numpy(dtype=float)
        assert_values_equal(y, np.where(np.isfinite(component_actual), y, np.nan), f"v1.8 mask/{horizon}")
        baseline_column = f"pred_C3_aligned_component_{horizon}" if horizon in HORIZONS[:2] else f"pred_C1_component_osi_{horizon}"
        baseline = v18_oof[baseline_column].to_numpy(dtype=float)
        composed = {model: compose_osi({component: component_predictions[(component, horizon)][model] for component in COMPONENTS}) for model in osi_models}
        composed["v1.8_rule"] = baseline
        oof_output[f"actual_{horizon}"] = y
        for model, prediction in composed.items():
            oof_output[f"pred_osi_{model}_{horizon}"] = prediction
            metrics = metric_values(y, prediction)
            base_metrics = metric_values(y, baseline)
            osi_summary.append({"horizon": horizon, "model": model, **metrics, "rmse_change_pct_vs_v1.8": 100 * (metrics["rmse"] / base_metrics["rmse"] - 1), "mae_change_pct_vs_v1.8": 100 * (metrics["mae"] / base_metrics["mae"] - 1)})
        osi_folds.extend(metric_rows(frame, y, composed, horizon, "fold"))
        osi_counties.extend(metric_rows(frame, y, composed, horizon, "fipsCode"))
        valid = np.isfinite(y)
        threshold = float(np.quantile(y[valid], 0.95))
        high = valid & (y >= threshold)
        for model, prediction in composed.items():
            osi_high.append({"horizon": horizon, "model": model, "threshold": threshold, **metric_values(y[high], prediction[high])})
        for method_index, method in enumerate(METHODS):
            osi_bootstrap.append({"horizon": horizon, "model": method, **paired_county_bootstrap(frame, y, baseline, composed[method], seed=20260913 + 100 + horizon_index * 10 + method_index)})

        test_composed = {model: compose_osi({component: component_test_predictions[(component, horizon)][model] for component in COMPONENTS}) for model in osi_models}
        for model, prediction in test_composed.items():
            test_output[f"pred_osi_{model}_{horizon}"] = prediction

    osi_summary_df = pd.DataFrame(osi_summary)
    osi_folds_df = pd.DataFrame(osi_folds)
    osi_baseline_fold = osi_folds_df[osi_folds_df.model == "v1.8_rule"][["horizon", "fold", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    osi_folds_df = osi_folds_df.merge(osi_baseline_fold, on=["horizon", "fold"], how="left")
    osi_folds_df["rmse_change_pct_vs_v1.8"] = 100 * (osi_folds_df.rmse / osi_folds_df.baseline_rmse - 1)
    osi_folds_df["improved_vs_v1.8"] = osi_folds_df.rmse < osi_folds_df.baseline_rmse

    component_summary_df.to_csv(RESULTS_DIR / "component_summary_metrics.csv", index=False)
    component_folds_df.to_csv(RESULTS_DIR / "component_fold_metrics.csv", index=False)
    pd.DataFrame(component_counties).to_csv(RESULTS_DIR / "component_county_metrics.csv", index=False)
    pd.DataFrame(distribution_metrics).to_csv(RESULTS_DIR / "component_zero_tail_metrics.csv", index=False)
    pd.DataFrame(residual_correlations).to_csv(RESULTS_DIR / "component_residual_correlations.csv", index=False)
    pd.DataFrame(weights_rows).to_csv(RESULTS_DIR / "weights.csv", index=False)
    osi_summary_df.to_csv(RESULTS_DIR / "osi_summary_metrics.csv", index=False)
    osi_folds_df.to_csv(RESULTS_DIR / "osi_fold_metrics.csv", index=False)
    pd.DataFrame(osi_counties).to_csv(RESULTS_DIR / "osi_county_metrics.csv", index=False)
    pd.DataFrame(osi_bootstrap).to_csv(RESULTS_DIR / "paired_county_bootstrap.csv", index=False)
    pd.DataFrame(osi_high).to_csv(RESULTS_DIR / "high_osi_metrics.csv", index=False)
    oof_output.to_parquet(ARTIFACTS_DIR / "oof_component_and_osi_predictions.parquet", index=False)
    test_output.to_csv(ARTIFACTS_DIR / "test_component_and_osi_predictions.csv", index=False)
    template = pd.read_csv(INPUTS["submission_template"], dtype={"fipsCode": str})
    for horizon in HORIZONS:
        expected_mask = np.isfinite(template[horizon].to_numpy(dtype=float))
        prediction = test_output[f"pred_osi_safe_convex_{horizon}"].to_numpy(dtype=float)
        template[horizon] = np.where(expected_mask, prediction, np.nan)
    template.to_csv(ARTIFACTS_DIR / "submission_v2.6_safe_convex_balanced_v1.csv", index=False)

    decisions: dict[str, object] = {}
    for horizon in HORIZONS:
        row = osi_summary_df[(osi_summary_df.horizon == horizon) & (osi_summary_df.model == "safe_convex")].iloc[0]
        boot = next(item for item in osi_bootstrap if item["horizon"] == horizon and item["model"] == "safe_convex")
        improved_folds = int(osi_folds_df[(osi_folds_df.horizon == horizon) & (osi_folds_df.model == "safe_convex")]["improved_vs_v1.8"].sum())
        rmse_change = float(row["rmse_change_pct_vs_v1.8"])
        decisions[horizon] = {"rmse_change_pct": rmse_change, "improved_folds": improved_folds, "bootstrap_rmse_ci": [boot["rmse_ci_low"], boot["rmse_ci_high"]], "passes": bool(rmse_change <= -0.5 and improved_folds >= 4 and boot["rmse_ci_high"] < 0.0)}
    promotion = {"primary_candidate": "safe_convex", "horizons": decisions, "promote": bool(all(value["passes"] for value in decisions.values()))}
    write_json(RESULTS_DIR / "promotion_decision.json", promotion)
    write_json(ARTIFACTS_DIR / "alignment_report.json", alignment)
    output_paths = list(MODELS_DIR.glob("*.json")) + list(RESULTS_DIR.glob("*")) + [ARTIFACTS_DIR / "oof_component_and_osi_predictions.parquet", ARTIFACTS_DIR / "test_component_and_osi_predictions.csv", ARTIFACTS_DIR / "submission_v2.6_safe_convex_balanced_v1.csv", ARTIFACTS_DIR / "alignment_report.json"]
    metadata = {**runtime_metadata(), "experiment_version": "v2.6_component_tree_ensemble", "feature_version": "v1.5.6", "feature_count": 163, "cv_version": "balanced_v1", "seed": SEED, "bootstrap_seed": 20260913, "bootstrap_replicates": 2000, "families": list(FAMILIES), "solver": "deterministic enumeration of every non-empty active set for simplex-constrained least squares", "component_post_process": {"clip": [0.0, 1.0], "zero_threshold": None}, "strict_nested_cv": "Each component/horizon outer fold is scored with weights and safe alpha fitted on the other four OOF folds.", "final_model_usage": "Full-OOF component weights are used only for test inference.", "component_formula": "max(0, 0.40*P + 0.35*N + 0.25*D - 0.10*R), then clip [0,0.65] and values <0.001 to zero", "runtime_seconds": time.perf_counter() - started, "input_hashes": {name: sha256_file(path) for name, path in INPUTS.items()}, "artifact_hashes": hash_tree(output_paths, PROJECT_ROOT)}
    write_json(ARTIFACTS_DIR / "run_metadata.json", metadata)
    print(osi_summary_df[osi_summary_df.model.isin(["v1.8_rule", *METHODS])].to_string(index=False))
    print(f"Completed v2.6 in {metadata['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    run()
