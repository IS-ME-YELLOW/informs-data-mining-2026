"""Strict nested tail-protected ensemble built on the v1.11 source predictions."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERSION_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "versions/v1.11_tree_ensemble"))

from ensemble_utils import (  # noqa: E402
    HORIZONS,
    SEED,
    hash_tree,
    load_row_folds,
    metric_rows,
    metric_values,
    paired_county_bootstrap,
    post_process,
    runtime_metadata,
    sha256_file,
    write_json,
)
from run_ensemble import CANDIDATES, INPUTS as SOURCE_INPUTS, load_oof_inputs, load_test_inputs  # noqa: E402

ARTIFACTS_DIR = VERSION_DIR / "artifacts"
MODELS_DIR = VERSION_DIR / "models"
RESULTS_DIR = VERSION_DIR / "results"
TAIL_QUANTILE = 0.95
GATED_HORIZONS = set(HORIZONS[2:])
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_REPLICATES = 2000


def fit_model(baseline: np.ndarray, horizon: str, train_mask: np.ndarray) -> dict[str, object]:
    if horizon not in GATED_HORIZONS:
        return {"mode": "baseline_only", "tail_quantile": None, "threshold": None}
    positive = baseline[train_mask & np.isfinite(baseline) & (baseline > 0.0)]
    if not len(positive):
        raise ValueError(f"No positive baseline predictions for {horizon}")
    return {
        "mode": "simple_average_below_threshold_else_v1.8",
        "tail_quantile": TAIL_QUANTILE,
        "threshold": float(np.quantile(positive, TAIL_QUANTILE)),
    }


def predict(model: dict[str, object], baseline: np.ndarray, simple_average: np.ndarray) -> np.ndarray:
    if model["mode"] == "baseline_only":
        return baseline.copy()
    return np.where(baseline > float(model["threshold"]), baseline, simple_average)


def run() -> None:
    started = time.perf_counter()
    for directory in (ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    frame, source, alignment = load_oof_inputs()
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame["fold"].to_numpy(dtype=int)
    test_frame, test_source = load_test_inputs()
    template = pd.read_csv(SOURCE_INPUTS["submission_template"], dtype={"fipsCode": str})
    oof_output = frame.copy()
    test_output = test_frame.copy()
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    county_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []

    for horizon_index, horizon in enumerate(HORIZONS):
        y = source[horizon]["actual"]
        baseline = source[horizon]["lgbm_component"]
        matrix = np.column_stack([source[horizon][name] for name in CANDIDATES])
        simple = post_process(np.mean(matrix, axis=1))
        protected = np.full(len(frame), np.nan)
        for fold in range(5):
            inner = (folds != fold) & np.isfinite(y) & np.isfinite(baseline) & np.isfinite(simple)
            outer = (folds == fold) & np.isfinite(y) & np.isfinite(baseline) & np.isfinite(simple)
            model = fit_model(baseline, horizon, inner)
            model.update({"horizon": horizon, "fold": fold, "inner_rows": int(inner.sum()), "outer_rows": int(outer.sum()), "candidate_names": list(CANDIDATES), "average_weights": [1.0 / len(CANDIDATES)] * len(CANDIDATES), "gate_inputs": ["v1.8_rule_prediction"], "uses_labels_for_gate": False})
            protected[outer] = predict(model, baseline[outer], simple[outer])
            write_json(MODELS_DIR / f"outer_fold{fold}_{horizon}.json", model)
            threshold_rows.append({"scope": "outer", "fold": fold, "horizon": horizon, "threshold": model["threshold"], "tail_quantile": model["tail_quantile"]})

        predictions = {"v1.8_rule": baseline, "simple_average": simple, "tail_protected": protected}
        base_metrics = metric_values(y, baseline)
        for name, values in predictions.items():
            metrics = metric_values(y, values)
            summary_rows.append({"horizon": horizon, "model": name, **metrics, "rmse_change_pct_vs_v1.8": 100 * (metrics["rmse"] / base_metrics["rmse"] - 1.0), "mae_change_pct_vs_v1.8": 100 * (metrics["mae"] / base_metrics["mae"] - 1.0)})
            oof_output[f"actual_{horizon}"] = y
            oof_output[f"pred_{name}_{horizon}"] = values
        fold_rows.extend(metric_rows(frame, y, predictions, horizon, "fold"))
        county_rows.extend(metric_rows(frame, y, predictions, horizon, "fipsCode"))
        bootstrap_rows.append({"horizon": horizon, "model": "tail_protected", **paired_county_bootstrap(frame, y, baseline, protected, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED + horizon_index)})

        valid = np.isfinite(y) & np.isfinite(protected)
        actual_threshold = float(np.quantile(y[valid], 0.95))
        for subset, subset_mask in {"all": valid, "actual_top5": valid & (y >= actual_threshold)}.items():
            for name, values in predictions.items():
                gate_rows.append({"horizon": horizon, "subset": subset, "actual_threshold": actual_threshold, "model": name, **metric_values(y[subset_mask], values[subset_mask])})

        final_model = fit_model(baseline, horizon, np.isfinite(y) & np.isfinite(baseline) & np.isfinite(simple))
        final_model.update({"horizon": horizon, "fold": "all", "train_rows": int(np.isfinite(y).sum()), "candidate_names": list(CANDIDATES), "average_weights": [1.0 / len(CANDIDATES)] * len(CANDIDATES), "gate_inputs": ["v1.8_rule_prediction"], "uses_labels_for_gate": False, "evaluation_use": "test inference only"})
        write_json(MODELS_DIR / f"final_{horizon}.json", final_model)
        threshold_rows.append({"scope": "final", "fold": "all", "horizon": horizon, "threshold": final_model["threshold"], "tail_quantile": final_model["tail_quantile"]})
        test_matrix = np.column_stack([test_source[horizon][name] for name in CANDIDATES])
        test_simple = post_process(np.mean(test_matrix, axis=1))
        test_prediction = predict(final_model, test_source[horizon]["lgbm_component"], test_simple)
        expected_mask = np.isfinite(template[horizon].to_numpy(dtype=float))
        test_output[f"pred_tail_protected_{horizon}"] = np.where(expected_mask, test_prediction, np.nan)

    summary = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)
    baseline_fold = folds_df[folds_df.model == "v1.8_rule"][["horizon", "fold", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    folds_df = folds_df.merge(baseline_fold, on=["horizon", "fold"], how="left")
    folds_df["rmse_change_pct_vs_v1.8"] = 100 * (folds_df.rmse / folds_df.baseline_rmse - 1.0)
    folds_df["improved_vs_v1.8"] = folds_df.rmse < folds_df.baseline_rmse
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    folds_df.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(county_rows).to_csv(RESULTS_DIR / "county_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS_DIR / "paired_county_bootstrap.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(RESULTS_DIR / "tail_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(RESULTS_DIR / "thresholds.csv", index=False)
    oof_output.to_parquet(ARTIFACTS_DIR / "oof_predictions.parquet", index=False)
    test_output.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
    for horizon in HORIZONS:
        template[horizon] = test_output[f"pred_tail_protected_{horizon}"].to_numpy()
    template.to_csv(ARTIFACTS_DIR / "submission_v1.12_tail_protected_balanced_v1.csv", index=False)

    decision: dict[str, object] = {"primary_candidate": "tail_protected", "predeclared_rule": "v1.8 for t+1/t+6; simple six-model average below inner positive-prediction q95 and v1.8 above it for t+24/t+48", "horizons": {}}
    for horizon in HORIZONS:
        row = summary[(summary.horizon == horizon) & (summary.model == "tail_protected")].iloc[0]
        boot = next(item for item in bootstrap_rows if item["horizon"] == horizon)
        wins = int(folds_df[(folds_df.horizon == horizon) & (folds_df.model == "tail_protected")]["improved_vs_v1.8"].sum())
        unchanged = horizon not in GATED_HORIZONS
        passes = unchanged or (row["rmse_change_pct_vs_v1.8"] <= -0.5 and wins >= 4 and boot["rmse_ci_high"] < 0.0)
        decision["horizons"][horizon] = {"rmse_change_pct": float(row["rmse_change_pct_vs_v1.8"]), "improved_folds": wins, "bootstrap_rmse_ci": [boot["rmse_ci_low"], boot["rmse_ci_high"]], "unchanged_by_design": unchanged, "passes": bool(passes)}
    decision["promote"] = bool(all(value["passes"] for value in decision["horizons"].values()))
    write_json(RESULTS_DIR / "promotion_decision.json", decision)
    alignment.update({"fold_alignment": "outer county fold excluded when computing q95", "tail_quantile": TAIL_QUANTILE, "gated_horizons": sorted(GATED_HORIZONS)})
    write_json(ARTIFACTS_DIR / "alignment_report.json", alignment)
    output_paths = list(MODELS_DIR.glob("*.json")) + list(RESULTS_DIR.glob("*")) + [ARTIFACTS_DIR / "oof_predictions.parquet", ARTIFACTS_DIR / "test_predictions.csv", ARTIFACTS_DIR / "submission_v1.12_tail_protected_balanced_v1.csv", ARTIFACTS_DIR / "alignment_report.json"]
    metadata = {**runtime_metadata(), "experiment_version": "v1.12_tail_protected_ensemble", "feature_version": "v1.5.6", "feature_count": 163, "cv_version": "balanced_v1", "seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "tail_quantile": TAIL_QUANTILE, "gated_horizons": sorted(GATED_HORIZONS), "strict_nested_cv": "Each outer threshold is computed from positive baseline predictions in the other four county folds; no labels enter the gate.", "runtime_seconds": time.perf_counter() - started, "input_hashes": {name: sha256_file(path) for name, path in SOURCE_INPUTS.items()}, "artifact_hashes": hash_tree(output_paths, PROJECT_ROOT)}
    write_json(ARTIFACTS_DIR / "run_metadata.json", metadata)
    print(summary[summary.model.isin(["v1.8_rule", "tail_protected"])].to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
