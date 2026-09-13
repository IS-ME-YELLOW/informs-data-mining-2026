"""Three-leaf risk gate with strictly nested, region-wise safe shrinkage."""

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
    assert_aligned,
    hash_tree,
    load_row_folds,
    metric_rows,
    metric_values,
    normalize_frame,
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
V112_DIR = PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble"
GATED_HORIZONS = set(HORIZONS[2:])
RISK_QUANTILES = (0.90, 0.95)
REFERENCE_ALPHAS = (1.0, 1.0, 0.0)
MIN_INNER_FOLD_WINS = 3
ROBUST_STD_PENALTY = 0.25
BOOTSTRAP_SEED = 20260917

INPUTS = {
    **SOURCE_INPUTS,
    "v112_oof": V112_DIR / "artifacts/oof_predictions.parquet",
    "v112_test": V112_DIR / "artifacts/test_predictions.csv",
    "v112_metadata": V112_DIR / "artifacts/run_metadata.json",
}


def region_ids(risk: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return np.where(risk > thresholds[1], 2, np.where(risk > thresholds[0], 1, 0)).astype(int)


def fit_alpha(y: np.ndarray, baseline: np.ndarray, simple: np.ndarray, mask: np.ndarray) -> float:
    direction = simple[mask] - baseline[mask]
    denominator = float(direction @ direction)
    if denominator <= 1e-18:
        return 1.0
    return float(np.clip(direction @ (y[mask] - baseline[mask]) / denominator, 0.0, 1.0))


def predict(model: dict[str, object], baseline: np.ndarray, simple: np.ndarray) -> np.ndarray:
    if model["mode"] == "v1.12_unchanged":
        return baseline.copy()
    regions = region_ids(baseline, [float(value) for value in model["thresholds"]])
    alphas = np.asarray(model["alphas"], dtype=float)
    return post_process(baseline + alphas[regions] * (simple - baseline))


def fit_model(
    horizon: str,
    y: np.ndarray,
    baseline: np.ndarray,
    simple: np.ndarray,
    folds: np.ndarray,
    training_folds: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if horizon not in GATED_HORIZONS:
        return {"mode": "v1.12_unchanged", "horizon": horizon, "training_folds": training_folds}, []
    train = np.isin(folds, training_folds) & np.isfinite(y) & np.isfinite(baseline) & np.isfinite(simple)
    positive = baseline[train & (baseline > 0.0)]
    thresholds = [float(np.quantile(positive, quantile)) for quantile in RISK_QUANTILES]
    regions = region_ids(baseline, thresholds)
    alphas = list(REFERENCE_ALPHAS)
    diagnostics: list[dict[str, object]] = []
    for region in (0, 1):
        mask = train & (regions == region)
        proposed = fit_alpha(y, baseline, simple, mask)
        reference = baseline + REFERENCE_ALPHAS[region] * (simple - baseline)
        candidate = baseline + proposed * (simple - baseline)
        ratios: list[float] = []
        wins = 0
        for fold in training_folds:
            valid = (folds == fold) & mask
            if not valid.any():
                continue
            reference_mse = float(np.mean((reference[valid] - y[valid]) ** 2))
            candidate_mse = float(np.mean((candidate[valid] - y[valid]) ** 2))
            ratio = candidate_mse / reference_mse
            ratios.append(ratio)
            wins += int(ratio < 1.0)
        score = float(np.mean(ratios) + ROBUST_STD_PENALTY * np.std(ratios))
        safe = proposed < REFERENCE_ALPHAS[region] - 1e-12 and wins >= MIN_INNER_FOLD_WINS and score < 1.0
        if safe:
            alphas[region] = proposed
        diagnostics.append({"horizon": horizon, "region": region, "proposed_alpha": proposed, "selected_alpha": alphas[region], "inner_fold_wins": wins, "robust_score": score, "fold_mse_ratios": ratios, "safe_update": safe})
    return {
        "mode": "three_leaf_risk_gate",
        "horizon": horizon,
        "training_folds": training_folds,
        "risk_quantiles": list(RISK_QUANTILES),
        "thresholds": thresholds,
        "alphas": alphas,
        "reference_alphas": list(REFERENCE_ALPHAS),
        "leaf_definition": ["risk<=q90", "q90<risk<=q95", "risk>q95"],
        "gate_inputs": ["v1.8 deployable prediction"],
        "uses_labels_only_for_inner_alpha_fit": True,
    }, diagnostics


def run() -> None:
    started = time.perf_counter()
    for directory in (ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    frame, source, alignment = load_oof_inputs()
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame.fold.to_numpy(dtype=int)
    test_frame, test_source = load_test_inputs()
    v112 = normalize_frame(pd.read_parquet(INPUTS["v112_oof"]))
    v112_test = normalize_frame(pd.read_csv(INPUTS["v112_test"], dtype={"fipsCode": str}))
    assert_aligned(frame, v112, "v1.12 OOF")
    assert_aligned(test_frame, v112_test, "v1.12 test")
    template = pd.read_csv(INPUTS["submission_template"], dtype={"fipsCode": str})
    oof_output = frame.copy()
    test_output = test_frame.copy()
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    county_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    high_rows: list[dict[str, object]] = []
    leaf_rows: list[dict[str, object]] = []

    for horizon_index, horizon in enumerate(HORIZONS):
        y = source[horizon]["actual"]
        baseline = source[horizon]["lgbm_component"]
        simple = post_process(np.mean(np.column_stack([source[horizon][name] for name in CANDIDATES]), axis=1))
        current = v112[f"pred_tail_protected_{horizon}"].to_numpy(dtype=float)
        nested = np.full(len(frame), np.nan)
        for outer_fold in range(5):
            training_folds = [fold for fold in range(5) if fold != outer_fold]
            model, diagnostics = fit_model(horizon, y, baseline, simple, folds, training_folds)
            model.update({"scope": "outer", "outer_fold": outer_fold})
            write_json(MODELS_DIR / f"outer_fold{outer_fold}_{horizon}.json", model)
            leaf_rows.extend({**row, "scope": "outer", "outer_fold": outer_fold} for row in diagnostics)
            outer = folds == outer_fold
            if horizon not in GATED_HORIZONS:
                nested[outer] = current[outer]
            else:
                nested[outer] = predict(model, baseline[outer], simple[outer])
        final_model, diagnostics = fit_model(horizon, y, baseline, simple, folds, list(range(5)))
        final_model.update({"scope": "final", "evaluation_use": "test inference only"})
        write_json(MODELS_DIR / f"final_{horizon}.json", final_model)
        leaf_rows.extend({**row, "scope": "final", "outer_fold": "all"} for row in diagnostics)
        if horizon not in GATED_HORIZONS:
            test_nested = v112_test[f"pred_tail_protected_{horizon}"].to_numpy(dtype=float)
        else:
            test_matrix = np.column_stack([test_source[horizon][name] for name in CANDIDATES])
            test_simple = post_process(np.mean(test_matrix, axis=1))
            test_nested = predict(final_model, test_source[horizon]["lgbm_component"], test_simple)
        expected = np.isfinite(template[horizon].to_numpy(dtype=float))
        test_output[f"pred_shallow_gate_{horizon}"] = np.where(expected, test_nested, np.nan)

        predictions = {"v1.12": current, "shallow_risk_gate": nested}
        base_metrics = metric_values(y, current)
        oof_output[f"actual_{horizon}"] = y
        for name, values in predictions.items():
            metrics = metric_values(y, values)
            summary_rows.append({"horizon": horizon, "model": name, **metrics, "rmse_change_pct_vs_v1.12": 100 * (metrics["rmse"] / base_metrics["rmse"] - 1.0), "mae_change_pct_vs_v1.12": 100 * (metrics["mae"] / base_metrics["mae"] - 1.0)})
            oof_output[f"pred_{name}_{horizon}"] = values
        fold_rows.extend(metric_rows(frame, y, predictions, horizon, "fold"))
        county_rows.extend(metric_rows(frame, y, predictions, horizon, "fipsCode"))
        bootstrap_rows.append({"horizon": horizon, "model": "shallow_risk_gate", **paired_county_bootstrap(frame, y, current, nested, seed=BOOTSTRAP_SEED + horizon_index)})
        valid = np.isfinite(y)
        actual_cut = float(np.quantile(y[valid], 0.95))
        for name, values in predictions.items():
            high_rows.append({"horizon": horizon, "model": name, "subset": "actual_top5", **metric_values(y[valid & (y >= actual_cut)], values[valid & (y >= actual_cut)])})

    summary = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)
    baseline_fold = folds_df[folds_df.model == "v1.12"][["horizon", "fold", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    folds_df = folds_df.merge(baseline_fold, on=["horizon", "fold"], how="left")
    folds_df["rmse_change_pct_vs_v1.12"] = 100 * (folds_df.rmse / folds_df.baseline_rmse - 1.0)
    folds_df["improved_vs_v1.12"] = folds_df.rmse < folds_df.baseline_rmse
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    folds_df.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(county_rows).to_csv(RESULTS_DIR / "county_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS_DIR / "paired_county_bootstrap.csv", index=False)
    pd.DataFrame(high_rows).to_csv(RESULTS_DIR / "high_osi_metrics.csv", index=False)
    pd.DataFrame(leaf_rows).to_json(RESULTS_DIR / "leaf_diagnostics.json", orient="records", indent=2)
    oof_output.to_parquet(ARTIFACTS_DIR / "oof_predictions.parquet", index=False)
    test_output.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
    for horizon in HORIZONS:
        template[horizon] = test_output[f"pred_shallow_gate_{horizon}"].to_numpy()
    template.to_csv(ARTIFACTS_DIR / "submission_v1.15_shallow_gate_balanced_v1.csv", index=False)

    decision: dict[str, object] = {"baseline": "v1.12", "primary_candidate": "shallow_risk_gate", "horizons": {}}
    for horizon in HORIZONS:
        row = summary[(summary.horizon == horizon) & (summary.model == "shallow_risk_gate")].iloc[0]
        boot = next(item for item in bootstrap_rows if item["horizon"] == horizon)
        wins = int(folds_df[(folds_df.horizon == horizon) & (folds_df.model == "shallow_risk_gate")]["improved_vs_v1.12"].sum())
        unchanged = horizon not in GATED_HORIZONS
        passes = unchanged or (row["rmse_change_pct_vs_v1.12"] <= -0.5 and wins >= 4 and boot["rmse_ci_high"] < 0.0)
        decision["horizons"][horizon] = {"rmse_change_pct": float(row["rmse_change_pct_vs_v1.12"]), "improved_folds": wins, "bootstrap_rmse_ci": [boot["rmse_ci_low"], boot["rmse_ci_high"]], "unchanged_by_design": unchanged, "passes": bool(passes)}
    decision["promote"] = bool(all(value["passes"] for value in decision["horizons"].values()))
    write_json(RESULTS_DIR / "promotion_decision.json", decision)
    alignment.update({"risk_quantiles": list(RISK_QUANTILES), "leaf_count": 3, "gated_horizons": sorted(GATED_HORIZONS), "outer_alpha_fit": "other four county folds only"})
    write_json(ARTIFACTS_DIR / "alignment_report.json", alignment)
    outputs = list(MODELS_DIR.glob("*.json")) + list(RESULTS_DIR.glob("*")) + [ARTIFACTS_DIR / "oof_predictions.parquet", ARTIFACTS_DIR / "test_predictions.csv", ARTIFACTS_DIR / "submission_v1.15_shallow_gate_balanced_v1.csv", ARTIFACTS_DIR / "alignment_report.json"]
    metadata = {**runtime_metadata(), "experiment_version": "v1.15_shallow_risk_gate", "feature_version": "v1.5.6", "feature_count": 163, "cv_version": "balanced_v1", "seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED, "risk_quantiles": list(RISK_QUANTILES), "reference_alphas": list(REFERENCE_ALPHAS), "strict_nested_cv": "Every outer q90/q95 threshold and safe leaf alpha is fit on the other four county folds; final models are test-only.", "runtime_seconds": time.perf_counter() - started, "input_hashes": {name: sha256_file(path) for name, path in INPUTS.items()}, "artifact_hashes": hash_tree(outputs, PROJECT_ROOT)}
    write_json(ARTIFACTS_DIR / "run_metadata.json", metadata)
    print(summary.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
