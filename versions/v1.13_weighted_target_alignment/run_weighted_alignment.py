"""Learn robust lead-time weights for v1.8 C3 target-time alignment."""

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
    COMPONENTS,
    COMPONENT_WEIGHTS,
    HORIZONS,
    KEYS,
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
from run_ensemble import load_oof_inputs, load_test_inputs  # noqa: E402

ARTIFACTS_DIR = VERSION_DIR / "artifacts"
MODELS_DIR = VERSION_DIR / "models"
RESULTS_DIR = VERSION_DIR / "results"
V18_DIR = PROJECT_ROOT / "versions/v1.8"
V112_DIR = PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble"
LEAD_HOURS = (1, 6, 24, 48)
SHORT_HORIZONS = HORIZONS[:2]
GRID_STEP = 0.05
ROBUST_STD_PENALTY = 0.25
BOOTSTRAP_SEED = 20260915

INPUTS = {
    "v18_oof_components": V18_DIR / "oof_component_predictions.parquet",
    "v18_test_components": V18_DIR / "test_component_predictions.parquet",
    "v18_oof_controls": V18_DIR / "oof_predictions.parquet",
    "v18_test_controls": V18_DIR / "test_control_predictions.parquet",
    "v112_oof": V112_DIR / "artifacts/oof_predictions.parquet",
    "v112_test": V112_DIR / "artifacts/test_predictions.csv",
    "v112_metadata": V112_DIR / "artifacts/run_metadata.json",
    "cv": PROJECT_ROOT / "cv/cv_assignments_balanced_v1_seed42.csv",
    "submission_template": PROJECT_ROOT / "data/sample_submission.csv",
}


def component_column(component: str, horizon: str) -> str:
    return f"pred_{component}_target_{horizon.rsplit('_', 1)[-1]}"


def build_aligned_raw_matrices(meta: pd.DataFrame, component_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    meta = normalize_frame(meta)
    component_frame = normalize_frame(component_frame)
    assert_aligned(meta, component_frame, "component predictions")
    series_by_lead: dict[int, pd.Series] = {}
    for horizon, hours in zip(HORIZONS, LEAD_HOURS):
        raw = np.zeros(len(meta), dtype=float)
        finite = np.ones(len(meta), dtype=bool)
        for component in COMPONENTS:
            values = component_frame[component_column(component, horizon)].to_numpy(dtype=float)
            finite &= np.isfinite(values)
            raw += COMPONENT_WEIGHTS[component] * np.clip(values, 0.0, 1.0)
        valid = finite & (meta.hour_idx.to_numpy(dtype=int) + hours <= 215)
        index = pd.MultiIndex.from_arrays(
            [meta.loc[valid, "fipsCode"], meta.loc[valid, "timestamp_et"] + pd.to_timedelta(hours, unit="h")],
            names=["fipsCode", "target_timestamp"],
        )
        candidate = pd.Series(raw[valid], index=index)
        if candidate.index.duplicated().any():
            raise ValueError(f"Duplicate aligned source keys for lead {hours}")
        series_by_lead[hours] = candidate
    matrices: dict[str, np.ndarray] = {}
    for horizon, target_hours in zip(HORIZONS, LEAD_HOURS):
        target_index = pd.MultiIndex.from_arrays(
            [meta.fipsCode, meta.timestamp_et + pd.to_timedelta(target_hours, unit="h")],
            names=["fipsCode", "target_timestamp"],
        )
        matrix = np.column_stack([series_by_lead[hours].reindex(target_index).to_numpy(dtype=float) for hours in LEAD_HOURS])
        invalid = meta.hour_idx.to_numpy(dtype=int) + target_hours > 215
        matrix[invalid, :] = np.nan
        matrices[horizon] = matrix
    return matrices


def aligned_prediction(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(matrix)
    denominator = finite.astype(float) @ weights
    numerator = np.nan_to_num(matrix, nan=0.0) @ weights
    raw = np.full(len(matrix), np.nan)
    valid = denominator > 0.0
    raw[valid] = numerator[valid] / denominator[valid]
    return post_process(raw)


def weight_grid() -> np.ndarray:
    units = int(round(1.0 / GRID_STEP))
    candidates = []
    for lead1 in range(1, units + 1):
        for lead6 in range(units - lead1 + 1):
            for lead24 in range(units - lead1 - lead6 + 1):
                lead48 = units - lead1 - lead6 - lead24
                candidates.append(np.array([lead1, lead6, lead24, lead48], dtype=float) / units)
    return np.vstack(candidates)


def fit_weights(
    matrices: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    folds: np.ndarray,
    training_folds: list[int],
) -> tuple[np.ndarray, float, list[float]]:
    best_weights = None
    best_score = np.inf
    best_ratios: list[float] = []
    for weights in weight_grid():
        ratios: list[float] = []
        feasible = True
        for horizon in SHORT_HORIZONS:
            prediction = aligned_prediction(matrices[horizon], weights)
            for fold in training_folds:
                mask = (folds == fold) & np.isfinite(truth[horizon]) & np.isfinite(prediction)
                if not mask.any():
                    feasible = False
                    break
                candidate_mse = float(np.mean((prediction[mask] - truth[horizon][mask]) ** 2))
                baseline_mse = float(np.mean((baseline[horizon][mask] - truth[horizon][mask]) ** 2))
                ratios.append(candidate_mse / baseline_mse)
            if not feasible:
                break
        if not feasible:
            continue
        score = float(np.mean(ratios) + ROBUST_STD_PENALTY * np.std(ratios))
        distance = float(np.sum((weights - 0.25) ** 2))
        if score < best_score - 1e-15 or (abs(score - best_score) <= 1e-15 and best_weights is not None and distance < np.sum((best_weights - 0.25) ** 2)):
            best_score, best_weights, best_ratios = score, weights.copy(), ratios
    if best_weights is None:
        raise RuntimeError("No feasible lead-time weights")
    return best_weights, best_score, best_ratios


def run() -> None:
    started = time.perf_counter()
    for directory in (ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    frame, source, alignment = load_oof_inputs()
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame.fold.to_numpy(dtype=int)
    test_frame, _ = load_test_inputs()
    component_oof = normalize_frame(pd.read_parquet(INPUTS["v18_oof_components"]))
    component_test = normalize_frame(pd.read_parquet(INPUTS["v18_test_components"]))
    matrices = build_aligned_raw_matrices(frame[KEYS], component_oof)
    test_matrices = build_aligned_raw_matrices(test_frame[KEYS], component_test)
    v18_oof = normalize_frame(pd.read_parquet(INPUTS["v18_oof_controls"]))
    v112_oof = normalize_frame(pd.read_parquet(INPUTS["v112_oof"]))
    v112_test = normalize_frame(pd.read_csv(INPUTS["v112_test"], dtype={"fipsCode": str}))
    assert_aligned(frame, v18_oof, "v1.8 controls")
    assert_aligned(frame, v112_oof, "v1.12 OOF")
    assert_aligned(test_frame, v112_test, "v1.12 test")
    truth = {h: source[h]["actual"] for h in HORIZONS}
    v18_rule = {h: source[h]["lgbm_component"] for h in HORIZONS}
    v112 = {h: v112_oof[f"pred_tail_protected_{h}"].to_numpy(dtype=float) for h in HORIZONS}
    equal = np.full(4, 0.25)
    for horizon in HORIZONS:
        expected = v18_oof[f"pred_C3_aligned_component_{horizon}"].to_numpy(dtype=float)
        actual = aligned_prediction(matrices[horizon], equal)
        valid = np.isfinite(expected)
        if not np.array_equal(valid, np.isfinite(actual)) or np.max(np.abs(expected[valid] - actual[valid])) > 1e-12:
            raise ValueError(f"Equal-weight reconstruction does not reproduce C3 for {horizon}")

    weighted_oof = {h: np.full(len(frame), np.nan) for h in HORIZONS}
    weight_rows: list[dict[str, object]] = []
    for outer_fold in range(5):
        training_folds = [fold for fold in range(5) if fold != outer_fold]
        weights, score, ratios = fit_weights(matrices, truth, v18_rule, folds, training_folds)
        model = {"scope": "outer", "outer_fold": outer_fold, "training_folds": training_folds, "lead_hours": list(LEAD_HOURS), "weights": weights.tolist(), "grid_step": GRID_STEP, "objective": "mean inner fold/horizon MSE ratio to v1.8 C3 + 0.25*std", "objective_value": score, "inner_ratios": ratios}
        write_json(MODELS_DIR / f"outer_fold{outer_fold}.json", model)
        outer = folds == outer_fold
        for horizon in HORIZONS:
            values = aligned_prediction(matrices[horizon], weights)
            weighted_oof[horizon][outer] = values[outer]
        for lead, weight in zip(LEAD_HOURS, weights):
            weight_rows.append({"scope": "outer", "fold": outer_fold, "lead_hours": lead, "weight": weight, "objective_value": score})

    final_weights, final_score, final_ratios = fit_weights(matrices, truth, v18_rule, folds, list(range(5)))
    final_model = {"scope": "final", "training_folds": list(range(5)), "lead_hours": list(LEAD_HOURS), "weights": final_weights.tolist(), "grid_step": GRID_STEP, "objective": "mean fold/horizon MSE ratio to v1.8 C3 + 0.25*std", "objective_value": final_score, "inner_ratios": final_ratios, "evaluation_use": "test inference only"}
    write_json(MODELS_DIR / "final.json", final_model)
    for lead, weight in zip(LEAD_HOURS, final_weights):
        weight_rows.append({"scope": "final", "fold": "all", "lead_hours": lead, "weight": weight, "objective_value": final_score})

    combined = {h: weighted_oof[h] if h in SHORT_HORIZONS else v112[h] for h in HORIZONS}
    oof_output = frame.copy()
    test_output = test_frame.copy()
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    county_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    template = pd.read_csv(INPUTS["submission_template"], dtype={"fipsCode": str})
    for horizon_index, horizon in enumerate(HORIZONS):
        predictions = {"v1.12": v112[horizon], "weighted_alignment": weighted_oof[horizon], "weighted_alignment_plus_tail": combined[horizon]}
        base_metrics = metric_values(truth[horizon], v112[horizon])
        oof_output[f"actual_{horizon}"] = truth[horizon]
        for name, values in predictions.items():
            metrics = metric_values(truth[horizon], values)
            summary_rows.append({"horizon": horizon, "model": name, **metrics, "rmse_change_pct_vs_v1.12": 100 * (metrics["rmse"] / base_metrics["rmse"] - 1.0), "mae_change_pct_vs_v1.12": 100 * (metrics["mae"] / base_metrics["mae"] - 1.0)})
            oof_output[f"pred_{name}_{horizon}"] = values
        fold_rows.extend(metric_rows(frame, truth[horizon], predictions, horizon, "fold"))
        county_rows.extend(metric_rows(frame, truth[horizon], predictions, horizon, "fipsCode"))
        bootstrap_rows.append({"horizon": horizon, "model": "weighted_alignment_plus_tail", **paired_county_bootstrap(frame, truth[horizon], v112[horizon], combined[horizon], seed=BOOTSTRAP_SEED + horizon_index)})
        valid = np.isfinite(truth[horizon])
        threshold = float(np.quantile(truth[horizon][valid], 0.95))
        high = valid & (truth[horizon] >= threshold)
        for name, values in predictions.items():
            tail_rows.append({"horizon": horizon, "model": name, "threshold": threshold, **metric_values(truth[horizon][high], values[high])})
        test_weighted = aligned_prediction(test_matrices[horizon], final_weights)
        if horizon in SHORT_HORIZONS:
            test_combined = test_weighted
        else:
            test_combined = v112_test[f"pred_tail_protected_{horizon}"].to_numpy(dtype=float)
        mask = np.isfinite(template[horizon].to_numpy(dtype=float))
        test_output[f"pred_weighted_alignment_{horizon}"] = np.where(mask, test_weighted, np.nan)
        test_output[f"pred_weighted_alignment_plus_tail_{horizon}"] = np.where(mask, test_combined, np.nan)

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
    pd.DataFrame(tail_rows).to_csv(RESULTS_DIR / "high_osi_metrics.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(RESULTS_DIR / "lead_weights.csv", index=False)
    oof_output.to_parquet(ARTIFACTS_DIR / "oof_predictions.parquet", index=False)
    test_output.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
    for horizon in HORIZONS:
        template[horizon] = test_output[f"pred_weighted_alignment_plus_tail_{horizon}"].to_numpy()
    template.to_csv(ARTIFACTS_DIR / "submission_v1.13_weighted_alignment_balanced_v1.csv", index=False)

    decision: dict[str, object] = {"baseline": "v1.12", "primary_candidate": "weighted_alignment_plus_tail", "horizons": {}}
    for horizon in HORIZONS:
        row = summary[(summary.horizon == horizon) & (summary.model == "weighted_alignment_plus_tail")].iloc[0]
        boot = next(item for item in bootstrap_rows if item["horizon"] == horizon)
        wins = int(folds_df[(folds_df.horizon == horizon) & (folds_df.model == "weighted_alignment_plus_tail")]["improved_vs_v1.12"].sum())
        unchanged = horizon not in SHORT_HORIZONS
        passes = unchanged or (row["rmse_change_pct_vs_v1.12"] <= -0.5 and wins >= 4 and boot["rmse_ci_high"] < 0.0)
        decision["horizons"][horizon] = {"rmse_change_pct": float(row["rmse_change_pct_vs_v1.12"]), "improved_folds": wins, "bootstrap_rmse_ci": [boot["rmse_ci_low"], boot["rmse_ci_high"]], "unchanged_by_design": unchanged, "passes": bool(passes)}
    decision["promote"] = bool(all(item["passes"] for item in decision["horizons"].values()))
    write_json(RESULTS_DIR / "promotion_decision.json", decision)
    alignment.update({"equal_weight_reproduces_v1.8_C3": True, "lead_hours": list(LEAD_HOURS), "grid_step": GRID_STEP, "robust_std_penalty": ROBUST_STD_PENALTY})
    write_json(ARTIFACTS_DIR / "alignment_report.json", alignment)
    outputs = list(MODELS_DIR.glob("*.json")) + list(RESULTS_DIR.glob("*")) + [ARTIFACTS_DIR / "oof_predictions.parquet", ARTIFACTS_DIR / "test_predictions.csv", ARTIFACTS_DIR / "submission_v1.13_weighted_alignment_balanced_v1.csv", ARTIFACTS_DIR / "alignment_report.json"]
    metadata = {**runtime_metadata(), "experiment_version": "v1.13_weighted_target_alignment", "feature_version": "v1.5.6", "cv_version": "balanced_v1", "seed": SEED, "lead_hours": list(LEAD_HOURS), "grid_step": GRID_STEP, "grid_candidate_count": int(len(weight_grid())), "robust_std_penalty": ROBUST_STD_PENALTY, "strict_nested_cv": "Every outer lead-weight vector is selected on the other four county folds; final weights are test-only.", "runtime_seconds": time.perf_counter() - started, "input_hashes": {name: sha256_file(path) for name, path in INPUTS.items()}, "artifact_hashes": hash_tree(outputs, PROJECT_ROOT)}
    write_json(ARTIFACTS_DIR / "run_metadata.json", metadata)
    print(summary[summary.model.isin(["v1.12", "weighted_alignment_plus_tail"])].to_string(index=False))
    print("final lead weights", dict(zip(LEAD_HOURS, final_weights)))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
