"""Strict nested reuse of the existing v1.9 loss-specialist predictions."""

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

ARTIFACTS_DIR = VERSION_DIR / "artifacts"
MODELS_DIR = VERSION_DIR / "models"
RESULTS_DIR = VERSION_DIR / "results"
V18_DIR = PROJECT_ROOT / "versions/v1.8"
V19_DIR = PROJECT_ROOT / "versions/v1.9"
V112_DIR = PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble"
TAIL_QUANTILE = 0.95
ROBUST_STD_PENALTY = 0.25
MIN_INNER_FOLD_WINS = 3
BOOTSTRAP_SEED = 20260916
SPECIALIST_CONFIGS = {
    "base_pd_weighted_l2": {"P_t": "weighted_l2", "N_t": "huber", "D_t": "weighted_l2", "R_t": "huber"},
    "profile_pd_weighted_l2": {"P_t": "weighted_l2", "N_t": "huber", "D_t": "weighted_l2", "R_t": "huber"},
}
SPECIALISTS = tuple(SPECIALIST_CONFIGS)

INPUTS = {
    "v18_oof": V18_DIR / "oof_predictions.parquet",
    "v19_oof_components": V19_DIR / "oof_component_candidates.parquet",
    "v19_test_components": V19_DIR / "test_component_candidates.parquet",
    "v19_test_controls": V19_DIR / "test_predictions.parquet",
    "v19_metadata": V19_DIR / "run_metadata.json",
    "v112_oof": V112_DIR / "artifacts/oof_predictions.parquet",
    "v112_test": V112_DIR / "artifacts/test_predictions.csv",
    "v112_metadata": V112_DIR / "artifacts/run_metadata.json",
    "cv": PROJECT_ROOT / "cv/cv_assignments_balanced_v1_seed42.csv",
    "submission_template": PROJECT_ROOT / "data/sample_submission.csv",
}


def component_column(feature_set: str, component: str, objective: str, horizon: str) -> str:
    return f"pred__{feature_set}__{component}__{objective}__{horizon}"


def build_fixed_experts(frame: pd.DataFrame, horizon: str) -> dict[str, np.ndarray]:
    experts: dict[str, np.ndarray] = {}
    for name, config in SPECIALIST_CONFIGS.items():
        feature_set = name.split("_", 1)[0]
        raw = np.zeros(len(frame), dtype=float)
        for component, weight in {"P_t": 0.4, "N_t": 0.35, "D_t": 0.25, "R_t": -0.1}.items():
            values = frame[component_column(feature_set, component, config[component], horizon)].to_numpy(dtype=float)
            raw += weight * np.clip(values, 0.0, 1.0)
        experts[name] = post_process(raw)
    return experts


def risk_column(horizon: str) -> str:
    if horizon in HORIZONS[:2]:
        return f"pred_C3_aligned_component_{horizon}"
    return f"pred_C1_component_osi_{horizon}"


def fit_alpha(y: np.ndarray, baseline: np.ndarray, expert: np.ndarray, mask: np.ndarray) -> float:
    direction = expert[mask] - baseline[mask]
    denominator = float(direction @ direction)
    if denominator <= 1e-18:
        return 0.0
    alpha = float(direction @ (y[mask] - baseline[mask]) / denominator)
    return float(np.clip(alpha, 0.0, 1.0))


def apply_model(model: dict[str, object], current: np.ndarray, expert: np.ndarray, risk: np.ndarray) -> np.ndarray:
    if model["mode"] == "fallback_v1.12":
        return current.copy()
    gated = risk > float(model["threshold"])
    output = current.copy()
    output[gated] = post_process(
        current[gated] + float(model["alpha"]) * (expert[gated] - current[gated])
    )
    return output


def fit_model(
    horizon: str,
    y: np.ndarray,
    current: np.ndarray,
    risk: np.ndarray,
    experts: dict[str, np.ndarray],
    folds: np.ndarray,
    training_folds: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train = np.isin(folds, training_folds) & np.isfinite(y) & np.isfinite(current) & np.isfinite(risk)
    positive = risk[train & (risk > 0.0)]
    if not len(positive):
        raise ValueError(f"No positive risk prediction for {horizon}")
    threshold = float(np.quantile(positive, TAIL_QUANTILE))
    tail = train & (risk > threshold)
    candidate_rows: list[dict[str, object]] = []
    for name, expert in experts.items():
        fit_mask = tail & np.isfinite(expert)
        alpha = fit_alpha(y, current, expert, fit_mask)
        candidate = apply_model(
            {"mode": "tail_specialist", "threshold": threshold, "alpha": alpha}, current, expert, risk
        )
        ratios: list[float] = []
        tail_ratios: list[float] = []
        wins = 0
        for fold in training_folds:
            valid = (folds == fold) & np.isfinite(y) & np.isfinite(current) & np.isfinite(candidate)
            base_mse = float(np.mean((current[valid] - y[valid]) ** 2))
            cand_mse = float(np.mean((candidate[valid] - y[valid]) ** 2))
            ratio = cand_mse / base_mse
            ratios.append(ratio)
            wins += int(ratio < 1.0)
            valid_tail = valid & (risk > threshold)
            if valid_tail.any():
                base_tail = float(np.mean((current[valid_tail] - y[valid_tail]) ** 2))
                cand_tail = float(np.mean((candidate[valid_tail] - y[valid_tail]) ** 2))
                tail_ratios.append(cand_tail / base_tail)
        score = float(np.mean(ratios) + ROBUST_STD_PENALTY * np.std(ratios))
        candidate_rows.append(
            {
                "horizon": horizon,
                "specialist": name,
                "alpha": alpha,
                "threshold": threshold,
                "inner_fold_wins": wins,
                "inner_score": score,
                "mean_rmse_squared_ratio": float(np.mean(ratios)),
                "mean_tail_rmse_squared_ratio": float(np.mean(tail_ratios)),
                "inner_fold_ratios": ratios,
            }
        )
    best = min(candidate_rows, key=lambda row: (row["inner_score"], -row["inner_fold_wins"], row["specialist"]))
    safe = best["alpha"] > 0.0 and best["inner_fold_wins"] >= MIN_INNER_FOLD_WINS and best["inner_score"] < 1.0
    model = {
        "mode": "tail_specialist" if safe else "fallback_v1.12",
        "horizon": horizon,
        "training_folds": training_folds,
        "tail_quantile": TAIL_QUANTILE,
        "threshold": threshold,
        "specialist": best["specialist"],
        "alpha": best["alpha"] if safe else 0.0,
        "inner_fold_wins": best["inner_fold_wins"],
        "inner_score": best["inner_score"],
        "fallback_reason": None if safe else "requires alpha>0, at least 3 inner-fold wins, and robust score<1",
    }
    return model, candidate_rows


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v18 = normalize_frame(pd.read_parquet(INPUTS["v18_oof"]))
    v19 = normalize_frame(pd.read_parquet(INPUTS["v19_oof_components"]))
    v19_test = normalize_frame(pd.read_parquet(INPUTS["v19_test_components"]))
    v19_test_controls = normalize_frame(pd.read_parquet(INPUTS["v19_test_controls"]))
    v112 = normalize_frame(pd.read_parquet(INPUTS["v112_oof"]))
    v112_test = normalize_frame(pd.read_csv(INPUTS["v112_test"], dtype={"fipsCode": str}))
    assert_aligned(v18, v19, "v1.9 component OOF")
    assert_aligned(v18, v112, "v1.12 OOF")
    assert_aligned(v19_test, v19_test_controls, "v1.9 test controls")
    assert_aligned(v19_test, v112_test, "v1.12 test")
    return v18, v19, v19_test, v19_test_controls, v112, v112_test


def run() -> None:
    started = time.perf_counter()
    for directory in (ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    v18, v19, v19_test, v19_test_controls, v112, v112_test = load_inputs()
    frame = v18[KEYS].copy()
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame.fold.to_numpy(dtype=int)
    template = pd.read_csv(INPUTS["submission_template"], dtype={"fipsCode": str})
    oof_output = frame.copy()
    test_output = v19_test[KEYS].copy()
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    county_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    high_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for horizon_index, horizon in enumerate(HORIZONS):
        y = v112[f"actual_{horizon}"].to_numpy(dtype=float)
        current = v112[f"pred_tail_protected_{horizon}"].to_numpy(dtype=float)
        risk = v18[risk_column(horizon)].to_numpy(dtype=float)
        experts = build_fixed_experts(v19, horizon)
        nested = np.full(len(frame), np.nan)
        for outer_fold in range(5):
            training_folds = [fold for fold in range(5) if fold != outer_fold]
            model, rows = fit_model(horizon, y, current, risk, experts, folds, training_folds)
            model.update({"scope": "outer", "outer_fold": outer_fold})
            write_json(MODELS_DIR / f"outer_fold{outer_fold}_{horizon}.json", model)
            selection_rows.extend({**row, "scope": "outer", "outer_fold": outer_fold} for row in rows)
            outer = folds == outer_fold
            nested[outer] = apply_model(model, current[outer], experts[str(model["specialist"])][outer], risk[outer])

        final_model, rows = fit_model(horizon, y, current, risk, experts, folds, list(range(5)))
        final_model.update({"scope": "final", "evaluation_use": "test inference only"})
        write_json(MODELS_DIR / f"final_{horizon}.json", final_model)
        selection_rows.extend({**row, "scope": "final", "outer_fold": "all"} for row in rows)
        test_current = v112_test[f"pred_tail_protected_{horizon}"].to_numpy(dtype=float)
        test_risk = v19_test_controls[f"pred__V18_current_rule__{horizon}"].to_numpy(dtype=float)
        test_expert = build_fixed_experts(v19_test, horizon)[str(final_model["specialist"])]
        test_nested = apply_model(final_model, test_current, test_expert, test_risk)
        expected = np.isfinite(template[horizon].to_numpy(dtype=float))
        test_output[f"pred_tail_specialist_{horizon}"] = np.where(expected, test_nested, np.nan)

        predictions = {"v1.12": current, "tail_specialist_nested": nested}
        base_metrics = metric_values(y, current)
        oof_output[f"actual_{horizon}"] = y
        for name, values in predictions.items():
            metrics = metric_values(y, values)
            summary_rows.append({"horizon": horizon, "model": name, **metrics, "rmse_change_pct_vs_v1.12": 100 * (metrics["rmse"] / base_metrics["rmse"] - 1.0), "mae_change_pct_vs_v1.12": 100 * (metrics["mae"] / base_metrics["mae"] - 1.0)})
            oof_output[f"pred_{name}_{horizon}"] = values
        fold_rows.extend(metric_rows(frame, y, predictions, horizon, "fold"))
        county_rows.extend(metric_rows(frame, y, predictions, horizon, "fipsCode"))
        bootstrap_rows.append({"horizon": horizon, "model": "tail_specialist_nested", **paired_county_bootstrap(frame, y, current, nested, seed=BOOTSTRAP_SEED + horizon_index)})
        valid = np.isfinite(y) & np.isfinite(current)
        actual_cut = float(np.quantile(y[valid], 0.95))
        for subset, subset_mask in {"actual_top5": valid & (y >= actual_cut), "predicted_top5": valid & (risk > np.quantile(risk[valid & (risk > 0)], TAIL_QUANTILE))}.items():
            for name, values in predictions.items():
                high_rows.append({"horizon": horizon, "subset": subset, "model": name, **metric_values(y[subset_mask], values[subset_mask])})
        residual_current = y[valid] - current[valid]
        for name, expert in experts.items():
            diagnostic_rows.append({
                "horizon": horizon,
                "specialist": name,
                "residual_correlation_with_v1.12": float(np.corrcoef(residual_current, y[valid] - expert[valid])[0, 1]),
                "mean_absolute_prediction_disagreement": float(np.mean(np.abs(expert[valid] - current[valid]))),
            })

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
    pd.DataFrame(diagnostic_rows).to_csv(RESULTS_DIR / "specialist_diagnostics.csv", index=False)
    pd.DataFrame(selection_rows).to_json(RESULTS_DIR / "inner_selection_diagnostics.json", orient="records", indent=2)
    oof_output.to_parquet(ARTIFACTS_DIR / "oof_predictions.parquet", index=False)
    test_output.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
    for horizon in HORIZONS:
        template[horizon] = test_output[f"pred_tail_specialist_{horizon}"].to_numpy()
    template.to_csv(ARTIFACTS_DIR / "submission_v1.14_tail_specialist_balanced_v1.csv", index=False)

    decision: dict[str, object] = {"baseline": "v1.12", "primary_candidate": "tail_specialist_nested", "horizons": {}}
    for horizon in HORIZONS:
        row = summary[(summary.horizon == horizon) & (summary.model == "tail_specialist_nested")].iloc[0]
        boot = next(item for item in bootstrap_rows if item["horizon"] == horizon)
        wins = int(folds_df[(folds_df.horizon == horizon) & (folds_df.model == "tail_specialist_nested")]["improved_vs_v1.12"].sum())
        change = float(row["rmse_change_pct_vs_v1.12"])
        unchanged = bool(np.isclose(change, 0.0, atol=1e-12))
        passes = unchanged or (change <= -0.5 and wins >= 4 and boot["rmse_ci_high"] < 0.0)
        decision["horizons"][horizon] = {"rmse_change_pct": change, "improved_folds": wins, "bootstrap_rmse_ci": [boot["rmse_ci_low"], boot["rmse_ci_high"]], "unchanged_by_safe_fallback": unchanged, "passes": bool(passes)}
    decision["promote"] = bool(any(value["rmse_change_pct"] <= -0.5 for value in decision["horizons"].values()) and all(value["passes"] for value in decision["horizons"].values()))
    write_json(RESULTS_DIR / "promotion_decision.json", decision)
    alignment = {"row_alignment": "exact", "feature_version": "v1.5.6", "cv_version": "balanced_v1", "tail_gate_inputs": ["v1.8 deployable prediction"], "gate_uses_labels": False, "outer_selection": "specialist and alpha selected only on the other four county folds"}
    write_json(ARTIFACTS_DIR / "alignment_report.json", alignment)
    outputs = list(MODELS_DIR.glob("*.json")) + list(RESULTS_DIR.glob("*")) + [ARTIFACTS_DIR / "oof_predictions.parquet", ARTIFACTS_DIR / "test_predictions.csv", ARTIFACTS_DIR / "submission_v1.14_tail_specialist_balanced_v1.csv", ARTIFACTS_DIR / "alignment_report.json"]
    metadata = {**runtime_metadata(), "experiment_version": "v1.14_tail_specialist_reuse", "feature_version": "v1.5.6", "cv_version": "balanced_v1", "seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED, "specialists": SPECIALIST_CONFIGS, "specialist_preselection": "fixed unscreened P/D weighted-L2 with N/R Huber; does not use v1.9 best-subset selection", "tail_quantile": TAIL_QUANTILE, "strict_nested_cv": "Every outer specialist, tail threshold, and alpha is learned on the other four county folds; final models are test-only.", "runtime_seconds": time.perf_counter() - started, "input_hashes": {name: sha256_file(path) for name, path in INPUTS.items()}, "artifact_hashes": hash_tree(outputs, PROJECT_ROOT)}
    write_json(ARTIFACTS_DIR / "run_metadata.json", metadata)
    print(summary.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
