"""Nested primary evaluation, promotion gates, repeat blocking, and finalization."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from data_contract import (
    clip_component, fill_submission, load_inputs, post_process,
    recompose_with_replacements, sha256_file,
)
from distributional_config import (
    ALPHAS, ARTIFACTS_DIR, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
    COMPONENTS, HORIZONS, MANIFESTS_DIR, MODELS_DIR, MODES, PREDICTIONS_DIR,
    PRIMARY_SPLIT, RESULTS_DIR, V112_OOF, V112_TEST, candidate_names,
    ensure_output_dirs, horizon_suffix, resolved_spec, stable_fingerprint,
)
from folds import load_fold_assignments
from train_distributional import artifact_paths


def metrics(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if mask is not None:
        valid &= mask
    if not valid.any():
        return {"rmse": np.nan, "mae": np.nan, "n": 0}
    error = prediction[valid] - truth[valid]
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))), "n": int(valid.sum())}


def load_component_oof(inputs, split: str, horizon: str, component: str, candidate: str, row_folds: np.ndarray) -> np.ndarray:
    result = np.full(len(inputs.data.X_train), np.nan)
    for fold in range(5):
        path = artifact_paths(split, component, horizon, candidate, fold)["oof"]
        if not path.exists():
            raise FileNotFoundError(f"Missing OOF shard: {path}")
        shard = pd.read_parquet(path)
        index = shard["row_index"].to_numpy(dtype=int)
        if not np.all(row_folds[index] == fold):
            raise ValueError(f"OOF shard contains rows outside fold {fold}: {path}")
        if np.isfinite(result[index]).any():
            raise ValueError(f"OOF overlap in {path}")
        result[index] = shard["prediction_raw"].to_numpy(float)
    target_name = f"{component}_target_{horizon_suffix(horizon)}"
    expected = inputs.component_targets[target_name].notna().to_numpy()
    if np.isnan(result[expected]).any() or np.isfinite(result[~expected]).any():
        raise ValueError(f"OOF coverage differs from target mask: {split}/{component}/{horizon}/{candidate}")
    return clip_component(result)


def replacement_names() -> list[tuple[str, str | None, str | None]]:
    names = list(candidate_names())
    result = [(f"P__{p}", p, None) for p in names]
    result += [(f"D__{d}", None, d) for d in names]
    result += [(f"PD__{p}__{d}", p, d) for p in names for d in names]
    if len(result) != 48:
        raise AssertionError("Expected exactly 48 replacement candidates")
    return result


def blended(component_prediction: np.ndarray, baseline: np.ndarray, alpha: float,
            mode: str, threshold: float) -> np.ndarray:
    value = (1.0 - alpha) * baseline + alpha * component_prediction
    if mode == "protected":
        value = np.where(baseline >= threshold, baseline, value)
    return post_process(value)


def _positive_q95(values: np.ndarray, mask: np.ndarray) -> float:
    positive = values[mask & np.isfinite(values) & (values > 0)]
    if not len(positive):
        return np.inf
    return float(np.quantile(positive, 0.95))


def _select_on_outer_train(truth: np.ndarray, component_pred: np.ndarray, baseline: np.ndarray,
                           outer_train: np.ndarray, candidate_name: str) -> dict:
    threshold = _positive_q95(baseline, outer_train)
    choices = []
    for alpha in ALPHAS:
        for mode in MODES:
            pred = blended(component_pred, baseline, alpha, mode, threshold)
            score = metrics(truth, pred, outer_train)
            choices.append({"alpha": alpha, "mode": mode, "threshold": threshold, "candidate": candidate_name, **score})
    # RMSE, smaller alpha, protected, candidate name.
    choices.sort(key=lambda x: (x["rmse"], x["alpha"], 0 if x["mode"] == "protected" else 1, x["candidate"]))
    return choices[0]


def _county_bootstrap(meta: pd.DataFrame, truth: np.ndarray, candidate: np.ndarray,
                      baseline: np.ndarray, replicates: int, seed: int) -> dict:
    counties = meta["fipsCode"].drop_duplicates().to_numpy()
    mapping = {county: i for i, county in enumerate(counties)}
    county_idx = meta["fipsCode"].map(mapping).to_numpy(int)
    valid = np.isfinite(truth) & np.isfinite(candidate) & np.isfinite(baseline)
    n = np.bincount(county_idx[valid], minlength=len(counties))
    sse_c = np.bincount(county_idx[valid], weights=(candidate[valid] - truth[valid]) ** 2, minlength=len(counties))
    sse_b = np.bincount(county_idx[valid], weights=(baseline[valid] - truth[valid]) ** 2, minlength=len(counties))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(counties), size=(replicates, len(counties)))
    counts = n[sampled].sum(axis=1)
    diff = np.sqrt(sse_c[sampled].sum(axis=1) / counts) - np.sqrt(sse_b[sampled].sum(axis=1) / counts)
    low, high = np.quantile(diff, [0.025, 0.975])
    return {"difference": float(metrics(truth, candidate)["rmse"] - metrics(truth, baseline)["rmse"]), "ci_2_5": float(low), "ci_97_5": float(high), "improvement_probability": float(np.mean(diff < 0)), "replicates": replicates, "unit": "county_full_trajectory"}


def _candidate_component_predictions(inputs, split: str, horizon: str, row_folds: np.ndarray) -> dict[str, np.ndarray]:
    p = {name: load_component_oof(inputs, split, horizon, "P_t", name, row_folds) for name in candidate_names()}
    d = {name: load_component_oof(inputs, split, horizon, "D_t", name, row_folds) for name in candidate_names()}
    result = {}
    for name, p_name, d_name in replacement_names():
        replacements = {}
        if p_name:
            replacements["P_t"] = p[p_name]
        if d_name:
            replacements["D_t"] = d[d_name]
        result[name] = recompose_with_replacements(inputs.v18_oof_components, horizon, replacements)
    return result


def evaluate_primary(bootstrap_replicates: int) -> dict:
    inputs = load_inputs(require_v112=True)
    _, row_folds, _ = load_fold_assignments(PRIMARY_SPLIT, inputs.data.meta_train)
    nested_frame = inputs.data.meta_train[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    selection_rows, summary_rows, fold_rows, county_rows, high_rows, correlation_rows, bootstrap_rows = [], [], [], [], [], [], []
    decisions = {}
    for h_index, horizon in enumerate(HORIZONS):
        truth = inputs.data.y_train[horizon].to_numpy(float)
        baseline = inputs.v112_oof[f"pred_tail_protected_{horizon}"].to_numpy(float)
        component_candidates = _candidate_component_predictions(inputs, PRIMARY_SPLIT, horizon, row_folds)
        nested_predictions: dict[str, np.ndarray] = {}
        for candidate_name, component_pred in component_candidates.items():
            nested = np.full(len(truth), np.nan)
            for outer_fold in range(5):
                outer_train = (row_folds != outer_fold) & np.isfinite(truth)
                outer_valid = (row_folds == outer_fold) & np.isfinite(truth)
                choice = _select_on_outer_train(truth, component_pred, baseline, outer_train, candidate_name)
                prediction = blended(component_pred, baseline, choice["alpha"], choice["mode"], choice["threshold"])
                nested[outer_valid] = prediction[outer_valid]
                selection_rows.append({"horizon": horizon, "candidate": candidate_name, "outer_fold": outer_fold, **choice})
            if np.isnan(nested[np.isfinite(truth)]).any():
                raise ValueError(f"Nested OOF coverage incomplete: {horizon}/{candidate_name}")
            nested_predictions[candidate_name] = nested
            nested_frame[f"pred_{horizon_suffix(horizon)}__{candidate_name}"] = nested
            score = metrics(truth, nested)
            base_score = metrics(truth, baseline)
            summary_rows.append({"horizon": horizon, "candidate": candidate_name, **score, "baseline_rmse": base_score["rmse"], "baseline_mae": base_score["mae"], "rmse_change_pct": 100 * (score["rmse"] / base_score["rmse"] - 1), "mae_change_pct": 100 * (score["mae"] / base_score["mae"] - 1)})
            for fold in range(5):
                hit = row_folds == fold
                cs, bs = metrics(truth, nested, hit), metrics(truth, baseline, hit)
                fold_rows.append({"horizon": horizon, "candidate": candidate_name, "fold": fold, "candidate_rmse": cs["rmse"], "baseline_rmse": bs["rmse"], "rmse_difference": cs["rmse"] - bs["rmse"], "rmse_change_pct": 100 * (cs["rmse"] / bs["rmse"] - 1), "candidate_mae": cs["mae"], "baseline_mae": bs["mae"]})
            for county in inputs.data.meta_train["fipsCode"].drop_duplicates():
                hit = inputs.data.meta_train["fipsCode"].to_numpy() == county
                cs, bs = metrics(truth, nested, hit), metrics(truth, baseline, hit)
                county_rows.append({"horizon": horizon, "candidate": candidate_name, "fipsCode": county, "candidate_rmse": cs["rmse"], "baseline_rmse": bs["rmse"], "rmse_difference": cs["rmse"] - bs["rmse"], "candidate_mae": cs["mae"], "baseline_mae": bs["mae"]})
            valid_truth = truth[np.isfinite(truth)]
            top5_threshold = float(np.quantile(valid_truth, 0.95))
            top5_mask = truth >= top5_threshold
            top_cs, top_bs = metrics(truth, nested, top5_mask), metrics(truth, baseline, top5_mask)
            high_rows.append({"horizon": horizon, "candidate": candidate_name, "actual_top5_threshold": top5_threshold, "candidate_rmse": top_cs["rmse"], "baseline_rmse": top_bs["rmse"], "rmse_change_pct": 100 * (top_cs["rmse"] / top_bs["rmse"] - 1), "candidate_mae": top_cs["mae"], "baseline_mae": top_bs["mae"], "mae_change_pct": 100 * (top_cs["mae"] / top_bs["mae"] - 1), "n": top_cs["n"]})
            valid = np.isfinite(truth) & np.isfinite(nested) & np.isfinite(baseline)
            candidate_error, baseline_error = nested[valid] - truth[valid], baseline[valid] - truth[valid]
            correlation_rows.append({"horizon": horizon, "candidate": candidate_name, "pearson": float(pearsonr(candidate_error, baseline_error).statistic), "spearman": float(spearmanr(candidate_error, baseline_error).statistic), "n": int(valid.sum())})
            bootstrap = _county_bootstrap(inputs.data.meta_train, truth, nested, baseline, bootstrap_replicates, BOOTSTRAP_SEED + h_index)
            bootstrap_rows.append({"horizon": horizon, "candidate": candidate_name, **bootstrap})
        winner = min(summary_rows[-48:], key=lambda x: (x["rmse"], x["candidate"]))
        winner_name = winner["candidate"]
        winner_folds = [row for row in fold_rows if row["horizon"] == horizon and row["candidate"] == winner_name]
        fold_wins = sum(row["candidate_rmse"] < row["baseline_rmse"] for row in winner_folds)
        winner_high = next(row for row in high_rows if row["horizon"] == horizon and row["candidate"] == winner_name)
        winner_bootstrap = next(row for row in bootstrap_rows if row["horizon"] == horizon and row["candidate"] == winner_name)
        parts = winner_name.split("__")
        p_candidate = parts[1] if parts[0] in ("P", "PD") else None
        d_candidate = parts[-1] if parts[0] in ("D", "PD") else None
        promoted = (
            winner["rmse_change_pct"] <= -0.5 and fold_wins >= 4 and winner_bootstrap["ci_97_5"] < 0
            and winner_high["rmse_change_pct"] <= 0.25 and winner["mae_change_pct"] <= 1.0
        )
        full_choice = _select_on_outer_train(truth, component_candidates[winner_name], baseline, np.isfinite(truth), winner_name)
        decisions[horizon] = {
            "winner": winner_name, "p_candidate": p_candidate, "d_candidate": d_candidate,
            "pooled_rmse_change_pct": winner["rmse_change_pct"], "pooled_mae_change_pct": winner["mae_change_pct"],
            "fold_wins": int(fold_wins), "bootstrap_ci_97_5": winner_bootstrap["ci_97_5"],
            "actual_top5_rmse_change_pct": winner_high["rmse_change_pct"], "promoted": bool(promoted),
            "full_data_alpha": full_choice["alpha"], "full_data_mode": full_choice["mode"],
            "full_data_protection_threshold": full_choice["threshold"],
        }
    nested_frame.to_parquet(PREDICTIONS_DIR / "primary_all_candidates_nested_oof.parquet", index=False)
    tables = {
        "nested_selection_by_outer_fold.csv": selection_rows, "all_candidate_summary.csv": summary_rows,
        "primary_winner_fold_metrics.csv": fold_rows, "primary_winner_county_metrics.csv": county_rows,
        "primary_winner_high_osi_metrics.csv": high_rows, "primary_winner_residual_correlations.csv": correlation_rows,
        "primary_winner_county_bootstrap.csv": bootstrap_rows,
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(RESULTS_DIR / name, index=False)
    payload = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "primary", "horizons": decisions,
        "repeat_policy": "Repeats require independently retrained v1.12 baselines on each repeat split.",
    }
    (MANIFESTS_DIR / "primary_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate_repeats() -> dict:
    primary_path = MANIFESTS_DIR / "primary_decision.json"
    if not primary_path.exists():
        raise FileNotFoundError("Run primary evaluation first")
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    any_promoted = any(item.get("promoted") for item in primary["horizons"].values())
    payload = {
        "schema_version": 1,
        "stage": "repeats",
        "status": "blocked_requires_baseline_retraining" if any_promoted else "not_run_primary_not_promoted",
        "final_promoted": False,
        "reason": (
            "v1.12 original OOF predictions are tied to balanced_v1 folds and cannot serve as independent "
            "baselines for repeat folds. Retrain the v1.12 baseline on each repeat assignment before making "
            "repeat or final-promotion claims."
        ) if any_promoted else "No primary winner passed all gates.",
        "prohibited_baseline": str(V112_OOF),
    }
    (MANIFESTS_DIR / "repeat_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def finalize(num_threads: int, quick: bool) -> None:
    repeat_path = MANIFESTS_DIR / "repeat_decision.json"
    if not repeat_path.exists():
        raise FileNotFoundError("Repeat decision is required before finalization")
    repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
    if repeat.get("status") == "blocked_requires_baseline_retraining":
        raise RuntimeError("Finalization blocked: repeat baselines must be retrained; final promotion cannot be inferred from original OOF")
    if not repeat.get("final_promoted", False):
        raise RuntimeError("Finalization refused because final promotion is false")
    # This path is intentionally reachable only after a future independent-repeat implementation
    # writes a valid final decision with horizon winners.
    final_decision = repeat.get("horizons", {})
    inputs = load_inputs(require_v112=True)
    submissions = {}
    model_manifest = []
    for horizon, decision in final_decision.items():
        replacements = {}
        for component, key in (("P_t", "p_candidate"), ("D_t", "d_candidate")):
            candidate = decision.get(key)
            if not candidate:
                continue
            sidecars = [json.loads(artifact_paths(PRIMARY_SPLIT, component, horizon, candidate, fold)["sidecar"].read_text(encoding="utf-8")) for fold in range(5)]
            rounds = max(1, int(round(np.mean([item["best_iteration"] for item in sidecars]))))
            spec = resolved_spec(candidate, num_threads, quick)
            target = f"{component}_target_{horizon_suffix(horizon)}"
            valid = inputs.component_targets[target].notna().to_numpy()
            booster = lgb.train(spec["params"], lgb.Dataset(inputs.data.X_train.loc[valid], label=inputs.component_targets.loc[valid, target]), num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
            path = MODELS_DIR / "final" / f"final__{component}__{horizon_suffix(horizon)}__{candidate}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(path), num_iteration=rounds)
            replacements[component] = clip_component(booster.predict(inputs.data.X_test, num_iteration=rounds))
            model_manifest.append({"horizon": horizon, "component": component, "candidate": candidate, "rounds": rounds, "path": str(path), "sha256": sha256_file(path)})
        component_test = recompose_with_replacements(inputs.v18_test_components, horizon, replacements)
        baseline_test = inputs.v112_test[f"pred_tail_protected_{horizon}"].to_numpy(float)
        submissions[horizon] = blended(component_test, baseline_test, float(decision["alpha"]), decision["mode"], float(decision["protection_threshold"]))
    submission = fill_submission(inputs.data.meta_test, submissions)
    output = ARTIFACTS_DIR / "submission_v1.20_lgbm_distributional.csv"
    submission.to_csv(output, index=False)
    (MANIFESTS_DIR / "final_model_manifest.json").write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("primary", "repeats", "finalize"))
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--num-threads", type=int, default=-1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    if args.stage == "primary":
        if args.bootstrap_replicates < 100:
            parser.error("bootstrap replicates must be at least 100")
        print(json.dumps(evaluate_primary(args.bootstrap_replicates), indent=2))
    elif args.stage == "repeats":
        evaluate_repeats()
    else:
        finalize(args.num_threads, args.quick)


if __name__ == "__main__":
    main()
