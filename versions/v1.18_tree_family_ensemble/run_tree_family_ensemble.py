"""Strict outer-fold ensemble of v1.12, ExtraTrees and Random Forest OOF."""

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
    HORIZONS, fit_simplex_least_squares, hash_tree, metric_rows, metric_values,
    paired_county_bootstrap, post_process, runtime_metadata, sha256_file, write_json,
)
from run_ensemble import INPUTS as BOOSTING_INPUTS, load_oof_inputs, load_test_inputs  # noqa: E402

ARTIFACTS_DIR = VERSION_DIR / "artifacts"
MODELS_DIR = VERSION_DIR / "models"
RESULTS_DIR = VERSION_DIR / "results"
LONG_HORIZONS = HORIZONS[2:]
TAIL_QUANTILE = 0.95
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260918
INPUTS = {
    "v112_oof": PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/oof_predictions.parquet",
    "v112_test": PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/test_predictions.csv",
    "v112_submission": PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/submission_v1.12_tail_protected_balanced_v1.csv",
    "et_oof": PROJECT_ROOT / "versions/v1.16_extratrees/artifacts/oof_predictions.parquet",
    "et_test": PROJECT_ROOT / "versions/v1.16_extratrees/artifacts/test_predictions.parquet",
    "rf_oof": PROJECT_ROOT / "versions/v1.17_random_forest/artifacts/oof_predictions.parquet",
    "rf_test": PROJECT_ROOT / "versions/v1.17_random_forest/artifacts/test_predictions.parquet",
}
INPUTS.update({f"boosting_{name}": path for name, path in BOOSTING_INPUTS.items()})
FAMILIES = {
    "v1.12": ("v1.12",),
    "v1.12+ExtraTrees": ("v1.12", "ExtraTrees"),
    "v1.12+RandomForest": ("v1.12", "RF"),
    "v1.12+ExtraTrees+RandomForest": ("v1.12", "ExtraTrees", "RF"),
}


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["fipsCode"] = out.fipsCode.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    out["timestamp_et"] = pd.to_datetime(out.timestamp_et)
    out["hour_idx"] = pd.to_numeric(out.hour_idx).astype(int)
    out["stateAbbr"] = out.stateAbbr.astype(str)
    return out


def aligned(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    keys = ["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]
    a, b = normalize(reference[keys]), normalize(other[keys])
    if len(a) != len(b) or any(not np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in keys):
        raise ValueError(f"{label} key alignment failed")


def choose_et(y: np.ndarray, et_a: np.ndarray, et_b: np.ndarray, mask: np.ndarray) -> str:
    a = metric_values(y[mask], et_a[mask])["rmse"]
    b = metric_values(y[mask], et_b[mask])["rmse"]
    return "ET-A" if a <= b else "ET-B"


def fit_fold_model(
    family: str, y: np.ndarray, sources: dict[str, np.ndarray], train: np.ndarray
) -> dict[str, object]:
    positive = sources["v1.8"][train & np.isfinite(sources["v1.8"]) & (sources["v1.8"] > 0)]
    threshold = float(np.quantile(positive, TAIL_QUANTILE))
    fit_mask = train & (sources["v1.8"] <= threshold)
    selected_et = choose_et(y, sources["ET-A"], sources["ET-B"], fit_mask) if "ExtraTrees" in FAMILIES[family] else None
    names = [selected_et if name == "ExtraTrees" else name for name in FAMILIES[family]]
    matrix = np.column_stack([sources[name] for name in names])
    weights = np.array([1.0]) if len(names) == 1 else fit_simplex_least_squares(matrix[fit_mask], y[fit_mask])
    return {"family": family, "source_names": names, "weights": weights.tolist(), "selected_extratrees": selected_et, "tail_quantile": TAIL_QUANTILE, "risk_source": "v1.8", "risk_threshold": threshold, "fit_rows_below_threshold": int(fit_mask.sum()), "uses_outer_labels": False}


def predict(model: dict[str, object], sources: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.column_stack([sources[name] for name in model["source_names"]])
    blended = post_process(matrix @ np.asarray(model["weights"], float))
    return np.where(sources["v1.8"] > float(model["risk_threshold"]), sources["v1.8"], blended)


def run() -> None:
    started = time.perf_counter()
    for directory in (ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    v112 = normalize(pd.read_parquet(INPUTS["v112_oof"]))
    et = normalize(pd.read_parquet(INPUTS["et_oof"]))
    rf = normalize(pd.read_parquet(INPUTS["rf_oof"]))
    aligned(v112, et, "ExtraTrees OOF")
    aligned(v112, rf, "RF OOF")
    folds = v112.fold.to_numpy(int)
    if set(folds) != set(range(5)):
        raise ValueError("Expected outer folds 0..4")
    v112_test = normalize(pd.read_csv(INPUTS["v112_test"], dtype={"fipsCode": str}))
    et_test = normalize(pd.read_parquet(INPUTS["et_test"]))
    rf_test = normalize(pd.read_parquet(INPUTS["rf_test"]))
    aligned(v112_test, et_test, "ExtraTrees test")
    aligned(v112_test, rf_test, "RF test")
    boosting_frame, boosting, boosting_alignment = load_oof_inputs()
    aligned(v112, boosting_frame, "boosting OOF")
    boosting_test_frame, boosting_test = load_test_inputs()
    aligned(v112_test, boosting_test_frame, "boosting test")

    output = v112[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr", "fold"]].copy()
    test_output = v112_test[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    summary_rows, fold_rows, county_rows, bootstrap_rows = [], [], [], []
    weight_rows, high_rows, corr_rows, disagreement_rows, oracle_rows, extreme_rows = [], [], [], [], [], []

    for hi, horizon in enumerate(HORIZONS):
        y = v112[f"actual_{horizon}"].to_numpy(float)
        baseline = v112[f"pred_v1.8_rule_{horizon}"].to_numpy(float)
        current = v112[f"pred_tail_protected_{horizon}"].to_numpy(float)
        output[f"actual_{horizon}"] = y
        output[f"pred_v1.8_{horizon}"] = baseline
        output[f"pred_v1.12_{horizon}"] = current
        if horizon not in LONG_HORIZONS:
            for family in FAMILIES:
                output[f"pred_{family}_{horizon}"] = current
                test_output[f"pred_{family}_{horizon}"] = v112_test[f"pred_tail_protected_{horizon}"].to_numpy(float)
            continue
        sources = {"v1.8": baseline, "v1.12": current, "ET-A": et[f"pred_ET-A_{horizon}"].to_numpy(float), "ET-B": et[f"pred_ET-B_{horizon}"].to_numpy(float), "RF": rf[f"pred_RF_{horizon}"].to_numpy(float)}
        test_sources = {"v1.8": boosting_test[horizon]["lgbm_component"], "v1.12": v112_test[f"pred_tail_protected_{horizon}"].to_numpy(float), "ET-A": et_test[f"pred_ET-A_{horizon}"].to_numpy(float), "ET-B": et_test[f"pred_ET-B_{horizon}"].to_numpy(float), "RF": rf_test[f"pred_RF_{horizon}"].to_numpy(float)}
        predictions: dict[str, np.ndarray] = {"v1.8": baseline, "v1.12": current}
        for family in FAMILIES:
            nested = np.full(len(y), np.nan)
            for fold in range(5):
                train = (folds != fold) & np.isfinite(y)
                outer = (folds == fold) & np.isfinite(y)
                model = fit_fold_model(family, y, sources, train)
                model.update({"scope": "outer", "fold": fold, "horizon": horizon, "train_rows": int(train.sum()), "outer_rows": int(outer.sum())})
                nested[outer] = predict(model, {k: v[outer] for k, v in sources.items()})
                write_json(MODELS_DIR / f"outer_fold{fold}_{horizon}_{family.replace('+', '_').replace('.', '')}.json", model)
                for source, weight in zip(model["source_names"], model["weights"]):
                    weight_rows.append({"scope": "outer", "fold": fold, "horizon": horizon, "family": family, "source": source, "weight": weight, "risk_threshold": model["risk_threshold"]})
            predictions[family] = nested
            output[f"pred_{family}_{horizon}"] = nested
            final = fit_fold_model(family, y, sources, np.isfinite(y))
            final.update({"scope": "final_test_only", "fold": "all", "horizon": horizon, "train_rows": int(np.isfinite(y).sum())})
            write_json(MODELS_DIR / f"final_{horizon}_{family.replace('+', '_').replace('.', '')}.json", final)
            test_output[f"pred_{family}_{horizon}"] = predict(final, test_sources)
            for source, weight in zip(final["source_names"], final["weights"]):
                weight_rows.append({"scope": "final", "fold": "all", "horizon": horizon, "family": family, "source": source, "weight": weight, "risk_threshold": final["risk_threshold"]})

        base_score = metric_values(y, baseline)
        current_score = metric_values(y, current)
        for name, values in predictions.items():
            score = metric_values(y, values)
            summary_rows.append({"horizon": horizon, "model": name, **score, "rmse_change_pct_vs_v1.8": 100 * (score["rmse"] / base_score["rmse"] - 1), "rmse_change_pct_vs_v1.12": 100 * (score["rmse"] / current_score["rmse"] - 1), "mae_change_pct_vs_v1.8": 100 * (score["mae"] / base_score["mae"] - 1)})
        fold_rows.extend(metric_rows(output, y, predictions, horizon, "fold"))
        county_rows.extend(metric_rows(output, y, predictions, horizon, "fipsCode"))
        threshold = float(np.nanquantile(y, 0.95))
        high = np.isfinite(y) & (y >= threshold)
        for name, values in predictions.items():
            high_rows.append({"horizon": horizon, "model": name, "actual_threshold": threshold, **metric_values(y[high], values[high])})
        for fi, family in enumerate(FAMILIES):
            bootstrap_rows.append({"horizon": horizon, "model": family, "reference": "v1.12", **paired_county_bootstrap(output, y, current, predictions[family], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + hi * 10 + fi)})

        diagnostic_sources = {"v1.12": current, "ET-A": sources["ET-A"], "ET-B": sources["ET-B"], "RF": sources["RF"]}
        for name in ("lgbm_direct", "lgbm_component", "xgboost_direct", "catboost_direct", "xgboost_component", "catboost_component"):
            diagnostic_sources[name] = boosting[horizon][name]
        names = list(diagnostic_sources)
        valid = np.isfinite(y)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                e1, e2 = diagnostic_sources[first][valid] - y[valid], diagnostic_sources[second][valid] - y[valid]
                delta = diagnostic_sources[first][valid] - diagnostic_sources[second][valid]
                corr_rows.append({"horizon": horizon, "model_a": first, "model_b": second, "residual_correlation": float(np.corrcoef(e1, e2)[0, 1])})
                disagreement_rows.append({"horizon": horizon, "model_a": first, "model_b": second, "mean_absolute_disagreement": float(np.mean(np.abs(delta))), "rmse_disagreement": float(np.sqrt(np.mean(delta ** 2)))})
        for label, members in {"v1.12_ET": ["v1.12", "ET-A", "ET-B"], "v1.12_RF": ["v1.12", "RF"], "v1.12_ET_RF": ["v1.12", "ET-A", "ET-B", "RF"]}.items():
            matrix = np.column_stack([diagnostic_sources[name][valid] for name in members])
            low, high_bound = matrix.min(axis=1), matrix.max(axis=1)
            oracle = np.clip(y[valid], low, high_bound)
            oracle_rows.append({"horizon": horizon, "family": label, "coverage_pct": 100 * float(np.mean((y[valid] >= low) & (y[valid] <= high_bound))), **metric_values(y[valid], oracle)})
        county_actual = output.loc[valid, ["fipsCode"]].copy()
        county_actual["actual_sq"] = y[valid] ** 2
        extremes = county_actual.groupby("fipsCode").actual_sq.mean().pow(0.5).nlargest(10).index
        for county in extremes:
            mask = valid & (output.fipsCode.to_numpy() == county)
            for name, values in predictions.items():
                extreme_rows.append({"horizon": horizon, "fipsCode": county, "model": name, **metric_values(y[mask], values[mask])})

    summary_df = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)
    current_fold = folds_df[folds_df.model == "v1.12"][["horizon", "fold", "rmse"]].rename(columns={"rmse": "v112_rmse"})
    folds_df = folds_df.merge(current_fold, on=["horizon", "fold"], how="left")
    folds_df["improved_vs_v1.12"] = folds_df.rmse < folds_df.v112_rmse
    folds_df["rmse_change_pct_vs_v1.12"] = 100 * (folds_df.rmse / folds_df.v112_rmse - 1)
    summary_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    folds_df.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(county_rows).to_csv(RESULTS_DIR / "county_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS_DIR / "paired_county_bootstrap.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(RESULTS_DIR / "weights.csv", index=False)
    pd.DataFrame(high_rows).to_csv(RESULTS_DIR / "high_osi_metrics.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / "residual_correlations.csv", index=False)
    pd.DataFrame(disagreement_rows).to_csv(RESULTS_DIR / "prediction_disagreement.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(RESULTS_DIR / "convex_hull_oracle.csv", index=False)
    pd.DataFrame(extreme_rows).to_csv(RESULTS_DIR / "extreme_counties.csv", index=False)
    output.to_parquet(ARTIFACTS_DIR / "oof_predictions.parquet", index=False)
    test_output.to_parquet(ARTIFACTS_DIR / "test_predictions.parquet", index=False)

    best_by_horizon: dict[str, object] = {}
    for horizon in LONG_HORIZONS:
        candidates = summary_df[(summary_df.horizon == horizon) & summary_df.model.isin(FAMILIES)]
        best = candidates.sort_values("rmse").iloc[0]
        boot = next(row for row in bootstrap_rows if row["horizon"] == horizon and row["model"] == best.model)
        wins = int(folds_df[(folds_df.horizon == horizon) & (folds_df.model == best.model)]["improved_vs_v1.12"].sum())
        best_by_horizon[horizon] = {"exploratory_best": best["model"], "rmse": float(best["rmse"]), "rmse_change_pct_vs_v1.8": float(best["rmse_change_pct_vs_v1.8"]), "rmse_change_pct_vs_v1.12": float(best["rmse_change_pct_vs_v1.12"]), "improved_folds_vs_v1.12": wins, "bootstrap_ci_vs_v1.12": [boot["rmse_ci_low"], boot["rmse_ci_high"]], "passes": bool(best["rmse_change_pct_vs_v1.8"] <= -0.5 and wins >= 4 and boot["rmse_ci_high"] < 0)}
    decision = {"reference": "v1.12", "selection_warning": "The best family is selected from the same five outer-fold scores and is exploratory; promotion requires all stated stability criteria.", "horizons": best_by_horizon, "promote": bool(all(v["passes"] for v in best_by_horizon.values()))}
    write_json(RESULTS_DIR / "promotion_decision.json", decision)
    submission = pd.read_csv(INPUTS["v112_submission"], dtype={"fipsCode": str})
    for horizon in LONG_HORIZONS:
        chosen = best_by_horizon[horizon]["exploratory_best"]
        values = test_output[f"pred_{chosen}_{horizon}"].to_numpy(float)
        mask = submission[horizon].notna().to_numpy()
        submission.loc[mask, horizon] = values[mask]
    submission.to_csv(ARTIFACTS_DIR / "submission_v1.18_exploratory_best_balanced_v1.csv", index=False)
    alignment_report = {"status": "pass", "row_count": len(output), "test_row_count": len(test_output), "keys": ["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"], "folds": "exact balanced_v1 labels inherited from verified v1.12 OOF", "feature_version": "v1.5.6", "boosting_alignment": boosting_alignment}
    write_json(ARTIFACTS_DIR / "alignment_report.json", alignment_report)
    generated = list(MODELS_DIR.glob("*.json")) + list(RESULTS_DIR.glob("*")) + [ARTIFACTS_DIR / "oof_predictions.parquet", ARTIFACTS_DIR / "test_predictions.parquet", ARTIFACTS_DIR / "submission_v1.18_exploratory_best_balanced_v1.csv", ARTIFACTS_DIR / "alignment_report.json"]
    metadata = {**runtime_metadata(), "experiment_version": "v1.18_tree_family_ensemble", "feature_version": "v1.5.6", "feature_count": 163, "cv_version": "balanced_v1", "seed": 42, "horizons_trained": list(LONG_HORIZONS), "short_horizons": "unchanged v1.12", "families": {k: list(v) for k, v in FAMILIES.items()}, "strict_nested_cv": "Every outer-fold ExtraTrees variant choice, q95 fallback threshold, and simplex weight is learned only from the other four county folds.", "tail_fallback": "v1.8 when its prediction exceeds the inner-training positive-prediction q95", "runtime_seconds": time.perf_counter() - started, "input_hashes": {name: sha256_file(path) for name, path in INPUTS.items()}, "artifact_hashes": hash_tree(generated, PROJECT_ROOT)}
    write_json(ARTIFACTS_DIR / "run_metadata.json", metadata)
    print(summary_df.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
