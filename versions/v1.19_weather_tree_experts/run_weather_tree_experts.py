"""Strictly nested shallow weather-mechanism tree selecting forest experts."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERSION_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "versions"))
sys.path.insert(0, str(PROJECT_ROOT / "versions/v1.11_tree_ensemble"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

from bagging_tree_experiment import FEATURE_VERSION, load_experiment_data  # noqa: E402
from ensemble_utils import fit_safe_alpha, hash_tree, metric_rows, metric_values, paired_county_bootstrap, post_process, sha256_file, write_json  # noqa: E402

ARTIFACTS_DIR = VERSION_DIR / "artifacts"
MODELS_DIR = VERSION_DIR / "models"
RESULTS_DIR = VERSION_DIR / "results"
HORIZONS = ("osi_target_t24h", "osi_target_t48h")
DEPTHS = (2, 3)
EXPERTS = ("v1.12", "ET-A", "ET-B", "RF")
TAIL_QUANTILE = 0.95
MIN_SAMPLES_LEAF = 500
SEED = 42
BOOTSTRAP_SEED = 20260919
INPUTS = {
    "v112_oof": PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/oof_predictions.parquet",
    "v112_test": PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/test_predictions.csv",
    "v112_submission": PROJECT_ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/submission_v1.12_tail_protected_balanced_v1.csv",
    "v18_test": PROJECT_ROOT / "versions/v1.8/test_control_predictions.parquet",
    "et_oof": PROJECT_ROOT / "versions/v1.16_extratrees/artifacts/oof_predictions.parquet",
    "et_test": PROJECT_ROOT / "versions/v1.16_extratrees/artifacts/test_predictions.parquet",
    "rf_oof": PROJECT_ROOT / "versions/v1.17_random_forest/artifacts/oof_predictions.parquet",
    "rf_test": PROJECT_ROOT / "versions/v1.17_random_forest/artifacts/test_predictions.parquet",
    "features_train": PROJECT_ROOT / f"cache/features_train_{FEATURE_VERSION}.parquet",
    "features_test": PROJECT_ROOT / f"cache/features_test_{FEATURE_VERSION}.parquet",
    "meta_train": PROJECT_ROOT / f"cache/meta_train_{FEATURE_VERSION}.parquet",
    "meta_test": PROJECT_ROOT / f"cache/meta_test_{FEATURE_VERSION}.parquet",
    "feature_names": PROJECT_ROOT / f"cache/feature_names_{FEATURE_VERSION}.json",
    "cv": PROJECT_ROOT / "cv/cv_assignments_balanced_v1_seed42.csv",
}


def feature_names(horizon: str) -> list[str]:
    hours = "24h" if horizon.endswith("24h") else "48h"
    return [
        "last_osi", "t2m_t", "tp_t", "csnow_t", "is_snowing",
        "gust_t", "gust_exceed_30", "gust_max_last6h_obs", "gust_trend_last6h_obs",
        "cumul_gust_since_onset", "cumul_tp_since_onset",
        f"gust_max_next_{hours}", f"gust_mean_next_{hours}", f"total_tp_next_{hours}",
        f"min_t2m_next_{hours}", f"gust_gt30_frac_next_{hours}",
    ]


def assert_keys(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    left = reference[["fipsCode", "timestamp_et"]].copy()
    right = other[["fipsCode", "timestamp_et"]].copy()
    for frame in (left, right):
        frame["fipsCode"] = frame.fipsCode.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        frame["timestamp_et"] = pd.to_datetime(frame.timestamp_et)
    if len(left) != len(right) or not np.array_equal(left.to_numpy(), right.to_numpy()):
        raise ValueError(f"{label} county/timestamp alignment failed")


def build(depth: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gate", DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=SEED)),
    ])


def source_dict(v112: pd.DataFrame, et: pd.DataFrame, rf: pd.DataFrame, horizon: str) -> dict[str, np.ndarray]:
    return {
        "v1.8": v112[f"pred_v1.8_rule_{horizon}"].to_numpy(float),
        "v1.12": v112[f"pred_tail_protected_{horizon}"].to_numpy(float),
        "ET-A": et[f"pred_ET-A_{horizon}"].to_numpy(float),
        "ET-B": et[f"pred_ET-B_{horizon}"].to_numpy(float),
        "RF": rf[f"pred_RF_{horizon}"].to_numpy(float),
    }


def expert_prediction(labels: np.ndarray, sources: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.column_stack([sources[name] for name in EXPERTS])
    return matrix[np.arange(len(labels)), np.asarray(labels, int)]


def predict(bundle: dict[str, object], X: pd.DataFrame, sources: dict[str, np.ndarray]) -> np.ndarray:
    labels = bundle["pipeline"].predict(X[bundle["feature_names"]])
    selected = expert_prediction(labels, sources)
    blended = post_process(sources["v1.12"] + float(bundle["safe_alpha"]) * (selected - sources["v1.12"]))
    return np.where(sources["v1.8"] > float(bundle["risk_threshold"]), sources["v1.8"], blended)


def fit_bundle(depth: int, horizon: str, X: pd.DataFrame, y: np.ndarray, sources: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, object]:
    positive = sources["v1.8"][mask & (sources["v1.8"] > 0)]
    threshold = float(np.quantile(positive, TAIL_QUANTILE))
    fit_mask = mask & (sources["v1.8"] <= threshold)
    matrix = np.column_stack([sources[name] for name in EXPERTS])
    labels = np.argmin((matrix - y[:, None]) ** 2, axis=1)
    pipeline = build(depth)
    names = feature_names(horizon)
    pipeline.fit(X.loc[fit_mask, names], labels[fit_mask])
    chosen = expert_prediction(pipeline.predict(X.loc[fit_mask, names]), {k: v[fit_mask] for k, v in sources.items()})
    alpha = fit_safe_alpha(y[fit_mask], sources["v1.12"][fit_mask], post_process(chosen))
    return {"pipeline": pipeline, "depth": depth, "feature_names": names, "experts": list(EXPERTS), "safe_alpha": alpha, "risk_threshold": threshold, "tail_quantile": TAIL_QUANTILE, "fit_rows": int(fit_mask.sum()), "uses_severity_tier": False}


def path(horizon: str, depth: int, scope: str) -> Path:
    return MODELS_DIR / f"gate_depth{depth}_{horizon}_{scope}.joblib"


def run() -> None:
    started = time.perf_counter()
    for directory in (ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    data = load_experiment_data(FEATURE_VERSION)
    v112 = pd.read_parquet(INPUTS["v112_oof"])
    et = pd.read_parquet(INPUTS["et_oof"])
    rf = pd.read_parquet(INPUTS["rf_oof"])
    v112_test = pd.read_csv(INPUTS["v112_test"], dtype={"fipsCode": str})
    et_test = pd.read_parquet(INPUTS["et_test"])
    rf_test = pd.read_parquet(INPUTS["rf_test"])
    v18_test = pd.read_parquet(INPUTS["v18_test"])
    for frame, label in ((v112, "v1.12 OOF"), (et, "ET OOF"), (rf, "RF OOF")):
        assert_keys(data.meta_train, frame, label)
    for frame, label in ((v112_test, "v1.12 test"), (et_test, "ET test"), (rf_test, "RF test"), (v18_test, "v1.8 test")):
        assert_keys(data.meta_test, frame, label)
    folds = v112.fold.to_numpy(int)
    output = v112[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr", "fold"]].copy()
    test_output = v112_test[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    summary_rows, fold_rows, county_rows, high_rows, bootstrap_rows, model_rows, leaf_rows = [], [], [], [], [], [], []
    for hi, horizon in enumerate(HORIZONS):
        y = v112[f"actual_{horizon}"].to_numpy(float)
        sources = source_dict(v112, et, rf, horizon)
        test_sources = {"v1.8": v18_test[f"pred_C1_component_osi_{horizon}"].to_numpy(float), "v1.12": v112_test[f"pred_tail_protected_{horizon}"].to_numpy(float), "ET-A": et_test[f"pred_ET-A_{horizon}"].to_numpy(float), "ET-B": et_test[f"pred_ET-B_{horizon}"].to_numpy(float), "RF": rf_test[f"pred_RF_{horizon}"].to_numpy(float)}
        predictions = {"v1.8": sources["v1.8"], "v1.12": sources["v1.12"]}
        output[f"actual_{horizon}"] = y
        for depth in DEPTHS:
            name = f"weather_gate_depth{depth}"
            oof = np.full(len(y), np.nan)
            for fold in range(5):
                train = (folds != fold) & np.isfinite(y)
                outer = (folds == fold) & np.isfinite(y)
                bundle = fit_bundle(depth, horizon, data.X_train, y, sources, train)
                oof[outer] = predict(bundle, data.X_train.loc[outer], {k: v[outer] for k, v in sources.items()})
                bundle.update({"scope": "outer", "fold": fold, "horizon": horizon, "train_rows": int(train.sum()), "outer_rows": int(outer.sum())})
                joblib.dump(bundle, path(horizon, depth, f"fold{fold}"), compress=3)
                model_rows.append({"scope": "outer", "fold": fold, "horizon": horizon, "model": name, "safe_alpha": bundle["safe_alpha"], "risk_threshold": bundle["risk_threshold"], "fit_rows": bundle["fit_rows"]})
                counts = pd.Series(bundle["pipeline"].predict(data.X_train.loc[outer, bundle["feature_names"]])).value_counts()
                for label, count in counts.items():
                    leaf_rows.append({"scope": "outer_predictions", "fold": fold, "horizon": horizon, "model": name, "expert": EXPERTS[int(label)], "rows": int(count)})
            predictions[name] = oof
            output[f"pred_{name}_{horizon}"] = oof
            final = fit_bundle(depth, horizon, data.X_train, y, sources, np.isfinite(y))
            final.update({"scope": "final_test_only", "fold": "all", "horizon": horizon})
            joblib.dump(final, path(horizon, depth, "final"), compress=3)
            test_output[f"pred_{name}_{horizon}"] = predict(final, data.X_test, test_sources)
            model_rows.append({"scope": "final", "fold": "all", "horizon": horizon, "model": name, "safe_alpha": final["safe_alpha"], "risk_threshold": final["risk_threshold"], "fit_rows": final["fit_rows"]})
            (RESULTS_DIR / f"tree_rules_{name}_{horizon}.txt").write_text(export_text(final["pipeline"].named_steps["gate"], feature_names=final["feature_names"]), encoding="utf-8")
        base, current = metric_values(y, sources["v1.8"]), metric_values(y, sources["v1.12"])
        for name, values in predictions.items():
            score = metric_values(y, values)
            summary_rows.append({"horizon": horizon, "model": name, **score, "rmse_change_pct_vs_v1.8": 100 * (score["rmse"] / base["rmse"] - 1), "rmse_change_pct_vs_v1.12": 100 * (score["rmse"] / current["rmse"] - 1), "mae_change_pct_vs_v1.12": 100 * (score["mae"] / current["mae"] - 1)})
        fold_rows.extend(metric_rows(output, y, predictions, horizon, "fold"))
        county_rows.extend(metric_rows(output, y, predictions, horizon, "fipsCode"))
        threshold = float(np.nanquantile(y, 0.95)); high = np.isfinite(y) & (y >= threshold)
        for name, values in predictions.items():
            high_rows.append({"horizon": horizon, "model": name, "actual_threshold": threshold, **metric_values(y[high], values[high])})
        for di, depth in enumerate(DEPTHS):
            name = f"weather_gate_depth{depth}"
            bootstrap_rows.append({"horizon": horizon, "model": name, "reference": "v1.12", **paired_county_bootstrap(output, y, sources["v1.12"], predictions[name], 2000, BOOTSTRAP_SEED + hi * 10 + di)})
    summary = pd.DataFrame(summary_rows); folds_df = pd.DataFrame(fold_rows)
    ref = folds_df[folds_df.model == "v1.12"][["horizon", "fold", "rmse"]].rename(columns={"rmse": "v112_rmse"})
    folds_df = folds_df.merge(ref, on=["horizon", "fold"]); folds_df["improved_vs_v1.12"] = folds_df.rmse < folds_df.v112_rmse
    folds_df["rmse_change_pct_vs_v1.12"] = 100 * (folds_df.rmse / folds_df.v112_rmse - 1)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False); folds_df.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(county_rows).to_csv(RESULTS_DIR / "county_metrics.csv", index=False); pd.DataFrame(high_rows).to_csv(RESULTS_DIR / "high_osi_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS_DIR / "paired_county_bootstrap.csv", index=False); pd.DataFrame(model_rows).to_csv(RESULTS_DIR / "gate_models.csv", index=False); pd.DataFrame(leaf_rows).to_csv(RESULTS_DIR / "expert_usage.csv", index=False)
    output.to_parquet(ARTIFACTS_DIR / "oof_predictions.parquet", index=False); test_output.to_parquet(ARTIFACTS_DIR / "test_predictions.parquet", index=False)
    decision = {"reference": "v1.12", "horizons": {}, "promote": False}
    for horizon in HORIZONS:
        rows = summary[(summary.horizon == horizon) & summary.model.str.startswith("weather")]
        best = rows.sort_values("rmse").iloc[0]; name = best["model"]
        wins = int(folds_df[(folds_df.horizon == horizon) & (folds_df.model == name)]["improved_vs_v1.12"].sum())
        boot = next(x for x in bootstrap_rows if x["horizon"] == horizon and x["model"] == name)
        decision["horizons"][horizon] = {"exploratory_best": name, "rmse": float(best.rmse), "rmse_change_pct_vs_v1.12": float(best["rmse_change_pct_vs_v1.12"]), "improved_folds": wins, "bootstrap_ci": [boot["rmse_ci_low"], boot["rmse_ci_high"]], "passes": bool(best["rmse_change_pct_vs_v1.8"] <= -0.5 and wins >= 4 and boot["rmse_ci_high"] < 0)}
    decision["promote"] = bool(all(x["passes"] for x in decision["horizons"].values()))
    write_json(RESULTS_DIR / "promotion_decision.json", decision)
    submission = pd.read_csv(INPUTS["v112_submission"], dtype={"fipsCode": str})
    for horizon in HORIZONS:
        name = decision["horizons"][horizon]["exploratory_best"]
        values = test_output[f"pred_{name}_{horizon}"].to_numpy(float); mask = submission[horizon].notna().to_numpy(); submission.loc[mask, horizon] = values[mask]
    submission.to_csv(ARTIFACTS_DIR / "submission_v1.19_exploratory_best_balanced_v1.csv", index=False)
    write_json(ARTIFACTS_DIR / "alignment_report.json", {"status": "pass", "train_rows": len(output), "test_rows": len(test_output), "feature_version": FEATURE_VERSION, "feature_count": 163, "folds": "balanced_v1", "severity_tier_used": False})
    generated = list(MODELS_DIR.glob("*.joblib")) + list(RESULTS_DIR.glob("*")) + list(ARTIFACTS_DIR.glob("*.parquet")) + list(ARTIFACTS_DIR.glob("*.csv")) + [ARTIFACTS_DIR / "alignment_report.json"]
    metadata = {"experiment_version": "v1.19_weather_tree_experts", "feature_version": FEATURE_VERSION, "feature_count": 163, "cv_version": "balanced_v1", "seed": SEED, "depths": list(DEPTHS), "min_samples_leaf": MIN_SAMPLES_LEAF, "experts": list(EXPERTS), "strict_nested_cv": "Gate imputer, depth-2/3 tree, risk threshold and safe alpha are fit on the other four county folds only.", "uses_severity_tier": False, "runtime_seconds": time.perf_counter() - started, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"), "libraries": {"sklearn": sklearn.__version__, "joblib": joblib.__version__}, "input_hashes": {name: sha256_file(p) for name, p in INPUTS.items()}, "artifact_hashes": hash_tree(generated, PROJECT_ROOT)}
    write_json(ARTIFACTS_DIR / "run_metadata.json", metadata)
    print(summary.to_string(index=False)); print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
