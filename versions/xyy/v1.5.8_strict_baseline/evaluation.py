"""Descriptive evaluation of complete, fixed-mask nested OOF predictions."""
import numpy as np
import pandas as pd
from config import HORIZONS, HORIZON_HOURS, COMPONENTS, COMPONENT_WEIGHTS
from protocol import metrics, clip_component, component_target_name, MODES
MODELS = MODES

def percent_change(value, reference):
    return 100 * (value / reference - 1) if reference != 0 else np.nan

def _evaluation_tables(data, targets, controls, component_oof) -> dict[str, pd.DataFrame]:
    summary_rows = []
    fold_rows = []
    county_rows = []
    severity_rows = []
    day_rows = []
    correlation_rows = []
    contribution_rows = []
    component_rows = []

    c0_metrics = {
        horizon: metrics(data.y_train[horizon], controls["C0_direct_osi"][horizon])
        for horizon in HORIZONS
    }
    for model in MODELS:
        for horizon in HORIZONS:
            truth = data.y_train[horizon].to_numpy(dtype=float)
            pred = controls[model][horizon]
            score = metrics(truth, pred)
            base = c0_metrics[horizon]
            summary_rows.append({
                "model": model,
                "horizon": horizon,
                **score,
                "rmse_change_pct_vs_C0": percent_change(score["rmse"], base["rmse"]),
                "mae_change_pct_vs_C0": percent_change(score["mae"], base["mae"]),
                "prediction_mean": float(np.nanmean(pred)),
                "prediction_max": float(np.nanmax(pred)),
                "zero_pct": float(100 * np.mean(pred[np.isfinite(pred)] == 0)),
            })
            for fold in range(5):
                hit = data.row_folds == fold
                fold_rows.append({"model": model, "horizon": horizon, "fold": fold,
                                  **metrics(truth[hit], pred[hit])})
            for fips in data.meta_train["fipsCode"].drop_duplicates():
                hit = data.meta_train["fipsCode"].to_numpy() == fips
                county_rows.append({"model": model, "horizon": horizon, "fipsCode": fips,
                                    **metrics(truth[hit], pred[hit])})

            target_time = data.meta_train["timestamp_et"] + pd.to_timedelta(HORIZON_HOURS[horizon], unit="h")
            metrics(truth, pred)
            valid = np.isfinite(truth)
            bins = np.select(
                [truth == 0, (truth > 0) & (truth <= 0.01), (truth > 0.01) & (truth <= 0.05), truth > 0.05],
                ["zero", "0_to_0.01", "0.01_to_0.05", "above_0.05"], default="invalid"
            )
            for severity in ("zero", "0_to_0.01", "0.01_to_0.05", "above_0.05"):
                hit = valid & (bins == severity)
                if hit.any():
                    severity_rows.append({"model": model, "horizon": horizon, "severity": severity,
                                          **metrics(truth[hit], pred[hit])})
            for day in sorted(target_time[valid].dt.strftime("%Y-%m-%d").unique()):
                hit = valid & (target_time.dt.strftime("%Y-%m-%d").to_numpy() == day)
                day_rows.append({"model": model, "horizon": horizon, "target_day": day,
                                 **metrics(truth[hit], pred[hit])})

    for horizon in HORIZONS:
        truth = data.y_train[horizon].to_numpy(dtype=float)
        valid = np.isfinite(truth)
        e0 = controls["C0_direct_osi"][horizon][valid] - truth[valid]
        e1 = controls["C1_component_osi"][horizon][valid] - truth[valid]
        correlation_rows.append({
            "horizon": horizon,
            "pearson_error_correlation_C0_C1": float(np.corrcoef(e0, e1)[0, 1]),
            "sign_disagreement_pct": float(100 * np.mean(np.sign(e0) != np.sign(e1))),
        })
        osi_error = e1
        for component in COMPONENTS:
            target_name = component_target_name(component, horizon)
            actual_component = targets[target_name].to_numpy(dtype=float)
            pred_component = clip_component(component_oof[horizon][component])
            component_rows.append({
                "horizon": horizon,
                "component": component,
                **metrics(actual_component, pred_component),
            })
            contribution = COMPONENT_WEIGHTS[component] * (pred_component[valid] - actual_component[valid])
            contribution_rows.append({
                "horizon": horizon,
                "component": component,
                "weight": COMPONENT_WEIGHTS[component],
                "contribution_rmse": float(np.sqrt(np.mean(contribution ** 2))),
                "contribution_mae": float(np.mean(np.abs(contribution))),
                "correlation_with_C1_osi_error": float(np.corrcoef(contribution, osi_error)[0, 1]),
            })

    return {
        "summary_metrics.csv": pd.DataFrame(summary_rows),
        "control_fold_metrics.csv": pd.DataFrame(fold_rows),
        "county_metrics.csv": pd.DataFrame(county_rows),
        "severity_metrics.csv": pd.DataFrame(severity_rows),
        "target_day_metrics.csv": pd.DataFrame(day_rows),
        "error_correlation.csv": pd.DataFrame(correlation_rows),
        "component_contribution.csv": pd.DataFrame(contribution_rows),
        "component_summary_metrics.csv": pd.DataFrame(component_rows),
    }


def _paired_bootstrap(data, controls, replicates: int, seed: int) -> pd.DataFrame:
    comparisons = [
        ("C1_component_osi", "C0_direct_osi"),
        ("C2_equal_blend", "C0_direct_osi"),
        ("C2_equal_blend", "C1_component_osi"),
        ("C3_aligned_component", "C0_direct_osi"),
        ("C3_aligned_component", "C1_component_osi"),
        ("v18_rule", "C0_direct_osi"),
        ("v18_rule", "C1_component_osi"),
    ]
    counties = data.meta_train["fipsCode"].drop_duplicates().to_numpy()
    county_index = {fips: idx for idx, fips in enumerate(counties)}
    row_county = data.meta_train["fipsCode"].map(county_index).to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(counties), size=(replicates, len(counties)))
    rows = []
    for horizon in HORIZONS:
        truth = data.y_train[horizon].to_numpy(dtype=float)
        stats = {}
        for model in MODELS:
            pred = controls[model][horizon]
            metrics(truth, pred)
            valid = np.isfinite(truth)
            err = pred - truth
            stats[model] = {
                "sse": np.bincount(row_county[valid], weights=err[valid] ** 2, minlength=len(counties)),
                "sae": np.bincount(row_county[valid], weights=np.abs(err[valid]), minlength=len(counties)),
                "n": np.bincount(row_county[valid], minlength=len(counties)),
            }
        for model, reference in comparisons:
            left, right = stats[model], stats[reference]
            left_n = left["n"][sampled].sum(axis=1)
            right_n = right["n"][sampled].sum(axis=1)
            metric_values = {
                "rmse": (
                    np.sqrt(left["sse"][sampled].sum(axis=1) / left_n),
                    np.sqrt(right["sse"][sampled].sum(axis=1) / right_n),
                    np.sqrt(left["sse"].sum() / left["n"].sum()),
                    np.sqrt(right["sse"].sum() / right["n"].sum()),
                ),
                "mae": (
                    left["sae"][sampled].sum(axis=1) / left_n,
                    right["sae"][sampled].sum(axis=1) / right_n,
                    left["sae"].sum() / left["n"].sum(),
                    right["sae"].sum() / right["n"].sum(),
                ),
            }
            for metric_name, (left_rep, right_rep, left_point, right_point) in metric_values.items():
                diff = left_rep - right_rep
                low, high = np.quantile(diff, [0.025, 0.975])
                rows.append({
                    "model": model,
                    "reference": reference,
                    "horizon": horizon,
                    "metric": metric_name,
                    "difference": float(left_point - right_point),
                    "change_pct": float(percent_change(left_point, right_point)),
                    "ci_2_5": float(low),
                    "ci_97_5": float(high),
                    "improvement_probability": float(np.mean(diff < 0)),
                    "replicates": replicates,
                    "unit": "county_full_trajectory",
                })
    return pd.DataFrame(rows)
