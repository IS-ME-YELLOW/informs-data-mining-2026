"""v3.2 M4 的严格 OOF 评价、权重诊断和县级 bootstrap。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZON_HOURS, HORIZONS
from dataset import TimelineBundle, broadcast_unique
from evaluate import metrics, postprocess


def _model_horizon_arrays(
    baseline: np.ndarray,
    unique_predictions: dict[str, np.ndarray],
    timeline: TimelineBundle,
) -> dict[str, np.ndarray]:
    result = {"B0_lightgbm": postprocess(baseline)}
    for name, values in unique_predictions.items():
        result[name] = broadcast_unique(postprocess(values), timeline)
    return result


def build_evaluation_tables(
    source_targets: np.ndarray,
    unique_targets: np.ndarray,
    baseline: np.ndarray,
    unique_predictions: dict[str, np.ndarray],
    timeline: TimelineBundle,
    county_fips: np.ndarray,
    folds: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = _model_horizon_arrays(baseline, unique_predictions, timeline)
    summary_rows = []
    fold_rows = []
    county_rows = []
    baseline_metrics = {
        idx: metrics(source_targets[:, :, idx], predictions["B0_lightgbm"][:, :, idx])
        for idx in range(len(HORIZONS))
    }
    for model, mapped in predictions.items():
        model_horizon_metrics = []
        for horizon_idx, (horizon, hours) in enumerate(zip(HORIZONS, HORIZON_HOURS)):
            y = source_targets[:, :, horizon_idx]
            p = mapped[:, :, horizon_idx]
            result = metrics(y, p)
            model_horizon_metrics.append(result)
            valid = np.isfinite(y) & np.isfinite(p)
            values = p[valid]
            error = np.where(valid, p - y, 0.0)
            county_sse = np.sum(error ** 2, axis=1)
            worst_idx = int(np.argmax(county_sse))
            total_sse = float(county_sse.sum())
            summary_rows.append({
                "model": model,
                "scope": "horizon",
                "horizon": horizon,
                "horizon_hours": hours,
                **result,
                "rmse_change_vs_B0_pct": 100 * (result["rmse"] / baseline_metrics[horizon_idx]["rmse"] - 1),
                "mae_change_vs_B0_pct": 100 * (result["mae"] / baseline_metrics[horizon_idx]["mae"] - 1),
                "prediction_mean": float(values.mean()),
                "prediction_max": float(values.max()),
                "zero_pct": float(100 * np.mean(values == 0)),
                "worst_county": str(county_fips[worst_idx]),
                "worst_county_sse_share_pct": 100 * float(county_sse[worst_idx]) / total_sse,
                "evaluation_status": "strict_nested_cv",
            })
            for fold in sorted(np.unique(folds)):
                hit = folds == fold
                fold_result = metrics(y[hit], p[hit])
                fold_rows.append({
                    "model": model,
                    "outer_fold": int(fold),
                    "scope": "horizon",
                    "horizon": horizon,
                    "horizon_hours": hours,
                    **fold_result,
                    "evaluation_status": "strict_nested_cv",
                })
            for county_idx, fips in enumerate(county_fips):
                county_result = metrics(y[county_idx], p[county_idx])
                valid_county = np.isfinite(y[county_idx]) & np.isfinite(p[county_idx])
                county_error = p[county_idx][valid_county] - y[county_idx][valid_county]
                county_rows.append({
                    "model": model,
                    "fipsCode": str(fips),
                    "outer_fold": int(folds[county_idx]),
                    "horizon": horizon,
                    **county_result,
                    "sse": float(np.sum(county_error ** 2)),
                    "sae": float(np.sum(np.abs(county_error))),
                    "evaluation_status": "strict_nested_cv",
                })
        summary_rows.append({
            "model": model,
            "scope": "mean_horizon",
            "horizon": "mean_horizon",
            "horizon_hours": 0,
            "rmse": float(np.mean([item["rmse"] for item in model_horizon_metrics])),
            "mae": float(np.mean([item["mae"] for item in model_horizon_metrics])),
            "n": int(sum(item["n"] for item in model_horizon_metrics)),
            "rmse_change_vs_B0_pct": np.nan,
            "mae_change_vs_B0_pct": np.nan,
            "prediction_mean": np.nan,
            "prediction_max": np.nan,
            "zero_pct": np.nan,
            "worst_county": "",
            "worst_county_sse_share_pct": np.nan,
            "evaluation_status": "strict_nested_cv",
        })

    for model, unique in unique_predictions.items():
        result = metrics(unique_targets, postprocess(unique))
        summary_rows.append({
            "model": model,
            "scope": "unique_trajectory",
            "horizon": "unique_trajectory",
            "horizon_hours": 0,
            **result,
            "rmse_change_vs_B0_pct": np.nan,
            "mae_change_vs_B0_pct": np.nan,
            "prediction_mean": float(postprocess(unique).mean()),
            "prediction_max": float(postprocess(unique).max()),
            "zero_pct": float(100 * np.mean(postprocess(unique) == 0)),
            "worst_county": "",
            "worst_county_sse_share_pct": np.nan,
            "evaluation_status": "strict_nested_cv",
        })

    stage_rows = []
    candidate_counts = timeline.candidate_mask.sum(axis=1)
    for model, unique in unique_predictions.items():
        prediction = postprocess(unique)
        for count in range(1, 5):
            hit = candidate_counts == count
            result = metrics(unique_targets[:, hit], prediction[:, hit])
            stage_rows.append({
                "model": model,
                "candidate_count": count,
                "target_hour_start": int(timeline.target_hours[hit].min()),
                "target_hour_end": int(timeline.target_hours[hit].max()),
                **result,
                "evaluation_status": "strict_nested_cv",
            })
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(county_rows),
        pd.DataFrame(stage_rows),
    )


def paired_county_bootstrap(
    source_targets: np.ndarray,
    baseline: np.ndarray,
    unique_predictions: dict[str, np.ndarray],
    timeline: TimelineBundle,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    predictions = _model_horizon_arrays(baseline, unique_predictions, timeline)
    comparisons = [
        ("B1_equal_mean", "B0_lightgbm"),
        ("B2_static_convex", "B1_equal_mean"),
        ("B3_mlp_convex", "B1_equal_mean"),
        ("B4_gru_no_future", "B1_equal_mean"),
        ("B5_gru_future", "B1_equal_mean"),
        ("B5_gru_future", "B2_static_convex"),
        ("B5_gru_future", "B3_mlp_convex"),
        ("B5_gru_future", "B4_gru_no_future"),
    ]
    rng = np.random.default_rng(seed)
    county_count = source_targets.shape[0]
    sampled = rng.integers(0, county_count, size=(replicates, county_count))
    rows = []
    for horizon_idx, horizon in enumerate(HORIZONS):
        y = source_targets[:, :, horizon_idx]
        stats = {}
        for model, mapped in predictions.items():
            p = mapped[:, :, horizon_idx]
            mask = np.isfinite(y) & np.isfinite(p)
            error = np.where(mask, p - y, 0.0)
            stats[model] = {
                "sse": np.sum(error ** 2, axis=1),
                "sae": np.sum(np.abs(error), axis=1),
                "n": np.sum(mask, axis=1),
            }
        for model, reference in comparisons:
            left = stats[model]
            right = stats[reference]
            n_left = left["n"][sampled].sum(axis=1)
            n_right = right["n"][sampled].sum(axis=1)
            left_rmse = np.sqrt(left["sse"][sampled].sum(axis=1) / n_left)
            right_rmse = np.sqrt(right["sse"][sampled].sum(axis=1) / n_right)
            left_mae = left["sae"][sampled].sum(axis=1) / n_left
            right_mae = right["sae"][sampled].sum(axis=1) / n_right
            for metric_name, differences in [
                ("rmse", left_rmse - right_rmse),
                ("mae", left_mae - right_mae),
            ]:
                low, high = np.quantile(differences, [0.025, 0.975])
                if metric_name == "rmse":
                    point_left = np.sqrt(left["sse"].sum() / left["n"].sum())
                    point_right = np.sqrt(right["sse"].sum() / right["n"].sum())
                else:
                    point_left = left["sae"].sum() / left["n"].sum()
                    point_right = right["sae"].sum() / right["n"].sum()
                rows.append({
                    "model": model,
                    "reference": reference,
                    "horizon": horizon,
                    "metric": metric_name,
                    "difference": float(point_left - point_right),
                    "ci_2_5": float(low),
                    "ci_97_5": float(high),
                    "bootstrap_improvement_probability": float(np.mean(differences < 0)),
                    "replicates": replicates,
                    "resampling_unit": "county_full_trajectory",
                    "evaluation_status": "strict_nested_cv",
                })
    return pd.DataFrame(rows)


def weights_frame(
    county_fips: np.ndarray,
    target_hours: np.ndarray,
    weight_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    frames = []
    for model, values in weight_arrays.items():
        for horizon_idx, horizon in enumerate(HORIZONS):
            frames.append(pd.DataFrame({
                "model": model,
                "fipsCode": np.repeat(county_fips.astype(str), len(target_hours)),
                "target_hour": np.tile(target_hours.astype(int), len(county_fips)),
                "candidate_horizon": horizon,
                "weight": values[:, :, horizon_idx].reshape(-1),
            }))
    return pd.concat(frames, ignore_index=True)
