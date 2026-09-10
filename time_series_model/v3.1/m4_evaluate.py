"""M4 严格 OOF 的配对评估与诊断工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZONS, HORIZON_HOURS
from evaluate import metrics, postprocess


def target_aligned_average(predictions: np.ndarray) -> np.ndarray:
    """对指向同一县、同一目标小时的各 horizon 原始预测取无权平均。"""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (144, len(HORIZONS)):
        raise ValueError("predictions 必须为 [county, 144, 4]")
    aligned = np.full_like(values, np.nan)
    for target_idx in range(144):
        sources = []
        locations = []
        for horizon_idx, hours in enumerate(HORIZON_HOURS):
            origin_idx = target_idx - hours
            if origin_idx >= 0:
                sources.append(values[:, origin_idx, horizon_idx])
                locations.append((origin_idx, horizon_idx))
        if not sources:
            continue
        mean_prediction = np.nanmean(np.column_stack(sources), axis=1)
        for origin_idx, horizon_idx in locations:
            aligned[:, origin_idx, horizon_idx] = mean_prediction
    return aligned


def _lag1_autocorrelation(errors: np.ndarray) -> float:
    left = errors[:, :-1]
    right = errors[:, 1:]
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return float("nan")
    x = left[mask]
    y = right[mask]
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_evaluation_tables(
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    county_fips: np.ndarray,
    county_folds: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    fold_rows = []
    county_rows = []
    baseline = postprocess(predictions["B0_lightgbm"])
    for model_name, raw_prediction in predictions.items():
        prediction = postprocess(raw_prediction)
        for horizon_idx, horizon in enumerate(HORIZONS):
            y = targets[:, :, horizon_idx]
            p = prediction[:, :, horizon_idx]
            pooled = metrics(y, p)
            valid = np.isfinite(y) & np.isfinite(p)
            valid_predictions = p[valid]
            errors = np.where(valid, p - y, np.nan)
            county_sse = np.nansum(errors ** 2, axis=1)
            total_sse = float(county_sse.sum())
            worst_idx = int(np.argmax(county_sse))
            baseline_metric = metrics(y, baseline[:, :, horizon_idx])
            summary_rows.append({
                "model": model_name,
                "horizon": horizon,
                "horizon_hours": HORIZON_HOURS[horizon_idx],
                **pooled,
                "rmse_change_vs_B0_pct": 100 * (pooled["rmse"] / baseline_metric["rmse"] - 1),
                "mae_change_vs_B0_pct": 100 * (pooled["mae"] / baseline_metric["mae"] - 1),
                "prediction_mean": float(valid_predictions.mean()),
                "prediction_max": float(valid_predictions.max()),
                "zero_pct": float(100 * np.mean(valid_predictions == 0)),
                "error_lag1_autocorrelation": _lag1_autocorrelation(errors),
                "worst_county": str(county_fips[worst_idx]),
                "worst_county_sse_share_pct": (
                    100 * float(county_sse[worst_idx]) / total_sse if total_sse else 0.0
                ),
            })
            for fold in sorted(np.unique(county_folds)):
                fold_mask = county_folds == fold
                fold_result = metrics(y[fold_mask], p[fold_mask])
                fold_rows.append({
                    "model": model_name,
                    "outer_fold": int(fold),
                    "horizon": horizon,
                    **fold_result,
                })
            for county_idx, fips in enumerate(county_fips):
                result = metrics(y[county_idx], p[county_idx])
                valid_county = np.isfinite(y[county_idx]) & np.isfinite(p[county_idx])
                error = p[county_idx][valid_county] - y[county_idx][valid_county]
                county_rows.append({
                    "model": model_name,
                    "fipsCode": str(fips),
                    "outer_fold": int(county_folds[county_idx]),
                    "horizon": horizon,
                    **result,
                    "sse": float(np.sum(error ** 2)),
                    "sae": float(np.sum(np.abs(error))),
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), pd.DataFrame(county_rows)


def paired_county_bootstrap(
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """按县整条轨迹配对重采样，返回相对 B0 的指标差区间。"""
    rng = np.random.default_rng(seed)
    county_count = targets.shape[0]
    sampled = rng.integers(0, county_count, size=(replicates, county_count))
    processed = {name: postprocess(values) for name, values in predictions.items()}
    rows = []
    for horizon_idx, horizon in enumerate(HORIZONS):
        y = targets[:, :, horizon_idx]
        statistics = {}
        for model_name, values in processed.items():
            p = values[:, :, horizon_idx]
            mask = np.isfinite(y) & np.isfinite(p)
            error = np.where(mask, p - y, 0.0)
            statistics[model_name] = {
                "sse": np.sum(error ** 2, axis=1),
                "sae": np.sum(np.abs(error), axis=1),
                "n": np.sum(mask, axis=1),
            }
        base = statistics["B0_lightgbm"]
        base_n = base["n"][sampled].sum(axis=1)
        base_rmse = np.sqrt(base["sse"][sampled].sum(axis=1) / base_n)
        base_mae = base["sae"][sampled].sum(axis=1) / base_n
        for model_name, stat in statistics.items():
            if model_name == "B0_lightgbm":
                continue
            n = stat["n"][sampled].sum(axis=1)
            model_rmse = np.sqrt(stat["sse"][sampled].sum(axis=1) / n)
            model_mae = stat["sae"][sampled].sum(axis=1) / n
            point_n = stat["n"].sum()
            point_rmse = np.sqrt(stat["sse"].sum() / point_n)
            point_mae = stat["sae"].sum() / point_n
            base_point_rmse = np.sqrt(base["sse"].sum() / base["n"].sum())
            base_point_mae = base["sae"].sum() / base["n"].sum()
            for metric_name, differences, point_difference in [
                ("rmse", model_rmse - base_rmse, point_rmse - base_point_rmse),
                ("mae", model_mae - base_mae, point_mae - base_point_mae),
            ]:
                low, high = np.quantile(differences, [0.025, 0.975])
                rows.append({
                    "model": model_name,
                    "reference": "B0_lightgbm",
                    "horizon": horizon,
                    "metric": metric_name,
                    "difference": float(point_difference),
                    "ci_2_5": float(low),
                    "ci_97_5": float(high),
                    "bootstrap_improvement_probability": float(np.mean(differences < 0)),
                    "replicates": int(replicates),
                    "resampling_unit": "county_full_trajectory",
                })
    return pd.DataFrame(rows)


def time_and_phase_metrics(
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    target_phase: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for model_name, raw_prediction in predictions.items():
        prediction = postprocess(raw_prediction)
        for horizon_idx, (horizon, hours) in enumerate(zip(HORIZONS, HORIZON_HOURS)):
            y = targets[:, :, horizon_idx]
            p = prediction[:, :, horizon_idx]
            target_hour = np.arange(144) + 72 + hours
            for origin_idx, hour in enumerate(target_hour):
                result = metrics(y[:, origin_idx], p[:, origin_idx])
                rows.append({
                    "model": model_name,
                    "horizon": horizon,
                    "group_type": "target_hour",
                    "group_value": int(hour),
                    **result,
                })
            phases = target_phase[:, :, horizon_idx]
            for phase in sorted(np.unique(phases[np.isfinite(phases)])):
                mask = phases == phase
                result = metrics(y[mask], p[mask])
                rows.append({
                    "model": model_name,
                    "horizon": horizon,
                    "group_type": "storm_phase",
                    "group_value": int(phase),
                    **result,
                })
    return pd.DataFrame(rows)


def target_time_consistency(predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model_name, raw_prediction in predictions.items():
        prediction = postprocess(raw_prediction)
        dispersions = []
        for target_idx in range(144):
            sources = []
            for horizon_idx, hours in enumerate(HORIZON_HOURS):
                origin_idx = target_idx - hours
                if origin_idx >= 0:
                    sources.append(prediction[:, origin_idx, horizon_idx])
            if len(sources) >= 2:
                stacked = np.column_stack(sources)
                dispersions.extend(np.nanstd(stacked, axis=1).tolist())
        values = np.asarray(dispersions, dtype=float)
        values = values[np.isfinite(values)]
        rows.append({
            "model": model_name,
            "mean_within_target_std": float(values.mean()),
            "median_within_target_std": float(np.median(values)),
            "p90_within_target_std": float(np.quantile(values, 0.9)),
            "county_target_pairs": int(values.size),
        })
    return pd.DataFrame(rows)


def build_target_phase(feature_frame: pd.DataFrame, row_index: np.ndarray) -> np.ndarray:
    origin_phase = feature_frame["storm_phase"].to_numpy(dtype=float)[row_index]
    result = np.full((*row_index.shape, len(HORIZONS)), np.nan, dtype=float)
    for horizon_idx, hours in enumerate(HORIZON_HOURS):
        valid_origins = np.arange(144 - hours)
        result[:, valid_origins, horizon_idx] = origin_phase[:, valid_origins + hours]
    return result
