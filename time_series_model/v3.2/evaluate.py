"""v3.2 唯一轨迹映射、指标、表格和提交工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZON_HOURS, HORIZONS, N_HORIZONS, OSI_MAX, ZERO_THRESHOLD
from dataset import TimelineBundle, broadcast_unique, county_matrix_to_rows


def postprocess(values: np.ndarray, threshold: float = ZERO_THRESHOLD) -> np.ndarray:
    result = np.clip(np.asarray(values, dtype=float), 0.0, OSI_MAX)
    return np.where(np.isfinite(result) & (result < threshold), 0.0, result)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
    error = p[mask] - y[mask]
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "n": int(mask.sum()),
    }


def horizon_metrics(
    source_targets: np.ndarray,
    unique_prediction: np.ndarray,
    timeline: TimelineBundle,
) -> list[dict]:
    mapped = broadcast_unique(unique_prediction, timeline)
    rows = []
    for idx, (name, hours) in enumerate(zip(HORIZONS, HORIZON_HOURS)):
        rows.append({"horizon": name, "horizon_hours": hours, **metrics(source_targets[:, :, idx], mapped[:, :, idx])})
    return rows


def primary_score(
    source_targets: np.ndarray,
    unique_prediction: np.ndarray,
    timeline: TimelineBundle,
) -> float:
    return float(np.mean([row["rmse"] for row in horizon_metrics(source_targets, postprocess(unique_prediction), timeline)]))


def apply_alpha(equal_raw: np.ndarray, dynamic_raw: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * equal_raw + alpha * dynamic_raw).astype(np.float32)


def build_summary(
    source_targets: np.ndarray,
    unique_targets: np.ndarray,
    unique_predictions: dict[str, np.ndarray],
    timeline: TimelineBundle,
    evaluation_status: str,
) -> pd.DataFrame:
    rows = []
    for model, raw in unique_predictions.items():
        for stage, values in {
            "raw": raw,
            "clipped": np.clip(raw, 0.0, OSI_MAX),
            "postprocessed": postprocess(raw),
        }.items():
            horizon_rows = horizon_metrics(source_targets, values, timeline)
            for row in horizon_rows:
                valid_values = broadcast_unique(values, timeline)[:, :, HORIZONS.index(row["horizon"])]
                valid_values = valid_values[np.isfinite(valid_values)]
                rows.append({
                    "model": model,
                    "stage": stage,
                    **row,
                    "prediction_mean": float(valid_values.mean()),
                    "prediction_max": float(valid_values.max()),
                    "zero_pct": float(100 * np.mean(valid_values == 0)),
                    "evaluation_status": evaluation_status,
                })
            unique_result = metrics(unique_targets, values)
            rows.append({
                "model": model,
                "stage": stage,
                "horizon": "unique_trajectory",
                "horizon_hours": 0,
                **unique_result,
                "prediction_mean": float(values.mean()),
                "prediction_max": float(values.max()),
                "zero_pct": float(100 * np.mean(values == 0)),
                "evaluation_status": evaluation_status,
            })
    return pd.DataFrame(rows)


def build_prediction_frame(
    county_fips: np.ndarray,
    target_hours: np.ndarray,
    unique_targets: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    county = np.repeat(county_fips.astype(str), len(target_hours))
    hours = np.tile(target_hours.astype(int), len(county_fips))
    result = pd.DataFrame({
        "fipsCode": county,
        "target_hour": hours,
        "target": unique_targets.reshape(-1),
    })
    for name, values in predictions.items():
        result[f"{name}_raw"] = values.reshape(-1)
        result[f"{name}_postprocessed"] = postprocess(values).reshape(-1)
    return result


def build_submission(
    template: pd.DataFrame,
    meta_test: pd.DataFrame,
    row_index: np.ndarray,
    mapped_predictions: np.ndarray,
) -> pd.DataFrame:
    prediction_rows = county_matrix_to_rows(mapped_predictions, row_index)
    meta = meta_test.reset_index(drop=True).copy()
    meta["fipsCode"] = meta["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    output = template.copy()
    output["fipsCode"] = output["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    key_to_meta = {
        (row.fipsCode, row.timestamp_et): idx
        for idx, row in meta[["fipsCode", "timestamp_et"]].iterrows()
    }
    for template_idx, row in output[["fipsCode", "timestamp_et"]].iterrows():
        key = (row.fipsCode, row.timestamp_et)
        if key not in key_to_meta:
            raise ValueError(f"提交模板键不在 meta_test 中: {key}")
        meta_idx = key_to_meta[key]
        origin_hour = int(meta.loc[meta_idx, "hour_idx"])
        for horizon_idx, (horizon, hours) in enumerate(zip(HORIZONS, HORIZON_HOURS)):
            output.at[template_idx, horizon] = (
                np.nan if origin_hour + hours > 215 else prediction_rows[meta_idx, horizon_idx]
            )
    return output


def assert_unique_mapping(unique: np.ndarray, mapped: np.ndarray, timeline: TimelineBundle) -> None:
    rebuilt = broadcast_unique(unique, timeline)
    np.testing.assert_array_equal(mapped, rebuilt)
    for target_idx in range(len(timeline.target_hours)):
        locations = []
        for origin_idx in range(144):
            for horizon_idx in range(N_HORIZONS):
                if timeline.origin_horizon_to_target[origin_idx, horizon_idx] == target_idx:
                    locations.append(mapped[:, origin_idx, horizon_idx])
        stacked = np.stack(locations, axis=1)
        if not np.all(stacked == stacked[:, :1]):
            raise AssertionError(f"target_hour={timeline.target_hours[target_idx]} 映射后不一致")
