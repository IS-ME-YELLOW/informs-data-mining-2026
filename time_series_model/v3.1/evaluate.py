"""v3.1 预测合成、指标计算和提交映射。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZONS, HORIZON_HOURS, OSI_MAX, ZERO_THRESHOLD
from dataset import county_matrix_to_rows


def postprocess(values: np.ndarray, threshold: float = ZERO_THRESHOLD) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.clip(values, 0.0, OSI_MAX)
    result = np.where(np.isfinite(result) & (result < threshold), 0.0, result)
    return result


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
    error = y_pred[mask] - y_true[mask]
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "n": int(mask.sum()),
    }


def summarize_predictions(
    targets: np.ndarray,
    baseline_raw: np.ndarray,
    combined_raw: np.ndarray,
) -> pd.DataFrame:
    rows = []
    stages = {
        "baseline_raw": baseline_raw,
        "baseline_clipped": np.clip(baseline_raw, 0.0, OSI_MAX),
        "baseline_postprocessed": postprocess(baseline_raw),
        "combined_raw": combined_raw,
        "combined_clipped": np.clip(combined_raw, 0.0, OSI_MAX),
        "combined_postprocessed": postprocess(combined_raw),
    }
    for horizon_idx, horizon in enumerate(HORIZONS):
        for stage, prediction in stages.items():
            result = metrics(targets[:, :, horizon_idx], prediction[:, :, horizon_idx])
            valid_values = prediction[:, :, horizon_idx][
                np.isfinite(targets[:, :, horizon_idx])
                & np.isfinite(prediction[:, :, horizon_idx])
            ]
            rows.append({
                "horizon": horizon,
                "horizon_hours": HORIZON_HOURS[horizon_idx],
                "stage": stage,
                **result,
                "prediction_mean": float(valid_values.mean()) if valid_values.size else np.nan,
                "prediction_max": float(valid_values.max()) if valid_values.size else np.nan,
                "zero_pct": float(100 * np.mean(valid_values == 0)) if valid_values.size else np.nan,
            })
    return pd.DataFrame(rows)


def prediction_frame(
    meta: pd.DataFrame,
    row_index: np.ndarray,
    targets: np.ndarray,
    baseline_raw: np.ndarray,
    delta: np.ndarray,
    combined_raw: np.ndarray,
) -> pd.DataFrame:
    result = meta.reset_index(drop=True).copy()
    arrays = {
        "target": targets,
        "baseline_raw": baseline_raw,
        "delta": delta,
        "combined_raw": combined_raw,
        "combined_postprocessed": postprocess(combined_raw),
    }
    for prefix, values in arrays.items():
        rows = county_matrix_to_rows(values, row_index)
        for horizon_idx, horizon in enumerate(HORIZONS):
            result[f"{prefix}_{horizon}"] = rows[:, horizon_idx]
    return result


def build_submission(
    template: pd.DataFrame,
    meta_test: pd.DataFrame,
    row_index: np.ndarray,
    county_predictions: np.ndarray,
) -> pd.DataFrame:
    prediction_rows = county_matrix_to_rows(county_predictions, row_index)
    meta = meta_test.reset_index(drop=True).copy()
    meta["fipsCode"] = meta["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    template = template.copy()
    template["fipsCode"] = template["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)

    key_to_meta = {
        (row.fipsCode, row.timestamp_et): idx
        for idx, row in meta[["fipsCode", "timestamp_et"]].iterrows()
    }
    for template_idx, row in template[["fipsCode", "timestamp_et"]].iterrows():
        key = (row.fipsCode, row.timestamp_et)
        if key not in key_to_meta:
            raise ValueError(f"提交模板键不在 meta_test 中: {key}")
        meta_idx = key_to_meta[key]
        hour_idx = int(meta.loc[meta_idx, "hour_idx"])
        for horizon_idx, horizon in enumerate(HORIZONS):
            if hour_idx + HORIZON_HOURS[horizon_idx] > 215:
                template.at[template_idx, horizon] = np.nan
            else:
                template.at[template_idx, horizon] = prediction_rows[meta_idx, horizon_idx]
    return template

