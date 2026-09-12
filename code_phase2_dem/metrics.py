"""Metrics and safe OSI post-processing."""

import numpy as np


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"rmse": np.nan, "mae": np.nan}
    err = y_true[mask] - y_pred[mask]
    return {"rmse": float(np.sqrt(np.mean(err * err))), "mae": float(np.mean(np.abs(err)))}


def clip_osi(pred):
    return np.clip(np.asarray(pred, dtype=float), 0.0, 0.65)


def joint_score(y_true, y_pred):
    """Equal relative emphasis on RMSE and MAE for alpha selection."""
    m = metrics(y_true, y_pred)
    return m["rmse"] + m["mae"]
