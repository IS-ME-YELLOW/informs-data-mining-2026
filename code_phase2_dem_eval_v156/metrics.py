"""Metrics and safe OSI post-processing."""

import numpy as np


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"n": 0, "mse": np.nan, "rmse": np.nan, "mae": np.nan,
                "medae": np.nan, "bias": np.nan, "error_std": np.nan,
                "r2": np.nan, "max_abs_error": np.nan}
    err = y_true[mask] - y_pred[mask]
    yt = y_true[mask]
    mse = float(np.mean(err * err))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum(err * err) / ss_tot) if ss_tot > 0 else np.nan
    return {
        "n": int(mask.sum()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "medae": float(np.median(np.abs(err))),
        "bias": float(np.mean(y_pred[mask] - yt)),
        "error_std": float(np.std(err)),
        "r2": r2,
        "max_abs_error": float(np.max(np.abs(err))),
    }


def clip_osi(pred):
    return np.clip(np.asarray(pred, dtype=float), 0.0, 0.65)


def joint_score(y_true, y_pred):
    """Backward-compatible alias for the official pooled RMSE score."""
    return metrics(y_true, y_pred)["rmse"]
