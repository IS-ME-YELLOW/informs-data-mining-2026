"""Strict scoring and fixed OSI post-processing."""

from __future__ import annotations

import numpy as np


def pooled_metrics(y_true, y_pred, expected_mask=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred shapes differ")
    if expected_mask is None:
        expected = np.ones(y_true.shape, dtype=bool)
    else:
        expected = np.asarray(expected_mask, dtype=bool)
        if expected.shape != y_true.shape:
            raise ValueError("expected_mask shape differs from score arrays")
    if not expected.any():
        raise ValueError("pooled score has no expected rows")
    if not np.isfinite(y_true[expected]).all():
        raise ValueError("official target is non-finite on an expected scoring row")
    if not np.isfinite(y_pred[expected]).all():
        raise FloatingPointError("prediction is non-finite on an expected scoring row")
    err = y_true[expected] - y_pred[expected]
    yt = y_true[expected]
    mse = float(np.mean(err * err))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return {
        "n": int(expected.sum()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "medae": float(np.median(np.abs(err))),
        "bias": float(np.mean(y_pred[expected] - yt)),
        "error_std": float(np.std(err)),
        "r2": float(1.0 - np.sum(err * err) / ss_tot) if ss_tot > 0 else np.nan,
        "max_abs_error": float(np.max(np.abs(err))),
    }


def metrics(y_true, y_pred, expected_mask=None):
    """Strict alias used by the formal protocol."""

    return pooled_metrics(y_true, y_pred, expected_mask=expected_mask)


def diagnostic_metrics(y_true, y_pred):
    """Opt-in permissive metrics for exploratory diagnostics only."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"n": 0, "mse": np.nan, "rmse": np.nan, "mae": np.nan}
    return pooled_metrics(y_true[mask], y_pred[mask])


def clip_osi(pred):
    values = np.asarray(pred, dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError("cannot post-process a non-finite OSI prediction")
    return np.clip(values, 0.0, 0.65)


def post_process_osi(pred):
    values = clip_osi(pred)
    return np.where(values < 0.001, 0.0, values)


def joint_score(y_true, y_pred):
    return pooled_metrics(y_true, y_pred)["rmse"]
