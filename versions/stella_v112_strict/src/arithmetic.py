from __future__ import annotations

import numpy as np

from .config import COMPONENTS, COMPONENT_WEIGHTS, OSI_MAX, QUANTILE, QUANTILE_METHOD, ZERO_THRESHOLD


def post_process(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.isinf(array).any():
        raise ValueError("Infinite raw OSI prediction")
    clipped = np.clip(array, 0.0, OSI_MAX)
    return np.where(clipped < ZERO_THRESHOLD, 0.0, clipped)


def clip_component(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.isinf(array).any():
        raise ValueError("Infinite raw component prediction")
    return np.clip(array, 0.0, 1.0)


def compose_component_osi(predictions: dict[str, np.ndarray]) -> np.ndarray:
    if set(predictions) != set(COMPONENTS):
        raise ValueError("Incorrect component set")
    arrays = {component: clip_component(predictions[component]) for component in COMPONENTS}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("Inconsistent component shapes")
    raw = sum(COMPONENT_WEIGHTS[component] * arrays[component] for component in COMPONENTS)
    return post_process(np.maximum(raw, 0.0))


def positive_quantile(values) -> tuple[float, int]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Inner anchor contains missing or non-finite values")
    positive = array[array > 0]
    if len(positive) == 0:
        raise ValueError("Inner anchor has no positive predictions")
    return float(np.quantile(positive, QUANTILE, method=QUANTILE_METHOD)), int(len(positive))


def gated_prediction(anchor, simple, theta: float) -> np.ndarray:
    anchor_array = np.asarray(anchor, dtype=float)
    simple_array = np.asarray(simple, dtype=float)
    if anchor_array.shape != simple_array.shape:
        raise ValueError("Gate inputs have different shapes")
    return np.where(anchor_array > float(theta), anchor_array, simple_array)


def six_source_mean(processed_sources: list[np.ndarray]) -> np.ndarray:
    if len(processed_sources) != 6:
        raise ValueError("Exactly six processed sources are required")
    shapes = {np.asarray(values).shape for values in processed_sources}
    if len(shapes) != 1:
        raise ValueError("Six-source shapes differ")
    return post_process(np.mean(np.stack(processed_sources, axis=0), axis=0))

