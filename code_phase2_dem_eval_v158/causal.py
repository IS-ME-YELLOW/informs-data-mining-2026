"""Pure feature-rebuild helpers for the Phase-1 causal contract.

The builder is supplied by the caller (the frozen Phase-1 v1.5.6 builder in a
full data rebuild). These helpers never write caches or mutate the raw frame,
which makes future-outage perturbation tests reproducible.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def rebuild_features_pure(raw_frame: pd.DataFrame, builder: Callable, *, is_train: bool):
    """Run a Phase-1 builder on a deep copy and reject input mutation."""

    if not isinstance(raw_frame, pd.DataFrame):
        raise TypeError("raw_frame must be a pandas DataFrame")
    original = raw_frame.copy(deep=True)
    result = builder(raw_frame.copy(deep=True), is_train=is_train)
    pd.testing.assert_frame_equal(raw_frame, original, check_exact=True)
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("Phase-1 builder must return (features, targets, metadata)")
    return result


def assert_future_outage_invariant(
    raw_frame: pd.DataFrame,
    builder: Callable,
    perturb: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    is_train: bool = False,
):
    """Assert that changing only prediction-window outage fields changes no X.

    The perturbation function must return a new frame and is responsible for
    changing only hours 72..215 outage/lag/target/delta columns. Targets are
    deliberately not compared; the assertion concerns model inputs only.
    """

    base_x, _, base_meta = rebuild_features_pure(raw_frame, builder, is_train=is_train)
    changed = perturb(raw_frame.copy(deep=True))
    changed_x, _, changed_meta = rebuild_features_pure(changed, builder, is_train=is_train)
    if list(base_x.columns) != list(changed_x.columns):
        raise AssertionError("feature schema changed under a causal perturbation")
    if not base_meta.reset_index(drop=True).equals(changed_meta.reset_index(drop=True)):
        raise AssertionError("metadata changed under a causal perturbation")
    left = base_x.to_numpy(dtype=float)
    right = changed_x.to_numpy(dtype=float)
    if not np.allclose(left, right, equal_nan=True, rtol=0.0, atol=0.0):
        difference = np.argwhere(~np.isclose(left, right, equal_nan=True, rtol=0.0, atol=0.0))
        raise AssertionError(f"future-outage perturbation changed features at {difference[:5].tolist()}")
    return True
