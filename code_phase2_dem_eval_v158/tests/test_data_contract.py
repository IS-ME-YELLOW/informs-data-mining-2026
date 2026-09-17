import numpy as np
import pytest

from data import load_component_targets, load_feature_bundle, load_supervision


def test_frozen_bundle_and_component_keys():
    bundle = load_feature_bundle()
    supervision = load_supervision(feature_bundle=bundle)
    assert bundle.X_train.shape == (34416, 163)
    assert bundle.X_test.shape == (9072, 163)
    assert list(bundle.X_train.columns) == list(bundle.feature_names)
    components = load_component_targets(bundle.meta_train)
    assert set(components) == set(supervision.y_train.columns)
    for horizon, values in components.items():
        expected_nan = bundle.meta_train["hour_idx"].to_numpy() + int(horizon[-3:-1]) > 215
        for component, target in values.items():
            assert target.shape == (34416,)
            assert np.array_equal(np.isnan(target), expected_nan)
