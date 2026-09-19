from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from src.arithmetic import gated_prediction, positive_quantile, post_process, six_source_mean
from src.planning import dependency_plan


class ProtocolTests(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(
            dependency_plan()["budget"],
            {
                "new_fit_calls": 820,
                "new_saved_models": 180,
                "referenced_models": 50,
                "outer_new_fit_calls": 500,
                "inner_new_fit_calls": 320,
            },
        )

    def test_gate_equality_uses_simple(self):
        np.testing.assert_array_equal(
            gated_prediction([0.03, 0.031], [0.02, 0.025], 0.03),
            [0.02, 0.031],
        )

    def test_post_after_average(self):
        np.testing.assert_array_equal(
            six_source_mean([np.asarray([value]) for value in [0, 0, 0, 0, 0.001, 0.003]]),
            [0.0],
        )

    def test_positive_quantile_rejects_no_positive(self):
        with self.assertRaises(ValueError):
            positive_quantile([0, 0, 0])

    def test_post_threshold_is_strict(self):
        np.testing.assert_array_equal(post_process([0.0009, 0.001]), [0.0, 0.001])


if __name__ == "__main__":
    unittest.main()
