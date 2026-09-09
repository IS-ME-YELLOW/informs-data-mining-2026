"""Checks for v1.5.6 land-cover and window-quality features."""
import json
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.python_packages'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'code_phase1'))

import numpy as np
import pandas as pd

from code_phase1.feature_dataset_v156 import (
    EXTRA_COLUMNS,
    MANIFEST,
    WINDOW_QUALITY_COLUMNS,
    frozen_schema,
    load_feature_dataset,
    sha256,
)
import feature_dataset as v155


class LandcoverFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_feature_dataset()
        cls.parent = v155.load_feature_dataset()
        cls.landcover = pd.read_csv(ROOT / 'data/county_landcover.csv', dtype={'fipsCode': str})
        cls.landcover['fipsCode'] = cls.landcover['fipsCode'].str.zfill(5)
        cls.landcover = cls.landcover.set_index('fipsCode')

    def test_schema_adds_expected_columns(self):
        self.assertEqual(len(frozen_schema()), 163)
        expected_new = set(EXTRA_COLUMNS) | set(WINDOW_QUALITY_COLUMNS)
        self.assertEqual(expected_new, set(self.data.X_train.columns) - set(self.parent.X_train.columns))
        pd.testing.assert_frame_equal(
            self.data.X_train[self.parent.X_train.columns],
            self.parent.X_train,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            self.data.X_test[self.parent.X_test.columns],
            self.parent.X_test,
            check_exact=True,
        )

    def test_formula_matches_source_landcover(self):
        for X, meta in [(self.data.X_train, self.data.meta_train), (self.data.X_test, self.data.meta_test)]:
            keys = meta.fipsCode.astype(str).str.zfill(5)
            expected_unmapped = keys.map(self.landcover['pct_unmapped']).to_numpy(dtype=float)
            expected_classified = (
                100.0 * keys.map(self.landcover['pct_forest']).to_numpy(dtype=float)
                / (100.0 - expected_unmapped)
            )
            np.testing.assert_allclose(X['pct_unmapped'], expected_unmapped, rtol=0, atol=0)
            np.testing.assert_allclose(X['pct_forest_classified'], expected_classified, rtol=1e-12, atol=1e-12)

    def test_window_quality_features_match_existing_weather_columns(self):
        for X, meta in [(self.data.X_train, self.data.meta_train), (self.data.X_test, self.data.meta_test)]:
            hour_idx = meta.hour_idx.to_numpy(dtype=int)
            for h in (1, 6, 24, 48):
                window_len = np.minimum(h + 1, 216 - hour_idx).astype(float)
                is_full = (hour_idx + h < 216).astype(int)
                np.testing.assert_allclose(X[f'window_len_next_{h}h'], window_len, rtol=0, atol=0)
                np.testing.assert_array_equal(X[f'is_full_next_{h}h'], is_full)
                np.testing.assert_array_equal(X[f'has_weather_at_t{h}h'], is_full)
                np.testing.assert_allclose(
                    X[f'tp_rate_next_{h}h'],
                    X[f'total_tp_next_{h}h'].to_numpy(dtype=float) / window_len,
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    X[f'gust_gt30_frac_next_{h}h'],
                    X[f'gust_gt30_next_{h}h'].to_numpy(dtype=float) / window_len,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_targets_and_metadata_are_parent_copies(self):
        pd.testing.assert_frame_equal(self.data.y_train, self.parent.y_train, check_exact=True)
        pd.testing.assert_frame_equal(self.data.meta_train, self.parent.meta_train, check_exact=True)
        pd.testing.assert_frame_equal(self.data.meta_test, self.parent.meta_test, check_exact=True)

    def test_manifest_hashes_current_outputs(self):
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        for name, digest in manifest['outputs_sha256'].items():
            self.assertEqual(sha256(ROOT / name), digest, name)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'successful': result.wasSuccessful(),
        'manifest_sha256': sha256(MANIFEST), 'test_script_sha256': sha256(Path(__file__))}
    (Path(__file__).parent / 'test_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
