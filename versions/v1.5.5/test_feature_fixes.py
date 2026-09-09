"""Regression and integration checks for the corrected feature release."""
import json
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from code_phase1.feature_dataset import (load_feature_dataset, load_train, load_test, preprocess,
    attach_external, build_feature_matrix, build_county_features, sha256, MANIFEST)
from data_loader import reconstruct_osi
from features import compute_observed_features
from build_county_features import aggregate_eia_utilities
import numpy as np
import pandas as pd

class FormulaTests(unittest.TestCase):
    def test_state_scoping_normalization_and_deduplication(self):
        ref = pd.DataFrame({'fipsCode': ['18001', '39001', '18141'],
            'stateAbbr': ['IN', 'OH', 'IN'], 'countyName': ['Adams', 'Adams', 'St. Joseph']}).set_index('fipsCode')
        src = pd.DataFrame({'State': ['IN', 'IN', 'OH', 'OH', 'IN'],
            'County': ['ADAMS', 'Adams', 'Adams', 'Adams', 'St Joseph'], 'Utility Number': [1, 1, 2, 3, 4]})
        result = aggregate_eia_utilities(ref, src)
        self.assertEqual(result.n_utilities.to_dict(), {'18001': 1, '18141': 1, '39001': 2})

    def test_unmatched_and_ambiguous_counties_raise(self):
        ref = pd.DataFrame({'fipsCode': ['18141'], 'stateAbbr': ['IN'], 'countyName': ['St. Joseph']}).set_index('fipsCode')
        src = pd.DataFrame({'State': ['OH'], 'County': ['St Joseph'], 'Utility Number': [1]})
        with self.assertRaisesRegex(ValueError, 'Unmatched'):
            aggregate_eia_utilities(ref, src)
        duplicate = ref.rename(index={'18141': '18999'}).copy()
        duplicate['countyName'] = 'St Joseph'
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            aggregate_eia_utilities(pd.concat([ref, duplicate]), src)

    def test_osi_train_test_symmetry_and_target_preservation(self):
        frame = pd.DataFrame({'_hour_idx': [0, 1, 2, 71, 72],
            'P_t': [0.00001, 0, 0.1, 0.12345, 0.9], 'N_t': [0, 0, 0.2, 0, 0.9],
            'D_t': [0, 0, 0.3, 0, 0.9], 'R_t': [0, 1, 0.1, 0, 0.9],
            'osi': [7., 7., 7., 7., 0.6543], 'osi_target_t01h': [0.1] * 5})
        untouched = frame.copy(deep=True)
        train = reconstruct_osi(frame)
        test = reconstruct_osi(frame.drop(columns='osi'))
        pd.testing.assert_series_equal(train.osi.iloc[:4], test.osi.iloc[:4])
        np.testing.assert_allclose(train.osi.iloc[:4], [0, 0, 0.175, 0.0494])
        self.assertEqual(train.osi.iloc[4], 0.6543)
        self.assertTrue(pd.isna(test.osi.iloc[4]))
        pd.testing.assert_series_equal(train.osi_target_t01h, frame.osi_target_t01h)
        pd.testing.assert_frame_equal(frame, untouched)

    def test_missing_observed_components_raise(self):
        frame = pd.DataFrame({'_hour_idx': [0], 'P_t': [np.nan], 'N_t': [0], 'D_t': [0], 'R_t': [0]})
        with self.assertRaisesRegex(ValueError, 'Observed outage components'):
            reconstruct_osi(frame)

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_feature_dataset()
        cls.county, cls.matches = build_county_features()

    def test_known_county_corrections(self):
        expected = {'18001': 4, '39001': 4, '42001': 2, '18087': 6, '18141': 8}
        for fips, count in expected.items():
            self.assertEqual(self.county.loc[fips, 'n_utilities'], count)
        self.assertEqual(len(self.matches), 302)

    def test_all_counties_observed_train_test_symmetry(self):
        raw = load_train()
        train = preprocess(raw)
        test = preprocess(raw.drop(columns='osi'))
        for fips, group in train.groupby('fipsCode'):
            a = pd.Series(compute_observed_features(group))
            b = pd.Series(compute_observed_features(test.loc[test.fipsCode.eq(fips)]))
            pd.testing.assert_series_equal(a, b, check_exact=True)

    def test_raw_rebuild_matches_cache_and_future_outages_cannot_change_features(self):
        for tag, loader, X, meta in [('train', load_train, self.data.X_train, self.data.meta_train),
                                     ('test', load_test, self.data.X_test, self.data.meta_test)]:
            raw = loader()
            # First county plus a zero-peak county (if present) exercise both finite/NaN peak paths.
            counties = [int(meta.fipsCode.iloc[0])]
            zero = meta.loc[X.osi_max_72h.eq(0), 'fipsCode']
            if len(zero) and int(zero.iloc[0]) not in counties:
                counties.append(int(zero.iloc[0]))
            raw = raw.loc[raw.fipsCode.isin(counties)].reset_index(drop=True)
            actual, _, actual_meta = build_feature_matrix(preprocess(raw), is_train=tag == 'train')
            actual = attach_external(actual, actual_meta, self.county)
            expected = X.loc[meta.fipsCode.isin(counties)].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
            if tag == 'train':
                changed = raw.copy()
                fields = ['osi', 'P_t', 'N_t', 'D_t', 'R_t', 'outageCount', 'outage_pct']
                fields += [c for c in changed if 'lag' in c or 'target' in c or 'delta' in c]
                changed.loc[changed._hour_idx.ge(72), fields] = 999.0
                rebuilt, _, rebuilt_meta = build_feature_matrix(preprocess(changed), is_train=True)
                rebuilt = attach_external(rebuilt, rebuilt_meta, self.county)
                pd.testing.assert_frame_equal(actual, rebuilt, check_exact=True)

    def test_cache_entry_and_partial_rebuild_protection(self):
        from cache import get_or_build, save_features
        X, y, meta = get_or_build(None, True)
        pd.testing.assert_frame_equal(X, self.data.X_train, check_exact=True)
        pd.testing.assert_frame_equal(y, self.data.y_train, check_exact=True)
        with self.assertRaises(ValueError):
            get_or_build(None, True, force_rebuild=True)
        with self.assertRaises(ValueError):
            save_features(X, y, meta, True)

    def test_original_inputs_and_sources_have_not_changed_since_build(self):
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        for section in ('inputs_sha256', 'source_sha256'):
            for name, digest in manifest[section].items():
                self.assertEqual(sha256(ROOT / name), digest, name)

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'successful': result.wasSuccessful(),
        'manifest_sha256': sha256(MANIFEST), 'test_script_sha256': sha256(Path(__file__))}
    (Path(__file__).parent / 'test_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
