"""Regression checks for review items 1--6; no model training or Torch needed.

Run directly with the project interpreter: python -B tests/test_repairs.py
All generated artifacts live in TemporaryDirectory, never in experiment outputs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import base_model as base
import data
import main as runner
from config import HORIZONS, HORIZON_HOURS, TRAIN_SCOREABLE_ROWS, OFFICIAL_SCOREABLE_ROWS


class DataRepairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = data.load_feature_bundle()
        cls.terrain = data.load_terrain_features()

    def test_frozen_manifest_and_supervision(self):
        bundle = self.bundle
        for filename, record in bundle.manifest["files"].items():
            self.assertEqual(
                hashlib.sha256((bundle.feature_dir / filename).read_bytes()).hexdigest(),
                record["sha256"].lower(),
            )
        y = data.load_supervision(feature_bundle=bundle).y_train
        self.assertEqual(y.notna().sum().to_dict(), TRAIN_SCOREABLE_ROWS)
        for horizon, hours in HORIZON_HOURS.items():
            expected = bundle.meta_train.hour_idx.to_numpy() + hours <= 215
            np.testing.assert_array_equal(y[horizon].notna(), expected)
            self.assertEqual(int((bundle.meta_test.hour_idx + hours <= 215).sum()),
                             OFFICIAL_SCOREABLE_ROWS[horizon])

    def test_pandas_nan_mask_is_writable_without_mutating_features(self):
        names = list(self.bundle.feature_names)
        # Exercise the genuinely missing peak-time value in the frozen data.
        row_id = int(np.flatnonzero(self.bundle.X_test.hours_since_peak.isna())[0])
        values = self.bundle.X_test.iloc[row_id].to_numpy(dtype=float)
        original = values.copy()
        cleaned = data._legal_nan_values(
            values, self.bundle.meta_test.iloc[row_id], names, peak_zero=True
        )
        self.assertEqual(cleaned[names.index("hours_since_peak")], 0.0)
        self.assertTrue(np.isfinite(cleaned).all())
        np.testing.assert_array_equal(values, original)

    def graph(self, drop_train_hour=False, duplicate_train_hour=False, overlap=False):
        b = self.bundle
        x = b.X_train.iloc[:144].reset_index(drop=True)
        xt = b.X_test.iloc[:144].reset_index(drop=True)
        m = b.meta_train.iloc[:144].reset_index(drop=True)
        mt = b.meta_test.iloc[:144].reset_index(drop=True)
        if drop_train_hour:
            x, m = x.iloc[1:].reset_index(drop=True), m.iloc[1:].reset_index(drop=True)
        if duplicate_train_hour:
            m.loc[1, "hour_idx"] = m.loc[0, "hour_idx"]
        if overlap:
            mt["fips_str"] = m.fips_str.iloc[0]
        return data.make_county_time_view(
            x, xt, m, mt, np.full(len(x), 0.02), np.full(len(xt), 0.03),
            HORIZONS[0], np.zeros((2, 2)), self.terrain, b.feature_names,
        )

    def test_graph_covers_disjoint_train_and_test_nodes(self):
        features, grid, train_rows, test_rows, fips, names = self.graph()
        np.testing.assert_array_equal(grid, features[..., 0])
        np.testing.assert_array_equal((train_rows >= 0) ^ (test_rows >= 0), True)
        self.assertEqual(int((train_rows >= 0).sum()), 144)
        self.assertEqual(int((test_rows >= 0).sum()), 144)
        self.assertEqual(len(fips), 2)
        result, extras = data.add_neighbor_feature_aggregates(
            features, names, np.array([[0, 1, 0, 1], [0, 1, 1, 0]]), HORIZONS[0]
        )
        self.assertEqual(result.shape, (144, 2, 205))
        self.assertEqual(len(extras), 32)

    def test_graph_rejects_missing_duplicate_and_overlapping_rows(self):
        for kwargs in [dict(drop_train_hour=True), dict(duplicate_train_hour=True), dict(overlap=True)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.graph(**kwargs)

    def test_cv_aggregation_accepts_training_counts_not_submission_counts(self):
        """Exercise the real CV aggregation/writes with precomputed fake outputs.

        Estimator calls and row production are replaced; no booster or GAT is
        fitted. This intentionally does not certify the missing end-to-end
        nested-label tests from the implementation specification.
        """
        b = self.bundle
        y = data.load_supervision(feature_bundle=b).y_train
        folds = base.load_cv_folds(b.meta_train)
        row_fold = np.empty(len(b.X_train), dtype=int)
        for k, (_, indices) in enumerate(folds):
            row_fold[indices] = k
        ctx = SimpleNamespace(bundle=b, row_fold=row_fold,
                              base_store=SimpleNamespace(supervision=None))
        args = SimpleNamespace(run_id="regression-no-training", base_mode="direct")

        def evaluated(ctx, k, horizon, mode):
            selection = SimpleNamespace(alpha=0.0, selection_id=f"selection-{k}-{horizon}", candidates=[])
            return dict(allowed_folds=tuple(i for i in range(5) if i != k), selection=selection,
                        stack=None, base_metric={}, gat_metric={})

        def append_outer(rows, stack, selection, k, horizon, mode, ctx):
            indices = np.flatnonzero(row_fold == k)
            hours = b.meta_train.hour_idx.to_numpy()[indices]
            valid = hours + HORIZON_HOURS[horizon] <= 215
            frame = pd.DataFrame({
                "fipsCode": b.meta_train.fips_str.to_numpy()[indices],
                "timestamp_et": b.meta_train.timestamp_et.to_numpy()[indices],
                "hour_idx": hours, "horizon": horizon, "outer_fold": k,
                "is_scoreable": valid, "y_true": y[horizon].to_numpy()[indices],
                "base_prediction": 0.002, "correction_osi": 0.0, "alpha": 0.0,
                "prediction": np.where(valid, 0.002, np.nan),
            })
            rows.extend(frame.to_dict("records"))

        def publish_fake_base_manifest(ctx, run_dir, args):
            pd.DataFrame({"model_id": ["regression-fixture"]}).to_parquet(
                run_dir / base.CV_BASE_MANIFEST, index=False
            )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run_manifest.json").write_text("{}")
            with patch.object(runner, "evaluate_outer", side_effect=evaluated), \
                 patch.object(runner, "_append_outer_records", side_effect=append_outer), \
                 patch.object(runner, "_append_inner_records"), \
                 patch.object(runner, "_base_manifest", side_effect=publish_fake_base_manifest):
                runner._run_cv(ctx, args, run_dir)
            summary = pd.read_csv(run_dir / "cv_summary.csv")
            actual = summary[summary.model == "gat"].set_index("horizon")["n"].to_dict()
            self.assertEqual(actual, TRAIN_SCOREABLE_ROWS)
            self.assertTrue((run_dir / "CV_COMPLETE").exists())
            self.assertEqual(len(pd.read_parquet(run_dir / "outer_oof.parquet")), 137664)


class JsonBooster:
    """Serialization stand-in; it never learns from labels."""
    def __init__(self, model_file):
        self.rounds = json.loads(Path(model_file).read_text())["rounds"]

    def current_iteration(self):
        return self.rounds


class ManifestRepairs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model_dir = self.root / "models" / "base"
        self.model_dir.mkdir(parents=True)
        self.X = pd.DataFrame({"feature": np.arange(5, dtype=float)})
        self.meta = pd.DataFrame({"hour_idx": np.full(5, 72)})
        self.folds = [(np.flatnonzero(np.arange(5) != k), np.array([k])) for k in range(5)]
        self.side_hash = "feature-side-fixture"
        self.full_hash = runner._package_identity(self.side_hash, "target-fixture")
        self.schema_hash = hashlib.sha256(b'["feature"]').hexdigest()

    def store(self, **overrides):
        identity = dict(feature_package_hash=self.full_hash, feature_side_hash=self.side_hash,
                        feature_schema_hash=self.schema_hash)
        identity.update(overrides)
        return base.BaseModelStore(self.X, self.X.iloc[:1], self.meta, self.folds,
                                   model_dir=self.model_dir, **identity)

    def ctx(self, store):
        return SimpleNamespace(base_store=store, row_fold=store.row_fold,
                               bundle=SimpleNamespace(meta_train=self.meta,
                                                      feature_names=("feature",), input_hash=self.side_hash))

    def add_fits(self, store, mode, sizes):
        components = ("osi",) if mode == "direct" else base.COMPONENTS
        for size in sizes:
            for scope in itertools.combinations(range(5), size):
                for horizon in HORIZONS:
                    for component in components:
                        filename = f"{component}_{horizon.replace('osi_target_', '')}_S{'-'.join(map(str, scope))}.txt"
                        (self.model_dir / filename).write_text('{"rounds": 3}')
                        probes = tuple(dict(probe_fold=k, train_folds=[r for r in scope if r != k],
                                            valid_folds=[k], best_iteration=3) for k in scope)
                        fit = base.BaseFit(
                            model=JsonBooster(self.model_dir / filename), scope=scope,
                            horizon=horizon, component=component, final_rounds=3,
                            best_iterations=tuple(3 for _ in scope), probe_records=probes,
                            model_id=f"base:{component}:{horizon}:S{','.join(map(str, scope))}",
                            seed=base.scoped_seed(42, "base_refit", scope, horizon, component, "refit"),
                        )
                        store._fits[(horizon, component, scope)] = fit

    def test_cv_final_manifest_roundtrip_both_modes(self):
        for mode, factor in [("direct", 1), ("component_v158", 4)]:
            with self.subTest(mode=mode):
                # Separate mode directories, as required by the real runner.
                mode_root = self.root / mode
                mode_root.mkdir()
                self.model_dir = mode_root / "models" / "base"
                self.model_dir.mkdir(parents=True)
                store = self.store()
                self.add_fits(store, mode, [2, 3, 4])
                args = SimpleNamespace(base_mode=mode)
                runner._base_manifest(self.ctx(store), mode_root, args)
                cv_path = mode_root / base.CV_BASE_MANIFEST
                before = cv_path.read_bytes()
                (mode_root / "CV_COMPLETE").write_text("frozen-fixture")
                reloaded = self.store()
                with patch.object(base.lgb, "Booster", side_effect=JsonBooster):
                    reloaded.load_from_run(mode_root)
                self.assertEqual(len(reloaded.fits), 100 * factor)
                for key, fit in store.fits.items():
                    self.assertEqual(fit.best_iterations, reloaded.fits[key].best_iterations)
                    self.assertEqual(fit.probe_records, reloaded.fits[key].probe_records)
                self.add_fits(reloaded, mode, [5])
                runner._base_manifest(self.ctx(reloaded), mode_root, args, final=True)
                self.assertEqual(before, cv_path.read_bytes())
                final = self.store()
                with patch.object(base.lgb, "Booster", side_effect=JsonBooster):
                    final.load_from_run(mode_root)
                self.assertEqual(len(final.fits), 104 * factor)
                frame = pd.read_parquet(mode_root / base.FINAL_BASE_MANIFEST)
                self.assertEqual(set(frame.feature_package_hash), {self.full_hash})
                self.assertEqual(set(frame.feature_side_hash), {self.side_hash})
                with self.assertRaises(FileExistsError):
                    runner._base_manifest(self.ctx(final), mode_root, args)

    def publish_cv(self):
        store = self.store()
        self.add_fits(store, "direct", [2])
        runner._base_manifest(self.ctx(store), self.root, SimpleNamespace(base_mode="direct"))
        return store

    def test_rejects_wrong_full_or_feature_side_identity(self):
        self.publish_cv()
        for override in [dict(feature_package_hash=self.side_hash), dict(feature_side_hash="wrong")]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.store(**override).load_from_run(self.root)

    def test_missing_registered_model_fails_but_orphan_is_not_loaded(self):
        original = self.publish_cv()
        (self.model_dir / "orphan.txt").write_text("unfinished fit")
        restored = self.store()
        with patch.object(base.lgb, "Booster", side_effect=JsonBooster):
            restored.load_from_run(self.root)
        self.assertEqual(set(original.fits), set(restored.fits))
        next(self.model_dir.glob("osi_*.txt")).unlink()
        with patch.object(base.lgb, "Booster", side_effect=JsonBooster), self.assertRaises(FileNotFoundError):
            self.store().load_from_run(self.root)

    def test_tampered_model_is_rejected(self):
        self.publish_cv()
        next(self.model_dir.glob("osi_*.txt")).write_text('{"rounds": 4}')
        with patch.object(base.lgb, "Booster", side_effect=JsonBooster), self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.store().load_from_run(self.root)

    def test_context_passes_same_identity_to_base_and_gat(self):
        bundle = SimpleNamespace(X_train=self.X, X_test=self.X.iloc[:1], meta_train=self.meta,
                                 feature_names=("feature",), input_hash=self.side_hash)
        preflight = dict(bundle=bundle, supervision=SimpleNamespace(y_train=None), component_targets=None,
                         feature_package_hash=self.full_hash, feature_side_hash=self.side_hash,
                         terrain=None, coords=None, edge_index=None, edge_attr=None, graph_hash="graph")
        args = SimpleNamespace(seed=42, parquet_engine="pyarrow", device="cpu", epochs=220,
                               patience=35, time_stride=1, gat_loss="mse")
        with patch.object(runner, "_load_training_modules", return_value=(base.BaseModelStore, lambda m: self.folds, SimpleNamespace)), \
             patch.object(runner, "_resolve_device", return_value="cpu"):
            ctx = runner._make_context(args, preflight, self.root)
        self.assertEqual(ctx.feature_package_hash, ctx.base_store.feature_package_hash)
        self.assertEqual(ctx.feature_package_hash, self.full_hash)
        self.assertEqual(ctx.feature_side_hash, ctx.base_store.feature_side_hash)

    def test_real_lightgbm_text_reload_without_training(self):
        """Use historical weights only as a file-format fixture, not CV evidence."""
        bundle = data.load_feature_bundle()
        folds = base.load_cv_folds(bundle.meta_train)
        schema_hash = hashlib.sha256(
            json.dumps(list(bundle.feature_names), separators=(",", ":")).encode()
        ).hexdigest()
        full_hash = runner._package_identity(bundle.input_hash, "serialization-test-only")

        def new_store():
            return base.BaseModelStore(
                bundle.X_train, bundle.X_test, bundle.meta_train, folds,
                model_dir=self.model_dir, feature_package_hash=full_hash,
                feature_side_hash=bundle.input_hash, feature_schema_hash=schema_hash,
            )

        store = new_store()
        ctx = SimpleNamespace(bundle=bundle, base_store=store, row_fold=store.row_fold)
        with patch.object(base.lgb, "train", side_effect=AssertionError("training is prohibited")):
            for horizon in HORIZONS:
                fixture = PACKAGE / "outputs" / "models" / f"lightgbm_{horizon}.txt"
                model = base.lgb.Booster(model_file=str(fixture))
                rounds = model.current_iteration()
                scope = (0, 1)
                filename = f"osi_{horizon.replace('osi_target_', '')}_S0-1.txt"
                model.save_model(str(self.model_dir / filename), num_iteration=rounds)
                store._fits[(horizon, "osi", scope)] = base.BaseFit(
                    model, scope, horizon, "osi", rounds, (rounds, rounds), (),
                    f"base:osi:{horizon}:S0,1",
                    base.scoped_seed(42, "base_refit", scope, horizon, "osi", "refit"),
                )
            runner._base_manifest(ctx, self.root, SimpleNamespace(base_mode="direct"))
            loaded = new_store()
            loaded.load_from_run(self.root)
            for key, fit in store.fits.items():
                np.testing.assert_array_equal(
                    fit.predict(bundle.X_test.iloc[:10]),
                    loaded.fits[key].predict(bundle.X_test.iloc[:10]),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
