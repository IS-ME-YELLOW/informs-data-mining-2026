"""No-training identity, completion, recovery and live data-flow regressions."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import base_model as base
import config
import diagnostic_support as support
import main as runner
import model_records as records
import run_identity as identity
import stacking


def args(**changes):
    value = dict(seed=42, base_mode="direct", epochs=220, patience=35, time_stride=1,
                 gat_loss="mse", k=8, device="cpu", stage="cv", resume=False, run_id="fixture")
    value.update(changes)
    return SimpleNamespace(**value)


class IdentityTests(unittest.TestCase):
    def test_formal_runtime_loader_requires_torch(self):
        with patch.dict(sys.modules,{"torch":None}), self.assertRaisesRegex(RuntimeError,"missing training dependency: torch"):
            runner._load_training_modules()

    def test_missing_real_dependency_cannot_report_acceptance(self):
        import validate_runtime
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{"torch":None}), \
             patch.object(identity,"environment",return_value={"torch":None}):
            root = Path(tmp)/"missing-torch"
            options = validate_runtime.arguments(["--output-dir",str(root),"--backend","real"])
            with self.assertRaises(ModuleNotFoundError): validate_runtime.driver(options)
            report = json.loads((root/"diagnostic_report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["real_acceptance_passed"])
            self.assertIn("ModuleNotFoundError",report["error"])

    def test_each_identity_axis_rejects_changes(self):
        original = identity.build_identity(args(), {"targets_train_sha256": "labels"}, "cpu",
                         runtime={"torch": "fixture"}, sources={"main.py": "original"})
        identity.assert_identity(original, copy.deepcopy(original))
        for section, field in [("inputs", "targets_train_sha256"), ("configuration", "seed"),
                               ("environment", "torch"), ("sources", "main.py")]:
            changed = copy.deepcopy(original)
            changed["definition"][section][field] = "changed"
            changed["digest"] = identity.digest(changed["definition"])
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, field):
                identity.assert_identity(original, changed)
        with self.assertRaises(ValueError): identity.assert_identity(None, original)

    def test_identity_is_not_a_random_seed(self):
        before = base.scoped_seed(42, "base_refit", (1,2,3), config.HORIZONS[0], "osi", "refit")
        a = identity.build_identity(args(), {"target_hash": "one"}, "cpu", runtime={}, sources={})
        b = identity.build_identity(args(), {"target_hash": "two"}, "cpu", runtime={}, sources={})
        self.assertNotEqual(a["digest"], b["digest"])
        self.assertEqual(before, base.scoped_seed(42, "base_refit", (1,2,3), config.HORIZONS[0], "osi", "refit"))

    def test_run_identity_is_immutable_across_attempts(self):
        original_builder = identity.build_identity
        def build(a, i, d): return original_builder(a, i, d, runtime={"cpu": "fixture"}, sources={"source": "same"})
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "OUTPUT_ROOT", Path(tmp)), \
             patch.object(runner, "_resolve_device", return_value="cpu"), \
             patch.object(runner, "_input_manifest", return_value={"feature_side_hash": "same"}), \
             patch.object(identity, "build_identity", side_effect=build):
            folder, _ = runner._create_or_resume_run(args(), {})
            original = (folder/"run_manifest.json").read_bytes()
            runner._create_or_resume_run(args(resume=True), {})
            self.assertEqual(original, (folder/"run_manifest.json").read_bytes())
            self.assertEqual(len((folder/"attempts.jsonl").read_text().splitlines()), 2)
            with self.assertRaisesRegex(ValueError, "seed"):
                runner._create_or_resume_run(args(resume=True, seed=43), {})
            with self.assertRaises(FileNotFoundError):
                runner._create_or_resume_run(args(resume=True, run_id="missing"), {})

    def test_stage_dispatch_respects_frozen_markers(self):
        self.assertEqual(runner.stages_to_run("all", False, False), ["cv", "final"])
        self.assertEqual(runner.stages_to_run("all", True, False), ["final"])
        self.assertEqual(runner.stages_to_run("cv", True, False), [])
        self.assertEqual(runner.stages_to_run("all", True, True), [])
        with self.assertRaises(ValueError): runner.stages_to_run("final", False, False)
        with self.assertRaises(ValueError): runner.stages_to_run("all", False, True)


class CompletionTests(unittest.TestCase):
    def test_publish_commit_point_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/"model.txt"
            with self.assertRaises(InterruptedError):
                records.publish_model(model, "identity", "model-id", {},
                    lambda p: (_ for _ in ()).throw(InterruptedError()), lambda p: None)
            self.assertFalse(model.exists())
            with patch.object(records, "atomic_json", side_effect=InterruptedError()), self.assertRaises(InterruptedError):
                records.publish_model(model, "identity", "model-id", {}, lambda p: p.write_text("model"), lambda p: None)
            self.assertTrue(model.exists())
            self.assertIsNone(records.load_record(model, "identity"))
            records.publish_model(model, "identity", "model-id", {}, lambda p: p.write_text("model"), lambda p: None)
            self.assertIsNotNone(records.load_record(model, "identity", "model-id"))
            with self.assertRaises(FileExistsError):
                records.publish_model(model, "identity", "model-id", {}, lambda p: None, lambda p: None)
            with self.assertRaises(ValueError): records.load_record(model, "wrong")
            model.write_text("truncated")
            with self.assertRaises(ValueError): records.load_record(model, "identity")

    def test_actual_rounds_can_be_less_than_requested(self):
        model = support.ArithmeticBooster(mean=.01, rounds=1, columns=["x"])
        fit = base.BaseFit(model, (0,1), config.HORIZONS[0], "osi", 12, (12,12), (), "fixture", 42)
        self.assertEqual(fit.actual_rounds, 1)
        self.assertEqual(fit.final_rounds, 12)
        with self.assertRaises(ValueError):
            base.BaseFit(model, (0,1), config.HORIZONS[0], "osi", 0, (), (), "fixture", 42)


class ScopeAndRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = support.small_fixture(counties_per_fold=1, test_counties=1)

    def context(self, mode, **kwargs):
        return support.context(self.fixture, mode, epochs=2, base_rounds=6, base_patience=2, **kwargs)

    def test_base_scopes_exhaustively_block_outside_labels(self):
        with support.arithmetic_backend():
            for size in range(2,5):
                for S in itertools.combinations(range(5),size):
                    excluded = next(k for k in range(5) if k not in S)
                    left, right = self.context("component_v158"), self.context("component_v158", mutate_fold=excluded)
                    for component in ("osi", *config.COMPONENTS):
                        a = left.base_store.get(S, config.HORIZONS[0], component)
                        b = right.base_store.get(S, config.HORIZONS[0], component)
                        self.assertEqual(a.best_iterations, b.best_iterations)
                        np.testing.assert_array_equal(a.predict(left.bundle.X_train), b.predict(right.bundle.X_train))

    def test_component_perturbation_changes_all_four_targets(self):
        old = self.context("component_v158")
        new = self.context("component_v158",mutate_fold=0)
        for h in config.HORIZONS:
            valid = (self.fixture["row_fold"]==0) & self.fixture["targets"][h].notna().to_numpy()
            composed = np.zeros(int(valid.sum()))
            for c in config.COMPONENTS:
                before = old.base_store.supervision.training_values(h,c)[valid]
                after = new.base_store.supervision.training_values(h,c)[valid]
                self.assertTrue(np.all(before!=after))
                composed += config.COMPONENT_WEIGHTS[c]*after
            np.testing.assert_array_equal(new.base_store.supervision.training_values(h)[valid],np.maximum(composed,0).round(4))

    def test_outer_and_inner_actual_pipeline_label_isolation(self):
        # Both routes and all five outer positions; fresh contexts disable all
        # memoized results between the baseline and perturbed runs.
        with support.arithmetic_backend():
            for mode in ["direct", "component_v158"]:
                for k in range(5):
                    with self.subTest(mode=mode, outer=k):
                        a = support.snapshot(self.context(mode), config.HORIZONS[0], outer_fold=k)
                        b = support.snapshot(self.context(mode, mutate_fold=k), config.HORIZONS[0], outer_fold=k)
                        support.compare_snapshots(a, b, atol=0)
                        j = (k+1)%5
                        a = support.snapshot(self.context(mode), config.HORIZONS[0], kind="inner", outer_fold=k)
                        b = support.snapshot(self.context(mode, mutate_fold=j), config.HORIZONS[0], kind="inner", outer_fold=k)
                        support.compare_snapshots(a, b, atol=0)
                baseline = support.snapshot(self.context(mode), config.HORIZONS[0])
                changed = support.snapshot(self.context(mode, mutate_fold=2), config.HORIZONS[0])
                with self.assertRaises(AssertionError): support.compare_snapshots(baseline, changed)

    def test_model_level_resume_reuses_completed_base_and_gat(self):
        with tempfile.TemporaryDirectory() as tmp, support.arithmetic_backend():
            root = Path(tmp)
            ctx = self.context("component_v158", output=root)
            S, h = (2,3,4), config.HORIZONS[0]
            before = ctx.fit_stack(S,h,"component_v158")
            original_files = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            resumed = self.context("component_v158", output=root, resume=True)
            with patch.object(base,"fit_base",side_effect=AssertionError("base refitted")), \
                 patch.object(stacking,"fit_gat",side_effect=AssertionError("GAT refitted")):
                after = resumed.fit_stack(S,h,"component_v158")
            np.testing.assert_array_equal(before.correction_grid, after.correction_grid)
            np.testing.assert_array_equal(before.inputs.base_grid, after.inputs.base_grid)
            for name, content in original_files.items(): self.assertEqual(content,(root/name).read_bytes())

    def test_scoped_label_accessor_rejects_forbidden_reads(self):
        values = np.arange(5.)
        labels = base.ScopedLabels(values,np.arange(5),(1,2,3))
        np.testing.assert_array_equal(labels.take([1,2]),[1,2])
        with self.assertRaises(PermissionError): labels.take([0])

    def test_each_horizon_has_its_own_ordered_graph_schema(self):
        import data
        f = self.fixture
        schemas = {h:data.graph_feature_names(f["bundle"].feature_names,f["terrain"].columns,h) for h in config.HORIZONS}
        self.assertEqual(len({tuple(names) for names in schemas.values()}),4)
        for h,names in schemas.items():
            self.assertEqual(len(names),205)
            self.assertEqual(len(set(names)),205)
            self.assertIn(f"neighbor_mean_gust_max_next_{config.HORIZON_HOURS[h]}h",names)

    def test_static_artifacts_are_preserved_on_resume(self):
        f = self.fixture
        preflight = dict(f, versions={"numpy": "fixture"}, runtime_environment={"cpu": "fixture"},
                         all_fips=sorted(set(f["bundle"].meta_train.fips_str) | set(f["bundle"].meta_test.fips_str)),
                         graph_feature_names=["base", *f["bundle"].feature_names, *[f"extra{i}" for i in range(41)]])
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner,"_resolve_device",return_value="cpu"), \
             patch.object(runner,"_input_manifest",return_value={"identity":"fixture"}):
            root = Path(tmp)
            runner._write_static_artifacts(root,args(),preflight)
            original = {p.name:p.read_bytes() for p in root.iterdir()}
            runner._write_static_artifacts(root,args(resume=True),preflight)
            self.assertEqual(original,{p.name:p.read_bytes() for p in root.iterdir()})
            changed = dict(preflight,versions={"numpy":"different"})
            with self.assertRaises(ValueError): runner._write_static_artifacts(root,args(resume=True),changed)
            self.assertEqual(original,{p.name:p.read_bytes() for p in root.iterdir()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
