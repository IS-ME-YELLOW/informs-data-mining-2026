"""Numerical/negative artifact tests. Synthetic predictions only; no training."""
from __future__ import annotations

import hashlib
import io
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
import artifact_checks as ac
import base_model as bm
import data
import main as runner
import verify_artifacts as verifier
from config import HORIZONS, HORIZON_HOURS, PROTOCOL, EXPERIMENT_VERSION, FEATURE_VERSION, GAT_ALPHA_GRID
from metrics import pooled_metrics, post_process_osi


def fixture(mode="direct"):
    """Five synthetic training counties, one test county, all real time bounds."""
    def meta(counties):
        frame = pd.DataFrame([(str(f), h) for f in counties for h in range(72, 216)], columns=ac.KEY)
        frame["timestamp_et"] = (pd.Timestamp("2026-03-11") + pd.to_timedelta(frame.hour_idx, unit="h")).dt.strftime("%m/%d/%Y %H:%M")
        frame["stateAbbr"], frame["severity_tier"], frame["fips_str"] = "IN", 0, frame.fipsCode
        return frame
    train, test = meta(range(18001, 18006)), meta([18006])
    folds = np.repeat(np.arange(5), 144)
    targets = pd.DataFrame()
    outer, inner, alpha, final_alpha, test_rows, graph = [], [], [], [], [], []
    components = ("osi",) if mode == "direct" else ac.COMPONENTS

    def candidates(y, b, c, S, h, k, selection_type):
        rows = []
        for a in GAT_ALPHA_GRID:
            error = y - post_process_osi(b + a * c)
            sse = float(np.sum(error * error, dtype=np.float64))
            rows.append(dict(protocol=PROTOCOL, base_mode=mode, outer_fold=k, inner_fold="pooled", horizon=h,
                             selection_type=selection_type, selection_id=ac.identity("alpha", mode, h, S),
                             allowed_folds=json.dumps(S), alpha=a, n=len(y), sse=sse,
                             rmse=np.sqrt(sse / len(y)), mae=np.mean(np.abs(error)), selected=False))
        best = min(rows, key=lambda row: (row["rmse"], row["alpha"]))
        best["selected"] = True
        return rows, best["alpha"]

    def predictions(meta, h, scope_by_fold, row_folds):
        frame = meta.copy()
        frame["protocol"], frame["base_mode"], frame["run_id"] = PROTOCOL, mode, "fixture"
        frame["horizon"] = h
        frame["base_prediction"] = (.01 + (frame.hour_idx - 72) * .0001).astype(np.float32).astype(float)
        frame["correction_osi"] = .005
        frame["alpha"] = .2
        frame["is_scoreable"] = frame.hour_idx + HORIZON_HOURS[h] < 216
        frame["y_true"] = np.where(frame.is_scoreable, frame.base_prediction + .001, np.nan)
        frame["prediction_before_postprocess"] = frame.y_true
        frame["prediction"] = frame.y_true
        frame["target_timestamp"] = (pd.to_datetime(frame.timestamp_et) + pd.Timedelta(hours=HORIZON_HOURS[h])).astype(str)
        frame["outer_fold"] = row_folds
        for k in np.unique(row_folds):
            mask, S = row_folds == k, scope_by_fold[int(k)]
            frame.loc[mask, "base_source_id"] = ac.identity("base", mode, h, S, components[0])
            frame.loc[mask, "stack_id"] = ac.identity("stack", mode, h, S)
            frame.loc[mask, "inner_selection_id"] = ac.identity("alpha", mode, h, S)
            frame.loc[mask, "selection_id"] = ac.identity("alpha", mode, h, S)
            if mode == "component_v158":
                for c in components:
                    frame.loc[mask, f"base_{c}"] = frame.loc[mask, "base_prediction"] / .4 if c == "P_t" else 0.0
                    frame.loc[mask, f"base_source_{c}"] = ac.identity("base", mode, h, S, c)
        return frame

    for h in HORIZONS:
        S_by_fold = {k: tuple(i for i in range(5) if i != k) for k in range(5)}
        out = predictions(train, h, S_by_fold, folds)
        targets[h] = out.y_true
        outer.append(out)
        for k in range(5):
            S = S_by_fold[k]
            valid = (folds != k) & out.is_scoreable
            inside = out.loc[valid].copy()
            inside["outer_fold"] = k
            inside["inner_fold"] = folds[valid]
            inside["selection_id"] = ac.identity("alpha", mode, h, S)
            for j in S:
                allowed = tuple(i for i in S if i != j)
                mask = inside.inner_fold == j
                inside.loc[mask, "allowed_folds"] = json.dumps(allowed)
                inside.loc[mask, "base_source_id"] = ac.identity("base", mode, h, allowed, components[0])
                inside.loc[mask, "stack_id"] = ac.identity("stack", mode, h, allowed)
            inner.append(inside)
            rows, winner = candidates(inside.y_true.to_numpy(), inside.base_prediction.to_numpy(), inside.correction_osi.to_numpy(), S, h, k, "outer_train_inner_cv")
            assert winner == .2
            alpha.extend(rows)
        valid = out.is_scoreable
        rows, winner = candidates(out.loc[valid, "y_true"].to_numpy(), out.loc[valid, "base_prediction"].to_numpy(), out.loc[valid, "correction_osi"].to_numpy(), ac.ALL_FOLDS, h, "all", "full_train_cv_for_final")
        final_alpha.extend(rows)
        final_alpha.append(dict(protocol=PROTOCOL, base_mode=mode, selection_type="final_selected", horizon=h, alpha=winner))
        test_part = predictions(test, h, {-1: ac.ALL_FOLDS}, np.full(len(test), -1))
        test_rows.append(test_part.drop(columns="y_true"))
        for frame, split, row_fold in [(out, "train", folds), (test_part, "test", np.full(len(test), -1))]:
            part = frame[ac.KEY + ["horizon", "base_prediction", "base_source_id"]].copy()
            part["split"], part["county_fold"] = split, row_fold
            graph.append(part)
    refs = ac.References(train, test, targets, folds)
    tables = dict(outer=pd.concat(outer, ignore_index=True), inner=pd.concat(inner, ignore_index=True),
                  alpha=pd.DataFrame(alpha), final_alpha=pd.DataFrame(final_alpha),
                  test=pd.concat(test_rows, ignore_index=True), graph=pd.concat(graph, ignore_index=True))
    template = test[["fipsCode", "stateAbbr", "timestamp_et"]].copy()
    submission = template.copy()
    for h in HORIZONS:
        template[h] = np.where(test.hour_idx + HORIZON_HOURS[h] < 216, 0.0, np.nan)
        submission[h] = tables["test"].loc[tables["test"].horizon == h, "prediction"].to_numpy()
    return refs, tables, template, submission


class ArtifactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, cls.original, cls.template, cls.submission = fixture()

    def setUp(self):
        self.tables = {k: v.copy(deep=True) for k, v in self.original.items()}

    def validate(self, tables=None, mode="direct", refs=None):
        t = self.tables if tables is None else tables
        return ac.check_rows(t["outer"], t["inner"], t["alpha"], t["final_alpha"], t["test"], t["graph"],
                             self.refs if refs is None else refs, mode, "fixture")

    def test_good_direct_and_component(self):
        out, test, selected = self.validate()
        self.assertEqual(len(self.tables["alpha"]), 160)
        self.assertEqual(set(selected.values()), {.2})
        refs, tables, template, submission = fixture("component_v158")
        self.validate(tables, "component_v158", refs)
        ac.check_submission(submission, template, tables["test"], refs)

    def test_keys_allow_row_reordering(self):
        shuffled = {k: v.sample(frac=1, random_state=3).reset_index(drop=True) for k, v in self.tables.items()}
        self.validate(shuffled)

    def test_missing_duplicate_and_extra_alpha_rejected(self):
        for kind in ["missing", "duplicate", "extra"]:
            with self.subTest(kind=kind):
                t = dict(self.tables)
                t["alpha"] = self.tables["alpha"].copy()
                if kind == "missing": t["alpha"] = t["alpha"].iloc[1:]
                elif kind == "duplicate": t["alpha"].loc[1, "alpha"] = t["alpha"].loc[0, "alpha"]
                else: t["alpha"].loc[0, "alpha"] = .123
                with self.assertRaises(AssertionError): self.validate(t)

    def test_incorrect_candidate_score_and_selection_rejected(self):
        for field, value in [("rmse", 123.0), ("sse", 123.0), ("n", 1), ("selected", True)]:
            with self.subTest(field=field):
                t = dict(self.tables)
                t["alpha"] = self.tables["alpha"].copy()
                t["alpha"].loc[0, field] = value
                with self.assertRaises(AssertionError): self.validate(t)

    def test_alpha_tie_break_is_smaller_alpha(self):
        group = self.tables["alpha"].query('outer_fold == 0 and horizon == "osi_target_t01h"').copy()
        for c in ["sse", "rmse", "mae"]: group[c] = 0.0
        group["n"], group["selected"] = 2, False
        group.loc[group.alpha == 0, "selected"] = True
        result = ac.check_alpha_group(group, np.array([.01, .02]), np.array([.01, .02]), np.zeros(2),
                    S=(1,2,3,4), h=HORIZONS[0], mode="direct", selection_type="outer_train_inner_cv", outer_fold=0)
        self.assertEqual(result, 0.0)
        group["selected"] = group.alpha == .05
        with self.assertRaisesRegex(AssertionError, "winner"):
            ac.check_alpha_group(group, np.array([.01, .02]), np.array([.01, .02]), np.zeros(2),
                    S=(1,2,3,4), h=HORIZONS[0], mode="direct", selection_type="outer_train_inner_cv", outer_fold=0)

    def test_inner_missing_duplicate_and_label_scope_rejected(self):
        for action in ["missing", "duplicate", "scope", "label", "source"]:
            with self.subTest(action=action):
                t = dict(self.tables)
                frame = self.tables["inner"].copy()
                if action == "missing": frame = frame.iloc[1:]
                elif action == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
                elif action == "scope": frame.loc[0, "allowed_folds"] = "[0, 1, 2, 3, 4]"
                elif action == "label": frame.loc[0, "y_true"] += .01
                else: frame.loc[0, "base_source_id"] = frame.loc[0, "stack_id"]
                t["inner"] = frame
                with self.assertRaises(AssertionError): self.validate(t)

    def test_outer_and_test_prediction_algebra_rejected(self):
        for key, field in [("outer", "alpha"), ("outer", "prediction"), ("test", "prediction_before_postprocess")]:
            with self.subTest(key=key, field=field):
                t = dict(self.tables); t[key] = self.tables[key].copy()
                t[key].loc[0, field] += .001
                with self.assertRaises(AssertionError): self.validate(t)

    def test_final_graph_value_and_crossfit_source_rejected(self):
        for field, value in [("base_prediction", .5), ("base_source_id", "base:osi:osi_target_t01h:S0,1,2,3,4")]:
            t = dict(self.tables); t["graph"] = self.tables["graph"].copy()
            t["graph"].loc[0, field] = value
            with self.assertRaises(AssertionError): self.validate(t)

    def test_submission_values_nan_positions_and_identifiers(self):
        ac.check_submission(self.submission, self.template, self.tables["test"], self.refs)
        for kind in ["values", "nan_swap", "identifier"]:
            changed = self.submission.copy()
            if kind == "values": changed.loc[0, HORIZONS[0]] += .001
            elif kind == "nan_swap": changed.loc[[0,143], HORIZONS[0]] = [np.nan, .02]
            else: changed.loc[0, "stateAbbr"] = "OH"
            with self.assertRaises(AssertionError):
                ac.check_submission(changed, self.template, self.tables["test"], self.refs)

    def test_recompute_matches_producer_and_county_type_alignment(self):
        summary, folds, county, bootstrap = ac.recompute_metrics(self.tables["outer"])
        expected_county, expected_bootstrap = runner._county_metrics_and_bootstrap(self.tables["outer"])
        ac.compare_table(county, expected_county, ["horizon", "fipsCode"], "county")
        ac.compare_table(bootstrap, expected_bootstrap, ["horizon"], "bootstrap")
        # CSV inference changes county strings to integers; the checker normalises.
        csv_county = pd.read_csv(io.StringIO(county.to_csv(index=False)))
        self.assertTrue(pd.api.types.is_integer_dtype(csv_county.fipsCode))
        ac.compare_table(csv_county, county, ["horizon", "fipsCode"], "county CSV")
        for frame, keys in [(summary, ["horizon", "model"]), (folds, ["horizon", "outer_fold", "model"]),
                            (county, ["horizon", "fipsCode"]), (bootstrap, ["horizon"])]:
            changed = frame.copy()
            column = next(c for c in frame if c not in keys and pd.api.types.is_float_dtype(frame[c]))
            changed.loc[0, column] += .01
            with self.assertRaises(AssertionError): ac.compare_table(changed, frame, keys, "corrupt summary")

    def write_numeric_fixture(self, root):
        paths = {"outer": "outer_oof", "inner": "inner_oof", "alpha": "alpha_selection", "final_alpha": "alpha_selection_final",
                 "test": "test_predictions", "graph": "final_graph_base"}
        for key, filename in paths.items(): self.tables[key].to_parquet(root / f"{filename}.parquet", index=False)
        expected = ac.recompute_metrics(self.tables["outer"])
        for name, frame in zip(["cv_summary", "fold_metrics", "county_metrics", "county_bootstrap"], expected):
            frame.to_csv(root / f"{name}.csv", index=False)
        self.submission.to_csv(root / "submission_phase2_dem_gat.csv", index=False)
        self.template.to_csv(root / "template.csv", index=False)

    def test_numeric_file_pipeline_and_mutated_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.write_numeric_fixture(root)
            manifest = dict(base_mode="direct", run_id="fixture")
            with patch.object(ac, "SUBMISSION_FILE", root / "template.csv"):
                ac.check_numeric_artifacts(root, manifest, self.refs)
                bad = pd.read_csv(root / "cv_summary.csv"); bad.loc[0, "rmse"] += .1
                bad.to_csv(root / "cv_summary.csv", index=False)
                with self.assertRaises(AssertionError): ac.check_numeric_artifacts(root, manifest, self.refs)

    def test_inference_guard_blocks_labels_and_training(self):
        import lightgbm
        manifest = {"inputs": {"feature_dir": str(data.DEFAULT_FEATURE_DIR)}}
        with verifier._inference_guard(manifest):
            calls = [lambda: data.load_supervision(), lambda: runner.load_component_targets(None),
                     lambda: bm.fit_base(None, None, None, None, None, None), lambda: lightgbm.train({}),
                     lambda: pd.read_parquet(data.DEFAULT_FEATURE_DIR / "targets_train_v1.5.6.parquet"),
                     lambda: (data.DEFAULT_FEATURE_DIR / "targets_train_v1.5.6.parquet").read_bytes(),
                     lambda: pd.read_parquet("inner_oof.parquet")]
            for call in calls:
                with self.assertRaises(PermissionError): call()
            pd.read_parquet(data.DEFAULT_FEATURE_DIR / "features_test_v1.5.6.parquet")

    def test_failed_verification_never_writes_complete(self):
        for failing_stage in ["artifacts", "reload"]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(verifier, "_check_run", side_effect=AssertionError("bad") if failing_stage == "artifacts" else None,
                                  return_value={"run_id": "fixture"}), \
                     patch.object(verifier, "_independent_reload", side_effect=AssertionError("bad")):
                    with self.assertRaises(AssertionError): verifier.main([str(root)])
                self.assertFalse((root / "COMPLETE").exists())
                self.assertFalse((root / "verification.json").exists())

    def test_inner_writer_preserves_actual_base_source(self):
        S, h = (1, 2, 3, 4), HORIZONS[0]
        n = len(self.refs.train)
        stacks, bases = np.empty(n, dtype=object), np.empty(n, dtype=object)
        for j in S:
            allowed = tuple(k for k in S if k != j)
            stacks[self.refs.row_fold == j] = ac.identity("stack", "direct", h, allowed)
            bases[self.refs.row_fold == j] = ac.identity("base", "direct", h, allowed)
        selection = SimpleNamespace(scope=S, inner_base=np.full(n, .01), inner_correction=np.full(n, .005),
                                    inner_source_ids=stacks, inner_base_source_ids=bases,
                                    selection_id=ac.identity("alpha", "direct", h, S))
        store = bm.SupervisionStore(self.refs.row_fold, self.refs.targets)
        rows = []
        runner._append_inner_records(rows, selection, 0, h, "direct", SimpleNamespace(meta_train=self.refs.train), self.refs.row_fold, store)
        self.assertEqual(len(rows), 4 * 143)
        self.assertTrue(all(r["base_source_id"].startswith("base:") for r in rows))
        self.assertTrue(all(r["stack_id"].startswith("stack:") for r in rows))

    def test_base_manifest_probe_lineage_and_frozen_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "models/base").mkdir(parents=True)
            records = []
            for size in range(2, 6):
                for S in itertools.combinations(range(5), size):
                    for h in HORIZONS:
                        filename = f"models/base/osi_{h.replace('osi_target_', '')}_S{'-'.join(map(str,S))}.txt"
                        (root / filename).write_text("serialization fixture, no fitted estimator")
                        valid = self.refs.train.hour_idx.to_numpy() + HORIZON_HOURS[h] < 216
                        probes = [dict(probe_fold=q, train_folds=[k for k in S if k != q], valid_folds=[q],
                                       train_rows=int((valid & np.isin(self.refs.row_fold, [k for k in S if k != q])).sum()),
                                       valid_rows=int((valid & (self.refs.row_fold == q)).sum()), best_iteration=3,
                                       seed=bm.scoped_seed(42, "base_probe", S, h, "osi", q)) for q in S]
                        records.append(dict(protocol=PROTOCOL, base_mode="direct", model_id=ac.identity("base", "direct", h, S),
                            component="osi", horizon=h, allowed_folds=json.dumps(S), model_path=filename,
                            model_sha256=ac.sha256(root / filename), feature_package_hash="full", feature_side_hash="side",
                            valid_rows=int((valid & np.isin(self.refs.row_fold, S)).sum()), best_iterations=json.dumps([3]*len(S)),
                            final_rounds=3, actual_rounds=3, probe_records=json.dumps(probes), seed=bm.scoped_seed(42, "base_refit", S, h, "osi", "refit")))
            final = pd.DataFrame(records)
            cv = final[~final.model_id.str.endswith("S0,1,2,3,4")]
            final.to_parquet(root / "base_fit_manifest_final.parquet", index=False)
            cv.to_parquet(root / "base_fit_manifest.parquet", index=False)
            manifest = dict(base_mode="direct", seed=42, inputs=dict(feature_package_hash="full", feature_side_hash="side"))
            ac.check_base_manifests(root, manifest, self.refs)
            # A hidden held-out fold in early stopping must fail even when both
            # tables are changed consistently, so equality alone cannot pass it.
            bad = final.copy()
            probes = json.loads(bad.loc[0, "probe_records"]); probes[0]["valid_folds"] = [4]
            bad.loc[0, "probe_records"] = json.dumps(probes)
            bad.to_parquet(root / "base_fit_manifest_final.parquet", index=False)
            bad[~bad.model_id.str.endswith("S0,1,2,3,4")].to_parquet(root / "base_fit_manifest.parquet", index=False)
            with self.assertRaisesRegex(AssertionError, "label leakage"):
                ac.check_base_manifests(root, manifest, self.refs)

    def test_independent_reload_checks_submission_and_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.write_numeric_fixture(root)
            manifest = dict(run_id="fixture", base_mode="direct", seed=42,
                inputs=dict(feature_dir=str(data.DEFAULT_FEATURE_DIR), parquet_engine="pyarrow",
                            feature_package_hash="full", feature_side_hash="side"),
                parameters=dict(device="cpu", epochs=220, patience=35, time_stride=1, k=8, gat_loss="mse"))
            nodes = sorted(set(self.refs.train.fipsCode) | set(self.refs.test.fipsCode))

            def stack(S, h, mode):
                part = self.tables["graph"].query("horizon == @h")
                grid = np.zeros((144, len(nodes)))
                for row in part.itertuples(): grid[row.hour_idx-72, nodes.index(row.fipsCode)] = row.base_prediction
                return SimpleNamespace(inputs=SimpleNamespace(base_grid=grid, all_fips=nodes,
                    train_source_id=part[part.split == "train"].base_source_id.to_numpy(),
                    test_source_id=part[part.split == "test"].base_source_id.iloc[0]),
                    correction_grid=np.full_like(grid, .005))

            ctx = SimpleNamespace(row_fold=self.refs.row_fold, fit_stack=stack)
            bundle = SimpleNamespace(meta_train=self.refs.train, meta_test=self.refs.test)
            def preflight(args, *, load_supervision_data):
                self.assertFalse(load_supervision_data)
                return dict(bundle=bundle)
            def context(*args, **kwargs):
                self.assertTrue(kwargs["inference_only"] and kwargs["strict_resume"])
                return ctx
            with patch.object(runner, "_load_training_modules"), patch.object(runner, "_seed_process"), \
                 patch.object(runner, "_preflight", side_effect=preflight), patch.object(runner, "_make_context", side_effect=context), \
                 patch.object(verifier, "SUBMISSION_FILE", root / "template.csv"):
                result = verifier._independent_reload(root, manifest)
                self.assertTrue(all(result.values()))
                bad = self.submission.copy(); bad.loc[0, HORIZONS[0]] += .01
                bad.to_csv(root / "submission_phase2_dem_gat.csv", index=False)
                with self.assertRaisesRegex(AssertionError, "submission vs prediction"):
                    verifier._independent_reload(root, manifest)
                self.submission.to_csv(root / "submission_phase2_dem_gat.csv", index=False)
                ctx.fit_stack = lambda *args: data.load_supervision()
                with self.assertRaises(PermissionError): verifier._independent_reload(root, manifest)

    def test_frozen_cv_rejects_missing_inventory_and_modified_bytes(self):
        names = ["outer_oof.parquet", "inner_oof.parquet", "alpha_selection.parquet", "fold_metrics.csv", "cv_summary.csv",
                 "county_metrics.csv", "county_bootstrap.csv", "base_fit_manifest.parquet"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(run_id="fixture")
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            for name in names: (root / name).write_bytes(b"frozen fixture")
            hashes = {name: ac.sha256(root / name) for name in names}
            digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            cv = dict(protocol=PROTOCOL, experiment_version=EXPERIMENT_VERSION, run_id="fixture", artifact_hashes=hashes,
                      cv_hash=digest, run_manifest_sha256=ac.sha256(root / "run_manifest.json"))
            (root / "cv_manifest.json").write_text(json.dumps(cv)); (root / "CV_COMPLETE").write_text(digest)
            ac.check_frozen_cv(root, manifest)
            (root / names[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(AssertionError, "artifact changed"): ac.check_frozen_cv(root, manifest)
            cv["artifact_hashes"].pop(names[0]); (root / "cv_manifest.json").write_text(json.dumps(cv))
            with self.assertRaisesRegex(AssertionError, "inventory"): ac.check_frozen_cv(root, manifest)

    def test_all_gat_scope_metadata_and_crossfit_sources(self):
        """Test metadata with a Torch stand-in, not real model deserialization."""
        from config import GAT_HIDDEN, GAT_HEADS, GAT_DROPOUT, GAT_LR, GAT_WEIGHT_DECAY
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "models/gat").mkdir(parents=True)
            feature_names = [f"feature_{i}" for i in range(163)]
            (root / data.FEATURE_NAMES_FILE).write_text(json.dumps(feature_names))
            schemas = {h:data.graph_feature_names(feature_names,data.load_terrain_features().columns,h) for h in HORIZONS}
            phase_hash = hashlib.sha256(json.dumps(feature_names, separators=(",", ":")).encode()).hexdigest()
            (root / "feature_schema.json").write_text(json.dumps(dict(ordered_phase1_features=feature_names, graph_schemas=schemas, graph_input_dim=205)))
            pd.DataFrame({"feature_schema_hash": [phase_hash]}).to_parquet(root / "base_fit_manifest_final.parquet", index=False)
            nodes = sorted(set(self.refs.train.fipsCode) | set(self.refs.test.fipsCode))
            edges = np.array([np.arange(6), np.arange(6)]); attrs = np.ones((6,4), dtype=np.float32)
            graph_hash = runner._graph_identity(nodes, edges, attrs)
            np.savez(root / "graph.npz", all_fips=np.array(nodes), edge_index=edges, edge_attr=attrs)
            manifest = dict(base_mode="direct", seed=42, parameters=dict(epochs=220, patience=35, time_stride=1, gat_loss="mse"),
                            inputs=dict(feature_dir=str(root), feature_package_hash="full", feature_side_hash="side", graph_hash=graph_hash))
            identities = {}
            for size in [3,4,5]:
                for S in itertools.combinations(range(5), size):
                    for h in HORIZONS:
                        name = f"gat_direct_{h.replace('osi_target_', '')}_S{'-'.join(map(str,S))}.pt"
                        (root / "models/gat" / name).write_text("checkpoint metadata fixture")
                        identities[name] = S, h
            corrupt = [False]

            def load(path, **kwargs):
                S, h = identities[Path(path).name]
                graph_names = schemas[h]
                schema_hash = hashlib.sha256(json.dumps(graph_names, separators=(",", ":")).encode()).hexdigest()
                sources = [ac.identity("base", "direct", h, tuple(k for k in S if k != r) if r in S else S) for r in self.refs.row_fold]
                if corrupt[0]: sources[0] = ac.identity("base", "direct", h, ac.ALL_FOLDS)
                return dict(protocol=PROTOCOL, mode="direct", horizon=h, scope=S,
                    stack_id=ac.identity("stack", "direct", h, S), graph_hash=graph_hash, feature_package_hash="full", feature_side_hash="side",
                    feature_schema_hash=schema_hash, feature_names=graph_names,
                    architecture=dict(in_dim=205, hidden=GAT_HIDDEN, heads=GAT_HEADS, dropout=GAT_DROPOUT, edge_dim=4),
                    training_config=dict(epochs=220, patience=35, time_stride=1, loss_mode="mse", lr=GAT_LR, weight_decay=GAT_WEIGHT_DECAY),
                    seed=bm.scoped_seed(42,"gat",S,h,"direct","fit"), all_fips=nodes, time_order=list(range(72,216)),
                    edge_index=edges, edge_attr=attrs, feature_mean=np.zeros(205), feature_std=np.ones(205), residual_scale=.003,
                    state_dict={"gat1.linear.weight":np.zeros((96,205))}, base_source_id=sources,
                    base_source_by_component={"osi":sources}, test_base_source_by_component={"osi":ac.identity("base","direct",h,S)})
            with patch.dict(sys.modules, {"torch":SimpleNamespace(load=load,isfinite=np.isfinite)}):
                verifier._check_checkpoints(root, manifest, self.refs)
                corrupt[0] = True
                with self.assertRaisesRegex(AssertionError, "base scope"):
                    verifier._check_checkpoints(root, manifest, self.refs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
