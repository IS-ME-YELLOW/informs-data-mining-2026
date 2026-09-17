"""Recompute saved evidence, then reproduce submission without labels or fitting."""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager, ExitStack
import hashlib
import io
import itertools
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import artifact_checks as checks
import run_identity
from model_records import load_record
from config import (
    COMPONENTS, COMPONENT_TARGETS_FILE, DATA_DIR, EXPERIMENT_VERSION, FEATURE_NAMES_FILE,
    FEATURE_VERSION, GAT_DROPOUT, GAT_HEADS, GAT_HIDDEN, GAT_LR, GAT_WEIGHT_DECAY,
    HORIZONS, HORIZON_HOURS, PRED_START, PRED_END, PROTOCOL, SUBMISSION_FILE,
)
from metrics import post_process_osi


def _check_checkpoints(run_dir, manifest, refs):
    """Check every GAT's complete scope and node-source lineage."""
    import torch
    from base_model import scoped_seed
    from data import graph_feature_names, load_terrain_features

    mode, inputs = manifest["base_mode"], manifest["inputs"]
    schema = json.loads((run_dir / "feature_schema.json").read_text())
    frozen_names = json.loads((Path(inputs["feature_dir"]) / FEATURE_NAMES_FILE).read_text())
    checks.equal(schema["ordered_phase1_features"], frozen_names, "saved feature schema")
    schemas = schema["graph_schemas"]
    checks.require(set(schemas) == set(HORIZONS) and schema["graph_input_dim"] == 205, "GAT schema horizon coverage")
    terrain_names = list(load_terrain_features().columns)
    for h in HORIZONS:
        checks.equal(schemas[h], graph_feature_names(frozen_names, terrain_names, h), f"GAT ordered schema {h}")
        checks.require(len(schemas[h]) == 205, "GAT schema dimension mismatch")
    phase1_hash = hashlib.sha256(json.dumps(frozen_names, separators=(",", ":")).encode()).hexdigest()
    base_manifest = pd.read_parquet(run_dir / "base_fit_manifest_final.parquet")
    checks.require(base_manifest.feature_schema_hash.eq(phase1_hash).all(), "base feature schema mismatch")
    with np.load(run_dir / "graph.npz", allow_pickle=True) as graph:
        edges, attrs, nodes = graph["edge_index"], graph["edge_attr"], graph["all_fips"].tolist()
    checks.equal(nodes, sorted(set(refs.train.fipsCode) | set(refs.test.fipsCode)), "graph nodes")
    digest = hashlib.sha256()
    digest.update(json.dumps(nodes, separators=(",", ":")).encode())
    digest.update(np.asarray(edges, dtype=np.int64).tobytes())
    digest.update(np.asarray(attrs, dtype=np.float32).tobytes())
    checks.require(digest.hexdigest() == inputs["graph_hash"], "saved graph hash mismatch")
    expected_files = set()
    components = ("osi",) if mode == "direct" else COMPONENTS
    for size in [3, 4, 5]:
        for S in itertools.combinations(checks.ALL_FOLDS, size):
            for h in HORIZONS:
                names = schemas[h]
                schema_hash = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
                filename = f"gat_{mode}_{h.replace('osi_target_', '')}_S{'-'.join(map(str, S))}.pt"
                expected_files.add(filename)
                payload = torch.load(run_dir / "models/gat" / filename, map_location="cpu", weights_only=False)
                if "identity" in manifest:
                    receipt = load_record(run_dir / "models/gat" / filename, manifest["identity"]["digest"],
                                          checks.identity("stack", mode, h, S))
                    checks.require(receipt is not None and payload.get("run_identity") == manifest["identity"]["digest"], "GAT completion identity mismatch")
                    dependencies = payload.get("base_dependencies", {})
                    expected_ids = [checks.identity("base", mode, h, T, c)
                                    for T in [S] + [tuple(k for k in S if k != r) for r in S] for c in components]
                    expected_hashes = base_manifest.set_index("model_id").loc[expected_ids, "model_sha256"].to_dict()
                    checks.require(dependencies == expected_hashes == receipt["details"].get("base_dependencies"), "GAT upstream hashes mismatch")
                checks.require(payload["protocol"] == PROTOCOL and payload["mode"] == mode
                               and payload["horizon"] == h and tuple(payload["scope"]) == S, "GAT scope mismatch")
                checks.require(payload["stack_id"] == checks.identity("stack", mode, h, S), "GAT stack identity mismatch")
                for key in ["feature_package_hash", "feature_side_hash", "graph_hash"]:
                    checks.require(payload[key] == inputs[key], f"GAT {key} mismatch")
                checks.require(payload["feature_schema_hash"] == schema_hash and payload["feature_names"] == names, "GAT schema identity mismatch")
                checks.require(payload["architecture"] == dict(in_dim=205, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                               dropout=GAT_DROPOUT, edge_dim=4), "GAT architecture mismatch")
                params = manifest["parameters"]
                expected_config = dict(epochs=params["epochs"], patience=params["patience"],
                                       time_stride=params["time_stride"], loss_mode=params["gat_loss"],
                                       lr=GAT_LR, weight_decay=GAT_WEIGHT_DECAY)
                checks.require(payload["training_config"] == expected_config, "GAT training configuration mismatch")
                checks.require(payload["seed"] == scoped_seed(manifest["seed"], "gat", S, h, mode, "fit"), "GAT seed mismatch")
                checks.equal(payload["all_fips"], nodes, "GAT node order")
                checks.equal(payload["time_order"], np.arange(PRED_START, PRED_END), "GAT time order")
                checks.equal(payload["edge_index"], edges, "GAT edges")
                checks.equal(payload["edge_attr"], attrs, "GAT edge attributes")
                for name in ["feature_mean", "feature_std"]:
                    value = np.asarray(payload[name])
                    checks.require(value.shape == (205,) and np.isfinite(value).all(), "GAT scaling shape/value mismatch")
                checks.require((np.asarray(payload["feature_std"]) >= 1e-6).all()
                               and np.isfinite(payload["residual_scale"]) and payload["residual_scale"] >= .003, "GAT scaling invalid")
                checks.require(tuple(payload["state_dict"]["gat1.linear.weight"].shape) == (96, 205), "GAT first layer mismatch")
                for value in payload["state_dict"].values():
                    checks.require(bool(torch.isfinite(value).all()), "GAT weights non-finite")
                for c in components:
                    sources = [checks.identity("base", mode, h, tuple(k for k in S if k != r) if r in S else S, c)
                               for r in refs.row_fold]
                    checks.equal(payload["base_source_by_component"][c], sources, "GAT training-node base scope")
                    checks.require(payload["test_base_source_by_component"][c] == checks.identity("base", mode, h, S, c), "GAT test-node base scope")
                    if c == components[0]:
                        checks.equal(payload["base_source_id"], sources, "GAT primary base scope")
    checks.equal(sorted(expected_files), sorted(p.name for p in (run_dir / "models/gat").glob("*.pt")), "GAT checkpoint inventory")


def _check_run(run_dir: Path):
    required = [
        "run_manifest.json", "environment.json", "inputs_manifest.json", "feature_schema.json",
        "graph.npz", "folds.csv", "base_fit_manifest.parquet", "base_fit_manifest_final.parquet",
        "inner_oof.parquet", "outer_oof.parquet", "alpha_selection.parquet", "cv_summary.csv",
        "fold_metrics.csv", "county_metrics.csv", "county_bootstrap.csv", "cv_manifest.json",
        "CV_COMPLETE", "final_graph_base.parquet", "test_predictions.parquet", "alpha_selection_final.parquet",
        "submission_phase2_dem_gat.csv", "submission_audit.json", "verification.json", "REPORT.md", "FINAL_READY",
    ]
    checks.require(all((run_dir / name).is_file() for name in required), "required artifact missing")
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    checks.require(manifest.get("protocol") == PROTOCOL and manifest.get("experiment_version") == EXPERIMENT_VERSION
                   and manifest.get("feature_version") == FEATURE_VERSION, "run protocol/version mismatch")
    checks.require(manifest.get("base_mode") in {"direct", "component_v158"}, "invalid mode")
    run_identity.assert_current(manifest)
    inputs = json.loads((run_dir / "inputs_manifest.json").read_text())
    checks.check_frozen_cv(run_dir, manifest)
    refs = checks.load_references(manifest, inputs)
    folds = pd.read_csv(run_dir / "folds.csv", dtype={"fipsCode": str})
    reference = refs.train[["fipsCode", "stateAbbr", "severity_tier"]].copy()
    reference["row_id"] = np.arange(len(reference))
    reference["fold"] = refs.row_fold
    checks.compare_table(folds, reference, ["row_id"], "saved fold rows")
    checks.check_base_manifests(run_dir, manifest, refs)
    checks.check_numeric_artifacts(run_dir, manifest, refs)
    _check_checkpoints(run_dir, manifest, refs)
    return manifest


@contextmanager
def _inference_guard(manifest):
    """Fail if inference calls a training API or opens known supervision files."""
    import base_model
    import data
    import lightgbm
    import main as runner
    # StackContext imports Torch lazily. Register the real GAT entry point
    # before installing the guard; arithmetic-only tests may lack Torch.
    try:
        import gat_model
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise

    blocked_paths = {Path(manifest["inputs"]["feature_dir"]) / "targets_train_v1.5.6.parquet",
                     COMPONENT_TARGETS_FILE, DATA_DIR / "DM_Train.csv"}
    blocked_paths = {p.resolve() for p in blocked_paths}
    blocked_names = {"outer_oof.parquet", "inner_oof.parquet"}

    def forbidden(*args, **kwargs):
        raise PermissionError("independent inference cannot access supervision or train models")

    def guarded(original):
        def read(path, *args, **kwargs):
            candidate = path if isinstance(path, (str, bytes, os.PathLike)) else getattr(path, "name", None)
            if isinstance(candidate, (str, bytes, os.PathLike)):
                candidate = Path(os.fsdecode(candidate)).resolve()
                if candidate in blocked_paths or candidate.name in blocked_names:
                    forbidden()
            return original(path, *args, **kwargs)
        return read

    with ExitStack() as stack:
        for module, name in [(builtins, "open"), (io, "open"), (pd, "read_parquet"), (pd, "read_csv")]:
            stack.enter_context(patch.object(module, name, guarded(getattr(module, name))))
        for module, names in [
            (data, ["load_supervision", "load_component_targets"]),
            (runner, ["load_supervision", "load_component_targets"]),
            (base_model, ["fit_base", "_fit_probe", "_fit_fixed"]),
            (base_model.SupervisionStore, ["training_values", "scoped"]),
            (lightgbm, ["train"]),
        ]:
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        for name in ["gat_model", "stacking"]:
            if name in sys.modules:
                stack.enter_context(patch.object(sys.modules[name], "fit_gat", forbidden))
        yield


def _independent_reload(run_dir: Path, manifest: dict):
    import main as runner

    params = manifest["parameters"]
    args = SimpleNamespace(
        stage="final", run_id=manifest["run_id"], resume=True,
        feature_dir=manifest["inputs"]["feature_dir"], parquet_engine=manifest["inputs"]["parquet_engine"],
        base_mode=manifest["base_mode"], seed=int(manifest["seed"]), device=params["device"],
        epochs=int(params["epochs"]), patience=int(params["patience"]), time_stride=int(params["time_stride"]),
        k=int(params["k"]), gat_loss=params["gat_loss"],
        expected_feature_package_hash=manifest["inputs"]["feature_package_hash"],
        expected_feature_side_hash=manifest["inputs"]["feature_side_hash"],
        expected_run_identity=manifest.get("identity", {}).get("digest"),
    )
    runner._load_training_modules()
    runner._seed_process(args.seed, args.device)
    with _inference_guard(manifest):
        preflight = runner._preflight(args, load_supervision_data=False)
        ctx = runner._make_context(args, preflight, run_dir, resume=True, strict_resume=True, inference_only=True)
        bundle = preflight["bundle"]
        train, test_meta = checks.time_rows(bundle.meta_train, "inference train"), checks.time_rows(bundle.meta_test, "inference test")
        refs = checks.References(train, test_meta, pd.DataFrame(), ctx.row_fold)
        saved = checks.time_rows(pd.read_parquet(run_dir / "test_predictions.parquet"), "saved test")
        saved_graph = checks.time_rows(pd.read_parquet(run_dir / "final_graph_base.parquet"), "saved graph")
        final_alpha = pd.read_parquet(run_dir / "alpha_selection_final.parquet")
        selected = checks.align(final_alpha[final_alpha.selection_type == "final_selected"],
                                pd.DataFrame({"horizon": HORIZONS}), ["horizon"], "inference alpha")
        actual_rows = []
        for h in HORIZONS:
            stack = ctx.fit_stack(checks.ALL_FOLDS, h, args.base_mode)
            alpha = float(selected.loc[selected.horizon == h, "alpha"].iloc[0])
            node_of = {fips: i for i, fips in enumerate(stack.inputs.all_fips)}
            graph_part = saved_graph[saved_graph.horizon == h]
            for meta, split in [(train, "train"), (test_meta, "test")]:
                t = meta.hour_idx.to_numpy(dtype=int) - PRED_START
                node = np.array([node_of[fips] for fips in meta.fipsCode])
                base_values = np.asarray(stack.inputs.base_grid[t, node], dtype=float)
                source = stack.inputs.train_source_id if split == "train" else np.repeat(stack.inputs.test_source_id, len(meta))
                graph_rows = checks.align(graph_part[graph_part.split == split], meta, checks.KEY, "reloaded graph")
                checks.close(graph_rows.base_prediction, base_values, "reloaded graph base", atol=1e-7)
                checks.equal(graph_rows.base_source_id, source, "reloaded graph source")
                if split == "test":
                    correction = np.asarray(stack.correction_grid[t, node], dtype=float)
                    expected = meta.hour_idx.to_numpy() + HORIZON_HOURS[h] < PRED_END
                    values = np.full(len(meta), np.nan)
                    values[expected] = post_process_osi(base_values[expected] + alpha * correction[expected])
                    recorded = checks.align(saved[saved.horizon == h], meta, checks.KEY, "reloaded test")
                    checks.close(recorded.base_prediction, base_values, "reloaded test base", atol=1e-7)
                    checks.close(recorded.correction_osi, correction, "reloaded test correction", atol=1e-7)
                    checks.close(recorded.alpha, np.full(len(meta), alpha), "reloaded alpha", atol=0)
                    checks.close(recorded.prediction, values, "reloaded test predictions", atol=1e-7)
                    part = meta[checks.KEY].copy()
                    part["horizon"], part["prediction"] = h, values
                    actual_rows.append(part)
        actual = pd.concat(actual_rows, ignore_index=True)
        submission = pd.read_csv(run_dir / "submission_phase2_dem_gat.csv", dtype={"fipsCode": str})
        template = pd.read_csv(SUBMISSION_FILE, dtype={"fipsCode": str})
        checks.check_submission(submission, template, actual, refs, atol=1e-7)
    return dict(submission_reproduced=True, final_graph_reproduced=True,
                supervision_reads_blocked=True, training_calls_blocked=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    checks.require(not (run_dir / "COMPLETE").exists(), "COMPLETE already exists; verified runs are immutable")
    manifest = _check_run(run_dir)
    reload_checks = _independent_reload(run_dir, manifest)
    checks.require(set(reload_checks) == {"submission_reproduced", "final_graph_reproduced", "supervision_reads_blocked", "training_calls_blocked"}
                   and all(value is True for value in reload_checks.values()), "independent reload checks incomplete")
    verification = dict(protocol=PROTOCOL, run_id=manifest["run_id"], artifact_checks=True,
                        numeric_recomputation=True, lineage_checks=True, independent_reload=True,
                        **reload_checks, passed=True)
    tmp = run_dir / "verification.json.tmp"
    tmp.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    tmp.replace(run_dir / "verification.json")
    tmp = run_dir / "COMPLETE.tmp"
    tmp.write_text("verified", encoding="utf-8")
    tmp.replace(run_dir / "COMPLETE")
    print(json.dumps(verification, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
