"""Direct four-component GAT base-learner experiment.

This protocol intentionally has no LightGBM stage, residual target, alpha
selection, or correction channel.  Each outer county fold trains four direct
GAT predictors on the other four folds.  Their clipped component predictions
are recombined into official OSI for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COMPONENTS, COMPONENT_TARGETS_FILE, COMPONENT_WEIGHTS, CV_FILE, CV_FOLD_COUNTS,
    CV_VERSION, DEFAULT_FEATURE_DIR, EXPERIMENT_VERSION, FEATURE_NAMES_FILE,
    FEATURE_VERSION, FIXED_THREADS, GAT_DROPOUT, GAT_EPOCHS, GAT_HEADS, GAT_LOSS,
    GAT_HIDDEN, GAT_LR, GAT_PATIENCE, GAT_TIME_STRIDE, GAT_WEIGHT_DECAY, GEO_DBF,
    GRAPH_K, HORIZONS, HORIZON_HOURS, N_FOLDS, OFFICIAL_SCOREABLE_ROWS, OUTPUT_ROOT,
    PARQUET_ENGINE, PRED_END, PRED_START, PROTOCOL, SUBMISSION_FILE, TARGET_SCALE_FLOOR,
    TERRAIN_FILE, TRAIN_SCOREABLE_ROWS, component_weighted_osi,
)
from data import (
    add_neighbor_feature_aggregates, graph_feature_names, load_component_targets,
    load_feature_bundle, load_supervision, load_terrain_features, make_county_time_view,
    validate_component_recomposition,
)
from direct_gat import DirectGAT, fit_direct_gat, predict_direct_gat
from metrics import pooled_metrics, post_process_osi
from model_records import file_hash, load_record, publish_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "cv", "final", "all"), default="preflight")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--parquet-engine", choices=("pyarrow", "fastparquet"), default=PARQUET_ENGINE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=GAT_EPOCHS)
    parser.add_argument("--patience", type=int, default=GAT_PATIENCE)
    parser.add_argument("--time-stride", type=int, default=GAT_TIME_STRIDE)
    parser.add_argument("--k", type=int, default=GRAPH_K)
    parser.add_argument("--gat-loss", choices=(GAT_LOSS,), default=GAT_LOSS)
    args = parser.parse_args(argv)
    if args.run_id is None:
        args.run_id = f"{PROTOCOL}_component_seed{args.seed}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        raise ValueError("run-id may contain only letters, digits, '.', '_' and '-'")
    if args.epochs < 1 or args.patience < 1 or args.time_stride != 1 or args.k != 8:
        raise ValueError("this protocol fixes epochs/patience >=1, time_stride=1 and k=8")
    if args.stage == "preflight" and args.resume:
        raise ValueError("--resume is not valid for preflight")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_frame(path: Path, frame: pd.DataFrame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, engine="pyarrow", index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _versions():
    result = {}
    for name in ("numpy", "pandas", "pyarrow", "fastparquet", "shapely", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _resolve_device(requested):
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return requested


def _seed_process(seed, device):
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(FIXED_THREADS)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def scoped_seed(global_seed, stage, scope, horizon, component):
    payload = json.dumps([int(global_seed), stage, tuple(sorted(int(x) for x in scope)), horizon, component], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def load_cv_folds(meta):
    assignment = pd.read_csv(CV_FILE, dtype={"fipsCode": str})
    required = {"fipsCode", "stateAbbr", "severity_tier", "fold"}
    if set(assignment.columns) != required:
        raise ValueError(f"{CV_VERSION} CV schema must be exactly {sorted(required)}")
    assignment["fips_str"] = assignment["fipsCode"].astype(str).str.zfill(5)
    assignment["fold"] = pd.to_numeric(assignment["fold"], errors="raise").astype(int)
    assignment["severity_tier"] = pd.to_numeric(assignment["severity_tier"], errors="raise").astype(int)
    if assignment["fips_str"].duplicated().any() or set(assignment.fold) != set(range(N_FOLDS)):
        raise ValueError("invalid fixed county CV assignment")
    if assignment.fold.value_counts().sort_index().to_dict() != CV_FOLD_COUNTS:
        raise ValueError("fixed CV county counts differ")
    meta_county = meta[["fips_str", "stateAbbr", "severity_tier"]].drop_duplicates().set_index("fips_str").sort_index()
    cv_county = assignment.set_index("fips_str")[["stateAbbr", "severity_tier", "fold"]].sort_index()
    if not meta_county.index.equals(cv_county.index) or not meta_county[["stateAbbr", "severity_tier"]].equals(cv_county[["stateAbbr", "severity_tier"]]):
        raise ValueError("CV county/state/severity metadata mismatch")
    row_fold = meta["fips_str"].map(cv_county["fold"]).to_numpy(dtype=int)
    return row_fold, [(np.flatnonzero(row_fold != i), np.flatnonzero(row_fold == i)) for i in range(N_FOLDS)]


def _graph_hash(all_fips, edge_index, edge_attr):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(all_fips), separators=(",", ":")).encode())
    digest.update(np.asarray(edge_index, dtype=np.int64).tobytes())
    digest.update(np.asarray(edge_attr, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _preflight(args, load_supervision_data=True):
    versions = _versions()
    required = ("numpy", "pandas", "pyarrow", "shapely", "torch")
    missing = [name for name in required if versions.get(name) is None]
    if missing:
        raise RuntimeError(f"missing dependencies: {missing}")
    from spatial import build_spatial_graph
    bundle = load_feature_bundle(args.feature_dir, args.parquet_engine)
    supervision = load_supervision(feature_bundle=bundle, parquet_engine=args.parquet_engine) if load_supervision_data else None
    component_targets = load_component_targets(bundle.meta_train, args.parquet_engine) if load_supervision_data else None
    if args.parquet_engine == "fastparquet":
        reference = load_feature_bundle(args.feature_dir, "pyarrow")
        for left, right in ((bundle.X_train, reference.X_train), (bundle.X_test, reference.X_test),
                            (bundle.meta_train, reference.meta_train), (bundle.meta_test, reference.meta_test)):
            pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    row_fold, folds = load_cv_folds(bundle.meta_train)
    terrain = load_terrain_features()
    all_fips = sorted(set(bundle.meta_train.fips_str) | set(bundle.meta_test.fips_str))
    if len(all_fips) != 302:
        raise ValueError("expected 302 graph counties")
    coords, edge_index, edge_attr = build_spatial_graph(all_fips, GEO_DBF, GEO_DBF.with_suffix(".shp"), k=args.k)
    if edge_index.shape != (2, 3028) or edge_attr.shape != (3028, 4):
        raise ValueError(f"unexpected fixed graph shape: {edge_index.shape}/{edge_attr.shape}")
    zero_train = np.zeros(len(bundle.X_train), dtype=float)
    zero_test = np.zeros(len(bundle.X_test), dtype=float)
    raw, _, _, _, _, names = make_county_time_view(
        bundle.X_train, bundle.X_test, bundle.meta_train, bundle.meta_test,
        zero_train, zero_test, HORIZONS[0], coords, terrain, bundle.feature_names,
    )
    graph, extra_names = add_neighbor_feature_aggregates(raw, names, edge_index, HORIZONS[0])
    if graph.shape != (144, 302, 205) or not np.isfinite(graph).all():
        raise ValueError("direct GAT 205-dimensional input preflight failed")
    recomposition = validate_component_recomposition(component_targets, supervision.y_train) if load_supervision_data else None
    input_manifest = {
        "feature_version": FEATURE_VERSION, "experiment_version": EXPERIMENT_VERSION,
        "feature_dir": str(bundle.feature_dir), "parquet_engine": bundle.parquet_engine,
        "feature_side_hash": bundle.input_hash,
        "targets_train_sha256": None if supervision is None else supervision.target_hash,
        "component_targets": str(COMPONENT_TARGETS_FILE),
        "component_targets_sha256": _sha256(COMPONENT_TARGETS_FILE),
        "cv_file": str(CV_FILE), "cv_sha256": _sha256(CV_FILE),
        "terrain_sha256": _sha256(TERRAIN_FILE), "geo_dbf_sha256": _sha256(GEO_DBF),
        "geo_shp_sha256": _sha256(GEO_DBF.with_suffix(".shp")),
        "submission_sha256": _sha256(SUBMISSION_FILE),
        "graph_hash": _graph_hash(all_fips, edge_index, edge_attr), "all_fips": all_fips,
    }
    return {
        "bundle": bundle, "supervision": supervision, "component_targets": component_targets,
        "row_fold": row_fold, "folds": folds, "terrain": terrain, "coords": coords,
        "edge_index": edge_index, "edge_attr": edge_attr, "all_fips": all_fips,
        "graph_hash": input_manifest["graph_hash"], "graph_feature_names": [
            "base", *names, "latitude", "longitude", *list(terrain.columns), *extra_names,
        ], "input_manifest": input_manifest, "versions": versions, "recomposition": recomposition,
    }


class DirectGATContext:
    def __init__(self, args, preflight, run_dir, inference_only=False):
        self.args = args
        self.bundle = preflight["bundle"]
        self.supervision = preflight["supervision"]
        self.component_targets = preflight["component_targets"]
        self.row_fold = preflight["row_fold"]
        self.terrain = preflight["terrain"]
        self.coords = preflight["coords"]
        self.edge_index = preflight["edge_index"]
        self.edge_attr = preflight["edge_attr"]
        self.all_fips = preflight["all_fips"]
        self.graph_hash = preflight["graph_hash"]
        self.graph_feature_names = tuple(preflight["graph_feature_names"])
        self.run_dir = Path(run_dir)
        self.device = _resolve_device(args.device)
        self.inference_only = inference_only
        self.raw_cache = {}
        self.input_cache = {}
        self.pred_cache = {}
        self.model_dir = self.run_dir / "models" / "gat"

    def raw_graph(self, horizon):
        if horizon in self.raw_cache:
            return self.raw_cache[horizon]
        zeros_train = np.zeros(len(self.bundle.X_train), dtype=float)
        zeros_test = np.zeros(len(self.bundle.X_test), dtype=float)
        raw, base, train_rows, test_rows, all_fips, names = make_county_time_view(
            self.bundle.X_train, self.bundle.X_test, self.bundle.meta_train, self.bundle.meta_test,
            zeros_train, zeros_test, horizon, self.coords, self.terrain, self.bundle.feature_names,
        )
        graph, extra_names = add_neighbor_feature_aggregates(raw, names, self.edge_index, horizon)
        names = tuple(["base", *names, "latitude", "longitude", *list(self.terrain.columns), *extra_names])
        if names != self.graph_feature_names or graph.shape[-1] != 205:
            raise ValueError("direct GAT graph schema mismatch")
        if not np.allclose(graph[:, :, 0], 0.0) or not np.isfinite(graph).all():
            raise FloatingPointError("direct GAT graph contains invalid placeholder/input")
        result = (graph, train_rows, test_rows, all_fips, names)
        self.raw_cache[horizon] = result
        return result

    def _inputs(self, scope, horizon, component):
        scope = tuple(sorted(int(x) for x in scope))
        key = (scope, horizon, component)
        if key in self.input_cache:
            return self.input_cache[key]
        graph, train_rows, test_rows, all_fips, names = self.raw_graph(horizon)
        scope_set = set(scope)
        train_node_mask = np.asarray([
            fips in set(self.bundle.meta_train.fips_str)
            and self.row_fold[np.flatnonzero(self.bundle.meta_train.fips_str.to_numpy() == fips)[0]] in scope_set
            for fips in all_fips
        ], dtype=bool)
        valid_time = np.arange(PRED_START, PRED_END) + HORIZON_HOURS[horizon] <= PRED_END - 1
        stat_mask = valid_time[:, None] & train_node_mask[None, :]
        if self.inference_only:
            import torch
            checkpoint_path = self._path(scope, horizon, component)
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            mean = np.asarray(payload["feature_mean"], dtype=np.float64)
            std = np.asarray(payload["feature_std"], dtype=np.float64)
            scale = float(payload["target_scale"])
        else:
            values = graph.reshape(-1, graph.shape[-1]).astype(np.float64)
            train_values = values[stat_mask.reshape(-1)]
            mean = np.mean(train_values, axis=0, dtype=np.float64)
            std = np.std(train_values, axis=0, ddof=0, dtype=np.float64)
            std[std < 1e-6] = 1.0
            scale = np.nan
        scaled = ((graph.astype(np.float64) - mean) / std).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise FloatingPointError("direct GAT standardized input is non-finite")
        target = np.zeros(graph.shape[:2], dtype=np.float64)
        mask = np.zeros(graph.shape[:2], dtype=bool)
        if not self.inference_only:
            if self.component_targets is None:
                raise PermissionError("training component targets are unavailable")
            target_values = self.component_targets[horizon][component]
            for t in range(graph.shape[0]):
                if not valid_time[t]:
                    continue
                for node in np.flatnonzero(train_node_mask):
                    row_id = int(train_rows[t, node])
                    if row_id < 0:
                        raise ValueError("missing training row in direct GAT graph")
                    value = float(target_values[row_id])
                    if not np.isfinite(value):
                        raise ValueError("valid component supervision is missing")
                    target[t, node] = value
                    mask[t, node] = True
        if not self.inference_only and not mask.any():
            raise ValueError("direct GAT has no supervised cells")
        if mask.any():
            scale = float(max(np.std(target[mask], ddof=0), TARGET_SCALE_FLOOR))
        result = {
            "scaled": scaled, "target": target, "mask": mask, "target_scale": scale,
            "mean": mean, "std": std, "train_rows": train_rows, "test_rows": test_rows,
            "all_fips": all_fips, "names": names, "valid_time": valid_time,
        }
        self.input_cache[key] = result
        return result

    def _path(self, scope, horizon, component):
        return self.model_dir / f"gat_direct_{component}_{horizon.replace('osi_target_', '')}_S{'-'.join(map(str, scope))}.pt"

    def fit_one(self, scope, horizon, component):
        scope = tuple(sorted(int(x) for x in scope))
        key = (scope, horizon, component)
        if key in self.pred_cache:
            return self.pred_cache[key]
        inputs = self._inputs(scope, horizon, component)
        path = self._path(scope, horizon, component)
        model_id = f"direct_gat:{component}:{horizon}:S{','.join(map(str, scope))}"
        receipt = load_record(path, self.args.expected_run_identity, model_id) if path.exists() and self.args.resume else None
        if self.args.resume and receipt is None:
            raise FileNotFoundError(f"missing completed direct GAT model: {path}")
        seed = scoped_seed(self.args.seed, "direct_gat", scope, horizon, component)
        if receipt is not None:
            model, payload = self._load(path)
            self._check_payload(payload, inputs, scope, horizon, component, model_id, seed)
        else:
            model, _, fit_info = fit_direct_gat(
                inputs["scaled"], inputs["target"] / inputs["target_scale"], inputs["mask"], self.edge_index,
                epochs=self.args.epochs, patience=self.args.patience, hidden=GAT_HIDDEN,
                heads=GAT_HEADS, dropout=GAT_DROPOUT, lr=GAT_LR, weight_decay=GAT_WEIGHT_DECAY,
                seed=seed, device=self.device, edge_attr=self.edge_attr, loss_mode=GAT_LOSS,
            )
            payload = self._payload(model, fit_info, inputs, scope, horizon, component, model_id, seed)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            publish_model(
                path, self.args.expected_run_identity, model_id,
                {"scope": scope, "horizon": horizon, "component": component, "seed": seed},
                lambda temporary: self._save_payload(temporary, payload),
                lambda temporary: self._load(temporary),
            )
        normalized = predict_direct_gat(model, inputs["scaled"], self.edge_index, self.device, self.edge_attr)
        values = np.clip(normalized * inputs["target_scale"], 0.0, 1.0)
        if not np.isfinite(values).all():
            raise FloatingPointError("direct component prediction is non-finite")
        result = {"values": values, "model_id": model_id, "payload": payload, "inputs": inputs, "path": path}
        self.pred_cache[key] = result
        return result

    @staticmethod
    def _save_payload(path, payload):
        import torch
        torch.save(payload, path)

    def _payload(self, model, fit_info, inputs, scope, horizon, component, model_id, seed):
        return {
            "protocol": PROTOCOL, "run_identity": self.args.expected_run_identity, "model_id": model_id,
            "scope": scope, "horizon": horizon, "component": component, "seed": seed,
            "architecture": {"in_dim": 205, "hidden": GAT_HIDDEN, "heads": GAT_HEADS,
                              "dropout": GAT_DROPOUT, "edge_dim": int(self.edge_attr.shape[-1]),
                              "model": "DirectGAT", "output": "component_scaled"},
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "feature_names": list(inputs["names"]),
            "feature_schema_hash": _json_digest(list(inputs["names"])),
            "feature_mean": inputs["mean"], "feature_std": inputs["std"],
            "target_scale": inputs["target_scale"], "graph_hash": self.graph_hash,
            "all_fips": list(inputs["all_fips"]), "time_order": list(range(PRED_START, PRED_END)),
            "edge_index": self.edge_index, "edge_attr": self.edge_attr,
            "fit_info": fit_info,
            "training_config": {"epochs": self.args.epochs, "patience": self.args.patience,
                                 "time_stride": self.args.time_stride, "loss_mode": self.args.gat_loss,
                                 "lr": GAT_LR, "weight_decay": GAT_WEIGHT_DECAY},
        }

    def _load(self, path):
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        arch = payload["architecture"]
        model = DirectGAT(arch["in_dim"], arch["hidden"], arch["heads"], arch["dropout"], arch["edge_dim"]).to(self.device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        return model, payload

    def _check_payload(self, payload, inputs, scope, horizon, component, model_id, seed):
        if payload.get("protocol") != PROTOCOL or payload.get("run_identity") != self.args.expected_run_identity:
            raise ValueError("direct GAT checkpoint identity mismatch")
        for key, expected in (("model_id", model_id), ("scope", scope), ("horizon", horizon), ("component", component), ("seed", seed)):
            if payload.get(key) != expected:
                raise ValueError(f"direct GAT checkpoint {key} mismatch")
        if payload.get("graph_hash") != self.graph_hash or payload.get("feature_names") != list(inputs["names"]):
            raise ValueError("direct GAT checkpoint graph/schema mismatch")
        if not np.array_equal(np.asarray(payload["feature_mean"]), inputs["mean"]) or not np.array_equal(np.asarray(payload["feature_std"]), inputs["std"]):
            raise ValueError("direct GAT checkpoint feature scaling mismatch")
        if float(payload["target_scale"]) != float(inputs["target_scale"]):
            raise ValueError("direct GAT checkpoint target scale mismatch")


def _make_manifest(args, preflight, run_dir):
    device = _resolve_device(args.device)
    parameters = {"device": device, "epochs": args.epochs, "patience": args.patience,
                  "time_stride": args.time_stride, "k": args.k, "gat_loss": args.gat_loss,
                  "hidden": GAT_HIDDEN, "heads": GAT_HEADS, "dropout": GAT_DROPOUT,
                  "lr": GAT_LR, "weight_decay": GAT_WEIGHT_DECAY, "target_scale_floor": TARGET_SCALE_FLOOR}
    inputs = preflight["input_manifest"]
    identity = _json_digest({"protocol": PROTOCOL, "seed": args.seed, "parameters": parameters,
                             "inputs": inputs, "source_hashes": _source_hashes()})
    return {"protocol": PROTOCOL, "experiment_version": EXPERIMENT_VERSION, "run_id": args.run_id,
            "seed": args.seed, "parameters": parameters, "inputs": inputs,
            "feature_schema_hash": _json_digest(preflight["graph_feature_names"]),
            "source_hashes": _source_hashes(), "identity": identity,
            "created_utc": datetime.now(timezone.utc).isoformat()}


def _source_hashes():
    names = ("config.py", "data.py", "spatial.py", "metrics.py", "direct_gat.py", "main.py", "verify_artifacts.py")
    root = Path(__file__).resolve().parent
    return {name: _sha256(root / name) for name in names if (root / name).is_file()}


def _create_run(args, preflight):
    run_dir = OUTPUT_ROOT / "runs" / args.run_id
    manifest = _make_manifest(args, preflight, run_dir)
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists():
        if not args.resume:
            raise FileExistsError(f"run exists; use --resume only with unchanged code/config: {run_dir}")
        if not manifest_path.exists():
            raise ValueError("existing run lacks run_manifest.json")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("identity") != manifest["identity"]:
            raise ValueError("resume identity mismatch; use a new run-id after code changes")
        if (run_dir / "COMPLETE").exists():
            raise FileExistsError("COMPLETE run is immutable")
    else:
        run_dir.mkdir(parents=True)
        _atomic_json(manifest_path, manifest)
    (run_dir / "attempts.jsonl").open("a", encoding="utf-8").write(json.dumps({"stage": args.stage, "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    args.expected_run_identity = manifest["identity"]
    return run_dir, manifest


def _write_static(run_dir, args, preflight):
    bundle = preflight["bundle"]
    folds = []
    for row_id, fold in enumerate(preflight["row_fold"]):
        row = bundle.meta_train.iloc[row_id]
        folds.append({"row_id": row_id, "fipsCode": str(row.fips_str), "fold": int(fold),
                      "stateAbbr": str(row.stateAbbr), "severity_tier": int(row.severity_tier)})
    _atomic_frame(run_dir / "folds.csv", pd.DataFrame(folds)) if not (run_dir / "folds.csv").exists() else None
    if not (run_dir / "inputs_manifest.json").exists():
        _atomic_json(run_dir / "inputs_manifest.json", preflight["input_manifest"])
        _atomic_json(run_dir / "environment.json", {"versions": preflight["versions"], "device": _resolve_device(args.device), "threads": FIXED_THREADS})
        _atomic_json(run_dir / "feature_schema.json", {"ordered_phase1_features": list(bundle.feature_names),
            "graph_schema": preflight["graph_feature_names"], "graph_input_dim": 205,
            "base_channel": "constant_zero_placeholder_no_lightgbm"})
        np.savez_compressed(run_dir / "graph.npz", coords=preflight["coords"], edge_index=preflight["edge_index"], edge_attr=preflight["edge_attr"], all_fips=preflight["all_fips"])


def _row_prediction(ctx, h, row_id, component_grids):
    hour = int(ctx.bundle.meta_train.iloc[row_id].hour_idx)
    t = hour - PRED_START
    node = ctx.all_fips.index(str(ctx.bundle.meta_train.iloc[row_id].fips_str))
    parts = {c: float(component_grids[c][t, node]) for c in COMPONENTS}
    osi = float(post_process_osi(component_weighted_osi(parts)))
    scoreable = hour + HORIZON_HOURS[h] <= PRED_END - 1
    return parts, osi if scoreable else np.nan, scoreable


def _component_models(ctx, scope, h):
    return {c: ctx.fit_one(scope, h, c) for c in COMPONENTS}


def _run_cv(ctx, args, run_dir):
    if (run_dir / "CV_COMPLETE").exists():
        return
    outer_rows, fold_rows = [], []
    for h in HORIZONS:
        for outer_fold in range(N_FOLDS):
            scope = tuple(k for k in range(N_FOLDS) if k != outer_fold)
            models = _component_models(ctx, scope, h)
            grids = {c: models[c]["values"] for c in COMPONENTS}
            expected = (ctx.row_fold == outer_fold) & (ctx.bundle.meta_train.hour_idx.to_numpy(dtype=int) + HORIZON_HOURS[h] <= PRED_END - 1)
            row_ids = np.flatnonzero(ctx.row_fold == outer_fold)
            y = ctx.supervision.y_train[h].to_numpy(dtype=float)
            preds, truths = [], []
            for row_id in row_ids:
                parts, prediction, scoreable = _row_prediction(ctx, h, row_id, grids)
                row = ctx.bundle.meta_train.iloc[row_id]
                record = {"run_id": args.run_id, "protocol": PROTOCOL, "fipsCode": str(row.fips_str),
                          "timestamp_et": str(row.timestamp_et), "hour_idx": int(row.hour_idx),
                          "target_timestamp": str(pd.to_datetime(row.timestamp_et) + pd.Timedelta(hours=HORIZON_HOURS[h])),
                          "horizon": h, "outer_fold": outer_fold, "is_scoreable": scoreable,
                          "y_true": float(y[row_id]) if scoreable else np.nan, "prediction": prediction}
                for c in COMPONENTS:
                    record[f"prediction_{c}"] = parts[c]
                    record[f"model_id_{c}"] = models[c]["model_id"]
                outer_rows.append(record)
                if scoreable:
                    preds.append(prediction); truths.append(y[row_id])
            fold_rows.append({"horizon": h, "outer_fold": outer_fold, "model": "direct_component_gat",
                              **pooled_metrics(np.asarray(truths), np.asarray(preds))})
    outer = pd.DataFrame(outer_rows)
    if len(outer) != len(ctx.bundle.X_train) * len(HORIZONS):
        raise ValueError("outer OOF coverage is incomplete")
    for h in HORIZONS:
        subset = outer[outer.horizon == h]
        expected = subset.hour_idx.to_numpy(dtype=int) + HORIZON_HOURS[h] <= PRED_END - 1
        if int(expected.sum()) != TRAIN_SCOREABLE_ROWS[h] or not np.isfinite(subset.loc[expected, "prediction"]).all():
            raise FloatingPointError(f"invalid direct GAT OOF for {h}")
    summary = []
    for h, subset in outer[outer.is_scoreable].groupby("horizon", sort=True):
        summary.append({"horizon": h, "model": "direct_component_gat", **pooled_metrics(subset.y_true, subset.prediction)})
    county = []
    for (h, fips), group in outer[outer.is_scoreable].groupby(["horizon", "fipsCode"], sort=True):
        err = group.prediction.to_numpy() - group.y_true.to_numpy()
        county.append({"horizon": h, "fipsCode": fips, "outer_fold": int(group.outer_fold.iloc[0]), "n": len(group),
                       "sse": float(np.sum(err * err, dtype=np.float64)), "rmse": float(np.sqrt(np.mean(err * err)))})
    _atomic_frame(run_dir / "outer_oof.parquet", outer)
    _atomic_frame(run_dir / "fold_metrics.csv", pd.DataFrame(fold_rows))
    _atomic_frame(run_dir / "cv_summary.csv", pd.DataFrame(summary))
    _atomic_frame(run_dir / "county_metrics.csv", pd.DataFrame(county))
    manifest = {"protocol": PROTOCOL, "run_id": args.run_id,
                "outer_oof_sha256": _sha256(run_dir / "outer_oof.parquet"),
                "cv_summary_sha256": _sha256(run_dir / "cv_summary.csv")}
    _atomic_json(run_dir / "cv_manifest.json", manifest)
    (run_dir / "CV_COMPLETE").write_text(_json_digest(manifest), encoding="utf-8")


def _run_final(ctx, args, run_dir):
    if not (run_dir / "CV_COMPLETE").exists():
        raise ValueError("final requires CV_COMPLETE")
    rows = []
    full_scope = tuple(range(N_FOLDS))
    for h in HORIZONS:
        models = _component_models(ctx, full_scope, h)
        grids = {c: models[c]["values"] for c in COMPONENTS}
        for row_id, row in ctx.bundle.meta_test.iterrows():
            hour = int(row.hour_idx); t = hour - PRED_START; node = ctx.all_fips.index(str(row.fips_str))
            parts = {c: float(np.clip(grids[c][t, node], 0.0, 1.0)) for c in COMPONENTS}
            scoreable = hour + HORIZON_HOURS[h] <= PRED_END - 1
            prediction = float(post_process_osi(component_weighted_osi(parts))) if scoreable else np.nan
            record = {"run_id": args.run_id, "protocol": PROTOCOL, "fipsCode": str(row.fips_str),
                      "timestamp_et": str(row.timestamp_et), "hour_idx": hour, "horizon": h,
                      "is_scoreable": scoreable, "prediction": prediction}
            for c in COMPONENTS:
                record[f"prediction_{c}"] = parts[c]; record[f"model_id_{c}"] = models[c]["model_id"]
            rows.append(record)
    test = pd.DataFrame(rows)
    _atomic_frame(run_dir / "test_predictions.parquet", test)
    submission = pd.read_csv(SUBMISSION_FILE, dtype={"fipsCode": str})
    template_keys = list(zip(submission.fipsCode.astype(str).str.zfill(5), submission.timestamp_et.astype(str)))
    test_keys = list(zip(ctx.bundle.meta_test.fips_str.astype(str), ctx.bundle.meta_test.timestamp_et.astype(str)))
    if template_keys != test_keys:
        raise ValueError("submission key order mismatch")
    for h in HORIZONS:
        values = test.loc[test.horizon == h, "prediction"].to_numpy(dtype=float)
        if np.isfinite(values).sum() != OFFICIAL_SCOREABLE_ROWS[h]:
            raise ValueError(f"submission valid count mismatch for {h}")
        submission[h] = values
    _atomic_frame(run_dir / "submission_phase2_dem_gat.csv", submission)
    _atomic_json(run_dir / "submission_audit.json", {"rows": len(submission), "protocol": PROTOCOL,
        "finite_counts": {h: int(np.isfinite(submission[h]).sum()) for h in HORIZONS}, "passed": True})
    _atomic_json(run_dir / "final_manifest.json", {"protocol": PROTOCOL, "run_id": args.run_id,
        "test_predictions_sha256": _sha256(run_dir / "test_predictions.parquet")})
    (run_dir / "FINAL_READY").write_text("direct component GAT final artifacts written", encoding="utf-8")
    (run_dir / "REPORT.md").write_text(
        f"# {PROTOCOL}\n\n- run_id: `{args.run_id}`\n- no LightGBM\n- no residual learning\n"
        "- four direct component GAT models are clipped to [0,1] and recombined as official OSI\n",
        encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.stage == "preflight":
        _preflight(args)
        print("PREFLIGHT_OK")
        return 0
    preflight = _preflight(args)
    device = _resolve_device(args.device)
    _seed_process(args.seed, device)
    run_dir, manifest = _create_run(args, preflight)
    _write_static(run_dir, args, preflight)
    ctx = DirectGATContext(args, preflight, run_dir)
    cv_done = (run_dir / "CV_COMPLETE").exists()
    final_done = (run_dir / "FINAL_READY").exists()
    if args.stage in {"cv", "all"} and not cv_done:
        _run_cv(ctx, args, run_dir)
        cv_done = True
    if args.stage in {"final", "all"} and not final_done:
        if not cv_done:
            raise ValueError("final requires CV_COMPLETE")
        _run_final(ctx, args, run_dir)
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
