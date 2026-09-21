"""CLI for the dem_v158_nested_v2 leakage-isolated experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
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
    COMPONENT_TARGETS_FILE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CV_FILE,
    CV_VERSION,
    DEFAULT_FEATURE_DIR,
    EXPERIMENT_VERSION,
    FEATURE_VERSION,
    GAT_EPOCHS,
    GAT_LOSS,
    GAT_PATIENCE,
    GAT_TIME_STRIDE,
    GRAPH_INPUT_DIM,
    GRAPH_SCHEMA_VERSION,
    GEO_DBF,
    GRAPH_K,
    HORIZON_HOURS,
    HORIZONS,
    N_FOLDS,
    OFFICIAL_SCOREABLE_ROWS,
    OUTPUT_ROOT,
    PARQUET_ENGINE,
    PRED_END,
    PRED_START,
    PROTOCOL,
    SUBMISSION_FILE,
    TERRAIN_FILE,
    TRAIN_SCOREABLE_ROWS,
)
from data import (
    load_component_targets,
    load_feature_bundle,
    load_supervision,
    load_terrain_features,
    make_county_time_view,
    graph_feature_names,
)
from metrics import pooled_metrics, post_process_osi
import run_identity as identity


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "cv", "final", "all"), default="preflight")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--parquet-engine", choices=("pyarrow", "fastparquet"), default=PARQUET_ENGINE)
    parser.add_argument("--base-mode", choices=("direct", "component_v158"), default="direct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=GAT_EPOCHS)
    parser.add_argument("--patience", type=int, default=GAT_PATIENCE)
    parser.add_argument("--time-stride", type=int, default=GAT_TIME_STRIDE)
    parser.add_argument("--k", type=int, default=GRAPH_K)
    parser.add_argument("--gat-loss", choices=(GAT_LOSS,), default=GAT_LOSS)
    args = parser.parse_args(argv)
    if args.run_id is None:
        args.run_id = f"{PROTOCOL}_{args.base_mode}_seed{args.seed}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        raise ValueError("run-id may contain only letters, digits, '.', '_' and '-'")
    if args.stage == "preflight" and args.resume:
        raise ValueError("--resume is only valid for cv/final/all")
    if args.epochs < 1 or args.patience < 1 or args.time_stride < 1:
        raise ValueError("epochs, patience and time-stride must be >= 1")
    if args.time_stride != GAT_TIME_STRIDE:
        raise ValueError("dem_v158_nested_v2 fixes --time-stride=1")
    if args.k != GRAPH_K:
        raise ValueError("dem_v158_nested_v2 fixes --k=8")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_dataframe(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, engine="pyarrow", index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _versions():
    names = ("numpy", "pandas", "pyarrow", "fastparquet", "lightgbm", "torch", "shapely")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _version_tuple(value):
    if not value:
        return (0,)
    return tuple(int(part) if part.isdigit() else 0 for part in str(value).split(".")[:3])


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return requested


def _seed_process(seed: int, device: str):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:
        raise RuntimeError("the selected Torch device cannot satisfy deterministic algorithms") from exc


def _load_training_modules():
    """Lazy import so preflight can report missing training dependencies cleanly."""

    try:
        import torch  # validate the real runtime even though StackContext imports lazily
        from base_model import BaseModelStore, load_cv_folds
        from stacking import StackContext
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"missing training dependency: {exc.name}") from exc
    return BaseModelStore, load_cv_folds, StackContext


def _graph_identity(all_fips, edge_index, edge_attr):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(all_fips), separators=(",", ":")).encode())
    digest.update(np.asarray(edge_index, dtype=np.int64).tobytes())
    digest.update(np.asarray(edge_attr, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _package_identity(feature_side_hash: str, target_hash: str | None) -> str | None:
    """Derive the full five-table identity without reading labels in inference."""

    if target_hash is None:
        return None
    payload = json.dumps(
        {"feature_side_hash": str(feature_side_hash), "targets_train_sha256": str(target_hash)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _preflight(args, *, load_supervision_data=True):
    versions = _versions()
    missing = [name for name in ("numpy", "pandas", "pyarrow", "torch", "lightgbm", "shapely") if versions.get(name) is None]
    minimums = {"numpy": (2, 0), "pandas": (2, 2), "pyarrow": (15, 0), "lightgbm": (4, 0), "shapely": (2, 0), "torch": (2, 1)}
    incompatible = [name for name, minimum in minimums.items() if versions.get(name) is not None and _version_tuple(versions[name]) < minimum]
    if missing or incompatible:
        raise RuntimeError(f"preflight dependency check failed; missing={missing}, incompatible={incompatible}, versions={versions}")
    from spatial import build_spatial_graph
    bundle = load_feature_bundle(args.feature_dir, args.parquet_engine)
    expected_feature_side_hash = getattr(args, "expected_feature_side_hash", None)
    if expected_feature_side_hash is not None and bundle.input_hash != expected_feature_side_hash:
        raise ValueError("feature-side package hash differs from the frozen run manifest")
    supervision = (
        load_supervision(feature_bundle=bundle, parquet_engine=args.parquet_engine)
        if load_supervision_data else None
    )
    if args.parquet_engine == "fastparquet":
        reference = load_feature_bundle(args.feature_dir, "pyarrow")
        pairs = [
            (bundle.X_train, reference.X_train), (bundle.X_test, reference.X_test),
            (bundle.meta_train, reference.meta_train), (bundle.meta_test, reference.meta_test),
        ]
        if load_supervision_data:
            reference_supervision = load_supervision(feature_bundle=reference, parquet_engine="pyarrow")
            pairs.append((supervision.y_train, reference_supervision.y_train))
        for left, right in pairs:
            if list(left.columns) != list(right.columns) or left.shape != right.shape:
                raise ValueError("fastparquet and pyarrow schema/shape differ")
            pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    BaseModelStore, load_cv_folds, _ = _load_training_modules()
    folds = load_cv_folds(bundle.meta_train)
    terrain = load_terrain_features()
    all_fips = sorted(set(bundle.meta_train["fips_str"]) | set(bundle.meta_test["fips_str"]))
    if len(all_fips) != 302:
        raise ValueError("expected 302 graph counties")
    shp_path = GEO_DBF.with_suffix(".shp")
    coords, edge_index, edge_attr = build_spatial_graph(all_fips, GEO_DBF, shp_path, k=args.k)
    if edge_index.shape != (2, 3028) or edge_attr.shape != (3028, 4):
        raise ValueError(f"unexpected graph shape: {edge_index.shape}, {edge_attr.shape}")
    zeros_train = np.zeros(len(bundle.X_train), dtype=float)
    zeros_test = np.zeros(len(bundle.X_test), dtype=float)
    features, _, _, _, _, names = make_county_time_view(
        bundle.X_train, bundle.X_test, bundle.meta_train, bundle.meta_test,
        zeros_train, zeros_test, HORIZONS[0], coords, terrain, bundle.feature_names,
    )
    if features.shape != (144, 302, GRAPH_INPUT_DIM) or not np.isfinite(features).all():
        raise ValueError("D3 graph input preflight failed")
    component_targets = (
        load_component_targets(bundle.meta_train, args.parquet_engine)
        if load_supervision_data else None
    )
    component_recomposition = None
    if load_supervision_data:
        from data import validate_component_recomposition
        component_recomposition = validate_component_recomposition(
            component_targets, supervision.y_train
        )
    target_hash = None if supervision is None else supervision.target_hash
    package_hash = _package_identity(bundle.input_hash, target_hash)
    if package_hash is None:
        package_hash = getattr(args, "expected_feature_package_hash", None)
    return {
        "bundle": bundle,
        "folds": folds,
        "terrain": terrain,
        "all_fips": all_fips,
        "coords": coords,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "graph_hash": _graph_identity(all_fips, edge_index, edge_attr),
        "graph_feature_names": graph_feature_names(names, terrain.columns, HORIZONS[0]),
        "graph_feature_schemas": {h: graph_feature_names(names, terrain.columns, h) for h in HORIZONS},
        "component_target_count": 0 if component_targets is None else len(component_targets),
        "supervision": supervision,
        "component_targets": component_targets,
        "component_recomposition": component_recomposition,
        "feature_side_hash": bundle.input_hash,
        "target_hash": target_hash,
        "feature_package_hash": package_hash,
        "versions": versions,
        "graph_input_dim": GRAPH_INPUT_DIM,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
    }


def _input_manifest(preflight: dict):
    bundle = preflight["bundle"]
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_dir": str(bundle.feature_dir),
        "parquet_engine": bundle.parquet_engine,
        "package_manifest": bundle.manifest,
        "feature_side_hash": preflight["feature_side_hash"],
        "feature_package_hash": preflight["feature_package_hash"],
        "targets_train_sha256": preflight["target_hash"],
        "component_targets": str(COMPONENT_TARGETS_FILE),
        "component_targets_sha256": _sha256(COMPONENT_TARGETS_FILE),
        "component_target_version": "v1.5.8",
        "cv_file": str(CV_FILE),
        "cv_sha256": _sha256(CV_FILE),
        "terrain_file": str(TERRAIN_FILE),
        "terrain_sha256": _sha256(TERRAIN_FILE),
        "geo_dbf_sha256": _sha256(GEO_DBF),
        "geo_shp_sha256": _sha256(GEO_DBF.with_suffix(".shp")),
        "submission_template_sha256": _sha256(SUBMISSION_FILE),
        "graph_hash": preflight["graph_hash"],
        "all_fips": preflight["all_fips"],
        "graph_input_dim": GRAPH_INPUT_DIM,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
    }


def _create_or_resume_run(args, preflight: dict):
    run_dir = OUTPUT_ROOT / "runs" / args.run_id
    resolved_device = _resolve_device(args.device)
    manifest = {
        "protocol": PROTOCOL,
        "experiment_version": EXPERIMENT_VERSION,
        "run_id": args.run_id,
        "base_mode": args.base_mode,
        "feature_version": FEATURE_VERSION,
        "seed": args.seed,
        "stage": args.stage,
        "parameters": {
            "k": args.k, "epochs": args.epochs, "patience": args.patience,
            "time_stride": args.time_stride, "gat_loss": args.gat_loss,
            "device": resolved_device,
        },
        "inputs": _input_manifest(preflight),
    }
    manifest["identity"] = identity.build_identity(args, manifest["inputs"], resolved_device)
    preflight["run_identity"] = manifest["identity"]["digest"]
    preflight["runtime_environment"] = manifest["identity"]["definition"]["environment"]
    existing = run_dir.exists()
    if not existing and (args.resume or args.stage == "final"):
        raise FileNotFoundError("resume/final requires an existing run")
    if existing and not args.resume:
        raise FileExistsError(f"run already exists; use --resume only after identity validation: {run_dir}")
    if existing:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise ValueError("cannot resume a run without run_manifest.json")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity.assert_identity(old.get("identity"), manifest["identity"])
        if (run_dir / "COMPLETE").exists():
            raise FileExistsError("this run is already COMPLETE and immutable")
        comparable = {
            key: old.get(key)
            for key in ("protocol", "experiment_version", "run_id", "base_mode", "feature_version", "seed", "parameters", "inputs")
        }
        if comparable != {key: manifest.get(key) for key in comparable}:
            raise ValueError("resume manifest identity mismatch")
        if args.stage == "final" and not (run_dir / "CV_COMPLETE").exists():
            raise ValueError("final requires completed CV evidence")
    else:
        run_dir.mkdir(parents=True)
        _atomic_json(run_dir / "run_manifest.json", manifest)
    with (run_dir / "attempts.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "stage": args.stage,
                              "resume": args.resume, "identity": manifest["identity"]["digest"]}) + "\n")
    return run_dir, manifest


def _make_context(
    args, preflight: dict, run_dir: Path, resume=False, strict_resume=False,
    inference_only=False,
):
    BaseModelStore, load_cv_folds, StackContext = _load_training_modules()
    bundle = preflight["bundle"]
    folds = load_cv_folds(bundle.meta_train)
    model_dir = run_dir / "models" / "base"
    store = BaseModelStore(
        bundle.X_train, bundle.X_test, bundle.meta_train, folds,
        y_train=None if inference_only else preflight["supervision"].y_train,
        global_seed=args.seed, model_dir=model_dir, parquet_engine=args.parquet_engine,
        component_targets=None if inference_only else preflight["component_targets"],
        feature_package_hash=preflight["feature_package_hash"],
        feature_side_hash=preflight["feature_side_hash"],
        feature_schema_hash=hashlib.sha256(
            json.dumps(list(bundle.feature_names), separators=(",", ":")).encode()
        ).hexdigest(),
        protocol=PROTOCOL,
        run_identity=getattr(args, "expected_run_identity", preflight.get("run_identity")),
        reuse_existing=resume,
    )
    if resume:
        store.load_from_run(run_dir)
    return StackContext(
        bundle=bundle,
        terrain=preflight["terrain"],
        folds=folds,
        coords=preflight["coords"],
        edge_index=preflight["edge_index"],
        edge_attr=preflight["edge_attr"],
        base_store=store,
        global_seed=args.seed,
        device=_resolve_device(args.device),
        epochs=args.epochs,
        patience=args.patience,
        time_stride=args.time_stride,
        model_dir=run_dir / "models" / "gat",
        loss_mode=args.gat_loss,
        resume_only=strict_resume,
        reuse_existing=resume,
        inference_only=inference_only,
        graph_hash=preflight["graph_hash"],
        feature_package_hash=preflight["feature_package_hash"],
        feature_side_hash=preflight["feature_side_hash"],
        run_identity=getattr(args, "expected_run_identity", preflight.get("run_identity")),
    )


def _write_static_artifacts(run_dir: Path, args, preflight: dict):
    def frozen_json(name, payload):
        path = run_dir / name
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if identity.differences(old, payload):
                raise ValueError(f"immutable static artifact mismatch: {name}")
        else:
            _atomic_json(path, payload)

    bundle = preflight["bundle"]
    folds = preflight["folds"]
    fold_rows = []
    for fold, (_, valid) in enumerate(folds):
        for row_id in valid:
            fold_rows.append({
                "row_id": int(row_id),
                "fipsCode": str(bundle.meta_train.iloc[row_id]["fips_str"]),
                "fold": fold,
                "stateAbbr": str(bundle.meta_train.iloc[row_id]["stateAbbr"]),
                "severity_tier": int(bundle.meta_train.iloc[row_id]["severity_tier"]),
            })
    fold_frame = pd.DataFrame(fold_rows)
    if (run_dir / "folds.csv").exists():
        pd.testing.assert_frame_equal(pd.read_csv(run_dir / "folds.csv", dtype={"fipsCode": str}),
                                      fold_frame, check_dtype=False)
    else:
        _atomic_dataframe(run_dir / "folds.csv", fold_frame)
    frozen_json("environment.json", {
        "python": sys.version,
        "versions": preflight["versions"],
        "device": _resolve_device(args.device),
        "torch_threads": 1,
        "runtime": preflight["runtime_environment"],
    })
    frozen_json("inputs_manifest.json", _input_manifest(preflight))
    frozen_json("feature_schema.json", {
        "experiment_version": EXPERIMENT_VERSION,
        "feature_version": FEATURE_VERSION,
        "ordered_phase1_features": list(bundle.feature_names),
        "graph_schema": preflight["graph_feature_names"],
        "graph_schemas": {h: graph_feature_names(bundle.feature_names, preflight["terrain"].columns, h) for h in HORIZONS},
        "graph_input_dim": GRAPH_INPUT_DIM,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
    })
    graph = {name: preflight[name] for name in ("coords", "edge_index", "edge_attr", "all_fips")}
    if (run_dir / "graph.npz").exists():
        with np.load(run_dir / "graph.npz", allow_pickle=True) as old:
            if any(not np.array_equal(old[name], value) for name, value in graph.items()):
                raise ValueError("immutable graph artifact mismatch")
    else:
        np.savez_compressed(run_dir / "graph.npz", **graph)


def _base_manifest(ctx, run_dir: Path, args, *, final=False):
    from base_model import CV_BASE_MANIFEST, FINAL_BASE_MANIFEST

    if not final and (run_dir / "CV_COMPLETE").exists():
        raise FileExistsError("the CV base manifest is frozen")
    rows = []
    for fit in ctx.base_store.fits.values():
        safe = f"{fit.component}_{fit.horizon.replace('osi_target_', '')}_S{'-'.join(map(str, fit.scope))}.txt"
        valid_mask = (
            np.isin(ctx.row_fold, np.asarray(fit.scope, dtype=int))
            & (ctx.bundle.meta_train["hour_idx"].to_numpy(dtype=int) + HORIZON_HOURS[fit.horizon] <= PRED_END - 1)
        )
        rows.append({
            "protocol": PROTOCOL, "base_mode": args.base_mode,
            "horizon": fit.horizon, "component": fit.component,
            "allowed_folds": json.dumps(fit.scope),
            "valid_rows": int(valid_mask.sum()),
            "best_iterations": json.dumps(fit.best_iterations),
            "final_rounds": fit.final_rounds, "actual_rounds": fit.actual_rounds, "seed": fit.seed,
            "run_identity": ctx.base_store.run_identity,
            "model_id": fit.model_id, "model_path": str(Path("models/base") / safe),
            "model_sha256": _sha256(run_dir / "models" / "base" / safe),
            "feature_package_hash": ctx.base_store.feature_package_hash,
            "feature_side_hash": ctx.base_store.feature_side_hash,
            "feature_schema_hash": hashlib.sha256(
                json.dumps(list(ctx.bundle.feature_names), separators=(",", ":")).encode()
            ).hexdigest(),
            "probe_records": json.dumps(fit.probe_records, default=str),
        })
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated("model_id").any():
        raise ValueError("cannot publish an empty or duplicate base-fit manifest")
    name = FINAL_BASE_MANIFEST if final else CV_BASE_MANIFEST
    if final:
        # All CV entries must survive byte-for-byte at the value level; their
        # probe/round provenance cannot disappear during reload and re-export.
        cv = pd.read_parquet(run_dir / CV_BASE_MANIFEST, engine="pyarrow")
        current = frame.set_index("model_id")
        previous = cv.set_index("model_id")
        if not previous.index.isin(current.index).all():
            raise ValueError("final base manifest is missing CV models")
        pd.testing.assert_frame_equal(
            current.loc[previous.index, previous.columns], previous,
            check_dtype=False, check_exact=True,
        )
        expected_count = 26 * len(HORIZONS) * (1 if args.base_mode == "direct" else 4)
        if len(frame) != expected_count:
            raise ValueError(f"final base manifest has {len(frame)} models, expected {expected_count}")
    _atomic_dataframe(run_dir / name, frame)


def _append_inner_records(rows, selection, outer_fold, horizon, mode, bundle, row_fold, supervision):
    labels = supervision.scoped(selection.scope, horizon, "osi")
    for inner_fold in selection.scope:
        allowed = tuple(fold for fold in selection.scope if fold != inner_fold)
        mask = (row_fold == inner_fold) & (bundle.meta_train["hour_idx"].to_numpy(dtype=int) + HORIZON_HOURS[horizon] <= PRED_END - 1)
        for row_id in np.flatnonzero(mask):
            rows.append({
                "protocol": PROTOCOL, "base_mode": mode,
                "outer_fold": outer_fold, "inner_fold": inner_fold,
                "allowed_folds": json.dumps(allowed),
                "fipsCode": str(bundle.meta_train.iloc[row_id]["fips_str"]),
                "timestamp_et": str(bundle.meta_train.iloc[row_id]["timestamp_et"]),
                "hour_idx": int(bundle.meta_train.iloc[row_id]["hour_idx"]),
                "horizon": horizon,
                "is_scoreable": True,
                "y_true": float(labels.take(np.asarray([row_id]))[0]),
                "base_prediction": float(selection.inner_base[row_id]),
                "correction_osi": float(selection.inner_correction[row_id]),
                "base_source_id": str(selection.inner_base_source_ids[row_id]),
                "stack_id": str(selection.inner_source_ids[row_id]),
                "selection_id": selection.selection_id,
            })


def _serialise_alpha_candidate(candidate: dict) -> dict:
    result = dict(candidate)
    if "scope" in result:
        result["allowed_folds"] = json.dumps(tuple(int(value) for value in result.pop("scope")))
    return result


def _county_metrics_and_bootstrap(outer: pd.DataFrame):
    """Recompute county SSE/RMSE and paired county bootstrap from outer OOF."""

    scoreable = outer[outer["is_scoreable"].astype(bool)].copy()
    if scoreable.empty:
        raise ValueError("county metrics have no scoreable outer rows")
    county_rows = []
    for (horizon, fips), group in scoreable.groupby(["horizon", "fipsCode"], sort=True):
        if group["outer_fold"].nunique() != 1:
            raise ValueError(f"county belongs to multiple outer folds: {horizon}/{fips}")
        y = group["y_true"].to_numpy(dtype=float)
        base = post_process_osi(group["base_prediction"].to_numpy(dtype=float))
        gat = group["prediction"].to_numpy(dtype=float)
        if not (np.isfinite(y).all() and np.isfinite(base).all() and np.isfinite(gat).all()):
            raise FloatingPointError(f"non-finite county metric inputs for {horizon}/{fips}")
        base_error = base - y
        gat_error = gat - y
        county_rows.append({
            "horizon": horizon,
            "fipsCode": str(fips),
            "outer_fold": int(group["outer_fold"].iloc[0]),
            "n": int(len(group)),
            "base_sse": float(np.sum(base_error * base_error, dtype=np.float64)),
            "gat_sse": float(np.sum(gat_error * gat_error, dtype=np.float64)),
            "base_rmse": float(np.sqrt(np.mean(base_error * base_error))),
            "gat_rmse": float(np.sqrt(np.mean(gat_error * gat_error))),
            "gat_minus_base_sse": float(np.sum(gat_error * gat_error, dtype=np.float64) - np.sum(base_error * base_error, dtype=np.float64)),
            "gat_better": bool(np.sum(gat_error * gat_error) < np.sum(base_error * base_error)),
        })
    county = pd.DataFrame(county_rows)
    if county.duplicated(["horizon", "fipsCode"]).any():
        raise ValueError("county metrics contain duplicate horizon/county keys")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows = []
    for horizon, group in county.groupby("horizon", sort=True):
        sse_base = group["base_sse"].to_numpy(dtype=np.float64)
        sse_gat = group["gat_sse"].to_numpy(dtype=np.float64)
        counts = group["n"].to_numpy(dtype=np.int64)
        n_counties = len(group)
        draws = rng.integers(0, n_counties, size=(BOOTSTRAP_REPLICATES, n_counties))
        sampled_base = sse_base[draws].sum(axis=1, dtype=np.float64)
        sampled_gat = sse_gat[draws].sum(axis=1, dtype=np.float64)
        sampled_n = counts[draws].sum(axis=1, dtype=np.float64)
        base_rmse = np.sqrt(sampled_base / sampled_n)
        gat_rmse = np.sqrt(sampled_gat / sampled_n)
        delta = gat_rmse - base_rmse
        bootstrap_rows.append({
            "horizon": horizon,
            "replicates": int(BOOTSTRAP_REPLICATES),
            "seed": int(BOOTSTRAP_SEED),
            "n_counties": int(n_counties),
            "base_rmse": float(np.sqrt(group["base_sse"].sum() / group["n"].sum())),
            "gat_rmse": float(np.sqrt(group["gat_sse"].sum() / group["n"].sum())),
            "delta_rmse_gat_minus_base": float(np.sqrt(group["gat_sse"].sum() / group["n"].sum()) - np.sqrt(group["base_sse"].sum() / group["n"].sum())),
            "delta_ci_low": float(np.quantile(delta, 0.025)),
            "delta_ci_high": float(np.quantile(delta, 0.975)),
            "gat_better_probability": float(np.mean(delta < 0.0)),
        })
    return county, pd.DataFrame(bootstrap_rows)


def _append_outer_records(rows, stack, selection, outer_fold, horizon, mode, ctx):
    bundle = ctx.bundle
    row_fold = ctx.row_fold
    n_rows = len(bundle.X_train)
    base_rows = stack.train_base_rows(n_rows)
    correction_rows = stack.train_correction_rows(n_rows)
    labels = ctx.base_store.supervision.scoped((outer_fold,), horizon, "osi")
    for row_id in np.flatnonzero(row_fold == outer_fold):
        hour = int(bundle.meta_train.iloc[row_id]["hour_idx"])
        scoreable = hour + HORIZON_HOURS[horizon] <= PRED_END - 1
        before = float(base_rows[row_id] + selection.alpha * correction_rows[row_id]) if scoreable else np.nan
        prediction = float(post_process_osi(before)) if scoreable else np.nan
        record = {
            "run_id": ctx.run_id,
            "protocol": PROTOCOL, "base_mode": mode,
            "fipsCode": str(bundle.meta_train.iloc[row_id]["fips_str"]),
            "timestamp_et": str(bundle.meta_train.iloc[row_id]["timestamp_et"]),
            "hour_idx": hour,
            "target_timestamp": str(pd.to_datetime(bundle.meta_train.iloc[row_id]["timestamp_et"]) + pd.Timedelta(hours=HORIZON_HOURS[horizon])),
            "horizon": horizon, "outer_fold": outer_fold,
            "is_scoreable": scoreable,
            "y_true": float(labels.take(np.asarray([row_id]))[0]) if scoreable else np.nan,
            "base_prediction": float(base_rows[row_id]),
            "correction_osi": float(correction_rows[row_id]),
            "alpha": selection.alpha,
            "prediction_before_postprocess": before,
            "prediction": prediction,
            "base_source_id": str(stack.inputs.train_source_id[row_id]),
            "stack_id": stack.stack_id,
            "inner_selection_id": selection.selection_id,
        }
        if mode == "component_v158":
            for component, values in stack.inputs.base_parts["train_parts"].items():
                record[f"base_{component}"] = float(values[row_id])
                record[f"base_source_{component}"] = str(
                    stack.inputs.base_parts["train_source_by_component"][component][row_id]
                )
        rows.append(record)


def evaluate_outer(ctx, outer_fold: int, horizon: str, mode: str):
    """Evaluate one outer county fold with its own inner-selected alpha.

    This is the only public scoring path for an outer fold: training receives
    A=F-{outer_fold}, while the official labels of the held-out fold are read
    here solely for the final score and row-level evidence.
    """

    outer_fold = int(outer_fold)
    if outer_fold < 0 or outer_fold >= N_FOLDS:
        raise ValueError(f"invalid outer fold: {outer_fold}")
    allowed = tuple(fold for fold in range(N_FOLDS) if fold != outer_fold)
    selection = ctx.select_alpha(allowed, horizon, mode)
    stack = ctx.fit_stack(allowed, horizon, mode)
    n_rows = len(ctx.bundle.X_train)
    base_rows = stack.train_base_rows(n_rows)
    correction_rows = stack.train_correction_rows(n_rows)
    hours = ctx.bundle.meta_train["hour_idx"].to_numpy(dtype=int)
    expected = (ctx.row_fold == outer_fold) & (hours + HORIZON_HOURS[horizon] <= PRED_END - 1)
    if not expected.any():
        raise ValueError("outer fold has no expected scoring rows")
    labels = ctx.base_store.supervision.scoped((outer_fold,), horizon, "osi")
    alpha = float(selection.alpha)
    if not np.isfinite(alpha):
        raise FloatingPointError("outer alpha is non-finite")
    expected_rows = np.flatnonzero(expected)
    y_expected = labels.take(expected_rows)
    if not (np.isfinite(y_expected).all() and np.isfinite(base_rows[expected]).all()
            and np.isfinite(correction_rows[expected]).all()):
        raise FloatingPointError("outer score inputs are non-finite")
    base_prediction = post_process_osi(base_rows[expected])
    gat_prediction = post_process_osi(base_rows[expected] + alpha * correction_rows[expected])
    if not np.isfinite(gat_prediction).all():
        raise FloatingPointError("outer post-processed prediction is non-finite")
    return {
        "allowed_folds": allowed,
        "selection": selection,
        "stack": stack,
        "expected": expected,
        "base_metric": pooled_metrics(y_expected, base_prediction),
        "gat_metric": pooled_metrics(y_expected, gat_prediction),
    }


def _run_cv(ctx, args, run_dir):
    if (run_dir / "CV_COMPLETE").exists():
        raise FileExistsError("CV evidence already exists and is immutable")
    ctx.run_id = args.run_id
    outer_rows, inner_rows, alpha_rows, fold_rows, summary_rows = [], [], [], [], []
    row_fold = ctx.row_fold
    all_folds = tuple(range(N_FOLDS))
    for horizon in HORIZONS:
        for outer_fold in all_folds:
            evaluated = evaluate_outer(ctx, outer_fold, horizon, args.base_mode)
            allowed = evaluated["allowed_folds"]
            selection = evaluated["selection"]
            stack = evaluated["stack"]
            _append_inner_records(
                inner_rows, selection, outer_fold, horizon, args.base_mode,
                ctx.bundle, row_fold, ctx.base_store.supervision,
            )
            _append_outer_records(outer_rows, stack, selection, outer_fold, horizon, args.base_mode, ctx)
            fold_rows.extend([
                {"horizon": horizon, "outer_fold": outer_fold, "model": "base", "alpha": 0.0, **evaluated["base_metric"]},
                {"horizon": horizon, "outer_fold": outer_fold, "model": "gat", "alpha": selection.alpha, **evaluated["gat_metric"]},
            ])
            alpha_rows.extend({
                "protocol": PROTOCOL, "base_mode": args.base_mode,
                "selection_type": "outer_train_inner_cv", "outer_fold": outer_fold,
                "inner_fold": "pooled", "horizon": horizon,
                "selection_id": selection.selection_id,
                **_serialise_alpha_candidate(candidate),
            } for candidate in selection.candidates)
    outer = pd.DataFrame(outer_rows)
    inner = pd.DataFrame(inner_rows)
    if len(outer) != len(ctx.bundle.X_train) * len(HORIZONS):
        raise ValueError("outer OOF row count is not 34416*4")
    if outer.duplicated(["fipsCode", "timestamp_et", "horizon"]).any():
        raise ValueError("outer OOF county-time-horizon key is not unique")
    for horizon in HORIZONS:
        subset = outer[outer["horizon"] == horizon]
        expected = subset["hour_idx"].to_numpy(dtype=int) + HORIZON_HOURS[horizon] <= PRED_END - 1
        if not np.array_equal(subset["is_scoreable"].to_numpy(dtype=bool), expected):
            raise ValueError(f"outer OOF scoreable mask mismatch for {horizon}")
        if int(expected.sum()) != TRAIN_SCOREABLE_ROWS[horizon]:
            raise ValueError(f"outer OOF valid-row count mismatch for {horizon}")
        if not np.isfinite(subset["base_prediction"].to_numpy(dtype=float)).all():
            raise FloatingPointError(f"outer base prediction is non-finite for {horizon}")
        if not np.isfinite(subset["correction_osi"].to_numpy(dtype=float)).all():
            raise FloatingPointError(f"outer correction is non-finite for {horizon}")
        for column in ("y_true", "prediction"):
            values = subset[column].to_numpy(dtype=float)
            if not np.isfinite(values[expected]).all():
                raise FloatingPointError(f"outer {column} is non-finite on expected rows for {horizon}")
            if np.isfinite(values[~expected]).any():
                raise ValueError(f"outer {column} is finite in an unscoreable tail for {horizon}")
        if not np.isfinite(subset["alpha"].to_numpy(dtype=float)[expected]).all():
            raise FloatingPointError(f"outer alpha is non-finite on expected rows for {horizon}")
        summary_rows.extend([
            {"horizon": horizon, "model": "base", **pooled_metrics(subset.loc[expected, "y_true"].to_numpy(), post_process_osi(subset.loc[expected, "base_prediction"].to_numpy()))},
            {"horizon": horizon, "model": "gat", **pooled_metrics(subset.loc[expected, "y_true"].to_numpy(), subset.loc[expected, "prediction"].to_numpy())},
        ])
    county_metrics, county_bootstrap = _county_metrics_and_bootstrap(outer)
    _atomic_dataframe(run_dir / "outer_oof.parquet", outer)
    _atomic_dataframe(run_dir / "inner_oof.parquet", inner)
    _atomic_dataframe(run_dir / "alpha_selection.parquet", pd.DataFrame(alpha_rows))
    _atomic_dataframe(run_dir / "fold_metrics.csv", pd.DataFrame(fold_rows))
    _atomic_dataframe(run_dir / "cv_summary.csv", pd.DataFrame(summary_rows))
    _atomic_dataframe(run_dir / "county_metrics.csv", county_metrics)
    _atomic_dataframe(run_dir / "county_bootstrap.csv", county_bootstrap)
    _base_manifest(ctx, run_dir, args)
    cv_artifact_names = (
        "outer_oof.parquet", "inner_oof.parquet", "alpha_selection.parquet",
        "fold_metrics.csv", "cv_summary.csv", "county_metrics.csv",
        "county_bootstrap.csv", "base_fit_manifest.parquet",
    )
    cv_artifacts = {name: _sha256(run_dir / name) for name in cv_artifact_names}
    cv_hash = hashlib.sha256(json.dumps(cv_artifacts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _atomic_json(run_dir / "cv_manifest.json", {
        "run_id": args.run_id,
        "protocol": PROTOCOL,
        "experiment_version": EXPERIMENT_VERSION,
        "run_manifest_sha256": _sha256(run_dir / "run_manifest.json"),
        "artifact_hashes": cv_artifacts,
        "cv_hash": cv_hash,
    })
    (run_dir / "CV_COMPLETE").write_text(cv_hash, encoding="utf-8")


def _final_graph_records(ctx, stacks):
    rows = []
    train_fips = set(ctx.bundle.meta_train["fips_str"])
    for horizon, stack in stacks.items():
        for t in range(144):
            hour = PRED_START + t
            for node, fips in enumerate(stack.inputs.all_fips):
                train_row = int(stack.inputs.train_rows[t, node])
                test_row = int(stack.inputs.test_rows[t, node])
                if train_row >= 0:
                    source = stack.inputs.train_source_id[train_row]
                    fold = int(ctx.row_fold[train_row])
                    split = "train"
                    base = stack.inputs.base_grid[t, node]
                else:
                    source = stack.inputs.test_source_id
                    fold = -1
                    split = "test"
                    base = stack.inputs.base_grid[t, node]
                rows.append({
                    "fipsCode": fips, "hour_idx": hour, "horizon": horizon,
                    "base_prediction": float(base), "base_source_id": str(source),
                    "split": split, "county_fold": fold,
                })
    return pd.DataFrame(rows)


def _run_final(ctx, args, run_dir):
    cv_marker = run_dir / "CV_COMPLETE"
    if not cv_marker.exists():
        raise ValueError("final stage requires an immutable CV_COMPLETE marker")
    saved_cv = json.loads((run_dir / "cv_manifest.json").read_text(encoding="utf-8"))
    if saved_cv.get("protocol") != PROTOCOL or saved_cv.get("run_id") != args.run_id:
        raise ValueError("CV manifest identity mismatch")
    if saved_cv.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("CV experiment version mismatch")
    if (run_dir / "CV_COMPLETE").read_text(encoding="utf-8") != saved_cv.get("cv_hash"):
        raise ValueError("CV_COMPLETE marker does not match cv_manifest")
    if saved_cv.get("run_manifest_sha256") != _sha256(run_dir / "run_manifest.json"):
        raise ValueError("run manifest changed after CV completion")
    for name, digest in saved_cv.get("artifact_hashes", {}).items():
        if _sha256(run_dir / name) != digest:
            raise ValueError(f"frozen CV artifact changed: {name}")
    test_rows, graph_stacks, alpha_rows = [], {}, []
    for horizon in HORIZONS:
        selection = ctx.select_alpha(tuple(range(N_FOLDS)), horizon, args.base_mode)
        stack = ctx.fit_stack(tuple(range(N_FOLDS)), horizon, args.base_mode)
        graph_stacks[horizon] = stack
        alpha_rows.extend({
            "protocol": PROTOCOL, "base_mode": args.base_mode,
            "selection_type": "full_train_cv_for_final", "outer_fold": "all",
            "inner_fold": "pooled", "horizon": horizon,
            "selection_id": selection.selection_id,
            **_serialise_alpha_candidate(candidate),
        } for candidate in selection.candidates)
        for row_id, row in ctx.bundle.meta_test.iterrows():
            t = int(row["hour_idx"]) - PRED_START
            node = stack.inputs.all_fips.index(row["fips_str"])
            correction = float(stack.correction_grid[t, node])
            base = float(stack.inputs.base_grid[t, node])
            scoreable = int(row["hour_idx"]) + HORIZON_HOURS[horizon] <= PRED_END - 1
            before = base + selection.alpha * correction
            prediction = float(post_process_osi(before)) if scoreable else np.nan
            record = {
                "run_id": args.run_id, "protocol": PROTOCOL, "base_mode": args.base_mode,
                "fipsCode": str(row["fips_str"]), "timestamp_et": str(row["timestamp_et"]),
                "hour_idx": int(row["hour_idx"]), "horizon": horizon,
                "is_scoreable": scoreable, "base_prediction": base,
                "correction_osi": correction, "alpha": selection.alpha,
                "prediction_before_postprocess": before if scoreable else np.nan,
                "prediction": prediction, "base_source_id": stack.inputs.test_source_id,
                "stack_id": stack.stack_id, "selection_id": selection.selection_id,
            }
            if args.base_mode == "component_v158":
                for component, values in stack.inputs.base_parts["test_parts"].items():
                    record[f"base_{component}"] = float(values[row_id])
                    record[f"base_source_{component}"] = str(
                        stack.inputs.base_parts["test_source_by_component"][component]
                    )
            test_rows.append(record)
    _base_manifest(ctx, run_dir, args, final=True)
    _atomic_dataframe(run_dir / "alpha_selection_final.parquet", pd.DataFrame(alpha_rows + [
         {"protocol": PROTOCOL, "base_mode": args.base_mode, "selection_type": "final_selected",
          "experiment_version": EXPERIMENT_VERSION,
          "horizon": h, "alpha": ctx.alpha_cache[(args.base_mode, h, tuple(range(N_FOLDS)))].alpha}
        for h in HORIZONS
    ]))
    _atomic_dataframe(run_dir / "final_graph_base.parquet", _final_graph_records(ctx, graph_stacks))
    test_frame = pd.DataFrame(test_rows)
    _atomic_dataframe(run_dir / "test_predictions.parquet", test_frame)
    submission = pd.read_csv(SUBMISSION_FILE)
    required_columns = list(submission.columns)
    if len(submission) != 9072:
        raise ValueError("submission template does not have 9072 rows")
    template_keys = list(zip(submission["fipsCode"].astype(str).str.zfill(5), submission["timestamp_et"].astype(str)))
    test_keys = list(zip(ctx.bundle.meta_test["fips_str"].astype(str), ctx.bundle.meta_test["timestamp_et"].astype(str)))
    if template_keys != test_keys:
        raise ValueError("submission template key order differs from test metadata")
    for horizon in HORIZONS:
        values = test_frame.loc[test_frame["horizon"] == horizon, "prediction"].to_numpy(dtype=float)
        if len(values) != len(ctx.bundle.meta_test):
            raise ValueError("test prediction coverage is incomplete")
        submission[horizon] = values
        expected = OFFICIAL_SCOREABLE_ROWS[horizon]
        finite = np.isfinite(values)
        if int(finite.sum()) != expected or not np.isfinite(values[finite]).all() or not ((values[finite] >= 0) & (values[finite] <= 0.65)).all():
            raise ValueError(f"submission hard validation failed for {horizon}")
    if list(submission.columns) != required_columns:
        raise ValueError("submission columns changed")
    # Counts alone cannot detect shifted NaN positions or row/value swaps.
    from artifact_checks import References, check_submission, time_rows
    references = References(
        time_rows(ctx.bundle.meta_train, "submission train metadata"),
        time_rows(ctx.bundle.meta_test, "submission test metadata"),
        pd.DataFrame(), ctx.row_fold,
    )
    check_submission(submission, pd.read_csv(SUBMISSION_FILE), test_frame, references)
    _atomic_dataframe(run_dir / "submission_phase2_dem_gat.csv", submission)
    _atomic_json(run_dir / "submission_audit.json", {
        "rows": len(submission),
        "columns_match_template": list(submission.columns) == required_columns,
        "finite_counts": {h: int(np.isfinite(submission[h].to_numpy(dtype=float)).sum()) for h in HORIZONS},
        "expected_counts": OFFICIAL_SCOREABLE_ROWS,
        "hard_checks_passed": True,
    })
    _atomic_json(run_dir / "verification.json", {
        "protocol": PROTOCOL,
        "cv_complete": True,
        "submission_hard_checks": True,
        "independent_reload": "not_run_in_this_process",
        "note": "Run verify_artifacts.py in a new process before COMPLETE.",
    })
    cv_summary = pd.read_csv(run_dir / "cv_summary.csv")
    county_bootstrap = pd.read_csv(run_dir / "county_bootstrap.csv")
    report = [
        f"# {PROTOCOL}", "", f"- run_id: `{args.run_id}`",
        f"- base_mode: `{args.base_mode}`", f"- experiment_version: `{EXPERIMENT_VERSION}`",
        f"- feature_version: `{FEATURE_VERSION}`",
        f"- graph_input_dim: `{GRAPH_INPUT_DIM}`",
        f"- graph_schema_version: `{GRAPH_SCHEMA_VERSION}`", "- CV evidence: `CV_COMPLETE`", "",
        "## CV summary", "", "```text", cv_summary.to_string(index=False), "```", "",
        "## County-clustered paired bootstrap", "", "```text",
        county_bootstrap.to_string(index=False), "```", "",
        "County metrics and bootstrap are recomputed from `outer_oof.parquet`; "
        "the bootstrap preserves all scoreable hours within sampled counties. "
        "Spatial dependence is therefore an interval limitation, not an independent-row assumption.",
        "Final alpha selection is stored in `alpha_selection_final.parquet`; its internal CV score is not an independent outer score.",
        "This report is generated from the run manifest and saved row-level artifacts.",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (run_dir / "FINAL_READY").write_text("final artifacts written; independent verification pending", encoding="utf-8")


def stages_to_run(stage, cv_complete, final_ready):
    if final_ready and not cv_complete:
        raise ValueError("FINAL_READY without CV_COMPLETE is invalid")
    if stage == "cv":
        return [] if cv_complete else ["cv"]
    if stage == "final":
        if not cv_complete:
            raise ValueError("final requires CV_COMPLETE")
        return [] if final_ready else ["final"]
    if stage == "all":
        return ([] if cv_complete else ["cv"]) + ([] if final_ready else ["final"])
    raise ValueError(f"invalid training stage: {stage}")


def main(argv=None):
    args = parse_args(argv)
    identity.configure_determinism()
    if args.stage == "preflight":
        _preflight(args)
        print("PREFLIGHT_OK")
        return 0
    preflight = _preflight(args)
    device = _resolve_device(args.device)
    _seed_process(args.seed, device)
    run_dir, manifest = _create_or_resume_run(args, preflight)
    _write_static_artifacts(run_dir, args, preflight)
    if (run_dir / "CV_COMPLETE").exists():
        from artifact_checks import check_frozen_cv
        # The stage recorded at creation stays immutable across attempts.
        saved_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        check_frozen_cv(run_dir, saved_manifest)
    stages = stages_to_run(args.stage, (run_dir / "CV_COMPLETE").exists(), (run_dir / "FINAL_READY").exists())
    if stages:
        ctx = _make_context(args, preflight, run_dir, resume=args.resume)
        for stage in stages:
            (_run_cv if stage == "cv" else _run_final)(ctx, args, run_dir)
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
