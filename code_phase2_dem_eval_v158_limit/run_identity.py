"""Immutable experiment identity; never used to derive random seeds."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
from types import SimpleNamespace

import config as cfg

FORMAT_VERSION = 2
SOURCE_FILES = (
    "config.py", "data.py", "base_model.py", "stacking.py", "gat_model.py",
    "spatial.py", "main.py", "metrics.py", "artifact_checks.py",
    "verify_artifacts.py", "run_identity.py", "model_records.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def resolved_config(args):
    return {
        "seed": int(args.seed), "base_mode": args.base_mode,
        "lightgbm": cfg.make_lgbm_params(args.seed),
        "base_round_limit": cfg.LGBM_ROUNDS, "base_patience": cfg.LGBM_EARLY_STOPPING,
        "round_aggregation": "max(1,int(mean(best_iterations)))",
        "scope_seed_policy": "sha256(global_seed,stage,sorted_scope,horizon,component,probe)",
    "gat": {"input_dim": cfg.GRAPH_INPUT_DIM, "schema_version": cfg.GRAPH_SCHEMA_VERSION,
            "hidden": cfg.GAT_HIDDEN, "heads": cfg.GAT_HEADS, "dropout": cfg.GAT_DROPOUT,
                "lr": cfg.GAT_LR, "weight_decay": cfg.GAT_WEIGHT_DECAY, "optimizer": "AdamW",
                "epochs": args.epochs, "patience": args.patience, "loss": args.gat_loss,
                "time_stride": args.time_stride, "gradient_clip": 2.0,
                "checkpoint_metric": "post-step eval training MSE", "min_improvement": 1e-7},
        "graph_k": args.k, "graph_policy": "symmetric_knn_plus_shared_border",
        "preprocessing": {"std_ddof": 0, "std_floor": 1e-6, "residual_scale_floor": .003,
                          "scale_cells": "scope nodes and valid horizon times", "declared_nan_fill": 0},
        "postprocess": {"min": 0, "max": cfg.OSI_MAX, "zero_below": .001},
        "component_weights": cfg.COMPONENT_WEIGHTS, "component_clip": [0, 1],
        "horizons": cfg.HORIZON_HOURS, "alpha_grid": list(cfg.GAT_ALPHA_GRID),
        "alpha_rule": "pooled RMSE then smaller alpha", "threads": cfg.FIXED_THREADS,
        "bootstrap": {"seed": cfg.BOOTSTRAP_SEED, "replicates": cfg.BOOTSTRAP_REPLICATES},
    }


def configure_determinism():
    # Set before the first CUDA matrix multiplication, including diagnostics.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def environment(device):
    versions = {}
    for name in ("numpy", "pandas", "pyarrow", "fastparquet", "lightgbm", "torch", "shapely", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "implementation": platform.python_implementation(),
              "system": platform.system(), "machine": platform.machine(), "processor": platform.processor(),
              "versions": versions, "device": device, "threads": cfg.FIXED_THREADS,
              "deterministic_algorithms": True, "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if versions["torch"] is not None:
        import torch
        result["torch_cuda_build"] = torch.version.cuda
        result["cudnn"] = torch.backends.cudnn.version()
        if device == "cuda":
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            result["gpu"] = {"name": props.name, "capability": [props.major, props.minor]}
    return result


def build_identity(args, inputs, device, *, runtime=None, sources=None):
    # Input paths are already checked by the runner. Content digests, not paths
    # or this identity digest, control what may be cached; seeds stay separate.
    content = {key: value for key, value in inputs.items()
               if key.endswith("sha256") or key.endswith("hash") or key == "all_fips"}
    body = {"format_version": FORMAT_VERSION, "protocol": cfg.PROTOCOL,
            "inputs": content, "configuration": resolved_config(args),
            "sources": source_hashes() if sources is None else sources,
            "environment": environment(device) if runtime is None else runtime}
    return {"digest": digest(body), "definition": body}


def differences(expected, actual, prefix="identity"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                result.append(f"{prefix}.{key}: missing field")
            else:
                result.extend(differences(expected[key], actual[key], f"{prefix}.{key}"))
        return result
    return [] if expected == actual else [f"{prefix}: {expected!r} -> {actual!r}"]


def assert_identity(expected, actual):
    for value in (expected, actual):
        if not isinstance(value, dict) or value.get("digest") != digest(value.get("definition")):
            raise ValueError("missing/invalid immutable run identity; use a new run for old-format artifacts")
    changed = differences(expected["definition"], actual["definition"])
    if changed:
        raise ValueError("resume identity mismatch:\n" + "\n".join(changed[:20]))


def assert_current(manifest):
    configure_determinism()
    params = manifest["parameters"]
    args = SimpleNamespace(seed=manifest["seed"], base_mode=manifest["base_mode"],
                           epochs=params["epochs"], patience=params["patience"],
                           time_stride=params["time_stride"], gat_loss=params["gat_loss"], k=params["k"])
    assert_identity(manifest.get("identity"), build_identity(args, manifest["inputs"], params["device"]))
