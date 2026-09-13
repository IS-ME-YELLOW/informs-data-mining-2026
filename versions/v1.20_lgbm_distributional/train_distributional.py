"""Train resumable county-fold LightGBM distributional component models."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_contract import component_target_name, load_inputs, sha256_file
from distributional_config import (
    COMPONENTS, HORIZONS, MANIFESTS_DIR, MODELS_DIR, PREDICTIONS_DIR,
    PRIMARY_SPLIT, candidate_names, ensure_output_dirs, horizon_suffix,
    resolved_spec, stable_fingerprint,
)
from folds import load_fold_assignments


def job_id(split: str, component: str, horizon: str, candidate: str, fold: int) -> str:
    return f"{split}__{component}__{horizon_suffix(horizon)}__{candidate}__fold{fold}"


def artifact_paths(split: str, component: str, horizon: str, candidate: str, fold: int) -> dict[str, Path]:
    stem = job_id(split, component, horizon, candidate, fold)
    return {
        "model": MODELS_DIR / split / f"{stem}.txt",
        "sidecar": MODELS_DIR / split / f"{stem}.json",
        "oof": PREDICTIONS_DIR / split / f"{stem}.parquet",
    }


def build_job_payload(
    inputs, split: str, fold_file: Path, row_folds: np.ndarray, component: str,
    horizon: str, candidate: str, fold: int, num_threads: int, quick: bool,
) -> dict:
    spec = resolved_spec(candidate, num_threads, quick)
    target = component_target_name(component, horizon)
    valid_mask = inputs.component_targets[target].notna().to_numpy()
    train_idx = np.flatnonzero((row_folds != fold) & valid_mask)
    valid_idx = np.flatnonzero((row_folds == fold) & valid_mask)
    payload = {
        "schema_version": 1,
        "split": split,
        "fold": fold,
        "fold_file": str(fold_file),
        "fold_sha256": sha256_file(fold_file),
        "component": component,
        "horizon": horizon,
        "target": target,
        "candidate": candidate,
        "quick": bool(quick),
        "params": spec["params"],
        "rounds": spec["rounds"],
        "early_stopping": spec["early_stopping"],
        "feature_names": list(inputs.data.X_train.columns),
        "input_hashes": inputs.hashes,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
    }
    payload["parameter_fingerprint"] = stable_fingerprint(payload)
    return payload


def _validate_resume(paths: dict[str, Path], expected: dict) -> bool:
    present = {key: path.exists() for key, path in paths.items()}
    if not any(present.values()):
        return False
    if not all(present.values()):
        raise FileExistsError(f"Partial artifacts exist and will not be overwritten: {paths}")
    saved = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    if saved.get("parameter_fingerprint") != expected["parameter_fingerprint"]:
        raise FileExistsError(f"Existing artifacts have a different parameter fingerprint: {paths['sidecar']}")
    if saved.get("model_sha256") != sha256_file(paths["model"]):
        raise ValueError(f"Model hash mismatch: {paths['model']}")
    if saved.get("oof_sha256") != sha256_file(paths["oof"]):
        raise ValueError(f"OOF shard hash mismatch: {paths['oof']}")
    booster = lgb.Booster(model_file=str(paths["model"]))
    if int(booster.num_trees()) != int(saved["best_iteration"]):
        raise ValueError(f"Model tree count differs from sidecar: {paths['model']}")
    return True


def _atomic_save_model(booster: lgb.Booster, path: Path, iteration: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", dir=path.parent) as handle:
        temporary = Path(handle.name)
    try:
        booster.save_model(str(temporary), num_iteration=iteration)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def train_one(inputs, split: str, row_folds: np.ndarray, fold_file: Path, component: str,
              horizon: str, candidate: str, fold: int, num_threads: int, quick: bool,
              resume: bool) -> dict:
    paths = artifact_paths(split, component, horizon, candidate, fold)
    payload = build_job_payload(inputs, split, fold_file, row_folds, component, horizon, candidate, fold, num_threads, quick)
    if resume and _validate_resume(paths, payload):
        print(f"skip verified {job_id(split, component, horizon, candidate, fold)}", flush=True)
        return json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    if not resume and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Artifacts already exist; use --resume after verifying compatibility: {paths}")

    target = payload["target"]
    y = inputs.component_targets[target]
    valid_target = y.notna().to_numpy()
    train_idx = np.flatnonzero((row_folds != fold) & valid_target)
    valid_idx = np.flatnonzero((row_folds == fold) & valid_target)
    dtrain = lgb.Dataset(inputs.data.X_train.iloc[train_idx], label=y.iloc[train_idx], feature_name=list(inputs.data.X_train.columns))
    dvalid = lgb.Dataset(inputs.data.X_train.iloc[valid_idx], label=y.iloc[valid_idx], reference=dtrain)
    callbacks = [lgb.log_evaluation(0)]
    if payload["early_stopping"] is not None:
        callbacks.append(lgb.early_stopping(int(payload["early_stopping"]), verbose=False))
    booster = lgb.train(
        payload["params"], dtrain, num_boost_round=int(payload["rounds"]),
        valid_sets=[dvalid], callbacks=callbacks,
    )
    best_iteration = int(booster.best_iteration or payload["rounds"])
    predictions = booster.predict(inputs.data.X_train.iloc[valid_idx], num_iteration=best_iteration)
    shard = inputs.data.meta_train.iloc[valid_idx][["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    shard["row_index"] = valid_idx
    shard["split"] = split
    shard["fold"] = fold
    shard["component"] = component
    shard["horizon"] = horizon
    shard["candidate"] = candidate
    shard["target"] = y.iloc[valid_idx].to_numpy(float)
    shard["prediction_raw"] = predictions
    _atomic_save_model(booster, paths["model"], best_iteration)
    _atomic_parquet(shard, paths["oof"])
    payload.update({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_iteration": best_iteration,
        "model_path": str(paths["model"]),
        "oof_path": str(paths["oof"]),
        "model_sha256": sha256_file(paths["model"]),
        "oof_sha256": sha256_file(paths["oof"]),
    })
    _atomic_json(payload, paths["sidecar"])
    print(f"trained {job_id(split, component, horizon, candidate, fold)} best_iteration={best_iteration}", flush=True)
    return payload


def selected_jobs(primary_decision: Path) -> dict[str, set[tuple[str, str]]]:
    decision = json.loads(primary_decision.read_text(encoding="utf-8"))
    result: dict[str, set[tuple[str, str]]] = {}
    for horizon, record in decision["horizons"].items():
        if not record.get("promoted", False):
            continue
        pairs = set()
        if record.get("p_candidate"):
            pairs.add(("P_t", record["p_candidate"]))
        if record.get("d_candidate"):
            pairs.add(("D_t", record["d_candidate"]))
        result[horizon] = pairs
    return result


def run_training(split: str, num_threads: int, quick: bool, resume: bool,
                 components: Iterable[str], horizons: Iterable[str], candidates: Iterable[str]) -> None:
    ensure_output_dirs()
    inputs = load_inputs(require_v112=False)
    _, row_folds, fold_file = load_fold_assignments(split, inputs.data.meta_train)
    records = []
    for horizon in horizons:
        for component in components:
            for candidate in candidates:
                for fold in range(5):
                    records.append(train_one(inputs, split, row_folds, fold_file, component, horizon, candidate, fold, num_threads, quick, resume))
    manifest_path = MANIFESTS_DIR / f"training_manifest_{split}.json"
    _atomic_json({"split": split, "quick": quick, "jobs": records}, manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=PRIMARY_SPLIT)
    parser.add_argument("--num-threads", type=int, default=-1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--component", action="append", choices=COMPONENTS)
    parser.add_argument("--horizon", action="append", choices=HORIZONS)
    parser.add_argument("--candidate", action="append", choices=candidate_names())
    parser.add_argument("--primary-winner-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    components = tuple(args.component or COMPONENTS)
    horizons = tuple(args.horizon or HORIZONS)
    candidates = tuple(args.candidate or candidate_names())
    if args.primary_winner_only:
        decision_path = MANIFESTS_DIR / "primary_decision.json"
        if not decision_path.exists():
            raise FileNotFoundError("Primary decision is required for --primary-winner-only")
        jobs = selected_jobs(decision_path)
        if not jobs:
            print("Primary did not promote any horizon; repeat training is not run.")
            return
        if args.split == PRIMARY_SPLIT:
            raise ValueError("--primary-winner-only is intended for repeat splits")
        inputs = load_inputs(require_v112=False)
        _, row_folds, fold_file = load_fold_assignments(args.split, inputs.data.meta_train)
        for horizon, pairs in jobs.items():
            for component, candidate in sorted(pairs):
                for fold in range(5):
                    payload = build_job_payload(inputs, args.split, fold_file, row_folds, component, horizon, candidate, fold, args.num_threads, args.quick)
                    if args.dry_run:
                        print(json.dumps(payload, sort_keys=True))
                    else:
                        train_one(inputs, args.split, row_folds, fold_file, component, horizon, candidate, fold, args.num_threads, args.quick, args.resume)
        return
    if args.dry_run:
        inputs = load_inputs(require_v112=False)
        _, row_folds, fold_file = load_fold_assignments(args.split, inputs.data.meta_train)
        count = 0
        for horizon in horizons:
            for component in components:
                for candidate in candidates:
                    for fold in range(5):
                        build_job_payload(inputs, args.split, fold_file, row_folds, component, horizon, candidate, fold, args.num_threads, args.quick)
                        count += 1
        print(json.dumps({"split": args.split, "jobs": count, "quick": args.quick, "num_threads": args.num_threads}))
        return
    run_training(args.split, args.num_threads, args.quick, args.resume, components, horizons, candidates)


if __name__ == "__main__":
    main()
