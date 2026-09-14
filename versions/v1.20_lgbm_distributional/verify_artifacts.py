"""Verify v1.20 input, fold, model, prediction, and baseline contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_contract import (
    ALL_COMPONENTS, c1_component_predictions, compose_osi, load_inputs,
    post_process, sha256_file,
)
from distributional_config import (
    HORIZONS, MANIFESTS_DIR, MODELS_DIR, PREDICTIONS_DIR, PRIMARY_SPLIT,
    V112_OOF, V112_TEST, candidate_names, ensure_output_dirs,
)
from folds import REPEAT_SEEDS, load_fold_assignments, repeat_name, validate_fold_frame, repeat_path
from train_distributional import artifact_paths


def verify_inputs(inputs) -> dict:
    manifest_path = MANIFESTS_DIR / "input_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["input_hashes"] != inputs.hashes:
            raise ValueError("Current inputs differ from the recorded input manifest")
    return {"input_hashes": inputs.hashes, "v112_available": inputs.v112_oof is not None}


def verify_c1(inputs) -> dict:
    rows = []
    for label, frame in (("oof", inputs.v18_oof_components), ("test", inputs.v18_test_components)):
        for horizon in HORIZONS:
            recomposed = post_process(compose_osi(c1_component_predictions(frame, horizon)))
            if label == "oof":
                # v1.8 C1 is defined by same-horizon clipped P/N/D/R; compare to its saved OOF control.
                saved = pd.read_parquet(inputs.data.meta_train.attrs.get("unused", Path("."))) if False else None
                source = inputs.v18_oof_components
                valid = np.isfinite(recomposed)
            else:
                valid = np.isfinite(recomposed)
            if not valid.any():
                raise ValueError(f"No finite v1.8 C1 values for {label}/{horizon}")
            rows.append({"dataset": label, "horizon": horizon, "finite": int(valid.sum()), "min": float(np.nanmin(recomposed)), "max": float(np.nanmax(recomposed))})
    # Exact OOF consistency against v1.8's saved C1 columns when available.
    control_path = Path(__file__).resolve().parents[1] / "v1.8" / "oof_predictions.parquet"
    if control_path.exists():
        control = pd.read_parquet(control_path)
        for horizon in HORIZONS:
            expected = control[f"pred_C1_component_osi_{horizon}"].to_numpy(float)
            actual = post_process(compose_osi(c1_component_predictions(inputs.v18_oof_components, horizon)))
            if not np.allclose(actual, expected, equal_nan=True, rtol=0, atol=1e-12):
                raise ValueError(f"v1.8 C1 recomposition mismatch for {horizon}")
    return {"c1_checks": rows}


def verify_v112(inputs) -> dict:
    if inputs.v112_oof is None:
        return {"status": "missing", "oof_path": str(V112_OOF), "test_path": str(V112_TEST)}
    checks = []
    for horizon in HORIZONS:
        column = f"pred_tail_protected_{horizon}"
        values = inputs.v112_oof[column].to_numpy(float)
        target = inputs.data.y_train[horizon].to_numpy(float)
        if np.isnan(values[np.isfinite(target)]).any():
            raise ValueError(f"v1.12 baseline has NaN on evaluable rows: {horizon}")
        checks.append({"horizon": horizon, "finite": int(np.isfinite(values).sum())})
    return {"status": "aligned", "checks": checks, "oof_sha256": sha256_file(V112_OOF), "test_sha256": sha256_file(V112_TEST)}


def verify_folds(inputs) -> list[dict]:
    records = []
    for split in (PRIMARY_SPLIT, *(repeat_name(seed) for seed in REPEAT_SEEDS)):
        frame, row_folds, path = load_fold_assignments(split, inputs.data.meta_train)
        record = validate_fold_frame(frame, inputs.data.meta_train)
        for fold in range(5):
            train_counties = set(inputs.data.meta_train.loc[row_folds != fold, "fipsCode"])
            valid_counties = set(inputs.data.meta_train.loc[row_folds == fold, "fipsCode"])
            if train_counties & valid_counties:
                raise ValueError(f"County leakage for {split}/fold{fold}")
        records.append({"split": split, "path": str(path), "sha256": sha256_file(path), **record})
    return records


def verify_models(inputs, split: str, sample: int | None) -> dict:
    _, row_folds, fold_file = load_fold_assignments(split, inputs.data.meta_train)
    sidecars = sorted((MODELS_DIR / split).glob("*.json")) if (MODELS_DIR / split).exists() else []
    if sample is not None:
        sidecars = sidecars[:sample]
    checked = []
    for sidecar_path in sidecars:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        paths = artifact_paths(split, sidecar["component"], sidecar["horizon"], sidecar["candidate"], int(sidecar["fold"]))
        if sidecar["fold_sha256"] != sha256_file(fold_file):
            raise ValueError(f"Fold hash mismatch: {sidecar_path}")
        if sidecar["model_sha256"] != sha256_file(paths["model"]) or sidecar["oof_sha256"] != sha256_file(paths["oof"]):
            raise ValueError(f"Artifact hash mismatch: {sidecar_path}")
        fingerprint_payload = {key: value for key, value in sidecar.items() if key not in {"parameter_fingerprint", "created_at", "best_iteration", "model_path", "oof_path", "model_sha256", "oof_sha256"}}
        from distributional_config import stable_fingerprint
        if stable_fingerprint(fingerprint_payload) != sidecar["parameter_fingerprint"]:
            raise ValueError(f"Sidecar parameter fingerprint mismatch: {sidecar_path}")
        booster = lgb.Booster(model_file=str(paths["model"]))
        if int(booster.num_trees()) != int(sidecar["best_iteration"]):
            raise ValueError(f"Reloaded model tree count mismatch: {paths['model']}")
        shard = pd.read_parquet(paths["oof"])
        index = shard["row_index"].to_numpy(int)
        if not np.all(row_folds[index] == int(sidecar["fold"])):
            raise ValueError(f"OOF shard violates fold isolation: {paths['oof']}")
        prediction = booster.predict(inputs.data.X_train.iloc[index], num_iteration=int(sidecar["best_iteration"]))
        if not np.allclose(prediction, shard["prediction_raw"].to_numpy(float), rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Reloaded model predictions differ from OOF shard: {paths['model']}")
        checked.append(str(paths["model"]))
    return {"split": split, "available_sidecars": len(sorted((MODELS_DIR / split).glob("*.json"))) if (MODELS_DIR / split).exists() else 0, "checked_models": checked}


def verify_nested_predictions(inputs) -> dict:
    path = PREDICTIONS_DIR / "primary_all_candidates_nested_oof.parquet"
    if not path.exists():
        return {"status": "not_generated"}
    frame = pd.read_parquet(path)
    expected_columns = 2 * 48
    prediction_columns = [column for column in frame if column.startswith("pred_t24h__") or column.startswith("pred_t48h__")]
    if len(prediction_columns) != expected_columns:
        raise ValueError(f"Expected {expected_columns} nested prediction columns, got {len(prediction_columns)}")
    for horizon in HORIZONS:
        target_valid = np.isfinite(inputs.data.y_train[horizon].to_numpy(float))
        for column in [c for c in prediction_columns if c.startswith(f"pred_{horizon.rsplit('_', 1)[-1]}__")]:
            values = frame[column].to_numpy(float)
            if np.isnan(values[target_valid]).any() or np.isfinite(values[~target_valid]).any():
                raise ValueError(f"Nested prediction mask mismatch: {column}")
    return {"status": "verified", "path": str(path), "sha256": sha256_file(path), "columns": len(prediction_columns)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=PRIMARY_SPLIT)
    parser.add_argument("--model-sample", type=int, help="Verify only the first N existing models; default verifies all")
    parser.add_argument("--allow-missing-v112", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    inputs = load_inputs(require_v112=not args.allow_missing_v112)
    report = {
        "inputs": verify_inputs(inputs),
        "folds": verify_folds(inputs),
        "v18_c1": verify_c1(inputs),
        "v112": verify_v112(inputs),
        "models": verify_models(inputs, args.split, args.model_sample),
        "nested_predictions": verify_nested_predictions(inputs),
    }
    output = MANIFESTS_DIR / f"verification_{args.split}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
