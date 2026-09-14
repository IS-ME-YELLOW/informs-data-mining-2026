"""Independently reload every saved sklearn Pipeline and reproduce predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "versions"))
from bagging_tree_experiment import (  # noqa: E402
    FEATURE_VERSION, configurations, load_experiment_data, model_path, post_process, sha256_file, write_json
)


def compare(a: np.ndarray, b: np.ndarray, label: str) -> float:
    left, right = np.asarray(a, float), np.asarray(b, float)
    if not np.array_equal(np.isfinite(left), np.isfinite(right)):
        raise ValueError(f"{label}: finite masks differ")
    valid = np.isfinite(left)
    maximum = float(np.max(np.abs(left[valid] - right[valid]))) if valid.any() else 0.0
    if maximum > 1e-12:
        raise ValueError(f"{label}: max difference {maximum}")
    return maximum


def verify(version_dir: Path, family: str) -> None:
    artifacts, models = version_dir / "artifacts", version_dir / "models"
    metadata = json.loads((artifacts / "run_metadata.json").read_text(encoding="utf-8"))
    data = load_experiment_data(FEATURE_VERSION)
    saved_oof = pd.read_parquet(artifacts / "oof_predictions.parquet")
    saved_test = pd.read_parquet(artifacts / "test_predictions.parquet")
    maximum, loaded = 0.0, 0
    for horizon in metadata["horizons"]:
        y = data.y_train[horizon].to_numpy(float)
        finite = np.isfinite(y)
        for config in configurations(family):
            reproduced = np.full(len(y), np.nan)
            for fold, (_, valid_full) in enumerate(data.folds):
                valid = valid_full[finite[valid_full]]
                pipeline = joblib.load(model_path(models, config, horizon, f"fold{fold}"))
                loaded += 1
                reproduced[valid] = post_process(pipeline.predict(data.X_train.iloc[valid]))
            maximum = max(maximum, compare(saved_oof[f"pred_{config}_{horizon}"], reproduced, f"OOF/{config}/{horizon}"))
            final = joblib.load(model_path(models, config, horizon, "final"))
            loaded += 1
            test_pred = post_process(final.predict(data.X_test))
            maximum = max(maximum, compare(saved_test[f"pred_{config}_{horizon}"], test_pred, f"test/{config}/{horizon}"))
    bad_inputs = [rel for rel, digest in metadata["input_hashes"].items() if not (PROJECT_ROOT / rel).exists() or sha256_file(PROJECT_ROOT / rel) != digest]
    bad_artifacts = [rel for rel, digest in metadata["artifact_hashes"].items() if not (PROJECT_ROOT / rel).exists() or sha256_file(PROJECT_ROOT / rel) != digest]
    expected = len(metadata["horizons"]) * len(configurations(family)) * 6
    result = {"status": "pass" if loaded == expected and not bad_inputs and not bad_artifacts else "fail", "loaded_models": loaded, "expected_models": expected, "max_prediction_difference": maximum, "input_hashes_valid": not bad_inputs, "artifact_hashes_valid": not bad_artifacts, "bad_inputs": bad_inputs, "bad_artifacts": bad_artifacts}
    write_json(artifacts / "verification.json", result)
    if result["status"] != "pass":
        raise ValueError(str(result))
    print(json.dumps(result, indent=2))

