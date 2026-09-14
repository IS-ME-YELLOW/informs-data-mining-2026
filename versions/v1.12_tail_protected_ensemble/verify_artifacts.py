"""Reload v1.12 gate models and reproduce all saved OOF/test predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_tail_protected import (
    ARTIFACTS_DIR,
    CANDIDATES,
    HORIZONS,
    MODELS_DIR,
    PROJECT_ROOT,
    SOURCE_INPUTS,
    load_oof_inputs,
    load_row_folds,
    load_test_inputs,
    post_process,
    predict,
    sha256_file,
    write_json,
)


def compare(left: np.ndarray, right: np.ndarray, label: str) -> float:
    a, b = np.asarray(left, float), np.asarray(right, float)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise ValueError(f"{label}: missing masks differ")
    valid = np.isfinite(a)
    maximum = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0
    if maximum > 1e-12:
        raise ValueError(f"{label}: max difference={maximum}")
    return maximum


def main() -> None:
    frame, source, _ = load_oof_inputs()
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame.fold.to_numpy(int)
    _, test_source = load_test_inputs()
    saved_oof = pd.read_parquet(ARTIFACTS_DIR / "oof_predictions.parquet")
    saved_test = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv")
    template = pd.read_csv(SOURCE_INPUTS["submission_template"])
    maximum = 0.0
    loaded = 0
    for horizon in HORIZONS:
        y = source[horizon]["actual"]
        baseline = source[horizon]["lgbm_component"]
        matrix = np.column_stack([source[horizon][name] for name in CANDIDATES])
        simple = post_process(np.mean(matrix, axis=1))
        reproduced = np.full(len(frame), np.nan)
        for fold in range(5):
            model = json.loads((MODELS_DIR / f"outer_fold{fold}_{horizon}.json").read_text(encoding="utf-8"))
            loaded += 1
            outer = (folds == fold) & np.isfinite(y) & np.isfinite(simple)
            reproduced[outer] = predict(model, baseline[outer], simple[outer])
        maximum = max(maximum, compare(saved_oof[f"pred_tail_protected_{horizon}"], reproduced, f"OOF/{horizon}"))
        final = json.loads((MODELS_DIR / f"final_{horizon}.json").read_text(encoding="utf-8"))
        loaded += 1
        test_matrix = np.column_stack([test_source[horizon][name] for name in CANDIDATES])
        test_pred = predict(final, test_source[horizon]["lgbm_component"], post_process(np.mean(test_matrix, axis=1)))
        mask = np.isfinite(template[horizon].to_numpy(float))
        maximum = max(maximum, compare(saved_test[f"pred_tail_protected_{horizon}"], np.where(mask, test_pred, np.nan), f"test/{horizon}"))
    submission = pd.read_csv(ARTIFACTS_DIR / "submission_v1.12_tail_protected_balanced_v1.csv")
    for horizon in HORIZONS:
        maximum = max(maximum, compare(submission[horizon], saved_test[f"pred_tail_protected_{horizon}"], f"submission/{horizon}"))
    metadata = json.loads((ARTIFACTS_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    bad_inputs = [name for name, path in SOURCE_INPUTS.items() if sha256_file(path) != metadata["input_hashes"][name]]
    bad_artifacts = [relative for relative, digest in metadata["artifact_hashes"].items() if not (PROJECT_ROOT / relative).exists() or sha256_file(PROJECT_ROOT / relative) != digest]
    result = {"status": "pass" if loaded == 24 and not bad_inputs and not bad_artifacts else "fail", "loaded_models": loaded, "expected_models": 24, "max_prediction_difference": maximum, "input_hashes_valid": not bad_inputs, "artifact_hashes_valid": not bad_artifacts, "bad_inputs": bad_inputs, "bad_artifacts": bad_artifacts}
    write_json(ARTIFACTS_DIR / "verification.json", result)
    if result["status"] != "pass":
        raise ValueError(str(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
