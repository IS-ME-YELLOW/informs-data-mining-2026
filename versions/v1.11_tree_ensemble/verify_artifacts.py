"""Reload every saved v1.11 ensemble model and reproduce OOF/test predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_utils import HORIZONS, load_row_folds, post_process, sha256_file, write_json
from run_ensemble import ARTIFACTS_DIR, CANDIDATES, INPUTS, MODELS_DIR, PROJECT_ROOT, load_oof_inputs, load_test_inputs


def compare(expected: np.ndarray, actual: np.ndarray, label: str) -> float:
    left = np.asarray(expected, dtype=float)
    right = np.asarray(actual, dtype=float)
    if not np.array_equal(np.isfinite(left), np.isfinite(right)):
        raise ValueError(f"{label}: finite masks differ")
    valid = np.isfinite(left)
    maximum = float(np.max(np.abs(left[valid] - right[valid]))) if valid.any() else 0.0
    if maximum > 1e-12:
        raise ValueError(f"{label}: maximum difference {maximum} exceeds tolerance")
    return maximum


def main() -> None:
    frame, source, _ = load_oof_inputs()
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame["fold"].to_numpy(dtype=int)
    test_frame, test_source = load_test_inputs()
    saved_oof = pd.read_parquet(ARTIFACTS_DIR / "oof_predictions.parquet")
    saved_test = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv", dtype={"fipsCode": str})
    template = pd.read_csv(INPUTS["submission_template"])
    maximum = 0.0
    loaded_models = 0
    for horizon in HORIZONS:
        y = source[horizon]["actual"]
        matrix = np.column_stack([source[horizon][name] for name in CANDIDATES])
        baseline = source[horizon]["lgbm_component"]
        simple = post_process(np.mean(matrix, axis=1))
        maximum = max(maximum, compare(saved_oof[f"pred_simple_average_{horizon}"], simple, f"OOF simple/{horizon}"))
        static = np.full(len(frame), np.nan)
        safe = np.full(len(frame), np.nan)
        for fold in range(5):
            model = json.loads((MODELS_DIR / f"outer_fold{fold}_{horizon}.json").read_text(encoding="utf-8"))
            loaded_models += 1
            if model["candidate_names"] != list(CANDIDATES) or model["fold"] != fold:
                raise ValueError(f"Invalid model identity for {horizon}/fold {fold}")
            valid = (folds == fold) & np.isfinite(y) & np.isfinite(matrix).all(axis=1)
            static[valid] = post_process(matrix[valid] @ np.asarray(model["weights"]))
            safe[valid] = post_process(baseline[valid] + model["safe_alpha"] * (static[valid] - baseline[valid]))
        maximum = max(maximum, compare(saved_oof[f"pred_static_convex_{horizon}"], static, f"OOF static/{horizon}"))
        maximum = max(maximum, compare(saved_oof[f"pred_safe_convex_{horizon}"], safe, f"OOF safe/{horizon}"))

        final = json.loads((MODELS_DIR / f"final_{horizon}.json").read_text(encoding="utf-8"))
        loaded_models += 1
        test_matrix = np.column_stack([test_source[horizon][name] for name in CANDIDATES])
        test_static = post_process(test_matrix @ np.asarray(final["weights"]))
        test_safe = post_process(test_source[horizon]["lgbm_component"] + final["safe_alpha"] * (test_static - test_source[horizon]["lgbm_component"]))
        expected_mask = np.isfinite(template[horizon].to_numpy(dtype=float))
        maximum = max(maximum, compare(saved_test[f"pred_simple_average_{horizon}"], np.where(expected_mask, post_process(np.mean(test_matrix, axis=1)), np.nan), f"test simple/{horizon}"))
        maximum = max(maximum, compare(saved_test[f"pred_static_convex_{horizon}"], np.where(expected_mask, test_static, np.nan), f"test static/{horizon}"))
        maximum = max(maximum, compare(saved_test[f"pred_safe_convex_{horizon}"], np.where(expected_mask, test_safe, np.nan), f"test safe/{horizon}"))

    metadata = json.loads((ARTIFACTS_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    bad_inputs = [name for name, path in INPUTS.items() if sha256_file(path) != metadata["input_hashes"][name]]
    bad_artifacts = []
    for relative, expected_hash in metadata["artifact_hashes"].items():
        path = PROJECT_ROOT / relative
        if not path.exists() or sha256_file(path) != expected_hash:
            bad_artifacts.append(relative)
    submission = pd.read_csv(ARTIFACTS_DIR / "submission_v1.11_safe_convex_balanced_v1.csv")
    for horizon in HORIZONS:
        if not np.array_equal(np.isfinite(submission[horizon]), np.isfinite(template[horizon])):
            raise ValueError(f"submission/{horizon}: missing mask differs from official template")
        maximum = max(maximum, compare(submission[horizon], saved_test[f"pred_safe_convex_{horizon}"], f"submission/{horizon}"))
    result = {"status": "pass", "loaded_models": loaded_models, "expected_models": 24, "max_prediction_difference": maximum, "input_hashes_valid": not bad_inputs, "artifact_hashes_valid": not bad_artifacts, "bad_inputs": bad_inputs, "bad_artifacts": bad_artifacts, "oof_rows": len(saved_oof), "test_rows": len(test_frame)}
    if loaded_models != 24 or bad_inputs or bad_artifacts:
        result["status"] = "fail"
        write_json(ARTIFACTS_DIR / "verification.json", result)
        raise ValueError(str(result))
    write_json(ARTIFACTS_DIR / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
