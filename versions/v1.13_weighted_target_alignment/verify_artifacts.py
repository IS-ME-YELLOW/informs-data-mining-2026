"""Reload all v1.13 lead-weight models and reproduce OOF/test/submission."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from run_weighted_alignment import *  # noqa: F403


def compare(left, right, label):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise ValueError(f"{label}: masks differ")
    valid = np.isfinite(a)
    maximum = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0
    if maximum > 1e-12:
        raise ValueError(f"{label}: max difference {maximum}")
    return maximum


def main():
    frame, source, _ = load_oof_inputs()  # noqa: F405
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)  # noqa: F405
    folds = frame.fold.to_numpy(int)
    test_frame, _ = load_test_inputs()  # noqa: F405
    matrices = build_aligned_raw_matrices(frame[KEYS], normalize_frame(pd.read_parquet(INPUTS["v18_oof_components"])))  # noqa: F405
    test_matrices = build_aligned_raw_matrices(test_frame[KEYS], normalize_frame(pd.read_parquet(INPUTS["v18_test_components"])))  # noqa: F405
    v112_oof = normalize_frame(pd.read_parquet(INPUTS["v112_oof"]))  # noqa: F405
    v112_test = normalize_frame(pd.read_csv(INPUTS["v112_test"], dtype={"fipsCode": str}))  # noqa: F405
    saved_oof = pd.read_parquet(ARTIFACTS_DIR / "oof_predictions.parquet")  # noqa: F405
    saved_test = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv")  # noqa: F405
    template = pd.read_csv(INPUTS["submission_template"])  # noqa: F405
    maximum, loaded = 0.0, 0
    reproduced = {h: np.full(len(frame), np.nan) for h in HORIZONS}  # noqa: F405
    for fold in range(5):
        model = json.loads((MODELS_DIR / f"outer_fold{fold}.json").read_text(encoding="utf-8"))  # noqa: F405
        loaded += 1
        outer = folds == fold
        for horizon in HORIZONS:  # noqa: F405
            prediction = aligned_prediction(matrices[horizon], np.asarray(model["weights"]))  # noqa: F405
            reproduced[horizon][outer] = prediction[outer]
    final = json.loads((MODELS_DIR / "final.json").read_text(encoding="utf-8"))  # noqa: F405
    loaded += 1
    for horizon in HORIZONS:  # noqa: F405
        maximum = max(maximum, compare(saved_oof[f"pred_weighted_alignment_{horizon}"], reproduced[horizon], f"OOF weighted/{horizon}"))
        combined = reproduced[horizon] if horizon in SHORT_HORIZONS else v112_oof[f"pred_tail_protected_{horizon}"].to_numpy(float)  # noqa: F405
        maximum = max(maximum, compare(saved_oof[f"pred_weighted_alignment_plus_tail_{horizon}"], combined, f"OOF combined/{horizon}"))
        test_weighted = aligned_prediction(test_matrices[horizon], np.asarray(final["weights"]))  # noqa: F405
        test_combined = test_weighted if horizon in SHORT_HORIZONS else v112_test[f"pred_tail_protected_{horizon}"].to_numpy(float)  # noqa: F405
        mask = np.isfinite(template[horizon].to_numpy(float))
        maximum = max(maximum, compare(saved_test[f"pred_weighted_alignment_plus_tail_{horizon}"], np.where(mask, test_combined, np.nan), f"test/{horizon}"))
    submission = pd.read_csv(ARTIFACTS_DIR / "submission_v1.13_weighted_alignment_balanced_v1.csv")  # noqa: F405
    for horizon in HORIZONS:  # noqa: F405
        maximum = max(maximum, compare(submission[horizon], saved_test[f"pred_weighted_alignment_plus_tail_{horizon}"], f"submission/{horizon}"))
    metadata = json.loads((ARTIFACTS_DIR / "run_metadata.json").read_text(encoding="utf-8"))  # noqa: F405
    bad_inputs = [name for name, path in INPUTS.items() if sha256_file(path) != metadata["input_hashes"][name]]  # noqa: F405
    bad_artifacts = [relative for relative, digest in metadata["artifact_hashes"].items() if not (PROJECT_ROOT / relative).exists() or sha256_file(PROJECT_ROOT / relative) != digest]  # noqa: F405
    result = {"status": "pass" if loaded == 6 and not bad_inputs and not bad_artifacts else "fail", "loaded_models": loaded, "expected_models": 6, "max_prediction_difference": maximum, "input_hashes_valid": not bad_inputs, "artifact_hashes_valid": not bad_artifacts, "bad_inputs": bad_inputs, "bad_artifacts": bad_artifacts}
    write_json(ARTIFACTS_DIR / "verification.json", result)  # noqa: F405
    if result["status"] != "pass":
        raise ValueError(str(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
