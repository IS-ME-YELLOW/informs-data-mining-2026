"""Independently reload and reproduce the v1.15 shallow gate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_shallow_gate import (
    ARTIFACTS_DIR,
    CANDIDATES,
    GATED_HORIZONS,
    HORIZONS,
    INPUTS,
    MODELS_DIR,
    PROJECT_ROOT,
    assert_aligned,
    load_oof_inputs,
    load_row_folds,
    load_test_inputs,
    normalize_frame,
    post_process,
    predict,
    sha256_file,
)


def run() -> None:
    frame, source, _ = load_oof_inputs()
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame.fold.to_numpy(dtype=int)
    test_frame, test_source = load_test_inputs()
    v112 = normalize_frame(pd.read_parquet(INPUTS["v112_oof"]))
    v112_test = normalize_frame(pd.read_csv(INPUTS["v112_test"], dtype={"fipsCode": str}))
    saved_oof = normalize_frame(pd.read_parquet(ARTIFACTS_DIR / "oof_predictions.parquet"))
    saved_test = normalize_frame(pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv", dtype={"fipsCode": str}))
    assert_aligned(frame, saved_oof, "saved OOF")
    assert_aligned(test_frame, saved_test, "saved test")
    max_difference = 0.0
    loaded = 0
    for horizon in HORIZONS:
        baseline = source[horizon]["lgbm_component"]
        simple = post_process(np.mean(np.column_stack([source[horizon][name] for name in CANDIDATES]), axis=1))
        current = v112[f"pred_tail_protected_{horizon}"].to_numpy(dtype=float)
        reproduced = np.full(len(frame), np.nan)
        for outer_fold in range(5):
            model = json.loads((MODELS_DIR / f"outer_fold{outer_fold}_{horizon}.json").read_text(encoding="utf-8"))
            outer = folds == outer_fold
            reproduced[outer] = current[outer] if horizon not in GATED_HORIZONS else predict(model, baseline[outer], simple[outer])
            loaded += 1
        expected = saved_oof[f"pred_shallow_risk_gate_{horizon}"].to_numpy(dtype=float)
        valid = np.isfinite(expected)
        max_difference = max(max_difference, float(np.max(np.abs(reproduced[valid] - expected[valid]))))
        model = json.loads((MODELS_DIR / f"final_{horizon}.json").read_text(encoding="utf-8"))
        if horizon not in GATED_HORIZONS:
            reproduced_test = v112_test[f"pred_tail_protected_{horizon}"].to_numpy(dtype=float)
        else:
            matrix = np.column_stack([test_source[horizon][name] for name in CANDIDATES])
            reproduced_test = predict(model, test_source[horizon]["lgbm_component"], post_process(np.mean(matrix, axis=1)))
        expected_test = saved_test[f"pred_shallow_gate_{horizon}"].to_numpy(dtype=float)
        valid_test = np.isfinite(expected_test)
        max_difference = max(max_difference, float(np.max(np.abs(reproduced_test[valid_test] - expected_test[valid_test]))))
        loaded += 1
    metadata = json.loads((ARTIFACTS_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    bad_inputs = [name for name, expected in metadata["input_hashes"].items() if sha256_file(path=INPUTS[name]) != expected]
    bad_artifacts = []
    for relative, expected in metadata["artifact_hashes"].items():
        path = PROJECT_ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            bad_artifacts.append(relative)
    result = {"status": "pass" if max_difference <= 1e-12 and not bad_inputs and not bad_artifacts else "fail", "loaded_models": loaded, "expected_models": 24, "max_prediction_difference": max_difference, "input_hashes_valid": not bad_inputs, "artifact_hashes_valid": not bad_artifacts, "bad_inputs": bad_inputs, "bad_artifacts": bad_artifacts}
    (ARTIFACTS_DIR / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    run()
