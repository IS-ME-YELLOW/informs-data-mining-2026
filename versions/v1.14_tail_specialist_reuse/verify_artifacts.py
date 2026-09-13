"""Reload v1.14 JSON gate models and independently reproduce saved predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_tail_specialist import (
    ARTIFACTS_DIR,
    HORIZONS,
    INPUTS,
    MODELS_DIR,
    PROJECT_ROOT,
    SPECIALISTS,
    apply_model,
    build_fixed_experts,
    load_inputs,
    risk_column,
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> None:
    v18, v19, v19_test, v19_test_controls, v112, v112_test = load_inputs()
    saved_oof = pd.read_parquet(ARTIFACTS_DIR / "oof_predictions.parquet")
    saved_test = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv", dtype={"fipsCode": str})
    cv = pd.read_csv(INPUTS["cv"], dtype={"fipsCode": str})
    fold_map = cv.drop_duplicates("fipsCode").set_index("fipsCode")["fold"].astype(int)
    folds = v18.fipsCode.map(fold_map).to_numpy(dtype=int)
    max_difference = 0.0
    loaded = 0
    for horizon in HORIZONS:
        current = v112[f"pred_tail_protected_{horizon}"].to_numpy(dtype=float)
        risk = v18[risk_column(horizon)].to_numpy(dtype=float)
        reproduced = np.full(len(v18), np.nan)
        for outer_fold in range(5):
            model = json.loads((MODELS_DIR / f"outer_fold{outer_fold}_{horizon}.json").read_text(encoding="utf-8"))
            expert = build_fixed_experts(v19, horizon)[model["specialist"]]
            mask = folds == outer_fold
            reproduced[mask] = apply_model(model, current[mask], expert[mask], risk[mask])
            loaded += 1
        expected = saved_oof[f"pred_tail_specialist_nested_{horizon}"].to_numpy(dtype=float)
        valid = np.isfinite(expected)
        max_difference = max(max_difference, float(np.max(np.abs(reproduced[valid] - expected[valid]))))
        final_model = json.loads((MODELS_DIR / f"final_{horizon}.json").read_text(encoding="utf-8"))
        test_current = v112_test[f"pred_tail_protected_{horizon}"].to_numpy(dtype=float)
        test_risk = v19_test_controls[f"pred__V18_current_rule__{horizon}"].to_numpy(dtype=float)
        test_expert = build_fixed_experts(v19_test, horizon)[final_model["specialist"]]
        reproduced_test = apply_model(final_model, test_current, test_expert, test_risk)
        expected_test = saved_test[f"pred_tail_specialist_{horizon}"].to_numpy(dtype=float)
        valid_test = np.isfinite(expected_test)
        max_difference = max(max_difference, float(np.max(np.abs(reproduced_test[valid_test] - expected_test[valid_test]))))
        loaded += 1

    metadata = json.loads((ARTIFACTS_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    bad_inputs = [name for name, expected in metadata["input_hashes"].items() if sha256_file(INPUTS[name]) != expected]
    bad_artifacts = []
    for relative, expected in metadata["artifact_hashes"].items():
        path = PROJECT_ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            bad_artifacts.append(relative)
    result = {"status": "pass" if max_difference <= 1e-12 and not bad_inputs and not bad_artifacts else "fail", "loaded_models": loaded, "expected_models": 24, "max_prediction_difference": max_difference, "input_hashes_valid": not bad_inputs, "artifact_hashes_valid": not bad_artifacts, "bad_inputs": bad_inputs, "bad_artifacts": bad_artifacts, "specialists_available": list(SPECIALISTS)}
    (ARTIFACTS_DIR / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    run()
