"""Reload v1.18 JSON ensemble models and reproduce OOF/test predictions."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from run_tree_family_ensemble import ARTIFACTS_DIR, FAMILIES, INPUTS, LONG_HORIZONS, MODELS_DIR, PROJECT_ROOT, load_test_inputs, normalize, predict, sha256_file, write_json


def compare(a, b, label):
    x, y = np.asarray(a, float), np.asarray(b, float)
    if not np.array_equal(np.isfinite(x), np.isfinite(y)):
        raise ValueError(f"{label}: masks differ")
    valid = np.isfinite(x)
    maximum = float(np.max(np.abs(x[valid] - y[valid]))) if valid.any() else 0.0
    if maximum > 1e-12:
        raise ValueError(f"{label}: max difference={maximum}")
    return maximum


def main():
    v112 = normalize(pd.read_parquet(INPUTS["v112_oof"]))
    et = normalize(pd.read_parquet(INPUTS["et_oof"]))
    rf = normalize(pd.read_parquet(INPUTS["rf_oof"]))
    v112_test = normalize(pd.read_csv(INPUTS["v112_test"], dtype={"fipsCode": str}))
    et_test = normalize(pd.read_parquet(INPUTS["et_test"]))
    rf_test = normalize(pd.read_parquet(INPUTS["rf_test"]))
    _, boosting_test = load_test_inputs()
    saved = pd.read_parquet(ARTIFACTS_DIR / "oof_predictions.parquet")
    saved_test = pd.read_parquet(ARTIFACTS_DIR / "test_predictions.parquet")
    folds = v112.fold.to_numpy(int)
    maximum, loaded = 0.0, 0
    for horizon in LONG_HORIZONS:
        y = v112[f"actual_{horizon}"].to_numpy(float)
        sources = {"v1.8": v112[f"pred_v1.8_rule_{horizon}"].to_numpy(float), "v1.12": v112[f"pred_tail_protected_{horizon}"].to_numpy(float), "ET-A": et[f"pred_ET-A_{horizon}"].to_numpy(float), "ET-B": et[f"pred_ET-B_{horizon}"].to_numpy(float), "RF": rf[f"pred_RF_{horizon}"].to_numpy(float)}
        test_sources = {"v1.8": boosting_test[horizon]["lgbm_component"], "v1.12": v112_test[f"pred_tail_protected_{horizon}"].to_numpy(float), "ET-A": et_test[f"pred_ET-A_{horizon}"].to_numpy(float), "ET-B": et_test[f"pred_ET-B_{horizon}"].to_numpy(float), "RF": rf_test[f"pred_RF_{horizon}"].to_numpy(float)}
        for family in FAMILIES:
            reproduced = np.full(len(y), np.nan)
            slug = family.replace("+", "_").replace(".", "")
            for fold in range(5):
                model = json.loads((MODELS_DIR / f"outer_fold{fold}_{horizon}_{slug}.json").read_text(encoding="utf-8"))
                loaded += 1
                outer = (folds == fold) & np.isfinite(y)
                reproduced[outer] = predict(model, {k: v[outer] for k, v in sources.items()})
            maximum = max(maximum, compare(saved[f"pred_{family}_{horizon}"], reproduced, f"OOF/{family}/{horizon}"))
            final = json.loads((MODELS_DIR / f"final_{horizon}_{slug}.json").read_text(encoding="utf-8"))
            loaded += 1
            maximum = max(maximum, compare(saved_test[f"pred_{family}_{horizon}"], predict(final, test_sources), f"test/{family}/{horizon}"))
    metadata = json.loads((ARTIFACTS_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    bad_inputs = [name for name, path in INPUTS.items() if sha256_file(path) != metadata["input_hashes"][name]]
    bad_artifacts = [rel for rel, digest in metadata["artifact_hashes"].items() if not (PROJECT_ROOT / rel).exists() or sha256_file(PROJECT_ROOT / rel) != digest]
    expected = len(LONG_HORIZONS) * len(FAMILIES) * 6
    result = {"status": "pass" if loaded == expected and not bad_inputs and not bad_artifacts else "fail", "loaded_models": loaded, "expected_models": expected, "max_prediction_difference": maximum, "input_hashes_valid": not bad_inputs, "artifact_hashes_valid": not bad_artifacts, "bad_inputs": bad_inputs, "bad_artifacts": bad_artifacts}
    write_json(ARTIFACTS_DIR / "verification.json", result)
    if result["status"] != "pass":
        raise ValueError(str(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
