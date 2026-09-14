"""Reload all v2.6 component ensemble models and reproduce OOF/test/OSI outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VERSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERSION_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "versions/v1.11_tree_ensemble"))

from ensemble_utils import COMPONENTS, HORIZONS, clip_component, compose_osi, load_row_folds, sha256_file, write_json  # noqa: E402
from run_component_ensemble import ARTIFACTS_DIR, FAMILIES, INPUTS, MODELS_DIR, component_target, load_inputs  # noqa: E402


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
    frame, sources = load_inputs(test=False)
    test_frame, test_sources = load_inputs(test=True)
    frame["fold"] = load_row_folds(PROJECT_ROOT, frame)
    folds = frame["fold"].to_numpy(dtype=int)
    saved_oof = pd.read_parquet(ARTIFACTS_DIR / "oof_component_and_osi_predictions.parquet")
    saved_test = pd.read_csv(ARTIFACTS_DIR / "test_component_and_osi_predictions.csv", dtype={"fipsCode": str})
    maximum = 0.0
    loaded_models = 0
    oof_components: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    test_components: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for horizon in HORIZONS:
        for component in COMPONENTS:
            target = component_target(component, horizon)
            y = sources["lightgbm"][f"actual_{target}"].to_numpy(dtype=float)
            source = {family: sources[family][f"pred_{target}"].to_numpy(dtype=float) for family in FAMILIES}
            matrix = np.column_stack([source[family] for family in FAMILIES])
            outputs = {**source, "simple_average": clip_component(np.mean(matrix, axis=1)), "static_convex": np.full(len(frame), np.nan), "safe_convex": np.full(len(frame), np.nan)}
            for fold in range(5):
                model = json.loads((MODELS_DIR / f"outer_fold{fold}_{component}_{horizon}.json").read_text(encoding="utf-8"))
                loaded_models += 1
                valid = (folds == fold) & np.isfinite(y) & np.isfinite(matrix).all(axis=1)
                outputs["static_convex"][valid] = clip_component(matrix[valid] @ np.asarray(model["weights"]))
                outputs["safe_convex"][valid] = clip_component(source["lightgbm"][valid] + model["safe_alpha"] * (outputs["static_convex"][valid] - source["lightgbm"][valid]))
            oof_components[(component, horizon)] = outputs
            for method in ("simple_average", "static_convex", "safe_convex"):
                maximum = max(maximum, compare(saved_oof[f"pred_{method}_{target}"], outputs[method], f"OOF/{method}/{target}"))

            final = json.loads((MODELS_DIR / f"final_{component}_{horizon}.json").read_text(encoding="utf-8"))
            loaded_models += 1
            test_source = {family: test_sources[family][f"pred_{target}"].to_numpy(dtype=float) for family in FAMILIES}
            test_matrix = np.column_stack([test_source[family] for family in FAMILIES])
            test_static = clip_component(test_matrix @ np.asarray(final["weights"]))
            test_outputs = {**test_source, "simple_average": clip_component(np.mean(test_matrix, axis=1)), "static_convex": test_static, "safe_convex": clip_component(test_source["lightgbm"] + final["safe_alpha"] * (test_static - test_source["lightgbm"]))}
            test_components[(component, horizon)] = test_outputs
            for method in ("simple_average", "static_convex", "safe_convex"):
                maximum = max(maximum, compare(saved_test[f"pred_{method}_{target}"], test_outputs[method], f"test/{method}/{target}"))

    for horizon in HORIZONS:
        for method in (*FAMILIES, "simple_average", "static_convex", "safe_convex"):
            oof_osi = compose_osi({component: oof_components[(component, horizon)][method] for component in COMPONENTS})
            test_osi = compose_osi({component: test_components[(component, horizon)][method] for component in COMPONENTS})
            maximum = max(maximum, compare(saved_oof[f"pred_osi_{method}_{horizon}"], oof_osi, f"OOF OSI/{method}/{horizon}"))
            maximum = max(maximum, compare(saved_test[f"pred_osi_{method}_{horizon}"], test_osi, f"test OSI/{method}/{horizon}"))
    submission = pd.read_csv(ARTIFACTS_DIR / "submission_v2.6_safe_convex_balanced_v1.csv")
    template = pd.read_csv(INPUTS["submission_template"])
    for horizon in HORIZONS:
        expected_mask = np.isfinite(template[horizon].to_numpy(dtype=float))
        if not np.array_equal(np.isfinite(submission[horizon]), expected_mask):
            raise ValueError(f"submission/{horizon}: missing mask differs from official template")
        expected = np.where(expected_mask, saved_test[f"pred_osi_safe_convex_{horizon}"].to_numpy(dtype=float), np.nan)
        maximum = max(maximum, compare(submission[horizon], expected, f"submission/{horizon}"))

    metadata = json.loads((ARTIFACTS_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    bad_inputs = [name for name, path in INPUTS.items() if sha256_file(path) != metadata["input_hashes"][name]]
    bad_artifacts = []
    for relative, expected_hash in metadata["artifact_hashes"].items():
        path = PROJECT_ROOT / relative
        if not path.exists() or sha256_file(path) != expected_hash:
            bad_artifacts.append(relative)
    result = {"status": "pass", "loaded_models": loaded_models, "expected_models": 96, "max_prediction_difference": maximum, "input_hashes_valid": not bad_inputs, "artifact_hashes_valid": not bad_artifacts, "bad_inputs": bad_inputs, "bad_artifacts": bad_artifacts, "oof_rows": len(saved_oof), "test_rows": len(test_frame)}
    if loaded_models != 96 or bad_inputs or bad_artifacts:
        result["status"] = "fail"
        write_json(ARTIFACTS_DIR / "verification.json", result)
        raise ValueError(str(result))
    write_json(ARTIFACTS_DIR / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
