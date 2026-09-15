"""Data loading, alignment, and baseline contracts for v1.20."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from distributional_config import (
    ALL_COMPONENTS, HORIZONS, PRIMARY_FOLD_FILE, PROJECT_ROOT, V18_DIR,
    V18_OOF_COMPONENTS, V18_TEST_COMPONENTS, V112_OOF, V112_TEST,
)

if str(V18_DIR) not in sys.path:
    sys.path.insert(0, str(V18_DIR))
# v1.8 uses top-level imports named config/protocol.
_v18_config = importlib.import_module("config")
_v18_protocol = importlib.import_module("protocol")

load_data = _v18_protocol.load_data
build_component_targets = _v18_protocol.build_component_targets
component_target_name = _v18_protocol.component_target_name
compose_osi = _v18_protocol.compose_osi
clip_component = _v18_protocol.clip_component
post_process = _v18_protocol.post_process
fill_submission = _v18_protocol.fill_submission
sha256_file = _v18_protocol.sha256_file
normalize_fips = _v18_protocol.normalize_fips

KEYS = ("fipsCode", "timestamp_et")


@dataclass
class ExperimentInputs:
    data: Any
    component_targets: pd.DataFrame
    v18_oof_components: pd.DataFrame
    v18_test_components: pd.DataFrame
    v112_oof: pd.DataFrame | None
    v112_test: pd.DataFrame | None
    hashes: dict[str, str]


def _normalise_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing keys: {sorted(missing)}")
    result = frame.copy()
    result["fipsCode"] = normalize_fips(result["fipsCode"])
    result["timestamp_et"] = pd.to_datetime(result["timestamp_et"])
    if result[list(KEYS)].duplicated().any():
        raise ValueError(f"{label} has duplicate county-time keys")
    return result


def assert_aligned(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    left = _normalise_keys(reference[list(KEYS)], "reference")
    right = _normalise_keys(candidate, label)
    if len(left) != len(right):
        raise ValueError(f"{label} row count differs: {len(right)} != {len(left)}")
    if not left["fipsCode"].reset_index(drop=True).equals(right["fipsCode"].reset_index(drop=True)):
        raise ValueError(f"{label} fipsCode ordering differs from v1.5.6 metadata")
    left_time = left["timestamp_et"].to_numpy(dtype="datetime64[ns]")
    right_time = right["timestamp_et"].to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(left_time, right_time):
        raise ValueError(f"{label} timestamp_et ordering differs from v1.5.6 metadata")


def _read_v112_oof(meta_train: pd.DataFrame, required: bool) -> pd.DataFrame | None:
    if not V112_OOF.exists():
        if required:
            raise FileNotFoundError(f"Missing v1.12 OOF baseline: {V112_OOF}")
        return None
    frame = _normalise_keys(pd.read_parquet(V112_OOF), "v1.12 OOF")
    assert_aligned(meta_train, frame, "v1.12 OOF")
    for horizon in HORIZONS:
        column = f"pred_tail_protected_{horizon}"
        if column not in frame:
            raise ValueError(f"v1.12 OOF missing {column}")
    return frame


def _read_v112_test(meta_test: pd.DataFrame, required: bool) -> pd.DataFrame | None:
    if not V112_TEST.exists():
        if required:
            raise FileNotFoundError(f"Missing v1.12 test baseline: {V112_TEST}")
        return None
    frame = pd.read_csv(V112_TEST, dtype={"fipsCode": str})
    frame = _normalise_keys(frame, "v1.12 test")
    # A submission may use template order; explicitly align it to meta_test.
    indexed = frame.set_index(list(KEYS))
    keys = pd.MultiIndex.from_frame(_normalise_keys(meta_test[list(KEYS)], "test metadata"))
    aligned = indexed.reindex(keys).reset_index()
    if aligned[list(KEYS)].isna().any().any():
        raise ValueError("v1.12 test baseline cannot be aligned to test metadata")
    for horizon in HORIZONS:
        candidates = (f"pred_tail_protected_{horizon}", horizon)
        source = next((name for name in candidates if name in aligned), None)
        if source is None:
            raise ValueError(f"v1.12 test missing prediction for {horizon}")
        aligned[f"pred_tail_protected_{horizon}"] = pd.to_numeric(aligned[source], errors="coerce")
    return aligned


def _load_component_targets(data: Any) -> pd.DataFrame:
    """Reuse v1.8 targets while tolerating pandas datetime-unit differences."""
    path = V18_DIR / "component_targets_v1.8.parquet"
    targets = pd.read_parquet(path)
    expected_columns = ["fipsCode", "timestamp_et", "hour_idx"] + [
        component_target_name(component, horizon)
        for horizon in _v18_config.HORIZONS for component in _v18_config.COMPONENTS
    ]
    if list(targets.columns) != expected_columns or len(targets) != len(data.meta_train):
        raise ValueError("v1.8 component target columns or row count differ")
    if not normalize_fips(targets["fipsCode"]).reset_index(drop=True).equals(data.meta_train["fipsCode"].reset_index(drop=True)):
        raise ValueError("v1.8 component target county ordering differs")
    target_time = pd.to_datetime(targets["timestamp_et"]).to_numpy(dtype="datetime64[ns]")
    meta_time = pd.to_datetime(data.meta_train["timestamp_et"]).to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(target_time, meta_time):
        raise ValueError("v1.8 component target timestamp ordering differs")
    return targets


def load_inputs(require_v112: bool = True) -> ExperimentInputs:
    data = load_data()
    targets = _load_component_targets(data)
    v18_oof = _normalise_keys(pd.read_parquet(V18_OOF_COMPONENTS), "v1.8 OOF components")
    v18_test = _normalise_keys(pd.read_parquet(V18_TEST_COMPONENTS), "v1.8 test components")
    assert_aligned(data.meta_train, v18_oof, "v1.8 OOF components")
    assert_aligned(data.meta_test, v18_test, "v1.8 test components")
    required_component_columns = {
        f"pred_{component_target_name(component, horizon)}"
        for component in ALL_COMPONENTS for horizon in HORIZONS
    }
    for label, frame in (("v1.8 OOF", v18_oof), ("v1.8 test", v18_test)):
        missing = required_component_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing component columns: {sorted(missing)}")
    hashes = {
        "feature_names": sha256_file(PROJECT_ROOT / "cache" / "feature_names_v1.5.6.json"),
        "features_train": sha256_file(PROJECT_ROOT / "cache" / "features_train_v1.5.6.parquet"),
        "features_test": sha256_file(PROJECT_ROOT / "cache" / "features_test_v1.5.6.parquet"),
        "targets_train": sha256_file(PROJECT_ROOT / "cache" / "targets_train_v1.5.6.parquet"),
        "meta_train": sha256_file(PROJECT_ROOT / "cache" / "meta_train_v1.5.6.parquet"),
        "meta_test": sha256_file(PROJECT_ROOT / "cache" / "meta_test_v1.5.6.parquet"),
        "component_targets": sha256_file(V18_DIR / "component_targets_v1.8.parquet"),
        "v18_oof_components": sha256_file(V18_OOF_COMPONENTS),
        "v18_test_components": sha256_file(V18_TEST_COMPONENTS),
        "primary_folds": sha256_file(PRIMARY_FOLD_FILE),
    }
    if V112_OOF.exists():
        hashes["v112_oof"] = sha256_file(V112_OOF)
    if V112_TEST.exists():
        hashes["v112_test"] = sha256_file(V112_TEST)
    return ExperimentInputs(
        data=data,
        component_targets=targets,
        v18_oof_components=v18_oof,
        v18_test_components=v18_test,
        v112_oof=_read_v112_oof(data.meta_train, require_v112),
        v112_test=_read_v112_test(data.meta_test, require_v112),
        hashes=hashes,
    )


def c1_component_predictions(frame: pd.DataFrame, horizon: str) -> dict[str, np.ndarray]:
    return {
        component: clip_component(frame[f"pred_{component_target_name(component, horizon)}"].to_numpy(float))
        for component in ALL_COMPONENTS
    }


def recompose_with_replacements(
    baseline_frame: pd.DataFrame,
    horizon: str,
    replacements: dict[str, np.ndarray],
) -> np.ndarray:
    components = c1_component_predictions(baseline_frame, horizon)
    for component, values in replacements.items():
        if component not in ("P_t", "D_t"):
            raise ValueError(f"Only P_t/D_t replacements are allowed, got {component}")
        values = np.asarray(values, dtype=float)
        if len(values) != len(baseline_frame):
            raise ValueError(f"Replacement length mismatch for {component}")
        components[component] = clip_component(values)
    return post_process(compose_osi(components))


def write_input_manifest(inputs: ExperimentInputs, path: Path) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "project_root": str(PROJECT_ROOT),
        "input_hashes": inputs.hashes,
        "rows": {"train": len(inputs.data.X_train), "test": len(inputs.data.X_test)},
        "features": list(inputs.data.X_train.columns),
        "v112_available": inputs.v112_oof is not None and inputs.v112_test is not None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
