"""Read-only data contracts, fixed v1.8 controls and strict scoring."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from config import (CACHE_DIR, COMPONENTS, COMPONENT_WEIGHTS, CV_FILE, FEATURE_VERSION,
                    HORIZONS, HORIZON_HOURS, N_FOLDS, OSI_MAX, PROJECT_ROOT,
                    RAW_TRAIN_FILE, RAW_TEST_FILE, SUBMISSION_TEMPLATE, TARGETS_FILE,
                    ZERO_THRESHOLD)

MODES = ("C0_direct_osi", "C1_component_osi", "C2_equal_blend", "C3_aligned_component", "v18_rule")
TARGETS = tuple(HORIZONS) + tuple(f"{c}_target_{h.rsplit('_', 1)[-1]}" for h in HORIZONS for c in COMPONENTS)

@dataclass
class ExperimentData:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    meta_train: pd.DataFrame
    X_test: pd.DataFrame
    meta_test: pd.DataFrame
    folds: list
    row_folds: np.ndarray
    component_targets: pd.DataFrame = field(default_factory=pd.DataFrame)
    assignment: pd.DataFrame = field(default_factory=pd.DataFrame)
    input_paths: dict = field(default_factory=dict)
    loaded_hashes: dict = field(default_factory=dict)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

def resolve_input(path):
    path = Path(path)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

def normalize_fips(values):
    result = values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    if not result.str.fullmatch(r"\d{5}").all():
        raise ValueError("Invalid county FIPS")
    return result

def component_target_name(component, horizon):
    return f"{component}_target_{horizon.rsplit('_', 1)[-1]}"

def target_horizon(target):
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    return f"osi_target_{target.rsplit('_', 1)[-1]}"

def expected_mask(meta, horizon):
    hours = meta.hour_idx.to_numpy(dtype=int)
    return (hours >= 72) & (hours + HORIZON_HOURS[horizon] <= 215)

def require_predictions(pred, expected, *, exact_tail=False):
    pred, expected = np.asarray(pred, dtype=float), np.asarray(expected, dtype=bool)
    if pred.shape != expected.shape or not np.isfinite(pred[expected]).all():
        raise ValueError("Missing/non-finite prediction on required rows, or incorrect shape")
    if exact_tail and not np.isnan(pred[~expected]).all():
        raise ValueError("Non-NaN prediction outside the scoreable window")

def metrics(y_true, y_pred):
    truth, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape or np.isinf(truth).any():
        raise ValueError("Invalid metric input")
    valid = ~np.isnan(truth)
    require_predictions(pred, valid)
    if not valid.any():
        raise ValueError("No scoreable rows")
    error = pred[valid] - truth[valid]
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))), "n": int(valid.sum())}

def post_process(predictions):
    if np.isinf(np.asarray(predictions, dtype=float)).any():
        raise ValueError("Infinite raw OSI prediction")
    values = np.clip(np.asarray(predictions, dtype=float), 0., OSI_MAX)
    return np.where(values < ZERO_THRESHOLD, 0., values)

def clip_component(predictions):
    if np.isinf(np.asarray(predictions, dtype=float)).any():
        raise ValueError("Infinite raw component prediction")
    return np.clip(np.asarray(predictions, dtype=float), 0., 1.)

def compose_osi(predictions):
    if set(predictions) != set(COMPONENTS):
        raise ValueError("Incorrect component set")
    arrays = [np.asarray(predictions[c], dtype=float) for c in COMPONENTS]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("Inconsistent component shapes")
    return np.maximum(sum(COMPONENT_WEIGHTS[c] * a for c, a in zip(COMPONENTS, arrays)), 0.)

def load_cv_folds(meta_train, cv_file=CV_FILE):
    assignment = pd.read_csv(resolve_input(cv_file), dtype={"fipsCode": str, "stateAbbr": str})
    if set(assignment) != {"fipsCode", "stateAbbr", "severity_tier", "fold"}:
        raise ValueError("Incorrect CV columns")
    assignment["fipsCode"] = normalize_fips(assignment.fipsCode)
    for col in ["fold", "severity_tier"]:
        values = pd.to_numeric(assignment[col], errors="raise")
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError(f"Invalid integer CV field: {col}")
        assignment[col] = values.astype(int)
    if assignment.fipsCode.duplicated().any() or set(assignment.fold) != set(range(N_FOLDS)):
        raise ValueError("Duplicate county or missing fold")
    counties = meta_train[["fipsCode", "stateAbbr", "severity_tier"]].drop_duplicates()
    if counties.fipsCode.duplicated().any():
        raise ValueError("County has inconsistent stratification metadata")
    expected = counties.set_index("fipsCode").sort_index()
    actual = assignment.set_index("fipsCode").sort_index()
    pd.testing.assert_frame_equal(expected, actual[["stateAbbr", "severity_tier"]], check_dtype=False)
    counts = assignment.groupby("fold").size()
    strata = assignment.groupby(["stateAbbr", "severity_tier", "fold"]).size().unstack(fill_value=0).reindex(columns=range(5), fill_value=0)
    if counts.max() - counts.min() > 1 or ((strata.max(axis=1) - strata.min(axis=1)) > 1).any():
        raise ValueError("CV not balanced globally or within state x severity")
    row_folds = meta_train.fipsCode.map(actual.fold).to_numpy(dtype=int)
    folds = [(np.flatnonzero(row_folds != f), np.flatnonzero(row_folds == f)) for f in range(5)]
    return folds, row_folds, assignment.sort_values("fipsCode").reset_index(drop=True)

def _metadata(frame, counties):
    frame = frame.copy().reset_index(drop=True)
    frame["fipsCode"] = normalize_fips(frame.fipsCode)
    frame["timestamp_et"] = pd.to_datetime(frame.timestamp_et)
    if len(frame) != counties * 144 or frame.fipsCode.nunique() != counties:
        raise ValueError("Incorrect county/row coverage")
    if frame.duplicated(["fipsCode", "timestamp_et"]).any():
        raise ValueError("Duplicate county-time key")
    if not frame.equals(frame.sort_values(["fipsCode", "timestamp_et"]).reset_index(drop=True)):
        raise ValueError("Frozen metadata must be in canonical county/hour order")
    if not np.array_equal(frame.hour_idx, np.tile(np.arange(72, 216), counties)):
        raise ValueError("Missing or misordered forecast hours")
    times = pd.Timestamp("2026-03-11") + pd.to_timedelta(frame.hour_idx, unit="h")
    if not np.array_equal(times, frame.timestamp_et):
        raise ValueError("Timestamp/hour mismatch")
    return frame

def _raw(path, counties):
    frame = pd.read_csv(path, dtype={"fipsCode": str})
    frame["fipsCode"] = normalize_fips(frame.fipsCode)
    frame["timestamp_et"] = pd.to_datetime(frame.timestamp_et)
    frame = frame.sort_values(["fipsCode", "timestamp_et"]).reset_index(drop=True)
    hour = ((frame.timestamp_et - pd.Timestamp("2026-03-11")) / pd.Timedelta(hours=1)).to_numpy()
    if frame.fipsCode.nunique() != counties or not np.array_equal(hour, np.tile(np.arange(216), counties)):
        raise ValueError("Raw county/hour coverage mismatch")
    frame["hour_idx"] = hour.astype(int)
    return frame

def load_data(feature_dir=CACHE_DIR, cv_file=CV_FILE):
    """Read and validate complete input package; never builds or writes data."""
    feature_dir, cv_file = resolve_input(feature_dir), resolve_input(cv_file)
    paths = {name: feature_dir / f"{name}_{FEATURE_VERSION}.parquet" for name in
             ["features_train", "features_test", "targets_train", "meta_train", "meta_test"]}
    paths.update(feature_names=feature_dir/f"feature_names_{FEATURE_VERSION}.json", cv=cv_file,
                 component_targets=TARGETS_FILE, raw_train=RAW_TRAIN_FILE, raw_test=RAW_TEST_FILE,
                 submission=SUBMISSION_TEMPLATE)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    loaded_hashes = {key:sha256_file(path) for key,path in paths.items()}
    names = json.loads(paths["feature_names"].read_text())
    approved = json.loads((CACHE_DIR / f"feature_names_{FEATURE_VERSION}.json").read_text())
    if names != approved or len(names) != 163 or len(set(names)) != 163:
        raise ValueError("Feature schema is not frozen v1.5.6")
    tables = {key: pd.read_parquet(paths[key], engine="pyarrow") for key in
              ["features_train", "features_test", "targets_train", "meta_train", "meta_test"]}
    raw = {"train": _raw(RAW_TRAIN_FILE, 239), "test": _raw(RAW_TEST_FILE, 63)}
    for frame in raw.values():
        if not np.isfinite(frame.loc[frame.hour_idx<72,list(COMPONENTS)].to_numpy(dtype=float)).all():
            raise ValueError("Observed official components must all be finite")
    for split, count in [("train", 239), ("test", 63)]:
        x = tables[f"features_{split}"]
        meta = _metadata(tables[f"meta_{split}"], count)
        tables[f"meta_{split}"] = meta
        if list(x) != names or x.shape != (count*144, 163) or not x.index.equals(pd.RangeIndex(len(x))):
            raise ValueError("Frozen feature shape/order/index mismatch")
        if not all(pd.api.types.is_numeric_dtype(x[c]) for c in x) or np.isinf(x.to_numpy(dtype=float)).any():
            raise ValueError("Non-numeric or infinite feature")
        allowed = {f"{n}_at_t{h}h": meta.hour_idx + h > 215 for h in HORIZON_HOURS.values() for n in ["gust", "wind_speed", "t2m", "tp"]}
        allowed["hours_since_peak"] = x.osi_max_72h == 0
        for c in x:
            if not np.array_equal(x[c].isna(), allowed.get(c, np.zeros(len(x), dtype=bool))):
                raise ValueError(f"Unexpected missing feature: {c}")
        last = raw[split].query("hour_idx == 71").set_index("fipsCode")
        for c in COMPONENTS:
            if not np.array_equal(x[f"last_{c}"], meta.fipsCode.map(last[c])):
                raise ValueError(f"Last observed {c} no longer matches official hour71")
        forecast = raw[split].query("hour_idx >= 72").reset_index(drop=True)
        pd.testing.assert_frame_equal(meta[["fipsCode", "timestamp_et"]], forecast[["fipsCode", "timestamp_et"]])
    if set(tables["meta_train"].fipsCode) & set(tables["meta_test"].fipsCode):
        raise ValueError("Train/test county overlap")
    test = raw["test"]
    withheld = [c for c in test if c in COMPONENTS or c.startswith(("outage", "osi"))]
    if not test.loc[test.hour_idx >= 72, withheld].isna().all().all():
        raise ValueError("Test prediction-window outage data must remain missing")
    meta, y = tables["meta_train"], tables["targets_train"]
    if list(y) != list(HORIZONS) or not y.index.equals(pd.RangeIndex(len(meta))):
        raise ValueError("Official target schema/index mismatch")
    target_frame = pd.read_parquet(TARGETS_FILE, engine="pyarrow")
    target_frame["fipsCode"] = normalize_fips(target_frame.fipsCode)
    target_frame["timestamp_et"] = pd.to_datetime(target_frame.timestamp_et)
    component_columns = [component_target_name(c,h) for h in HORIZONS for c in COMPONENTS]
    if list(target_frame) != ["fipsCode", "timestamp_et", "hour_idx", *component_columns]:
        raise ValueError("Component target schema mismatch")
    pd.testing.assert_frame_equal(target_frame[["fipsCode", "timestamp_et", "hour_idx"]], meta[["fipsCode", "timestamp_et", "hour_idx"]], check_dtype=False)
    lookup = raw["train"].set_index(["fipsCode", "timestamp_et"])
    origins = pd.MultiIndex.from_frame(meta[["fipsCode", "timestamp_et"]])
    for h in HORIZONS:
        mask = expected_mask(meta, h)
        if not np.array_equal(np.isfinite(y[h]), mask) or not np.isnan(y.loc[~mask,h]).all():
            raise ValueError("Invalid official target mask")
        keys = pd.MultiIndex.from_arrays([meta.fipsCode, meta.timestamp_et + pd.to_timedelta(HORIZON_HOURS[h], unit="h")])
        np.testing.assert_array_equal(y[h], lookup.loc[origins,h])
        np.testing.assert_array_equal(y[h], lookup.osi.reindex(keys))
        parts = {}
        for c in COMPONENTS:
            parts[c] = target_frame[component_target_name(c,h)].to_numpy(dtype=float)
            np.testing.assert_array_equal(parts[c], lookup[c].reindex(keys))
        if np.max(np.abs(compose_osi(parts)[mask] - y[h].to_numpy()[mask])) > 5.1e-5:
            raise ValueError("Component composition differs beyond published precision")
    folds, row_folds, assignment = load_cv_folds(meta, cv_file)
    result = ExperimentData(tables["features_train"], y, meta, tables["features_test"], tables["meta_test"],
                            folds, row_folds, target_frame, assignment, paths, loaded_hashes)
    dummy = {h: np.where(expected_mask(result.meta_test,h), 0., np.nan) for h in HORIZONS}
    fill_submission(result.meta_test, dummy)
    if loaded_hashes != {key:sha256_file(path) for key,path in paths.items()}:
        raise ValueError("Inputs changed during preflight")
    return result

def build_component_targets(data, force=False):
    """Compatibility accessor: target generation is deliberately unavailable here."""
    if force:
        raise ValueError("This runner only reads frozen targets; build new packages separately")
    return data.component_targets.copy(), {"source": "frozen targets checked against raw data", "read_only": True}

def align_same_target(meta, predictions):
    if set(predictions) != set(HORIZONS) or meta.duplicated(["fipsCode", "timestamp_et"]).any():
        raise ValueError("Invalid alignment inputs")
    if "fold" in meta and meta.groupby("fipsCode").fold.nunique().max() != 1:
        raise ValueError("A county spans multiple outer folds")
    records = []
    for h in HORIZONS:
        mask = expected_mask(meta, h)
        require_predictions(predictions[h], mask)
        frame = meta.loc[mask, ["fipsCode", "timestamp_et"]].copy()
        frame["target_timestamp"] = frame.timestamp_et + pd.to_timedelta(HORIZON_HOURS[h], unit="h")
        frame["source_horizon"] = h
        frame["prediction"] = np.asarray(predictions[h])[mask]
        records.append(frame)
    long = pd.concat(records, ignore_index=True)
    if long.duplicated(["fipsCode", "target_timestamp", "source_horizon"]).any():
        raise ValueError("Duplicate alignment candidate")
    unique = long.groupby(["fipsCode", "target_timestamp"], sort=True, as_index=False).agg(
        prediction=("prediction", "mean"), candidate_count=("prediction", "size"),
        source_horizons=("source_horizon", lambda x: ",".join(sorted(x))))
    count = meta.fipsCode.nunique()
    if unique.candidate_count.value_counts().to_dict() != {k:v*count for k,v in {1:5,2:18,3:24,4:96}.items()}:
        raise ValueError("Missing expected target-hour candidates")
    lookup = unique.set_index(["fipsCode", "target_timestamp"]).prediction
    mapped = {}
    for h in HORIZONS:
        keys = pd.MultiIndex.from_arrays([meta.fipsCode, meta.timestamp_et + pd.to_timedelta(HORIZON_HOURS[h], unit="h")])
        values = lookup.reindex(keys).to_numpy(copy=True)
        values[~expected_mask(meta,h)] = np.nan
        mapped[h] = values
    return mapped, unique

def build_controls(meta, direct, components):
    aligned = {h:{} for h in HORIZONS}
    unique_frames = []
    for c in COMPONENTS:
        for h in HORIZONS:
            require_predictions(components[h][c], expected_mask(meta,h))
        values = {h:clip_component(components[h][c]) for h in HORIZONS}
        mapped, unique = align_same_target(meta, values)
        unique.insert(0,"component",c)
        unique_frames.append(unique)
        for h in HORIZONS:
            aligned[h][c] = mapped[h]
    controls = {m:{} for m in MODES}
    for h in HORIZONS:
        mask = expected_mask(meta,h)
        require_predictions(direct[h], mask)
        c0 = np.asarray(direct[h], dtype=float)
        c1 = compose_osi({c:clip_component(components[h][c]) for c in COMPONENTS})
        raw = (c0,c1,.5*(c0+c1),compose_osi(aligned[h]))
        for mode, values in zip(MODES[:4],raw):
            pred = post_process(values)
            pred[~mask] = np.nan
            controls[mode][h] = pred
        source = "C3_aligned_component" if HORIZON_HOURS[h] <= 6 else "C1_component_osi"
        controls["v18_rule"][h] = controls[source][h].copy()
    return controls, {"unique_components": pd.concat(unique_frames, ignore_index=True), "aligned_components": aligned}

def fill_submission(meta_test, predictions, template_file=SUBMISSION_TEMPLATE):
    submission = pd.read_csv(template_file, dtype={"fipsCode":str}, float_precision="round_trip")
    if len(submission) != len(meta_test) or set(predictions) != set(HORIZONS):
        raise ValueError("Submission coverage/columns mismatch")
    keys = pd.MultiIndex.from_arrays([normalize_fips(submission.fipsCode), pd.to_datetime(submission.timestamp_et)])
    source_keys = pd.MultiIndex.from_frame(meta_test[["fipsCode","timestamp_et"]])
    if keys.has_duplicates or source_keys.has_duplicates or set(keys) != set(source_keys):
        raise ValueError("Submission county-time keys mismatch")
    order = pd.Series(np.arange(len(meta_test)), index=source_keys).loc[keys].to_numpy()
    for h in HORIZONS:
        expected = expected_mask(meta_test,h)
        require_predictions(predictions[h], expected, exact_tail=True)
        if not np.array_equal(submission[h].notna(), expected[order]):
            raise ValueError("Official template target mask mismatch")
        values = np.asarray(predictions[h])[order]
        if ((values[expected[order]] < 0) | (values[expected[order]] > OSI_MAX)).any():
            raise ValueError("Submission value outside frozen output range")
        submission[h] = values
    return submission
