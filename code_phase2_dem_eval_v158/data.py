"""Validated v1.5.6 package loading and county-time graph construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COMPONENT_TARGETS_FILE,
    COMPONENT_WEIGHTS,
    DEFAULT_FEATURE_DIR,
    FEATURE_MANIFEST_CANDIDATES,
    FEATURE_NAMES_FILE,
    FEATURE_VERSION,
    FORBIDDEN_INPUT_COLUMNS,
    HORIZON_HOURS,
    PRED_END,
    PRED_START,
    TERRAIN_FILE,
)


TABLE_FILES = {
    "features_train": "features_train_v1.5.6.parquet",
    "features_test": "features_test_v1.5.6.parquet",
    "meta_train": "meta_train_v1.5.6.parquet",
    "meta_test": "meta_test_v1.5.6.parquet",
    "targets_train": "targets_train_v1.5.6.parquet",
}


@dataclass(frozen=True)
class FeatureBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    feature_names: tuple[str, ...]
    feature_dir: Path
    parquet_engine: str
    manifest: dict
    input_hash: str


@dataclass(frozen=True)
class SupervisionBundle:
    """Official OSI supervision kept separate from the feature bundle."""

    y_train: pd.DataFrame
    feature_dir: Path
    parquet_engine: str
    target_hash: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_fips(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.zfill(5)
    if (not values.str.fullmatch(r"\d{5}").all()) or values.eq("00000").any():
        raise ValueError("fipsCode must be a nonzero five-digit string")
    return values


def _timestamp_values(meta: pd.DataFrame) -> pd.Series:
    values = pd.to_datetime(meta["timestamp_et"], errors="raise")
    if getattr(values.dt, "tz", None) is not None:
        raise ValueError("timestamp_et must remain timezone-naive competition labels")
    return values


def _stable_key(meta: pd.DataFrame) -> pd.MultiIndex:
    fips = _normalise_fips(meta["fipsCode"])
    timestamps = _timestamp_values(meta).to_numpy()
    return pd.MultiIndex.from_arrays(
        [fips.to_numpy(dtype=object), timestamps],
        names=["fipsCode", "timestamp_et"],
    )


def _expected_feature_nan_mask(meta: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    hours = pd.to_numeric(meta["hour_idx"], errors="raise").to_numpy(dtype=int)
    allowed = np.zeros((len(meta), len(columns)), dtype=bool)
    for j, column in enumerate(columns):
        if column == "hours_since_peak":
            allowed[:, j] = True  # narrowed below using the observed OSI maximum
        for name in ("gust", "wind_speed", "t2m", "tp"):
            for horizon in (1, 6, 24, 48):
                if column == f"{name}_at_t{horizon}h":
                    allowed[:, j] = hours + horizon > PRED_END - 1
    if "hours_since_peak" in columns:
        j = columns.index("hours_since_peak")
        # The exact predicate is filled by validate_feature_nan_masks(), which
        # has access to the feature table. This placeholder prevents accidental
        # rejection before that column is handled.
        allowed[:, j] = False
    return pd.DataFrame(allowed, columns=columns, index=meta.index)


def _validate_feature_nan_masks(X: pd.DataFrame, meta: pd.DataFrame, source: str) -> None:
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(f"{source} contains Inf; only declared NaN is legal")
    expected = _expected_feature_nan_mask(meta, list(X.columns))
    if "hours_since_peak" in X:
        expected["hours_since_peak"] = X["osi_max_72h"].eq(0).to_numpy()
    actual = X.isna()
    mismatch = actual != expected
    if mismatch.to_numpy().any():
        where = np.argwhere(mismatch.to_numpy())[:5]
        examples = [(int(i), str(X.columns[j])) for i, j in where]
        raise ValueError(f"{source} has an invalid missing-value pattern, examples={examples}")


def _validate_meta(meta: pd.DataFrame, expected_rows: int, expected_counties: int, source: str) -> pd.DataFrame:
    if not isinstance(meta.index, pd.RangeIndex) or not np.array_equal(
        meta.index.to_numpy(), np.arange(len(meta))
    ):
        raise ValueError(f"{source} must have a RangeIndex starting at zero")
    required = {"fipsCode", "timestamp_et", "hour_idx", "stateAbbr", "hours_since_obs", "storm_phase"}
    missing = sorted(required - set(meta.columns))
    if missing:
        raise ValueError(f"{source} is missing metadata columns: {missing}")
    if len(meta) != expected_rows or meta["fipsCode"].nunique() != expected_counties:
        raise ValueError(f"{source} has unexpected shape or county count")
    out = meta.copy()
    out["fips_str"] = _normalise_fips(out["fipsCode"])
    out["hour_idx"] = pd.to_numeric(out["hour_idx"], errors="raise").astype(int)
    if out["hour_idx"].lt(PRED_START).any() or out["hour_idx"].ge(PRED_END).any():
        raise ValueError(f"{source} hour_idx must be in 72..215")
    if out.duplicated(["fips_str", "hour_idx"]).any() or _stable_key(out).duplicated().any():
        raise ValueError(f"{source} contains duplicate county-time keys")
    for _, group in out.groupby("fips_str", sort=False):
        if not np.array_equal(group["hour_idx"].to_numpy(), np.arange(PRED_START, PRED_END)):
            raise ValueError(f"{source} must contain exactly hours 72..215 per county")
    timestamps = _timestamp_values(out)
    expected = pd.Timestamp("2026-03-11 00:00") + pd.to_timedelta(out["hour_idx"], unit="h")
    if not np.array_equal(timestamps.to_numpy(), expected.to_numpy()):
        raise ValueError(f"{source} timestamp_et is not 2026-03-11 00:00 + hour_idx")
    return out


def validate_feature_columns(columns, source: str, expected: list[str] | tuple[str, ...] | None = None) -> None:
    names = [str(column) for column in columns]
    if len(names) != len(set(names)):
        raise ValueError(f"{source} contains duplicate feature names")
    forbidden = sorted(set(names) & FORBIDDEN_INPUT_COLUMNS)
    if forbidden:
        raise ValueError(f"{source} contains forbidden model inputs: {forbidden}")
    if expected is not None and names != list(expected):
        raise ValueError(f"{source} does not match the ordered frozen feature schema")


def _load_manifest(feature_dir: Path) -> dict:
    paths = [feature_dir / name for name in FEATURE_MANIFEST_CANDIDATES if (feature_dir / name).exists()]
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one package manifest in {feature_dir}: {FEATURE_MANIFEST_CANDIDATES}"
        )
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("feature package manifest has the wrong feature_version")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("feature package manifest must contain a files hash map")
    return manifest


def load_feature_bundle(
    feature_dir: str | Path | None = None,
    parquet_engine: str = "pyarrow",
    *,
    validate_targets: bool = False,
) -> FeatureBundle:
    """Read the feature/meta side of one immutable five-table package.

    The target table is required to exist and have a manifest entry, but its
    bytes are not read by the default feature-only path.  Call
    :func:`load_supervision` explicitly when a training/evaluation stage is
    allowed to access labels.
    """

    if parquet_engine not in {"pyarrow", "fastparquet"}:
        raise ValueError("parquet_engine must be pyarrow or fastparquet")
    root = Path(feature_dir) if feature_dir is not None else DEFAULT_FEATURE_DIR
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Feature package does not exist: {root}")
    manifest = _load_manifest(root)
    names_path = root / FEATURE_NAMES_FILE
    if not names_path.exists():
        raise FileNotFoundError(f"Missing frozen feature schema: {names_path}")
    feature_names = tuple(json.loads(names_path.read_text(encoding="utf-8")))
    if len(feature_names) != 163:
        raise ValueError(f"Expected 163 v1.5.6 features, found {len(feature_names)}")
    validate_feature_columns(feature_names, str(names_path))
    expected_names_entry = manifest["files"].get(names_path.name)
    if not isinstance(expected_names_entry, dict) or not expected_names_entry.get("sha256"):
        raise ValueError(f"manifest has no required hash for {names_path.name}")
    expected_names_hash = expected_names_entry["sha256"]
    if _sha256(names_path).lower() != str(expected_names_hash).lower():
        raise ValueError("hash mismatch for frozen feature_names schema")

    paths = {key: root / filename for key, filename in TABLE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete feature package, missing: {missing}")
    for key, path in paths.items():
        entry = manifest["files"].get(path.name)
        if not isinstance(entry, dict) or not entry.get("sha256"):
            raise ValueError(f"manifest has no required hash for {path.name}")
        # Independent inference must not read the target file, even merely to
        # recompute its hash.  load_supervision() performs that check in a
        # supervised process.
        if key == "targets_train" and not validate_targets:
            continue
        expected_hash = entry["sha256"]
        if _sha256(path).lower() != str(expected_hash).lower():
            raise ValueError(f"Hash mismatch for frozen package table {path.name}")

    # Deliberately do not read targets here.  A feature object must be safe to
    # hand to an inference-only consumer; labels are loaded by
    # load_supervision() after the caller has explicitly entered a supervised
    # stage.  The target file remains part of the package contract and is
    # explicitly checked by load_supervision().
    frames = {
        key: pd.read_parquet(path, engine=parquet_engine)
        for key, path in paths.items()
        if key != "targets_train" or validate_targets
    }
    X_train = frames["features_train"]
    X_test = frames["features_test"]
    meta_train = _validate_meta(frames["meta_train"], 34416, 239, "meta_train")
    meta_test = _validate_meta(frames["meta_test"], 9072, 63, "meta_test")
    y_train = frames.get("targets_train")
    if not isinstance(X_train.index, pd.RangeIndex) or not isinstance(X_test.index, pd.RangeIndex):
        raise ValueError("feature tables must have RangeIndex")
    if validate_targets and not isinstance(y_train.index, pd.RangeIndex):
        raise ValueError("targets_train must have RangeIndex")
    if validate_targets and list(y_train.columns) != list(HORIZON_HOURS):
        raise ValueError("targets_train schema/order does not match the frozen horizon contract")
    validate_feature_columns(X_train.columns, "features_train", feature_names)
    validate_feature_columns(X_test.columns, "features_test", feature_names)
    if len(X_train) != len(meta_train) or len(X_test) != len(meta_test):
        raise ValueError("features/meta/targets row counts do not match")
    if validate_targets and len(y_train) != len(meta_train):
        raise ValueError("features/meta/targets row counts do not match")
    if not _stable_key(meta_train).equals(_stable_key(meta_train).sort_values()):
        # The frozen package is expected to be county/hour ordered.  Rejecting
        # here avoids silently relying on a positional DataFrame index.
        raise ValueError("meta_train is not in deterministic county/time order")
    if not _stable_key(meta_test).equals(_stable_key(meta_test).sort_values()):
        raise ValueError("meta_test is not in deterministic county/time order")
    if set(meta_train["fips_str"]) & set(meta_test["fips_str"]):
        raise ValueError("train/test county sets overlap")
    _validate_feature_nan_masks(X_train, meta_train, "features_train")
    _validate_feature_nan_masks(X_test, meta_test, "features_test")
    table_shapes = manifest.get("tables", {})
    expected_shapes = {
        "features_train": (34416, 163),
        "features_test": (9072, 163),
        "meta_train": (34416, 7),
        "meta_test": (9072, 6),
        "targets_train": (34416, 4),
    }
    for name, expected_shape in expected_shapes.items():
        declared = table_shapes.get(name)
        if declared is not None:
            declared_shape = (int(declared.get("rows", -1)), int(declared.get("columns", -1)))
            if declared_shape != expected_shape:
                raise ValueError(f"manifest shape mismatch for {name}: {declared_shape}")
        if name == "targets_train" and not validate_targets:
            continue
        actual_shape = frames[name].shape
        if actual_shape != expected_shape:
            raise ValueError(f"unexpected frozen table shape for {name}: {actual_shape}")
    if validate_targets:
        _validate_official_targets(y_train, meta_train)

    digest = hashlib.sha256()
    for key in TABLE_FILES:
        if key == "targets_train":
            continue
        digest.update(_sha256(paths[key]).encode())
    digest.update(_sha256(names_path).encode())
    return FeatureBundle(
        X_train=X_train,
        X_test=X_test,
        meta_train=meta_train,
        meta_test=meta_test,
        feature_names=feature_names,
        feature_dir=root,
        parquet_engine=parquet_engine,
        manifest=manifest,
        input_hash=digest.hexdigest(),
    )


def _validate_official_targets(y_train: pd.DataFrame, meta_train: pd.DataFrame) -> None:
    if not isinstance(y_train.index, pd.RangeIndex):
        raise ValueError("targets_train must have RangeIndex")
    if list(y_train.columns) != list(HORIZON_HOURS) or len(y_train) != len(meta_train):
        raise ValueError("targets_train schema/order or row count is invalid")
    for horizon, hours in HORIZON_HOURS.items():
        expected_nan = meta_train["hour_idx"].to_numpy(dtype=int) + hours > PRED_END - 1
        actual_nan = y_train[horizon].isna().to_numpy()
        if not np.array_equal(actual_nan, expected_nan):
            raise ValueError(f"{horizon} target missing-value pattern is invalid")
        values = y_train[horizon].to_numpy(dtype=float)
        valid_values = values[~actual_nan]
        if (
            np.isinf(values).any()
            or not np.isfinite(valid_values).all()
            or (valid_values < 0).any()
            or (valid_values > 0.65).any()
        ):
            raise ValueError(f"{horizon} contains invalid values in its valid window")


def load_supervision(
    feature_dir: str | Path | None = None,
    parquet_engine: str = "pyarrow",
    *,
    feature_bundle: FeatureBundle | None = None,
) -> SupervisionBundle:
    """Load official OSI labels only for an explicitly supervised stage."""

    bundle = feature_bundle or load_feature_bundle(
        feature_dir=feature_dir, parquet_engine=parquet_engine, validate_targets=False
    )
    if bundle.parquet_engine != parquet_engine:
        raise ValueError("supervision and feature package engines differ")
    path = bundle.feature_dir / TABLE_FILES["targets_train"]
    manifest_entry = bundle.manifest.get("files", {}).get(path.name, {})
    expected_hash = str(manifest_entry.get("sha256", ""))
    if not expected_hash or _sha256(path).lower() != expected_hash.lower():
        raise ValueError("hash mismatch for frozen targets_train table")
    y_train = pd.read_parquet(path, engine=parquet_engine)
    _validate_official_targets(y_train, bundle.meta_train)
    return SupervisionBundle(
        y_train=y_train,
        feature_dir=bundle.feature_dir,
        parquet_engine=parquet_engine,
        target_hash=_sha256(path),
    )


def load_cached_data(
    feature_dir: str | Path | None = None,
    parquet_engine: str = "pyarrow",
    *,
    include_supervision: bool = False,
):
    """Compatibility loader; labels require an explicit opt-in."""

    bundle = load_feature_bundle(feature_dir=feature_dir, parquet_engine=parquet_engine)
    if include_supervision:
        supervision = load_supervision(feature_bundle=bundle, parquet_engine=parquet_engine)
        return bundle.X_train, bundle.X_test, bundle.meta_train, bundle.meta_test, supervision.y_train
    return bundle


def load_terrain_features() -> pd.DataFrame:
    if not TERRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing terrain file: {TERRAIN_FILE}")
    terrain = pd.read_csv(TERRAIN_FILE, dtype={"fipsCode": str})
    terrain["fips_str"] = _normalise_fips(terrain["fipsCode"])
    required = [
        "elevation_mean_m", "elevation_std_m", "elevation_min_m",
        "elevation_max_m", "slope_mean_deg", "slope_std_deg", "terrain_ruggedness",
    ]
    if terrain["fips_str"].duplicated().any() or not set(required).issubset(terrain.columns):
        raise ValueError("terrain coverage or schema is invalid")
    values = terrain[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("terrain contains non-finite values")
    return terrain.set_index("fips_str")[required].sort_index()


def _legal_nan_values(
    values: np.ndarray,
    meta_row: pd.Series,
    names: list[str],
    peak_zero: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    if np.isinf(values).any():
        raise ValueError("GAT input contains Inf")
    allowed = _expected_feature_nan_mask(pd.DataFrame([meta_row]), names).iloc[0].to_numpy()
    if "hours_since_peak" in names:
        allowed[names.index("hours_since_peak")] = bool(peak_zero)
    if not np.array_equal(np.isnan(values), allowed):
        raise ValueError("GAT input contains an invalid missing-value pattern")
    values[np.isnan(values)] = 0.0
    return values


def make_county_time_view(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    base_train: np.ndarray,
    base_test: np.ndarray,
    horizon: str,
    coords: np.ndarray,
    terrain: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...] | None = None,
):
    """Build a finite [144, 302, 173] tensor with explicit row mappings."""

    names = list(feature_names or X_train.columns)
    validate_feature_columns(X_train.columns, "GAT source features", names)
    validate_feature_columns(X_test.columns, "GAT source test features", names)
    if horizon not in HORIZON_HOURS:
        raise ValueError(f"unknown horizon: {horizon}")
    base_train = np.asarray(base_train, dtype=float)
    base_test = np.asarray(base_test, dtype=float)
    if len(base_train) != len(X_train) or len(base_test) != len(X_test):
        raise ValueError("base prediction length does not match its feature table")
    if not np.isfinite(base_train).all() or not np.isfinite(base_test).all():
        raise ValueError("base predictions must be finite for all 144 rows")
    all_fips = sorted(set(meta_train["fips_str"]) | set(meta_test["fips_str"]))
    node_of = {fips: i for i, fips in enumerate(all_fips)}
    n_time, n_nodes = PRED_END - PRED_START, len(all_fips)
    terrain_names = list(terrain.columns)
    if len(terrain_names) != 7 or set(all_fips) - set(terrain.index):
        raise ValueError("terrain does not cover exactly all graph nodes")
    features = np.empty((n_time, n_nodes, 1 + len(names) + 2 + len(terrain_names)), dtype=np.float32)
    train_rows = np.full((n_time, n_nodes), -1, dtype=np.int64)
    test_rows = np.full((n_time, n_nodes), -1, dtype=np.int64)

    def fill(X, meta, bases, rows):
        if not isinstance(meta.index, pd.RangeIndex):
            raise ValueError("metadata row positions must be an explicit RangeIndex")
        for row_id, row in meta.iterrows():
            t = int(row["hour_idx"]) - PRED_START
            node = node_of[row["fips_str"]]
            if rows[t, node] != -1:
                raise ValueError("duplicate county-time row while building graph input")
            source_row = X.iloc[row_id]
            values = _legal_nan_values(
                source_row[names].to_numpy(dtype=float), row, names,
                peak_zero=float(source_row["osi_max_72h"]) == 0.0,
            )
            features[t, node, 0] = float(bases[row_id])
            features[t, node, 1:1 + len(names)] = values
            features[t, node, 1 + len(names):1 + len(names) + 2] = coords[node]
            start = 1 + len(names) + 2
            features[t, node, start:start + len(terrain_names)] = terrain.loc[row["fips_str"], terrain_names]
            rows[t, node] = row_id

    fill(X_train, meta_train, base_train, train_rows)
    fill(X_test, meta_test, base_test, test_rows)
    if (train_rows < 0).any() or (test_rows < 0).any():
        raise ValueError("county-time graph construction has missing rows")
    if not np.isfinite(features).all():
        raise ValueError("graph input contains non-finite values")
    base = features[:, :, 0].copy()
    return features, base, train_rows, test_rows, all_fips, names


def add_neighbor_feature_aggregates(features, names, edge_index, horizon):
    """Append 16 approved neighbor means and 16 neighbor-minus-own deltas."""

    templates = [
        "last_osi", "last_P_t", "last_D_t", "last_N_t", "last_R_t",
        "osi_mean_72h", "osi_max_72h", "osi_trend_last6h",
        "gust_t", "wind_speed_t", "tp_t", "rain_t",
        f"gust_max_next_{HORIZON_HOURS[horizon]}h", f"gust_mean_next_{HORIZON_HOURS[horizon]}h",
        f"wind_speed_max_next_{HORIZON_HOURS[horizon]}h", f"total_tp_next_{HORIZON_HOURS[horizon]}h",
    ]
    if len(templates) != 16 or any(name not in names for name in templates):
        raise ValueError("approved 16-column neighbor source schema is incomplete")
    indices = [names.index(name) for name in templates]
    source = np.asarray(features[:, :, 1:1 + len(names)][:, :, indices], dtype=np.float32)
    src, dst = np.asarray(edge_index, dtype=np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    count = np.bincount(dst, minlength=features.shape[1]).astype(np.float32)
    if (count == 0).any():
        raise ValueError("every graph node must have a non-self neighbor")
    sums = np.zeros_like(source)
    for j in range(len(indices)):
        np.add.at(sums[:, :, j], (np.arange(features.shape[0])[:, None], dst[None, :]), source[:, src, j])
    neighbour = sums / count[None, :, None]
    delta = neighbour - source
    extra = np.concatenate([neighbour, delta], axis=2)
    result = np.concatenate([features, extra], axis=2)
    if result.shape[-1] != 205 or not np.isfinite(result).all():
        raise ValueError(f"expected finite 205-dimensional graph input, got {result.shape}")
    extra_names = [f"neighbor_mean_{n}" for n in templates] + [f"neighbor_delta_{n}" for n in templates]
    return result, extra_names


def load_component_targets(meta_train: pd.DataFrame, parquet_engine: str = "pyarrow") -> dict:
    """Load component supervision by a one-to-one normalized county-time key."""

    if not COMPONENT_TARGETS_FILE.exists():
        raise FileNotFoundError(f"Missing component target package: {COMPONENT_TARGETS_FILE}")
    frame = pd.read_parquet(COMPONENT_TARGETS_FILE, engine=parquet_engine)
    required = ["fipsCode", "timestamp_et", "hour_idx"]
    for horizon in HORIZON_HOURS:
        for component in ("P_t", "N_t", "D_t", "R_t"):
            required.append(f"{component}_target_{horizon.replace('osi_target_', '')}")
    if list(frame.columns) != required:
        raise ValueError("component target schema mismatch")
    frame = frame.copy()
    frame["fips_str"] = _normalise_fips(frame["fipsCode"])
    frame["hour_idx"] = pd.to_numeric(frame["hour_idx"], errors="raise").astype(int)
    if frame["hour_idx"].lt(PRED_START).any() or frame["hour_idx"].ge(PRED_END).any():
        raise ValueError("component target hour_idx must be in 72..215")
    timestamps = _timestamp_values(frame)
    expected_timestamps = pd.Timestamp("2026-03-11 00:00") + pd.to_timedelta(
        frame["hour_idx"], unit="h"
    )
    if not np.array_equal(timestamps.to_numpy(), expected_timestamps.to_numpy()):
        raise ValueError("component target timestamp_et is not aligned to hour_idx")
    keys = _stable_key(frame)
    if keys.duplicated().any():
        raise ValueError("component target county-time keys are not unique")
    target_keys = _stable_key(meta_train)
    if not target_keys.isin(keys).all() or len(target_keys) != len(keys):
        raise ValueError("component target keys do not exactly match training metadata")
    indexed = frame.copy()
    indexed.index = keys
    aligned = indexed.reindex(target_keys)
    if not aligned.index.equals(target_keys):
        raise ValueError("component target reindex failed")
    result = {}
    for horizon, hours in HORIZON_HOURS.items():
        result[horizon] = {}
        expected_nan = meta_train["hour_idx"].to_numpy() + hours > PRED_END - 1
        for component in ("P_t", "N_t", "D_t", "R_t"):
            column = f"{component}_target_{horizon.replace('osi_target_', '')}"
            values = aligned[column].to_numpy(dtype=float)
            valid = values[~expected_nan]
            if (
                np.isinf(values).any()
                or not np.array_equal(np.isnan(values), expected_nan)
                or not np.isfinite(valid).all()
                or (valid < 0).any()
                or (valid > 1).any()
            ):
                raise ValueError(f"invalid missing/non-finite component target: {column}")
            result[horizon][component] = values
    return result


def validate_component_recomposition(component_targets: dict, official_targets: pd.DataFrame) -> dict:
    """Check the public-data component reconstruction against official OSI."""

    if list(official_targets.columns) != list(HORIZON_HOURS):
        raise ValueError("official target schema is not the frozen horizon order")
    diagnostics = {}
    for horizon in HORIZON_HOURS:
        parts = {
            component: np.asarray(component_targets[horizon][component], dtype=float)
            for component in COMPONENT_WEIGHTS
        }
        shape = parts["P_t"].shape
        if any(values.shape != shape for values in parts.values()):
            raise ValueError(f"component target shapes differ for {horizon}")
        composed = np.zeros(shape, dtype=float)
        for component, weight in COMPONENT_WEIGHTS.items():
            composed += float(weight) * parts[component]
        composed = np.maximum(composed, 0.0)
        official = official_targets[horizon].to_numpy(dtype=float)
        expected_nan = np.isnan(official)
        component_nan = np.column_stack([np.isnan(parts[c]) for c in COMPONENT_WEIGHTS]).any(axis=1)
        if not np.array_equal(component_nan, expected_nan):
            raise ValueError(f"component/official missing mask differs for {horizon}")
        valid = ~expected_nan
        if not np.isfinite(composed[valid]).all() or not np.isfinite(official[valid]).all():
            raise ValueError(f"component/official values are non-finite for {horizon}")
        error = composed[valid] - official[valid]
        max_abs = float(np.max(np.abs(error))) if valid.any() else 0.0
        if max_abs > 5.1e-5:
            raise ValueError(f"component recomposition differs from official OSI by {max_abs}")
        diagnostics[horizon] = {
            "n": int(valid.sum()),
            "max_abs_error": max_abs,
            "rmse": float(np.sqrt(np.mean(error * error))) if valid.any() else 0.0,
        }
    return diagnostics
