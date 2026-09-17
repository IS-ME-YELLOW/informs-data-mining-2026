"""Scope-isolated LightGBM base models for dem_v158_nested_v2."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    COMPONENTS,
    COMPONENT_WEIGHTS,
    HORIZON_HOURS,
    HORIZONS,
    LGBM_EARLY_STOPPING,
    LGBM_ROUNDS,
    N_FOLDS,
    PRED_END,
    make_lgbm_params,
)
from data import load_component_targets, validate_feature_columns
from metrics import post_process_osi


def canonical_fold_set(S, *, minimum: int | None = None, maximum: int | None = None) -> tuple[int, ...]:
    """Normalize and validate a supervision scope."""

    try:
        result = tuple(sorted({int(value) for value in S}))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid fold scope: {S!r}") from exc
    if any(value < 0 or value >= N_FOLDS for value in result):
        raise ValueError(f"fold scope must use only 0..{N_FOLDS - 1}: {result}")
    if minimum is not None and len(result) < minimum:
        raise ValueError(f"scope {result} is too small; minimum is {minimum}")
    if maximum is not None and len(result) > maximum:
        raise ValueError(f"scope {result} is too large; maximum is {maximum}")
    return result


def scoped_seed(global_seed: int, stage: str, S, horizon: str, component: str, probe_fold: str | int) -> int:
    """Derive a stable seed without Python's process-randomized hash()."""

    scope = canonical_fold_set(S)
    payload = json.dumps(
        [int(global_seed), str(stage), scope, str(horizon), str(component), str(probe_fold)],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


class ScopedLabels:
    """A supervision view that actively refuses reads from out-of-scope rows."""

    def __init__(self, values: np.ndarray, row_fold: np.ndarray, allowed_folds):
        self._values = np.asarray(values, dtype=float)
        self._row_fold = np.asarray(row_fold, dtype=int)
        self.allowed_folds = canonical_fold_set(allowed_folds)
        if len(self._values) != len(self._row_fold):
            raise ValueError("label values and row_fold lengths differ")

    def __len__(self):
        return len(self._row_fold)

    def take(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=int)
        if rows.size and not np.isin(self._row_fold[rows], self.allowed_folds).all():
            raise PermissionError(
                f"supervision access outside allowed folds {self.allowed_folds}"
            )
        return self._values[rows]

    def finite_rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=int)
        values = self.take(rows)
        return rows[np.isfinite(values)]


class SupervisionStore:
    """Private label bank with scoped access and explicit evaluation access."""

    def __init__(self, row_fold: np.ndarray, y_train=None, component_targets=None):
        self._row_fold = np.asarray(row_fold, dtype=int)
        self._official = None if y_train is None else {
            str(column): np.asarray(y_train[column], dtype=float)
            for column in y_train.columns
        }
        self._components = component_targets

    @property
    def available(self) -> bool:
        return self._official is not None

    def training_values(self, horizon: str, component: str = "osi") -> np.ndarray:
        if component == "osi":
            if self._official is None or horizon not in self._official:
                raise PermissionError("official supervision is unavailable in this inference context")
            return self._official[horizon]
        if self._components is None:
            raise PermissionError("component supervision is unavailable in this inference context")
        try:
            return np.asarray(self._components[horizon][component], dtype=float)
        except KeyError as exc:
            raise ValueError(f"missing component supervision for {horizon}/{component}") from exc

    def scoped(self, S, horizon: str, component: str = "osi") -> ScopedLabels:
        return ScopedLabels(self.training_values(horizon, component), self._row_fold, S)


@dataclass
class BaseFit:
    model: lgb.Booster
    scope: tuple[int, ...]
    horizon: str
    component: str
    final_rounds: int
    best_iterations: tuple[int, ...]
    probe_records: tuple[dict, ...]
    model_id: str
    seed: int

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.model.predict(X, num_iteration=self.final_rounds), dtype=float)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite base prediction from {self.model_id}")
        if self.component != "osi":
            values = np.clip(values, 0.0, 1.0)
        return values


def _fit_probe(X_train, y_train, X_valid, y_valid, seed: int):
    if len(X_train) == 0 or len(X_valid) == 0:
        raise ValueError("LightGBM probe received an empty train or validation set")
    params = make_lgbm_params(seed)
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, free_raw_data=False)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=LGBM_ROUNDS,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
    )
    best = int(model.best_iteration or 0)
    if best < 1:
        raise RuntimeError("LightGBM did not produce a positive best_iteration")
    return model, best


def _fit_fixed(X_train, y_train, rounds: int, seed: int):
    if len(X_train) == 0:
        raise ValueError("LightGBM refit received an empty training set")
    model = lgb.train(
        make_lgbm_params(seed),
        lgb.Dataset(X_train, label=y_train, free_raw_data=False),
        num_boost_round=int(rounds),
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def _scope_rows(row_fold: np.ndarray, S) -> dict[int, np.ndarray]:
    scope = canonical_fold_set(S)
    return {fold: np.flatnonzero(row_fold == fold) for fold in range(N_FOLDS) if fold in scope}


def fit_base(
    X: pd.DataFrame,
    target: np.ndarray | ScopedLabels,
    row_fold: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    S,
    horizon: str,
    component: str = "osi",
    global_seed: int = 42,
    valid_mask: np.ndarray | None = None,
) -> BaseFit:
    """Fit one scalar base model using labels from exactly S.

    Every probe uses one validation fold in S, and the public predictor is a
    fixed-round refit on all valid rows in S. Probe models are never used for
    predictions returned to callers.
    """

    scope = canonical_fold_set(S, minimum=2, maximum=5)
    if horizon not in HORIZON_HOURS:
        raise ValueError(f"unknown horizon: {horizon}")
    if component != "osi" and component not in COMPONENTS:
        raise ValueError(f"unknown scalar base target: {component}")
    validate_feature_columns(X.columns, "LightGBM input")
    row_fold = np.asarray(row_fold, dtype=int)
    if isinstance(target, ScopedLabels):
        labels = target
        if labels.allowed_folds != scope:
            raise PermissionError("fit_base target accessor scope differs from declared S")
        target_length = len(labels)
    else:
        target = np.asarray(target, dtype=float)
        labels = ScopedLabels(target, row_fold, scope)
        target_length = len(target)
    if len(X) != target_length or len(X) != len(row_fold):
        raise ValueError("X, target and row_fold lengths differ")
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (target_length,):
            raise ValueError("valid_mask shape differs from target")
        scope_rows = np.flatnonzero(np.isin(row_fold, np.asarray(scope, dtype=int)))
        valid_scope_rows = scope_rows[valid_mask[scope_rows]]
        if not np.isfinite(labels.take(valid_scope_rows)).all():
            raise ValueError("valid-window supervision contains a missing or non-finite target")
    fold_rows = _scope_rows(row_fold, scope)
    probe_records = []
    best_iterations = []
    for q in scope:
        train_full = np.concatenate([fold_rows[r] for r in scope if r != q])
        valid_full = fold_rows[q]
        train_rows = labels.finite_rows(train_full)
        valid_rows = labels.finite_rows(valid_full)
        if len(train_rows) == 0 or len(valid_rows) == 0:
            raise ValueError(f"empty valid LightGBM probe for scope={scope}, q={q}")
        probe_seed = scoped_seed(global_seed, "base_probe", scope, horizon, component, q)
        _, best = _fit_probe(
            X.iloc[train_rows], labels.take(train_rows),
            X.iloc[valid_rows], labels.take(valid_rows), probe_seed,
        )
        best_iterations.append(best)
        probe_records.append({
            "probe_fold": q,
            "train_folds": tuple(r for r in scope if r != q),
            "valid_folds": (q,),
            "train_rows": int(len(train_rows)),
            "valid_rows": int(len(valid_rows)),
            "best_iteration": best,
            "seed": probe_seed,
        })
    final_rounds = max(1, int(np.mean(best_iterations)))
    final_rows = labels.finite_rows(np.concatenate([fold_rows[r] for r in scope]))
    final_seed = scoped_seed(global_seed, "base_refit", scope, horizon, component, "refit")
    model = _fit_fixed(X.iloc[final_rows], labels.take(final_rows), final_rounds, final_seed)
    model_id = f"base:{component}:{horizon}:S{','.join(map(str, scope))}"
    return BaseFit(
        model=model,
        scope=scope,
        horizon=horizon,
        component=component,
        final_rounds=final_rounds,
        best_iterations=tuple(best_iterations),
        probe_records=tuple(probe_records),
        model_id=model_id,
        seed=final_seed,
    )


def load_cv_folds(meta: pd.DataFrame, cv_file: str | Path | None = None, n_folds: int = N_FOLDS):
    """Load the fixed county folds and return row positions, not index labels."""

    from config import CV_FILE, CV_VERSION

    path = Path(cv_file) if cv_file is not None else CV_FILE
    assignment = pd.read_csv(path, dtype={"fipsCode": str})
    required = {"fipsCode", "stateAbbr", "severity_tier", "fold"}
    if set(assignment.columns) != required:
        raise ValueError(f"{CV_VERSION} CV schema must be exactly {sorted(required)}")
    assignment["fips_str"] = assignment["fipsCode"].astype(str).str.zfill(5)
    if not assignment["fips_str"].str.fullmatch(r"\d{5}").all():
        raise ValueError("CV fipsCode values must be five-digit strings")
    assignment["fold"] = pd.to_numeric(assignment["fold"], errors="raise").astype(int)
    assignment["severity_tier"] = pd.to_numeric(assignment["severity_tier"], errors="raise").astype(int)
    if assignment["fips_str"].duplicated().any() or set(assignment["fold"]) != set(range(n_folds)):
        raise ValueError("CV assignment has duplicate counties or wrong fold IDs")
    expected_counts = {0: 48, 1: 47, 2: 48, 3: 48, 4: 48}
    actual_counts = assignment["fold"].value_counts().sort_index().to_dict()
    if n_folds == 5 and actual_counts != expected_counts:
        raise ValueError(f"fixed CV county counts differ: {actual_counts}")
    meta_fips = meta["fips_str"].astype(str).str.zfill(5)
    county_meta = meta.assign(fips_str=meta_fips)[["fips_str", "stateAbbr", "severity_tier"]]
    county_meta = county_meta.drop_duplicates().set_index("fips_str").sort_index()
    assigned = assignment.set_index("fips_str")[["stateAbbr", "severity_tier", "fold"]].sort_index()
    if not county_meta.index.equals(assigned.index):
        raise ValueError("CV county set does not exactly match training metadata")
    if not county_meta[["stateAbbr", "severity_tier"]].equals(assigned[["stateAbbr", "severity_tier"]]):
        raise ValueError("CV state/severity metadata differs from the CV file")
    row_fold = meta_fips.map(assigned["fold"])
    if row_fold.isna().any():
        raise ValueError("CV assignment does not cover every training row")
    row_fold = row_fold.to_numpy(dtype=int)
    return [(np.flatnonzero(row_fold != i), np.flatnonzero(row_fold == i)) for i in range(n_folds)]


class BaseModelStore:
    """Memoize only complete, identity-keyed scope models within one run."""

    def __init__(
        self, X_train, X_test, meta_train, folds, y_train=None,
        global_seed=42, model_dir=None, parquet_engine="pyarrow",
        component_targets=None, supervision_store=None,
        feature_package_hash=None, feature_schema_hash=None,
        protocol="dem_v158_nested_v2",
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.meta_train = meta_train
        self.folds = folds
        self.row_fold = np.empty(len(meta_train), dtype=int)
        for fold, (_, valid) in enumerate(folds):
            self.row_fold[valid] = fold
        self.global_seed = int(global_seed)
        self.model_dir = Path(model_dir) if model_dir is not None else None
        if parquet_engine not in {"pyarrow", "fastparquet"}:
            raise ValueError("parquet_engine must be pyarrow or fastparquet")
        self.parquet_engine = parquet_engine
        self.feature_package_hash = feature_package_hash
        self.feature_schema_hash = feature_schema_hash
        self.protocol = str(protocol)
        self.supervision = supervision_store or SupervisionStore(
            self.row_fold, y_train=y_train, component_targets=component_targets
        )
        self._fits: dict[tuple[str, str, tuple[int, ...]], BaseFit] = {}

    def get(self, S, horizon: str, component: str = "osi") -> BaseFit:
        scope = canonical_fold_set(S, minimum=2, maximum=5)
        key = (horizon, component, scope)
        if key not in self._fits:
            target = self.supervision.scoped(scope, horizon, component)
            fit = fit_base(
                self.X_train, target, self.row_fold, self.folds, scope, horizon,
                component=component, global_seed=self.global_seed,
                valid_mask=(self.meta_train["hour_idx"].to_numpy(dtype=int)
                            + HORIZON_HOURS[horizon] <= PRED_END - 1),
            )
            self._fits[key] = fit
            if self.model_dir is not None:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                safe = f"{component}_{horizon.replace('osi_target_', '')}_S{'-'.join(map(str, scope))}"
                path = self.model_dir / f"{safe}.txt"
                temporary = path.with_name(f".{path.name}.tmp")
                try:
                    fit.model.save_model(str(temporary), num_iteration=fit.final_rounds)
                    os.replace(temporary, path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
        return self._fits[key]

    @property
    def fits(self):
        return dict(self._fits)

    def load_from_run(self, run_dir: str | Path) -> None:
        """Load only current-protocol fixed-round base boosters from a run."""

        run_root = Path(run_dir)
        root = run_root / "models" / "base"
        if not root.is_dir():
            raise FileNotFoundError(f"missing base model directory: {root}")
        manifest_path = run_root / "base_fit_manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing base-fit identity manifest: {manifest_path}")
        manifest = pd.read_parquet(manifest_path, engine=self.parquet_engine)
        required_manifest = {
            "model_path", "model_id", "protocol", "feature_package_hash",
            "feature_schema_hash", "model_sha256", "final_rounds", "seed",
            "component", "horizon",
        }
        if not required_manifest.issubset(manifest.columns):
            raise ValueError("base-fit manifest is missing cache identity fields")
        by_name = {}
        for record in manifest.to_dict("records"):
            model_path = str(record["model_path"]).replace("\\", "/")
            by_name[Path(model_path).name] = record
            if record["protocol"] != self.protocol:
                raise ValueError("base-fit manifest protocol mismatch")
            if self.feature_package_hash is not None and record["feature_package_hash"] != self.feature_package_hash:
                raise ValueError("base-fit manifest feature package mismatch")
            if self.feature_schema_hash is not None and record["feature_schema_hash"] != self.feature_schema_hash:
                raise ValueError("base-fit manifest feature schema mismatch")
        for path in sorted(root.glob("*.txt")):
            record = by_name.get(path.name)
            if record is None:
                raise ValueError(f"base model is absent from the identity manifest: {path.name}")
            if str(record["model_path"]).replace("\\", "/") != f"models/base/{path.name}":
                raise ValueError(f"base model path identity mismatch: {path.name}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if str(record.get("model_sha256", "")).lower() != digest.lower():
                raise ValueError(f"base model hash mismatch: {path.name}")
            stem = path.stem
            if "_S" not in stem:
                raise ValueError(f"unrecognized base model filename: {path.name}")
            left, scope_text = stem.rsplit("_S", 1)
            scope = tuple(int(value) for value in scope_text.split("-") if value != "")
            scope = canonical_fold_set(scope, minimum=2, maximum=5)
            component = next((value for value in COMPONENTS if left.startswith(value + "_")), None)
            if component is None:
                component = "osi" if left.startswith("osi_") else None
            if component is None:
                raise ValueError(f"unrecognized base model target: {path.name}")
            prefix = f"{component}_"
            horizon_short = left[len(prefix):]
            horizon = f"osi_target_t{horizon_short}h"
            if horizon not in HORIZON_HOURS:
                raise ValueError(f"unrecognized base model horizon: {path.name}")
            model = lgb.Booster(model_file=str(path))
            rounds = int(model.current_iteration())
            if rounds < 1:
                raise ValueError(f"base model has no positive iteration count: {path.name}")
            model_id = f"base:{component}:{horizon}:S{','.join(map(str, scope))}"
            if str(record["model_id"]) != model_id or int(record["final_rounds"]) != rounds:
                raise ValueError(f"base model training identity mismatch: {path.name}")
            expected_seed = scoped_seed(
                self.global_seed, "base_refit", scope, horizon, component, "refit"
            )
            if int(record["seed"]) != expected_seed:
                raise ValueError(f"base model seed identity mismatch: {path.name}")
            self._fits[(horizon, component, scope)] = BaseFit(
                model=model,
                scope=scope,
                horizon=horizon,
                component=component,
                final_rounds=rounds,
                best_iterations=(),
                probe_records=(),
                model_id=model_id,
                seed=scoped_seed(self.global_seed, "base_refit", scope, horizon, component, "refit"),
            )

    def scope_predictions(self, S, horizon: str, mode: str) -> dict:
        """Create cross-fitted train and S-fit test base predictions."""

        scope = canonical_fold_set(S, minimum=2, maximum=5)
        if mode not in {"direct", "component_v158"}:
            raise ValueError(f"unsupported base mode: {mode}")
        components = ("osi",) if mode == "direct" else COMPONENTS
        train_parts, test_parts, source_train, source_test = {}, {}, {}, {}
        source_train_by_component, source_test_by_component = {}, {}
        train_source = np.empty(len(self.X_train), dtype=object)
        for component in components:
            full_fit = self.get(scope, horizon, component)
            test_parts[component] = full_fit.predict(self.X_test)
            source_test[component] = full_fit.model_id
            source_test_by_component[component] = full_fit.model_id
            values = np.empty(len(self.X_train), dtype=float)
            source_values = np.empty(len(self.X_train), dtype=object)
            for fold in range(N_FOLDS):
                rows = np.flatnonzero(self.row_fold == fold)
                source_scope = tuple(r for r in scope if r != fold) if fold in scope else scope
                if fold in scope:
                    if len(source_scope) < 2:
                        raise ValueError("cross-fit source scope would be smaller than two folds")
                    source_fit = self.get(source_scope, horizon, component)
                else:
                    source_fit = full_fit
                values[rows] = source_fit.predict(self.X_train.iloc[rows])
                source_values[rows] = source_fit.model_id
                if component == components[0]:
                    train_source[rows] = source_fit.model_id
            train_parts[component] = values
            source_train_by_component[component] = source_values
        if mode == "direct":
            train_base = train_parts["osi"]
            test_base = test_parts["osi"]
        else:
            train_base = post_process_osi(self.compose(train_parts))
            test_base = post_process_osi(self.compose(test_parts))
        if not np.isfinite(train_base).all() or not np.isfinite(test_base).all():
            raise FloatingPointError("scope base prediction is non-finite")
        return {
            "scope": scope,
            "mode": mode,
            "horizon": horizon,
            "train_base": train_base,
            "test_base": test_base,
            "train_source_id": train_source,
            "test_source_id": source_test[components[0]],
            "train_source_by_component": source_train_by_component,
            "test_source_by_component": source_test_by_component,
            "train_parts": train_parts,
            "test_parts": test_parts,
        }

    @staticmethod
    def compose(parts: dict[str, np.ndarray]) -> np.ndarray:
        result = np.zeros_like(np.asarray(next(iter(parts.values())), dtype=float))
        for component, weight in COMPONENT_WEIGHTS.items():
            result += weight * np.asarray(parts[component], dtype=float)
        return result


# Kept as an explicit refusal so old v2.1 artifacts cannot silently enter v158.
def load_component_v21_base(*args, **kwargs):
    raise ValueError("component_v21 is a historical alias and is rejected by dem_v158_nested_v2")
