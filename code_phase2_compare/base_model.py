"""Scope-isolated LightGBM base models for the comparison protocol."""

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
from model_records import completed_paths, file_hash, load_record, publish_model


CV_BASE_MANIFEST = "base_fit_manifest.parquet"
FINAL_BASE_MANIFEST = "base_fit_manifest_final.parquet"


def _log_lgbm_round(env):
    """Print one progress line per completed LightGBM boosting round."""
    iteration = int(env.iteration) + 1
    total = int(env.end_iteration)
    metrics = " ".join(
        f"{dataset}:{metric}={value:.8f}"
        for dataset, metric, value, _ in (env.evaluation_result_list or ())
    )
    suffix = f" {metrics}" if metrics else ""
    print(f"[LightGBM] round={iteration}/{total}{suffix}", flush=True)


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
    actual_rounds: int | None = None

    def __post_init__(self):
        if self.actual_rounds is None:
            self.actual_rounds = int(self.model.current_iteration())
        if not 1 <= self.actual_rounds <= self.final_rounds:
            raise ValueError("actual LightGBM rounds must be positive and no greater than requested rounds")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.model.predict(X, num_iteration=self.actual_rounds), dtype=float)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite base prediction from {self.model_id}")
        if self.component != "osi":
            values = np.clip(values, 0.0, 1.0)
        return values


def _fit_probe(X_train, y_train, X_valid, y_valid, seed: int, *, round_limit=None, patience=None):
    if len(X_train) == 0 or len(X_valid) == 0:
        raise ValueError("LightGBM probe received an empty train or validation set")
    params = make_lgbm_params(seed)
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, free_raw_data=False)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=LGBM_ROUNDS if round_limit is None else int(round_limit),
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING if patience is None else int(patience), verbose=False), _log_lgbm_round],
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
        callbacks=[_log_lgbm_round],
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
    fit_options: dict | None = None,
) -> BaseFit:
    """Fit one scalar base model using labels from exactly S.

    Every probe uses one validation fold in S, and the public predictor is a
    fixed-round refit on all valid rows in S. Probe models are never used for
    predictions returned to callers.
    """

    scope = canonical_fold_set(S, minimum=2, maximum=5)
    options = {} if fit_options is None else dict(fit_options)
    if set(options) - {"round_limit", "patience"} or any(int(v) < 1 for v in options.values()):
        raise ValueError("invalid diagnostic base budget")
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
            **options,
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
        feature_package_hash=None, feature_schema_hash=None, feature_side_hash=None,
        protocol="dem_compare_v1_nested_v1",
        run_identity=None, reuse_existing=False, fit_options=None,
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
        self.feature_side_hash = feature_side_hash
        self.feature_schema_hash = feature_schema_hash
        self.protocol = str(protocol)
        self.run_identity = run_identity
        self.reuse_existing = bool(reuse_existing)
        self.fit_options = fit_options
        self.supervision = supervision_store or SupervisionStore(
            self.row_fold, y_train=y_train, component_targets=component_targets
        )
        self._fits: dict[tuple[str, str, tuple[int, ...]], BaseFit] = {}

    def _model_path(self, S, horizon, component):
        return self.model_dir / f"{component}_{horizon.replace('osi_target_', '')}_S{'-'.join(map(str, S))}.txt"

    def fit_record(self, fit, *, include_hash=True):
        path = self._model_path(fit.scope, fit.horizon, fit.component)
        valid = np.isin(self.row_fold, fit.scope) & (
            self.meta_train.hour_idx.to_numpy(dtype=int) + HORIZON_HOURS[fit.horizon] < PRED_END)
        record = {
            "protocol": self.protocol, "run_identity": self.run_identity,
            "horizon": fit.horizon, "component": fit.component,
            "allowed_folds": json.dumps(fit.scope), "valid_rows": int(valid.sum()),
            "best_iterations": json.dumps(fit.best_iterations), "probe_records": json.dumps(fit.probe_records),
            "final_rounds": fit.final_rounds, "actual_rounds": fit.actual_rounds,
            "seed": fit.seed, "model_id": fit.model_id,
            "model_path": f"models/base/{path.name}",
            "feature_package_hash": self.feature_package_hash, "feature_side_hash": self.feature_side_hash,
            "feature_schema_hash": self.feature_schema_hash or hashlib.sha256(
                json.dumps(list(self.X_train.columns), separators=(",", ":")).encode()).hexdigest(),
        }
        if include_hash:
            record["model_sha256"] = file_hash(path)
        return record

    def _restore_fit(self, path, record):
        scope = canonical_fold_set(json.loads(record["allowed_folds"]), minimum=2, maximum=5)
        horizon, component = record["horizon"], record["component"]
        if horizon not in HORIZONS or component not in ("osi", *COMPONENTS):
            raise ValueError("base model target identity mismatch")
        if path != self._model_path(scope, horizon, component):
            raise ValueError("base model filename/scope mismatch")
        expected_id = f"base:{component}:{horizon}:S{','.join(map(str, scope))}"
        if record["protocol"] != self.protocol or record["model_id"] != expected_id:
            raise ValueError("base model protocol/scope mismatch")
        if record["model_path"].replace(chr(92), "/") != f"models/base/{path.name}":
            raise ValueError("base model path identity mismatch")
        for field in ("feature_package_hash", "feature_side_hash", "feature_schema_hash"):
            expected = getattr(self, field)
            if expected is not None and record.get(field) != expected:
                raise ValueError(f"base-fit manifest {field} mismatch")
        if self.run_identity is not None and record.get("run_identity") != self.run_identity:
            raise ValueError("base model run identity mismatch")
        if not path.is_file():
            raise FileNotFoundError(f"registered base model missing: {path}")
        if file_hash(path) != record["model_sha256"]:
            raise ValueError("base model hash mismatch")
        expected_seed = scoped_seed(self.global_seed, "base_refit", scope, horizon, component, "refit")
        if record["seed"] != expected_seed:
            raise ValueError("base model seed mismatch")
        requested, actual = int(record["final_rounds"]), int(record["actual_rounds"])
        if not 1 <= actual <= requested:
            raise ValueError("invalid requested/actual base rounds")
        model = lgb.Booster(model_file=str(path))
        if int(model.current_iteration()) != actual:
            raise ValueError("base model actual round count mismatch")
        if hasattr(model, "num_model_per_iteration") and model.num_model_per_iteration() != 1:
            raise ValueError("only scalar one-tree-per-iteration boosters are supported")
        if hasattr(model, "feature_name") and model.feature_name() != list(self.X_train.columns):
            raise ValueError("base model feature schema mismatch")
        return BaseFit(model, scope, horizon, component, requested,
                       tuple(json.loads(record["best_iterations"])), tuple(json.loads(record["probe_records"])),
                       expected_id, expected_seed, actual)

    def _load_completed(self, path):
        receipt = load_record(path, self.run_identity)
        if receipt is None:
            return None
        record = dict(receipt["details"], model_sha256=receipt["sha256"])
        if receipt["model_id"] != record["model_id"]:
            raise ValueError("completion record model identity mismatch")
        return self._restore_fit(path, record)

    def get(self, S, horizon: str, component: str = "osi") -> BaseFit:
        scope = canonical_fold_set(S, minimum=2, maximum=5)
        key = (horizon, component, scope)
        if key in self._fits:
            return self._fits[key]
        path = None if self.model_dir is None else self._model_path(scope, horizon, component)
        if self.reuse_existing and path is not None:
            completed = self._load_completed(path)
            if completed is not None:
                self._fits[key] = completed
                return completed
        target = self.supervision.scoped(scope, horizon, component)
        fit = fit_base(
            self.X_train, target, self.row_fold, self.folds, scope, horizon,
            component=component, global_seed=self.global_seed,
            valid_mask=(self.meta_train.hour_idx.to_numpy(dtype=int) + HORIZON_HOURS[horizon] < PRED_END),
            fit_options=self.fit_options,
        )
        if path is not None:
            def validate(temporary):
                reloaded = lgb.Booster(model_file=str(temporary))
                if reloaded.current_iteration() != fit.actual_rounds:
                    raise ValueError("saved booster has an unexpected actual round count")
            publish_model(path, self.run_identity, fit.model_id, self.fit_record(fit, include_hash=False),
                          lambda temporary: fit.model.save_model(str(temporary), num_iteration=fit.actual_rounds), validate)
        # Only publish in memory after durable completion succeeds.
        self._fits[key] = fit
        return fit

    @property
    def fits(self):
        return dict(self._fits)

    def load_from_run(self, run_dir: str | Path) -> None:
        run_root = Path(run_dir)
        root = run_root / "models/base"
        manifest_path = run_root / FINAL_BASE_MANIFEST
        if not manifest_path.exists():
            manifest_path = run_root / CV_BASE_MANIFEST
        loaded = {}
        if self.run_identity is not None:
            # Partial CV has no aggregate manifest yet; per-model receipts are
            # sufficient to reuse exactly the completed fits.
            for path in completed_paths(root):
                fit = self._load_completed(path)
                loaded[(fit.horizon, fit.component, fit.scope)] = fit
            if manifest_path.exists():
                manifest = pd.read_parquet(manifest_path, engine=self.parquet_engine)
                for record in manifest.to_dict("records"):
                    key = (record["horizon"], record["component"], tuple(json.loads(record["allowed_folds"])))
                    if key not in loaded:
                        raise ValueError("frozen manifest model has no valid completion record")
                    actual = self.fit_record(loaded[key])
                    if any(actual.get(k) != record[k] for k in actual):
                        raise ValueError("completion record differs from frozen base manifest")
        else:
            # Standalone serialization tests can use an explicit manifest;
            # formal runs always require the immutable run identity and receipts.
            if not manifest_path.exists():
                raise FileNotFoundError(f"missing base-fit manifest: {manifest_path}")
            manifest = pd.read_parquet(manifest_path, engine=self.parquet_engine)
            if manifest.empty or manifest.duplicated("model_id").any() or manifest.duplicated("model_path").any():
                raise ValueError("base-fit manifest must contain unique, nonempty model identities")
            for record in manifest.to_dict("records"):
                path = root / Path(record["model_path"].replace(chr(92), "/")).name
                fit = self._restore_fit(path, record)
                loaded[(fit.horizon, fit.component, fit.scope)] = fit
        self._fits.update(loaded)

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
    raise ValueError("component_v21 is a historical alias and is rejected by the comparison protocol")
