"""Scope-isolated graph stacks and pooled alpha selection."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from base_model import BaseModelStore, canonical_fold_set, scoped_seed
from config import (
    GAT_DROPOUT,
    GAT_HEADS,
    GAT_HIDDEN,
    GAT_LOSS,
    GAT_LR,
    GAT_PATIENCE,
    GAT_TIME_STRIDE,
    GAT_WEIGHT_DECAY,
    GAT_ALPHA_GRID,
    HORIZON_HOURS,
    HORIZONS,
    N_FOLDS,
    PRED_END,
    PRED_START,
    PROTOCOL,
    COMPARE_VARIANTS,
    variant_config,
)
from data import add_neighbor_feature_aggregates, make_county_time_view
from metrics import post_process_osi
from model_records import file_hash, load_record, publish_model


def fit_gat(*args, **kwargs):
    from gat_model import fit_gat as implementation
    return implementation(*args, **kwargs)


def predict_gat(*args, **kwargs):
    from gat_model import predict_gat as implementation
    return implementation(*args, **kwargs)


def _read_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _write_checkpoint(path, payload):
    import torch
    torch.save(payload, path)


def _scope_node_mask(all_fips, train_fips_to_fold, S):
    scope = set(canonical_fold_set(S))
    return np.asarray(
        [fips in train_fips_to_fold and train_fips_to_fold[fips] in scope for fips in all_fips],
        dtype=bool,
    )


def _grid_to_rows(grid, rows, n_rows):
    values = np.full(n_rows, np.nan, dtype=float)
    coverage = np.zeros(n_rows, dtype=int)
    for t in range(rows.shape[0]):
        for node in range(rows.shape[1]):
            row_id = int(rows[t, node])
            if row_id >= 0:
                if coverage[row_id] != 0:
                    raise ValueError(f"duplicate grid coverage for row {row_id}")
                values[row_id] = grid[t, node]
                coverage[row_id] = 1
    return values, coverage


def _strict_standardize(features, train_node_mask, valid_time):
    features = np.asarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("raw graph features are non-finite before standardization")
    mask = np.asarray(valid_time, dtype=bool)[:, None] & np.asarray(train_node_mask, dtype=bool)[None, :]
    if not mask.any():
        raise ValueError("standardization has no allowed training cells")
    flat = features.reshape(-1, features.shape[-1])
    train_values = flat[mask.reshape(-1)]
    mean = np.mean(train_values, axis=0, dtype=np.float64)
    std = np.std(train_values, axis=0, ddof=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("feature standardization statistics are non-finite")
    std[std < 1e-6] = 1.0
    scaled = ((features.astype(np.float64) - mean) / std).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise FloatingPointError("standardized graph features are non-finite")
    return scaled, mean, std


def _standardize_with_stats(features, mean, std):
    """Apply checkpointed preprocessing without touching supervision."""

    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if values.ndim != 3 or mean.shape != (values.shape[-1],) or std.shape != mean.shape:
        raise ValueError("checkpointed graph preprocessing shape mismatch")
    if not np.isfinite(values).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("non-finite graph preprocessing input")
    if np.any(std < 1e-6):
        raise ValueError("checkpointed feature std contains an unnormalised small value")
    scaled = ((values - mean) / std).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise FloatingPointError("checkpointed graph standardization produced non-finite values")
    return scaled


def _apply_variant_input_policy(scaled, variant):
    """Apply only preregistered, label-free input handling."""

    policy = variant_config(variant)
    values = np.asarray(scaled, dtype=np.float32)
    clip = policy["input_clip_z"]
    if clip is not None:
        values = np.clip(values, -float(clip), float(clip)).astype(np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("variant input policy produced non-finite features")
    return values


def _residual_scale(residual_grid, mask):
    values = np.asarray(residual_grid, dtype=float)[np.asarray(mask, dtype=bool)]
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("residual scale has no finite allowed supervision")
    return float(max(np.std(values, ddof=0), 0.003))


@dataclass
class GraphInputs:
    scope: tuple[int, ...]
    horizon: str
    mode: str
    features: np.ndarray
    scaled_features: np.ndarray
    residual_grid: np.ndarray
    supervised_mask: np.ndarray
    base_grid: np.ndarray
    base_train: np.ndarray
    base_test: np.ndarray
    train_rows: np.ndarray
    test_rows: np.ndarray
    all_fips: list[str]
    train_source_id: np.ndarray
    test_source_id: str
    residual_scale: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: tuple[str, ...]
    base_parts: dict
    variant: str
    input_policy: dict


@dataclass
class StackFit:
    scope: tuple[int, ...]
    horizon: str
    mode: str
    inputs: GraphInputs
    model: torch.nn.Module
    correction_grid: np.ndarray
    fit_info: dict
    stack_id: str
    checkpoint_path: Path | None

    def train_base_rows(self, n_rows: int) -> np.ndarray:
        values, coverage = _grid_to_rows(self.inputs.base_grid, self.inputs.train_rows, n_rows)
        if not np.isfinite(values).all() or not np.all(coverage == 1):
            raise ValueError("base rows do not have exactly one finite graph coverage")
        return values

    def train_correction_rows(self, n_rows: int) -> np.ndarray:
        values, coverage = _grid_to_rows(self.correction_grid, self.inputs.train_rows, n_rows)
        if not np.isfinite(values).all() or not np.all(coverage == 1):
            raise ValueError("correction rows do not have exactly one finite graph coverage")
        return values


@dataclass
class AlphaSelection:
    scope: tuple[int, ...]
    horizon: str
    mode: str
    alpha: float
    candidates: list[dict]
    inner_base: np.ndarray
    inner_correction: np.ndarray
    inner_coverage: np.ndarray
    inner_source_ids: np.ndarray
    selection_id: str
    inner_base_source_ids: np.ndarray


class StackContext:
    def __init__(
        self,
        bundle,
        terrain,
        folds,
        coords,
        edge_index,
        edge_attr,
        base_store: BaseModelStore,
        global_seed: int,
        device: str,
        epochs: int,
        patience: int,
        time_stride: int,
        model_dir: str | Path | None = None,
        loss_mode: str = GAT_LOSS,
        resume_only: bool = False,
        reuse_existing: bool = False,
        inference_only: bool = False,
        graph_hash: str | None = None,
        feature_package_hash: str | None = None,
        feature_side_hash: str | None = None,
        run_identity: str | None = None,
        variant: str = "m0_full",
    ):
        if int(epochs) < 1 or int(patience) < 1 or int(time_stride) < 1:
            raise ValueError("epochs, patience and time_stride must be >= 1")
        if time_stride != GAT_TIME_STRIDE:
            raise ValueError("comparison protocol fixes time_stride=1")
        if loss_mode != GAT_LOSS:
            raise ValueError("comparison protocol fixes GAT loss to MSE")
        if variant not in COMPARE_VARIANTS:
            raise ValueError(f"invalid comparison variant: {variant}")
        self.bundle = bundle
        self.terrain = terrain
        self.folds = folds
        self.coords = coords
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.base_store = base_store
        self.global_seed = int(global_seed)
        self.device = device
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.time_stride = int(time_stride)
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.loss_mode = loss_mode
        self.resume_only = bool(resume_only)
        self.reuse_existing = bool(reuse_existing)
        self.inference_only = bool(inference_only)
        self.graph_hash = graph_hash
        self.feature_package_hash = feature_package_hash
        self.feature_side_hash = feature_side_hash
        self.run_identity = run_identity
        self.variant = variant
        self.input_policy = variant_config(variant)
        self.row_fold = base_store.row_fold
        self.train_fips_to_fold = {}
        for fold, (_, valid) in enumerate(folds):
            for fips in bundle.meta_train.iloc[valid]["fips_str"].unique():
                self.train_fips_to_fold[str(fips)] = fold
        self.stack_cache: dict[tuple[str, str, str, tuple[int, ...]], StackFit] = {}
        self.alpha_cache: dict[tuple[str, str, str, tuple[int, ...]], AlphaSelection] = {}

    def build_graph_inputs(self, S, horizon: str, mode: str) -> GraphInputs:
        scope = canonical_fold_set(S, minimum=3, maximum=5)
        if horizon not in HORIZONS or mode not in {"direct", "component_v158"}:
            raise ValueError("invalid horizon or base mode")
        base = self.base_store.scope_predictions(scope, horizon, mode)
        raw, base_grid, train_rows, test_rows, all_fips, names = make_county_time_view(
            self.bundle.X_train,
            self.bundle.X_test,
            self.bundle.meta_train,
            self.bundle.meta_test,
            base["train_base"],
            base["test_base"],
            horizon,
            self.coords,
            self.terrain,
            self.bundle.feature_names,
        )
        features, extra_names = add_neighbor_feature_aggregates(raw, names, self.edge_index, horizon)
        graph_names = tuple(
            ["base", *names, "latitude", "longitude", *list(self.terrain.columns), *extra_names]
        )
        if len(graph_names) != features.shape[-1] or len(graph_names) != 205:
            raise ValueError("graph feature names do not match the 205-dimensional tensor")
        if features.shape[-1] != 205 or not np.allclose(features[:, :, 0], base_grid):
            raise ValueError("graph base channel is not identical to scope base predictions")
        train_node_mask = _scope_node_mask(all_fips, self.train_fips_to_fold, scope)
        valid_time = np.arange(PRED_START, PRED_END) + HORIZON_HOURS[horizon] <= PRED_END - 1
        residual_grid = np.zeros(base_grid.shape, dtype=float)
        supervised_mask = np.zeros(base_grid.shape, dtype=bool)
        if self.inference_only:
            checkpoint_path = self._checkpoint_path(scope, horizon, mode)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"missing checkpointed preprocessing for {scope}/{horizon}/{mode}")
            if load_record(checkpoint_path, self.run_identity) is None:
                raise FileNotFoundError("inference requires a completed GAT record")
            payload = _read_checkpoint(checkpoint_path)
            mean = np.asarray(payload.get("feature_mean"), dtype=np.float64)
            std = np.asarray(payload.get("feature_std"), dtype=np.float64)
            scale = float(payload.get("residual_scale", np.nan))
            if not np.isfinite(scale) or scale < 0.003:
                raise ValueError("checkpointed residual scale is invalid")
            scaled = _apply_variant_input_policy(_standardize_with_stats(features, mean, std), self.variant)
        else:
            labels = self.base_store.supervision.scoped(scope, horizon, "osi")
            for t in range(base_grid.shape[0]):
                for node in range(base_grid.shape[1]):
                    row_id = int(train_rows[t, node])
                    if row_id < 0 or not train_node_mask[node] or not valid_time[t]:
                        continue
                    value = labels.take(np.asarray([row_id]))[0]
                    if not np.isfinite(value):
                        raise ValueError("valid in-window official target is missing")
                    residual_grid[t, node] = value - base_grid[t, node]
                    supervised_mask[t, node] = True
            if not supervised_mask.any():
                raise ValueError("scope graph has no supervised cells")
            scale = _residual_scale(residual_grid, supervised_mask)
            scaled, mean, std = _strict_standardize(features, train_node_mask, valid_time)
            scaled = _apply_variant_input_policy(scaled, self.variant)
        return GraphInputs(
            scope=scope,
            horizon=horizon,
            mode=mode,
            features=features,
            scaled_features=scaled,
            residual_grid=residual_grid,
            supervised_mask=supervised_mask,
            base_grid=base_grid,
            base_train=base["train_base"],
            base_test=base["test_base"],
            train_rows=train_rows,
            test_rows=test_rows,
            all_fips=all_fips,
            train_source_id=base["train_source_id"],
            test_source_id=base["test_source_id"],
            residual_scale=scale,
            feature_mean=mean,
            feature_std=std,
            feature_names=graph_names,
            base_parts=base,
            variant=self.variant,
            input_policy=dict(self.input_policy),
        )

    def fit_stack(self, S, horizon: str, mode: str) -> StackFit:
        scope = canonical_fold_set(S, minimum=3, maximum=5)
        key = (self.variant, mode, horizon, scope)
        if key in self.stack_cache:
            return self.stack_cache[key]
        checkpoint_path = self._checkpoint_path(scope, horizon, mode)
        expected_stack_id = f"stack:{self.variant}:{mode}:{horizon}:S{','.join(map(str, scope))}"
        receipt = (load_record(checkpoint_path, self.run_identity, expected_stack_id)
                   if self.resume_only or self.reuse_existing or self.inference_only else None)
        if (self.resume_only or self.inference_only) and receipt is None:
            raise FileNotFoundError(f"missing complete GAT record for {key}")
        inputs = self.build_graph_inputs(scope, horizon, mode)
        if receipt is not None:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"missing complete GAT checkpoint for {key}")
            model, payload = load_gat_checkpoint(checkpoint_path, self.device)
            if payload.get("run_identity") != self.run_identity:
                raise ValueError("GAT immutable run identity mismatch")
            if payload.get("base_dependencies") != self._base_dependencies(scope, horizon, mode):
                raise ValueError("GAT upstream base model hashes mismatch")
            if payload.get("stack_id") != expected_stack_id:
                raise ValueError("GAT checkpoint scope identity mismatch")
            if self.graph_hash is not None and payload.get("graph_hash") != self.graph_hash:
                raise ValueError("GAT checkpoint graph identity mismatch")
            if self.feature_package_hash is not None and payload.get("feature_package_hash") != self.feature_package_hash:
                raise ValueError("GAT checkpoint feature package identity mismatch")
            if self.feature_side_hash is not None and payload.get("feature_side_hash") != self.feature_side_hash:
                raise ValueError("GAT checkpoint feature-side identity mismatch")
            if (tuple(payload.get("scope", ())) != scope or payload.get("horizon") != horizon
                    or payload.get("mode") != mode or payload.get("variant") != self.variant):
                raise ValueError("GAT checkpoint horizon/mode/scope identity mismatch")
            if list(payload.get("all_fips", ())) != list(inputs.all_fips):
                raise ValueError("GAT checkpoint node order mismatch")
            if list(payload.get("time_order", ())) != list(range(PRED_START, PRED_END)):
                raise ValueError("GAT checkpoint time order mismatch")
            schema_hash = hashlib.sha256(
                json.dumps(list(inputs.feature_names), separators=(",", ":")).encode()
            ).hexdigest()
            if payload.get("feature_schema_hash") != schema_hash:
                raise ValueError("GAT checkpoint feature schema mismatch")
            arch = payload.get("architecture", {})
            if int(arch.get("edge_dim", -1)) != int(self.edge_attr.shape[-1]):
                raise ValueError("GAT checkpoint edge feature dimension mismatch")
            policy = payload.get("input_policy")
            if policy != self.input_policy:
                raise ValueError("GAT checkpoint input policy mismatch")
            expected_limit = (None if self.input_policy["correction_limit"] is None
                              else float(self.input_policy["correction_limit"]) / inputs.residual_scale)
            if bool(arch.get("skip_enabled")) != bool(self.input_policy["skip"]):
                raise ValueError("GAT checkpoint skip policy mismatch")
            saved_limit = arch.get("output_limit_normalized")
            if (saved_limit is None) != (expected_limit is None) or (
                    saved_limit is not None and not np.isclose(float(saved_limit), float(expected_limit), rtol=0, atol=0)):
                raise ValueError("GAT checkpoint correction bound mismatch")
            if not np.array_equal(np.asarray(payload.get("feature_mean")), inputs.feature_mean):
                raise ValueError("GAT checkpoint feature mean mismatch")
            if not np.array_equal(np.asarray(payload.get("feature_std")), inputs.feature_std):
                raise ValueError("GAT checkpoint feature std mismatch")
            if float(payload.get("residual_scale", np.nan)) != float(inputs.residual_scale):
                raise ValueError("GAT checkpoint residual scale mismatch")
            fit_info = payload.get("fit_info", {})
            training_config = payload.get("training_config", {})
            expected_training = {
                "epochs": self.epochs,
                "patience": self.patience,
                "time_stride": self.time_stride,
                "loss_mode": self.loss_mode,
                "lr": GAT_LR,
                "weight_decay": GAT_WEIGHT_DECAY,
            }
            if training_config != expected_training:
                raise ValueError("GAT checkpoint training configuration mismatch")
            if payload.get("seed") != scoped_seed(self.global_seed, "gat", scope, horizon, mode, "fit"):
                raise ValueError("GAT checkpoint seed mismatch")
            correction = predict_gat(model, inputs.scaled_features, self.edge_index, self.device, self.edge_attr)
            correction = correction * inputs.residual_scale
            self._check_variant_correction(correction)
            result = StackFit(
                scope, horizon, mode, inputs, model, correction, fit_info,
                payload["stack_id"], checkpoint_path,
            )
            self.stack_cache[key] = result
            return result
        if self.resume_only:
            raise FileNotFoundError(f"missing complete GAT checkpoint for {key}")
        fit_times = np.arange(0, inputs.features.shape[0], self.time_stride, dtype=int)
        if fit_times[-1] != inputs.features.shape[0] - 1:
            fit_times = np.unique(np.r_[fit_times, inputs.features.shape[0] - 1])
        gat_seed = scoped_seed(self.global_seed, "gat", scope, horizon, mode, "fit")
        model, _, fit_info = fit_gat(
            inputs.scaled_features[fit_times],
            (inputs.residual_grid / inputs.residual_scale)[fit_times],
            inputs.supervised_mask[fit_times],
            self.edge_index,
            epochs=self.epochs,
            patience=self.patience,
            hidden=GAT_HIDDEN,
            heads=GAT_HEADS,
            dropout=GAT_DROPOUT,
            lr=GAT_LR,
            weight_decay=GAT_WEIGHT_DECAY,
            seed=gat_seed,
            device=self.device,
            edge_attr=self.edge_attr,
            loss_mode=self.loss_mode,
            skip_enabled=bool(self.input_policy["skip"]),
            output_limit=(None if self.input_policy["correction_limit"] is None
                          else float(self.input_policy["correction_limit"]) / inputs.residual_scale),
        )
        correction = predict_gat(model, inputs.scaled_features, self.edge_index, self.device, self.edge_attr)
        correction = correction * inputs.residual_scale
        self._check_variant_correction(correction)
        if not np.isfinite(correction).all():
            raise FloatingPointError("scope GAT correction is non-finite")
        stack_id = f"stack:{self.variant}:{mode}:{horizon}:S{','.join(map(str, scope))}"
        if self.model_dir is not None:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.model_dir / f"gat_{mode}_{horizon.replace('osi_target_', '')}_S{'-'.join(map(str, scope))}.pt"
            payload = {
                "protocol": PROTOCOL,
                "run_identity": self.run_identity,
                "stack_id": stack_id,
                "variant": self.variant,
                "variant_description": __import__("config").VARIANT_DESCRIPTIONS[self.variant],
                "graph_hash": self.graph_hash,
                "feature_package_hash": self.feature_package_hash,
                "feature_side_hash": self.feature_side_hash,
                "architecture": {
                    "in_dim": int(inputs.features.shape[-1]),
                    "hidden": GAT_HIDDEN,
                    "heads": GAT_HEADS,
                    "dropout": GAT_DROPOUT,
                    "edge_dim": int(self.edge_attr.shape[-1]),
                    "skip_enabled": bool(self.input_policy["skip"]),
                    "output_limit_normalized": (
                        None if self.input_policy["correction_limit"] is None
                        else float(self.input_policy["correction_limit"]) / inputs.residual_scale
                    ),
                },
                "state_dict": model.state_dict(),
                "feature_schema_hash": hashlib.sha256(
                    json.dumps(list(inputs.feature_names), separators=(",", ":")).encode()
                ).hexdigest(),
                "feature_names": list(inputs.feature_names),
                "all_fips": inputs.all_fips,
                "time_order": list(range(PRED_START, PRED_END)),
                "feature_mean": inputs.feature_mean,
                "feature_std": inputs.feature_std,
                "residual_scale": inputs.residual_scale,
                "input_policy": dict(self.input_policy),
                "scope": scope,
                "horizon": horizon,
                "mode": mode,
                "seed": gat_seed,
                "fit_info": fit_info,
                "training_config": {
                    "epochs": self.epochs,
                    "patience": self.patience,
                    "time_stride": self.time_stride,
                    "loss_mode": self.loss_mode,
                    "lr": GAT_LR,
                    "weight_decay": GAT_WEIGHT_DECAY,
                },
                "edge_index": self.edge_index,
                "edge_attr": self.edge_attr,
                "base_source_id": inputs.train_source_id,
                "base_source_by_component": inputs.base_parts.get("train_source_by_component", {}),
                "test_base_source_by_component": inputs.base_parts.get("test_source_by_component", {}),
                "base_dependencies": self._base_dependencies(scope, horizon, mode),
            }
            publish_model(checkpoint_path, self.run_identity, stack_id,
                           {"scope": scope, "horizon": horizon, "mode": mode, "variant": self.variant,
                            "seed": gat_seed,
                           "base_dependencies": payload["base_dependencies"]},
                          lambda temporary: _write_checkpoint(temporary, payload),
                          lambda temporary: load_gat_checkpoint(temporary, "cpu"))
        result = StackFit(scope, horizon, mode, inputs, model, correction, fit_info, stack_id, checkpoint_path)
        self.stack_cache[key] = result
        return result

    def _check_variant_correction(self, correction):
        values = np.asarray(correction, dtype=float)
        if not np.isfinite(values).all():
            raise FloatingPointError("scope GAT correction is non-finite")
        limit = self.input_policy["correction_limit"]
        if limit is not None and np.any(np.abs(values) > float(limit) + 1e-6):
            raise FloatingPointError("bounded comparison variant exceeded its correction limit")

    def _base_dependencies(self, scope, horizon, mode):
        from config import COMPONENTS
        components = ("osi",) if mode == "direct" else COMPONENTS
        source_scopes = [scope] + [tuple(k for k in scope if k != r) for r in scope]
        dependencies = {}
        for S in source_scopes:
            for component in components:
                fit = self.base_store.fits[(horizon, component, S)]
                if self.base_store.model_dir is None:
                    raise ValueError("persistent GAT checkpoints require persistent base models")
                path = self.base_store._model_path(S, horizon, component)
                receipt = load_record(path, self.run_identity, fit.model_id)
                if receipt is None:
                    raise ValueError("GAT dependency has no complete base record")
                dependencies[fit.model_id] = file_hash(path)
        return dependencies

    def _checkpoint_path(self, scope, horizon: str, mode: str) -> Path:
        scope = canonical_fold_set(scope, minimum=3, maximum=5)
        if self.model_dir is None:
            return Path(f"gat_{mode}_{horizon.replace('osi_target_', '')}_S{'-'.join(map(str, scope))}.pt")
        return self.model_dir / f"gat_{mode}_{horizon.replace('osi_target_', '')}_S{'-'.join(map(str, scope))}.pt"

    def select_alpha(self, S, horizon: str, mode: str) -> AlphaSelection:
        if self.inference_only:
            raise PermissionError("alpha selection is unavailable in inference-only context")
        scope = canonical_fold_set(S, minimum=4, maximum=5)
        key = (self.variant, mode, horizon, scope)
        if key in self.alpha_cache:
            return self.alpha_cache[key]
        n_rows = len(self.bundle.X_train)
        inner_base = np.full(n_rows, np.nan, dtype=float)
        inner_correction = np.full(n_rows, np.nan, dtype=float)
        inner_coverage = np.zeros(n_rows, dtype=int)
        inner_source_ids = np.empty(n_rows, dtype=object)
        inner_base_source_ids = np.empty(n_rows, dtype=object)
        for valid_fold in scope:
            inner_scope = tuple(fold for fold in scope if fold != valid_fold)
            stack = self.fit_stack(inner_scope, horizon, mode)
            base_rows = stack.train_base_rows(n_rows)
            correction_rows, coverage = _grid_to_rows(stack.correction_grid, stack.inputs.train_rows, n_rows)
            valid_rows = np.flatnonzero(self.row_fold == valid_fold)
            if np.any(inner_coverage[valid_rows] != 0):
                raise ValueError("inner OOF row was generated more than once")
            inner_base[valid_rows] = base_rows[valid_rows]
            inner_correction[valid_rows] = correction_rows[valid_rows]
            inner_source_ids[valid_rows] = stack.stack_id
            inner_base_source_ids[valid_rows] = stack.inputs.train_source_id[valid_rows]
            inner_coverage[valid_rows] = coverage[valid_rows]
        expected = np.isin(self.row_fold, scope) & (
            self.bundle.meta_train["hour_idx"].to_numpy(dtype=int) + HORIZON_HOURS[horizon] <= PRED_END - 1
        )
        if not expected.any() or not np.all(inner_coverage[expected] == 1):
            raise ValueError("inner OOF coverage is incomplete")
        expected_rows = np.flatnonzero(expected)
        y_expected = self.base_store.supervision.scoped(scope, horizon, "osi").take(expected_rows)
        if not np.isfinite(y_expected).all() or not np.isfinite(inner_base[expected]).all() or not np.isfinite(inner_correction[expected]).all():
            raise FloatingPointError("inner alpha inputs are non-finite on expected rows")
        candidates = []
        for alpha in GAT_ALPHA_GRID:
            prediction = post_process_osi(inner_base[expected] + float(alpha) * inner_correction[expected])
            error = y_expected - prediction
            sse = float(np.sum(error * error, dtype=np.float64))
            candidates.append({
                "scope": scope,
                "alpha": float(alpha),
                "n": int(expected.sum()),
                "sse": sse,
                "rmse": float(np.sqrt(sse / expected.sum())),
                "mae": float(np.mean(np.abs(error))),
                "selected": False,
            })
        selected = min(candidates, key=lambda row: (row["rmse"], row["alpha"]))
        selected["selected"] = True
        selection_id = f"alpha:{self.variant}:{mode}:{horizon}:S{','.join(map(str, scope))}"
        result = AlphaSelection(
            scope, horizon, mode, float(selected["alpha"]), candidates,
            inner_base, inner_correction, inner_coverage, inner_source_ids, selection_id,
            inner_base_source_ids,
        )
        self.alpha_cache[key] = result
        return result


def load_gat_checkpoint(path: str | Path, device="cpu"):
    """Strictly reload one comparison checkpoint; never adapt old dimensions."""

    from gat_model import ResidualGAT
    payload = _read_checkpoint(path)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("checkpoint protocol mismatch")
    variant = payload.get("variant")
    if variant not in COMPARE_VARIANTS:
        raise ValueError("checkpoint comparison variant is missing or invalid")
    arch = payload["architecture"]
    if int(arch["in_dim"]) != 205:
        raise ValueError("only the current 205-dimensional GAT schema is accepted")
    model = ResidualGAT(
        arch["in_dim"], arch["hidden"], arch["heads"], arch["dropout"], arch["edge_dim"],
        skip_enabled=bool(arch.get("skip_enabled", variant_config(variant)["skip"])),
        output_limit=arch.get("output_limit_normalized"),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload
