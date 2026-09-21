"""Small fixed-county fixtures and explicit arithmetic stand-ins for tests.

This module is never imported by the formal training entry point. The real
runtime diagnostic calls the same BaseModelStore/StackContext as training.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import base_model as base
import config
import data
import stacking
import run_identity
from spatial import build_spatial_graph


class ArithmeticBooster:
    def __init__(self, model_file=None, *, mean=None, rounds=1, columns=None):
        value = json.loads(Path(model_file).read_text()) if model_file else dict(mean=mean, rounds=rounds, columns=columns)
        self.value = value

    def current_iteration(self): return int(self.value["rounds"])
    def num_model_per_iteration(self): return 1
    def feature_name(self): return self.value["columns"]
    def model_to_string(self, **kwargs): return json.dumps(self.value, sort_keys=True)
    def save_model(self, path, num_iteration=None): Path(path).write_text(self.model_to_string())
    def predict(self, X, num_iteration=None):
        return np.full(len(X), self.value["mean"]) + np.nan_to_num(X.iloc[:, 0].to_numpy(dtype=float)) * .001


class ArithmeticGAT:
    def __init__(self, coefficient): self.coefficient = float(coefficient)
    def state_dict(self): return {"coefficient": np.array([self.coefficient])}


def arithmetic_prediction(model, features, edges, device="cpu", edge_attr=None):
    # Propagate the entire graph's base channel, so wrong held-out dependencies
    # affect predictions even when the held-out node is masked from the loss.
    src, dst = edges
    neighbour = np.zeros(features.shape[:2], dtype=float)
    np.add.at(neighbour, (np.arange(len(features))[:, None], dst[None, :]), features[:, src, 0])
    neighbour /= np.maximum(np.bincount(dst, minlength=features.shape[1]), 1)[None, :]
    return (model.coefficient + .01 * neighbour).astype(np.float32)


@contextmanager
def arithmetic_backend():
    """No optimizer or estimator fitting; label-sensitive deterministic math."""
    def probe(X, y, Xv, yv, seed, **options):
        limit = options.get("round_limit", config.LGBM_ROUNDS)
        return None, 1 + int(abs(float(np.sum(y) + np.sum(yv))) * 1000) % min(limit, 7)

    def fixed(X, y, rounds, seed):
        return ArithmeticBooster(mean=float(np.mean(y)), rounds=rounds, columns=list(X.columns))

    def gat(features, residual, mask, edges, **kwargs):
        coefficient = float(np.mean(residual[mask]))
        model = ArithmeticGAT(coefficient)
        return model, arithmetic_prediction(model, features, edges), {
            "epochs": kwargs["epochs"], "best_epoch": 1, "best_loss": 0.0, "seed": kwargs["seed"]}

    def write(path, payload):
        with Path(path).open("wb") as stream: pickle.dump(payload, stream)

    def read(path):
        with Path(path).open("rb") as stream: return pickle.load(stream)

    def load(path, device="cpu"):
        payload = read(path)
        return ArithmeticGAT(payload["state_dict"]["coefficient"][0]), payload

    with ExitStack() as patches:
        for module, name, replacement in [
            (base, "_fit_probe", probe), (base, "_fit_fixed", fixed), (base.lgb, "Booster", ArithmeticBooster),
            (stacking, "fit_gat", gat), (stacking, "predict_gat", arithmetic_prediction),
            (stacking, "_write_checkpoint", write), (stacking, "_read_checkpoint", read),
            (stacking, "load_gat_checkpoint", load),
        ]:
            patches.enter_context(patch.object(module, name, replacement))
        yield


def small_fixture(counties_per_fold=2, test_counties=2, *, include_supervision=True):
    bundle = data.load_feature_bundle()
    targets = data.load_supervision(feature_bundle=bundle).y_train if include_supervision else None
    components = data.load_component_targets(bundle.meta_train) if include_supervision else None
    folds = base.load_cv_folds(bundle.meta_train)
    selected = []
    county_fold = {}
    for k, (_, rows) in enumerate(folds):
        counties = sorted(bundle.meta_train.iloc[rows].fips_str.unique())[:counties_per_fold]
        if len(counties) != counties_per_fold:
            raise ValueError("diagnostic county request exceeds a fold")
        selected.extend(counties)
        county_fold.update({fips: k for fips in counties})
    tr = np.flatnonzero(bundle.meta_train.fips_str.isin(selected))
    tests = sorted(bundle.meta_test.fips_str.unique())[:test_counties]
    te = np.flatnonzero(bundle.meta_test.fips_str.isin(tests))
    m, mt = bundle.meta_train.iloc[tr].reset_index(drop=True), bundle.meta_test.iloc[te].reset_index(drop=True)
    x, xt = bundle.X_train.iloc[tr].reset_index(drop=True), bundle.X_test.iloc[te].reset_index(drop=True)
    row_fold = m.fips_str.map(county_fold).to_numpy(dtype=int)
    split = [(np.flatnonzero(row_fold != k), np.flatnonzero(row_fold == k)) for k in range(5)]
    fixture = SimpleNamespace(X_train=x, X_test=xt, meta_train=m, meta_test=mt,
                              feature_names=bundle.feature_names, input_hash=bundle.input_hash)
    all_fips = sorted(set(m.fips_str) | set(mt.fips_str))
    coords, edges, attrs = build_spatial_graph(all_fips, config.GEO_DBF, config.GEO_DBF.with_suffix(".shp"), k=min(8, len(all_fips)-1))
    return dict(bundle=fixture, targets=None if targets is None else targets.iloc[tr].reset_index(drop=True),
                components=None if components is None else {h: {c: values[tr].copy() for c, values in parts.items()} for h, parts in components.items()},
                row_fold=row_fold, folds=split, coords=coords, edge_index=edges, edge_attr=attrs,
                terrain=data.load_terrain_features(), counties=selected, test_counties=tests)


def context(fixture, mode, *, device="cpu", epochs=3, base_rounds=12, base_patience=3,
            mutate_fold=None, output=None, resume=False, inference=False):
    labels = None if inference else fixture["targets"].copy(deep=True)
    components = None if inference else {h: {c: v.copy() for c, v in parts.items()} for h, parts in fixture["components"].items()}
    if inference and mutate_fold is not None:
        raise ValueError("inference diagnostic cannot perturb labels")
    if mutate_fold is not None:
        county_rows = fixture["row_fold"] == mutate_fold
        for h in config.HORIZONS:
            valid = county_rows & labels[h].notna().to_numpy()
            # Perturb only supervision; maintain component/OSI consistency and
            # all horizon tail masks. Observed inputs and fixed folds are frozen.
            constants = {"P_t": .25, "N_t": .1, "D_t": .2, "R_t": .05}
            for c in config.COMPONENTS:
                previous = components[h][c][valid]
                # Ensure every component changes even for all-zero N/R/D folds.
                components[h][c][valid] = np.where(previous == constants[c], constants[c]/2, constants[c])
            composed = sum(config.COMPONENT_WEIGHTS[c]*components[h][c][valid] for c in config.COMPONENTS)
            labels.loc[valid, h] = np.maximum(composed, 0).round(4)
    if inference:
        from model_records import completed_paths, record_path
        paths = completed_paths(Path(output)/"models/base")
        if not paths: raise FileNotFoundError("no completed diagnostic models")
        saved = json.loads(record_path(paths[0]).read_text())
        identity, target_digest = saved["run_identity"], saved["details"]["feature_package_hash"]
    else:
        payload = hashlib.sha256(labels.to_numpy(dtype=float).tobytes())
        for h in config.HORIZONS:
            for c in config.COMPONENTS: payload.update(components[h][c].tobytes())
        target_digest = payload.hexdigest()
        identity = run_identity.digest({"diagnostic": True, "mode": mode, "device": device,
            "epochs": epochs, "base_rounds": base_rounds, "patience": base_patience,
            "counties": fixture["counties"], "test_counties": fixture["test_counties"],
            "feature_hash": fixture["bundle"].input_hash, "targets": target_digest,
            "code": run_identity.source_hashes()})
    b = fixture["bundle"]
    root = None if output is None else Path(output)
    store = base.BaseModelStore(
        b.X_train, b.X_test, b.meta_train, fixture["folds"],
        y_train=None if inference else labels, component_targets=None if inference else components,
        model_dir=None if root is None else root/"models/base", global_seed=42,
        feature_side_hash=b.input_hash, feature_package_hash=target_digest,
        run_identity=identity, reuse_existing=resume,
        fit_options={"round_limit": base_rounds, "patience": base_patience},
    )
    if resume: store.load_from_run(root)
    ctx = stacking.StackContext(b, fixture["terrain"], fixture["folds"], fixture["coords"],
        fixture["edge_index"], fixture["edge_attr"], store, 42, device, epochs, max(1, min(epochs, 2)), 1,
        model_dir=None if root is None else root/"models/gat", reuse_existing=resume,
        resume_only=inference, inference_only=inference,
        graph_hash=run_identity.digest(fixture["edge_index"].tolist()),
        feature_side_hash=b.input_hash, feature_package_hash=target_digest, run_identity=identity)
    ctx.diagnostic_mode = mode
    return ctx


def snapshot(ctx, horizon, *, kind="outer", outer_fold=0, alpha_override=None):
    A = tuple(k for k in range(5) if k != outer_fold)
    inner = (outer_fold + 1) % 5
    r = (outer_fold + 2) % 5
    B = tuple(k for k in A if k != inner)
    if kind == "crossfit":
        S = tuple(k for k in B if k != r)
        components = ("osi",) if ctx.diagnostic_mode == "direct" else config.COMPONENTS
        for c in components: ctx.base_store.get(S, horizon, c)
        values = {}
    else:
        S = A if kind == "outer" else B
        values = {}
        if kind == "outer":
            if alpha_override is None:
                alpha = ctx.select_alpha(A, horizon, ctx.diagnostic_mode).alpha
            else:
                alpha = float(alpha_override)
                for j in A:
                    ctx.fit_stack(tuple(k for k in A if k != j), horizon, ctx.diagnostic_mode)
            values["alpha"] = np.asarray([alpha])
        fitted = ctx.fit_stack(S, horizon, ctx.diagnostic_mode)
        values.update(base=fitted.inputs.base_grid, correction=fitted.correction_grid,
                      mean=fitted.inputs.feature_mean, std=fitted.inputs.feature_std,
                      scale=np.asarray([fitted.inputs.residual_scale]))
    for key, fit in sorted(ctx.base_store.fits.items()):
        tag = fit.model_id
        values[tag+":rounds"] = np.array([fit.final_rounds, fit.actual_rounds, *fit.best_iterations])
        values[tag+":prediction"] = fit.predict(ctx.bundle.X_train)
        values[tag+":weights"] = np.asarray([hashlib.sha256(fit.model.model_to_string(num_iteration=fit.actual_rounds).encode()).hexdigest()])
    for key, fitted in sorted(ctx.stack_cache.items()):
        for name, tensor in fitted.model.state_dict().items():
            value = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
            values[fitted.stack_id+":"+name] = value
    return values


def compare_snapshots(left, right, *, atol=1e-7):
    if set(left) != set(right): raise AssertionError("snapshot model inventory differs")
    for key in left:
        a, b = left[key], right[key]
        if a.dtype.kind in "USO":
            if not np.array_equal(a, b): raise AssertionError(f"model digest differs: {key}")
        elif not np.allclose(a, b, rtol=0, atol=atol, equal_nan=True):
            raise AssertionError(f"numerical isolation/reload mismatch: {key}")
