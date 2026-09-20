"""Numerical and lineage checks on saved artifacts, without Torch or training."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, COMPONENTS, COMPONENT_WEIGHTS,
    COMPONENT_TARGETS_FILE, CV_FILE, EXPERIMENT_VERSION, FEATURE_VERSION,
    GAT_ALPHA_GRID, GEO_DBF, HORIZON_HOURS, HORIZONS, N_FOLDS, PRED_END, PRED_START,
    PROTOCOL, SUBMISSION_FILE, TERRAIN_FILE,
)
from metrics import post_process_osi

ALL_FOLDS = tuple(range(N_FOLDS))
KEY = ["fipsCode", "hour_idx"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalise_counties(frame):
    result = frame.copy()
    values = result["fipsCode"].astype("string").str.strip().str.zfill(5)
    require(values.str.fullmatch(r"\d{5}").fillna(False).all(), "invalid FIPS")
    result["fipsCode"] = values.astype(str)
    return result


def required_columns(frame, names, label):
    require(set(names).issubset(frame.columns), f"{label}: missing columns {set(names) - set(frame.columns)}")


def close(actual, expected, label, *, atol=1e-12):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape, f"{label}: shape mismatch")
    require(not np.isinf(a).any() and not np.isinf(b).any(), f"{label}: infinite values")
    require(np.array_equal(np.isnan(a), np.isnan(b)), f"{label}: NaN mask mismatch")
    require(np.allclose(a, b, atol=atol, rtol=0, equal_nan=True), f"{label}: numerical mismatch")


def equal(actual, expected, label):
    require(np.array_equal(np.asarray(actual), np.asarray(expected)), f"{label}: mismatch")


def booleans(values, label):
    require(values.notna().all() and values.isin([True, False]).all(), f"{label}: invalid boolean")
    return values.to_numpy(dtype=bool)


def align(frame, reference, keys, label):
    """Require the exact key set, then return rows in reference order."""
    required_columns(frame, keys, label)
    require(not frame[keys].isna().any().any(), f"{label}: null key")
    require(not frame.duplicated(keys).any(), f"{label}: duplicate key")
    wanted = pd.MultiIndex.from_frame(reference[keys])
    actual = pd.MultiIndex.from_frame(frame[keys])
    require(len(actual) == len(wanted) and wanted.isin(actual).all(), f"{label}: incomplete or unexpected coverage")
    result = frame.copy()
    result.index = actual
    return result.loc[wanted].reset_index(drop=True)


def time_rows(frame, label):
    required_columns(frame, KEY, label)
    result = normalise_counties(frame)
    hours = pd.to_numeric(result["hour_idx"], errors="raise").to_numpy(dtype=float)
    require(np.isfinite(hours).all() and np.equal(hours, np.floor(hours)).all(), f"{label}: invalid hour")
    require(((hours >= PRED_START) & (hours < PRED_END)).all(), f"{label}: hour outside prediction window")
    result["hour_idx"] = hours.astype(int)
    if "timestamp_et" in result:
        timestamps = pd.to_datetime(result["timestamp_et"], errors="raise")
        expected = pd.Timestamp("2026-03-11") + pd.to_timedelta(hours, unit="h")
        equal(timestamps.to_numpy(), expected.to_numpy(), f"{label} timestamp")
    return result


def identity(kind, mode, horizon, scope, component="osi", variant="m0_full"):
    middle = component if kind == "base" else f"{variant}:{mode}"
    return f"{kind}:{middle}:{horizon}:S{','.join(map(str, scope))}"


def scopes(values, expected, label):
    for raw in values.unique():
        parsed = json.loads(raw)
        require(parsed == list(expected), f"{label}: forbidden or incorrect label scope")


def metrics(y, prediction):
    y, p = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    require(y.shape == p.shape and y.size > 0, "metric coverage invalid")
    require(np.isfinite(y).all() and np.isfinite(p).all(), "metric inputs non-finite")
    error = y - p
    mse = float(np.mean(error ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return dict(n=len(y), mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(error))),
                medae=float(np.median(np.abs(error))), bias=float(np.mean(p - y)),
                error_std=float(np.std(error)), r2=1 - float(np.sum(error ** 2)) / total if total else np.nan,
                max_abs_error=float(np.max(np.abs(error))))


def compare_table(actual, expected, keys, label):
    if "fipsCode" in keys:
        actual, expected = normalise_counties(actual), normalise_counties(expected)
    required_columns(actual, expected.columns, label)
    actual = align(actual, expected, keys, label)
    for column in expected.columns:
        if column in keys:
            continue
        if pd.api.types.is_numeric_dtype(expected[column]):
            if pd.api.types.is_integer_dtype(expected[column]) or pd.api.types.is_bool_dtype(expected[column]):
                equal(actual[column], expected[column], f"{label}/{column}")
            else:
                close(actual[column], expected[column], f"{label}/{column}")
        else:
            equal(actual[column].astype(str), expected[column].astype(str), f"{label}/{column}")


@dataclass
class References:
    train: pd.DataFrame
    test: pd.DataFrame
    targets: pd.DataFrame
    row_fold: np.ndarray


def load_references(manifest, input_manifest):
    from base_model import load_cv_folds
    from data import load_feature_bundle, load_supervision

    require(manifest["inputs"] == input_manifest, "run/input manifest mismatch")
    bundle = load_feature_bundle(input_manifest["feature_dir"], input_manifest["parquet_engine"])
    supervised = load_supervision(feature_bundle=bundle, parquet_engine=input_manifest["parquet_engine"])
    require(bundle.input_hash == input_manifest["feature_side_hash"], "feature-side hash mismatch")
    require(supervised.target_hash == input_manifest["targets_train_sha256"], "target hash mismatch")
    payload = json.dumps({"feature_side_hash": bundle.input_hash, "targets_train_sha256": supervised.target_hash},
                         sort_keys=True, separators=(",", ":")).encode()
    require(hashlib.sha256(payload).hexdigest() == input_manifest["feature_package_hash"], "full package hash mismatch")
    for path, field in [(CV_FILE, "cv_sha256"), (COMPONENT_TARGETS_FILE, "component_targets_sha256"),
                        (TERRAIN_FILE, "terrain_sha256"), (GEO_DBF, "geo_dbf_sha256"),
                        (GEO_DBF.with_suffix(".shp"), "geo_shp_sha256"), (SUBMISSION_FILE, "submission_template_sha256")]:
        require(sha256(path) == input_manifest[field], f"input hash mismatch: {path.name}")
    row_fold = np.empty(len(bundle.meta_train), dtype=int)
    for k, (_, rows) in enumerate(load_cv_folds(bundle.meta_train)):
        row_fold[rows] = k
    train, test = time_rows(bundle.meta_train, "reference train"), time_rows(bundle.meta_test, "reference test")
    return References(train, test, supervised.y_train, row_fold)


def check_frozen_cv(run_dir, manifest):
    cv = json.loads((run_dir / "cv_manifest.json").read_text())
    expected_names = {"outer_oof.parquet", "inner_oof.parquet", "alpha_selection.parquet", "fold_metrics.csv",
                      "cv_summary.csv", "county_metrics.csv", "county_bootstrap.csv", "base_fit_manifest.parquet"}
    require(cv.get("protocol") == PROTOCOL and cv.get("run_id") == manifest["run_id"], "CV identity mismatch")
    require(cv.get("experiment_version") == EXPERIMENT_VERSION, "CV version mismatch")
    require(set(cv.get("artifact_hashes", {})) == expected_names, "CV hash inventory incomplete")
    digest = hashlib.sha256(json.dumps(cv["artifact_hashes"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(digest == cv.get("cv_hash") == (run_dir / "CV_COMPLETE").read_text(), "CV marker mismatch")
    require(cv.get("run_manifest_sha256") == sha256(run_dir / "run_manifest.json"), "frozen run manifest changed")
    for name, expected in cv["artifact_hashes"].items():
        require(sha256(run_dir / name) == expected, f"frozen CV artifact changed: {name}")


def check_base_manifests(run_dir, manifest, refs):
    from base_model import scoped_seed
    from model_records import load_record

    mode = manifest["base_mode"]
    components = ("osi",) if mode == "direct" else COMPONENTS
    expected_ids = {identity("base", mode, h, S, c)
                    for size in range(2, 6) for S in itertools.combinations(ALL_FOLDS, size)
                    for h in HORIZONS for c in components}
    final = pd.read_parquet(run_dir / "base_fit_manifest_final.parquet")
    cv = pd.read_parquet(run_dir / "base_fit_manifest.parquet")
    require(not final.model_id.duplicated().any() and set(final.model_id) == expected_ids, "base model scope inventory mismatch")
    cv_ids = {model_id for model_id in expected_ids if not model_id.endswith("S0,1,2,3,4")}
    require(not cv.model_id.duplicated().any() and set(cv.model_id) == cv_ids, "CV base inventory mismatch")
    compare_table(final[final.model_id.isin(cv_ids)], cv, ["model_id"], "immutable CV base records")
    names = []
    for row in final.to_dict("records"):
        S = tuple(json.loads(row["allowed_folds"]))
        h, c = row["horizon"], row["component"]
        require(row["model_id"] == identity("base", mode, h, S, c), "base lineage mismatch")
        require(row["protocol"] == PROTOCOL and row["base_mode"] == mode, "base protocol mismatch")
        for field in ["feature_side_hash", "feature_package_hash"]:
            require(row[field] == manifest["inputs"][field], f"base {field} mismatch")
        filename = f"models/base/{c}_{h.replace('osi_target_', '')}_S{'-'.join(map(str, S))}.txt"
        require(row["model_path"].replace("\\", "/") == filename, "base model path mismatch")
        require(sha256(run_dir / filename) == row["model_sha256"], "base model file hash mismatch")
        require(1 <= int(row["actual_rounds"]) <= int(row["final_rounds"]), "invalid actual/requested base rounds")
        if "identity" in manifest:
            receipt = load_record(run_dir / filename, manifest["identity"]["digest"], row["model_id"])
            require(receipt is not None, "base completion record missing")
            require(row.get("run_identity") == manifest["identity"]["digest"], "base run identity mismatch")
            for field, value in receipt["details"].items():
                require(row.get(field) == value, f"base completion metadata mismatch: {field}")
        names.append(Path(filename).name)
        valid = refs.train.hour_idx.to_numpy() + HORIZON_HOURS[h] < PRED_END
        require(row["valid_rows"] == int((valid & np.isin(refs.row_fold, S)).sum()), "base row count mismatch")
        iterations, probes = json.loads(row["best_iterations"]), json.loads(row["probe_records"])
        require(len(iterations) == len(S) and all(isinstance(v, int) and v > 0 for v in iterations), "base round provenance incomplete")
        require(row["final_rounds"] == max(1, int(np.mean(iterations))), "base round aggregation mismatch")
        require(len(probes) == len(S), "base probe coverage incomplete")
        for q, best, probe in zip(S, iterations, probes):
            require(probe["probe_fold"] == q and probe["train_folds"] == [k for k in S if k != q]
                    and probe["valid_folds"] == [q], "base probe label leakage")
            require(probe["best_iteration"] == best, "base probe iteration mismatch")
            require(probe["train_rows"] == int((valid & np.isin(refs.row_fold, [k for k in S if k != q])).sum())
                    and probe["valid_rows"] == int((valid & (refs.row_fold == q)).sum()), "base probe row mismatch")
            require(probe["seed"] == scoped_seed(manifest["seed"], "base_probe", S, h, c, q), "probe seed mismatch")
        require(row["seed"] == scoped_seed(manifest["seed"], "base_refit", S, h, c, "refit"), "base seed mismatch")
    equal(sorted(names), sorted(p.name for p in (run_dir / "models/base").glob("*.txt")), "base files")
    return final


def check_alpha_group(group, y, base, correction, *, S, h, mode, selection_type, outer_fold):
    required_columns(group, ["alpha", "n", "sse", "rmse", "mae", "selected", "selection_id", "allowed_folds"], "alpha")
    expected_keys = pd.DataFrame({"alpha": GAT_ALPHA_GRID})
    group = align(group, expected_keys, ["alpha"], "alpha grid")
    scopes(group.allowed_folds, S, "alpha")
    equal(group.selection_id, np.repeat(identity("alpha", mode, h, S), len(group)), "alpha selection id")
    equal(group.selection_type, np.repeat(selection_type, len(group)), "alpha selection type")
    equal(group.outer_fold.astype(str), np.repeat(str(outer_fold), len(group)), "alpha outer fold")
    scores = []
    for alpha in GAT_ALPHA_GRID:
        pred = post_process_osi(np.asarray(base) + alpha * np.asarray(correction))
        error = np.asarray(y) - pred
        require(len(error) > 0 and np.isfinite(error).all(), "alpha inputs non-finite")
        sse = float(np.sum(error * error, dtype=np.float64))
        scores.append(dict(n=len(error), sse=sse, rmse=np.sqrt(sse / len(error)), mae=np.mean(np.abs(error))))
    for col in ["n", "sse", "rmse", "mae"]:
        close(group[col], [r[col] for r in scores], f"alpha {h}/{outer_fold}/{col}")
    winner = min(range(len(scores)), key=lambda i: (scores[i]["rmse"], GAT_ALPHA_GRID[i]))
    selected = booleans(group.selected, "alpha selected")
    equal(selected, np.arange(len(scores)) == winner, "alpha winner/tie-break")
    return float(GAT_ALPHA_GRID[winner])


def _sources(frame, mode, h, S, *, inner=False, variant="m0_full"):
    component = "osi" if mode == "direct" else "P_t"
    equal(frame.base_source_id, np.repeat(identity("base", mode, h, S, component), len(frame)), "base source scope")
    equal(frame.stack_id, np.repeat(identity("stack", mode, h, S, variant=variant), len(frame)), "stack scope")
    if mode == "component_v158" and not inner:
        values = np.zeros(len(frame), dtype=float)
        for c, weight in COMPONENT_WEIGHTS.items():
            required_columns(frame, [f"base_{c}", f"base_source_{c}"], "component prediction")
            p = frame[f"base_{c}"].to_numpy(dtype=float)
            require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "invalid component prediction")
            equal(frame[f"base_source_{c}"], np.repeat(identity("base", mode, h, S, c), len(frame)), "component source scope")
            values += weight * p
        close(frame.base_prediction, post_process_osi(values).astype(np.float32), "component composition", atol=1e-7)


def _predictions(frame, expected, label):
    equal(booleans(frame.is_scoreable, label), expected, f"{label} scoreable")
    base = frame.base_prediction.to_numpy(dtype=float)
    correction = frame.correction_osi.to_numpy(dtype=float)
    alpha = frame.alpha.to_numpy(dtype=float)
    require(np.isfinite(base).all() and np.isfinite(correction).all() and np.isfinite(alpha).all(), f"{label}: non-finite prediction inputs")
    before = np.where(expected, base + alpha * correction, np.nan)
    close(frame.prediction_before_postprocess, before, f"{label} before postprocess")
    prediction = np.full(len(frame), np.nan)
    prediction[expected] = post_process_osi(before[expected])
    close(frame.prediction, prediction, f"{label} prediction")


def check_rows(outer, inner, alpha, alpha_final, test, graph, refs, mode, run_id, variant="m0_full"):
    outer, inner = time_rows(outer, "outer"), time_rows(inner, "inner")
    test, graph = time_rows(test, "test"), time_rows(graph, "graph")
    for label, frame in [("outer", outer), ("inner", inner), ("test", test), ("graph", graph)]:
        require(set(frame.horizon) == set(HORIZONS), f"{label}: horizon coverage")
    for label, frame in [("outer", outer), ("inner", inner), ("test", test), ("alpha", alpha), ("alpha_final", alpha_final)]:
        require(frame.protocol.eq(PROTOCOL).all() and frame.base_mode.eq(mode).all()
                and frame.variant.eq(variant).all(), f"{label}: protocol/mode/variant mismatch")
    for frame in [outer, test]:
        require(frame.run_id.eq(run_id).all(), "prediction run_id mismatch")
    # One pooled selection per outer fold/horizon: 5 * 4 * 8 = 160.
    require(len(alpha) == N_FOLDS * len(HORIZONS) * len(GAT_ALPHA_GRID), "outer alpha candidate coverage")
    require(set(alpha.outer_fold) == set(ALL_FOLDS) and set(alpha.horizon) == set(HORIZONS), "outer alpha group coverage")
    require(set(alpha_final.selection_type) == {"full_train_cv_for_final", "final_selected"}, "final alpha types")
    final_candidates = alpha_final[alpha_final.selection_type == "full_train_cv_for_final"]
    selected = alpha_final[alpha_final.selection_type == "final_selected"]
    require(len(final_candidates) == len(HORIZONS) * len(GAT_ALPHA_GRID), "final alpha candidate coverage")
    selected = align(selected, pd.DataFrame({"horizon": HORIZONS}), ["horizon"], "final selected alpha")
    final_alphas, aligned_outer, aligned_test = {}, [], []
    require(set(inner.outer_fold) == set(ALL_FOLDS), "inner outer-fold coverage")
    for h in HORIZONS:
        out = align(outer[outer.horizon == h], refs.train, KEY, f"outer {h}")
        valid = refs.train.hour_idx.to_numpy() + HORIZON_HOURS[h] < PRED_END
        equal(out.outer_fold, refs.row_fold, "outer county fold")
        close(out.y_true, refs.targets[h], "outer official labels", atol=0)
        expected_target_time = pd.Timestamp("2026-03-11") + pd.to_timedelta(out.hour_idx + HORIZON_HOURS[h], unit="h")
        equal(pd.to_datetime(out.target_timestamp).to_numpy(), expected_target_time.to_numpy(), "outer target timestamp")
        for k in ALL_FOLDS:
            S = tuple(i for i in ALL_FOLDS if i != k)
            mask = valid & (refs.row_fold != k)
            inside = align(inner[(inner.horizon == h) & (inner.outer_fold == k)], refs.train.loc[mask], KEY, f"inner {h}/{k}")
            equal(booleans(inside.is_scoreable, "inner"), np.ones(len(inside), dtype=bool), "inner valid mask")
            equal(inside.inner_fold, refs.row_fold[mask], "inner county fold")
            close(inside.y_true, refs.targets[h].to_numpy()[mask], "inner official labels", atol=0)
            require(np.isfinite(inside.base_prediction).all() and np.isfinite(inside.correction_osi).all(), "inner non-finite predictions")
            equal(inside.selection_id, np.repeat(identity("alpha", mode, h, S, variant=variant), len(inside)), "inner selection identity")
            for j in S:
                sub = inside[inside.inner_fold == j]
                allowed = tuple(i for i in S if i != j)
                scopes(sub.allowed_folds, allowed, "inner")
                _sources(sub, mode, h, allowed, inner=True, variant=variant)
            chosen = check_alpha_group(alpha[(alpha.horizon == h) & (alpha.outer_fold == k)],
                                       inside.y_true, inside.base_prediction, inside.correction_osi,
                                       S=S, h=h, mode=mode, selection_type="outer_train_inner_cv", outer_fold=k)
            fold_rows = out[out.outer_fold == k]
            close(fold_rows.alpha, np.full(len(fold_rows), chosen), "outer alpha", atol=0)
            equal(fold_rows.inner_selection_id, np.repeat(identity("alpha", mode, h, S, variant=variant), len(fold_rows)), "outer selection identity")
            _sources(fold_rows, mode, h, S, variant=variant)
        _predictions(out, valid, f"outer {h}")
        aligned_outer.append(out)
        final_alpha = check_alpha_group(final_candidates[final_candidates.horizon == h],
                                        out.loc[valid, "y_true"], out.loc[valid, "base_prediction"], out.loc[valid, "correction_osi"],
                                        S=ALL_FOLDS, h=h, mode=mode, selection_type="full_train_cv_for_final", outer_fold="all")
        close(selected[selected.horizon == h].alpha, [final_alpha], "final selected alpha", atol=0)
        final_alphas[h] = final_alpha
        prediction = align(test[test.horizon == h], refs.test, KEY, f"test {h}")
        expected = refs.test.hour_idx.to_numpy() + HORIZON_HOURS[h] < PRED_END
        close(prediction.alpha, np.full(len(prediction), final_alpha), "test alpha", atol=0)
        equal(prediction.selection_id, np.repeat(identity("alpha", mode, h, ALL_FOLDS, variant=variant), len(prediction)), "test selection")
        _sources(prediction, mode, h, ALL_FOLDS, variant=variant)
        _predictions(prediction, expected, f"test {h}")
        aligned_test.append(prediction)
        expected_graph = pd.concat([refs.train[KEY], refs.test[KEY]], ignore_index=True)
        saved_graph = align(graph[graph.horizon == h], expected_graph, KEY, f"final graph {h}")
        n = len(refs.train)
        equal(saved_graph.split, np.r_[np.repeat("train", n), np.repeat("test", len(refs.test))], "graph split")
        equal(saved_graph.county_fold, np.r_[refs.row_fold, np.repeat(-1, len(refs.test))], "graph county fold")
        close(saved_graph.base_prediction, np.r_[out.base_prediction, prediction.base_prediction], "final graph base")
        equal(saved_graph.base_source_id, np.r_[out.base_source_id, prediction.base_source_id], "final graph base source")
    return pd.concat(aligned_outer, ignore_index=True), pd.concat(aligned_test, ignore_index=True), final_alphas


def recompute_metrics(outer):
    summary, fold, county = [], [], []
    scoreable = outer[outer.is_scoreable.astype(bool)]
    for h, rows in scoreable.groupby("horizon", sort=True):
        for model in ["base", "gat"]:
            p = post_process_osi(rows.base_prediction.to_numpy()) if model == "base" else rows.prediction.to_numpy()
            summary.append(dict(horizon=h, model=model, **metrics(rows.y_true, p)))
        for k, sub in rows.groupby("outer_fold", sort=True):
            require(sub.alpha.nunique() == 1, "outer alpha varies within fold")
            for model in ["base", "gat"]:
                p = post_process_osi(sub.base_prediction.to_numpy()) if model == "base" else sub.prediction.to_numpy()
                fold.append(dict(horizon=h, outer_fold=k, model=model, alpha=0.0 if model == "base" else float(sub.alpha.iloc[0]), **metrics(sub.y_true, p)))
        for fips, sub in rows.groupby("fipsCode", sort=True):
            y = sub.y_true.to_numpy(dtype=float)
            be = post_process_osi(sub.base_prediction.to_numpy()) - y
            ge = sub.prediction.to_numpy(dtype=float) - y
            bs, gs = float(np.sum(be * be, dtype=np.float64)), float(np.sum(ge * ge, dtype=np.float64))
            county.append(dict(horizon=h, fipsCode=fips, outer_fold=int(sub.outer_fold.iloc[0]), n=len(sub),
                               base_sse=bs, gat_sse=gs, base_rmse=np.sqrt(np.mean(be * be)), gat_rmse=np.sqrt(np.mean(ge * ge)),
                               gat_minus_base_sse=gs-bs, gat_better=gs<bs))
    county_frame = pd.DataFrame(county)
    bootstrap = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for h, sub in county_frame.groupby("horizon", sort=True):
        b, g, n = [sub[c].to_numpy(dtype=float) for c in ["base_sse", "gat_sse", "n"]]
        draws = rng.integers(0, len(sub), size=(BOOTSTRAP_REPLICATES, len(sub)))
        total_n = n[draws].sum(axis=1)
        delta = np.sqrt(g[draws].sum(axis=1)/total_n) - np.sqrt(b[draws].sum(axis=1)/total_n)
        br, gr = float(np.sqrt(b.sum()/n.sum())), float(np.sqrt(g.sum()/n.sum()))
        bootstrap.append(dict(horizon=h, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, n_counties=len(sub),
                              base_rmse=br, gat_rmse=gr, delta_rmse_gat_minus_base=gr-br,
                              delta_ci_low=np.quantile(delta, .025), delta_ci_high=np.quantile(delta, .975),
                              gat_better_probability=np.mean(delta<0)))
    return pd.DataFrame(summary), pd.DataFrame(fold), county_frame, pd.DataFrame(bootstrap)


def check_submission(submission, template, test_predictions, refs, *, atol=1e-12):
    equal(list(submission.columns), list(template.columns), "submission columns")
    require(len(submission) == len(template) == len(refs.test), "submission row count")
    identifiers = [name for name in template.columns if name not in HORIZONS]
    pd.testing.assert_frame_equal(submission[identifiers].reset_index(drop=True), template[identifiers].reset_index(drop=True), check_dtype=False)
    template_rows = normalise_counties(template)
    timestamps = pd.to_datetime(template_rows.timestamp_et)
    hours = (timestamps - pd.Timestamp("2026-03-11")) / pd.Timedelta(hours=1)
    require(np.equal(hours, np.floor(hours)).all(), "template timestamp not hourly")
    template_rows["hour_idx"] = hours.astype(int)
    align(template_rows, refs.test, KEY, "template")
    for h in HORIZONS:
        expected = template_rows.hour_idx.to_numpy() + HORIZON_HOURS[h] < PRED_END
        require(np.array_equal(template[h].isna(), ~expected), f"template {h} mask does not match time contract")
        values = submission[h].to_numpy(dtype=float)
        require(np.array_equal(np.isnan(values), ~expected), f"submission {h}: NaN positions mismatch")
        require(np.isfinite(values[expected]).all() and ((values[expected]>=0)&(values[expected]<=.65)).all(), f"submission {h}: invalid values")
        pred = align(test_predictions[test_predictions.horizon == h], template_rows, KEY, f"submission predictions {h}")
        close(values, pred.prediction, f"submission vs prediction rows {h}", atol=atol)


def check_numeric_artifacts(run_dir, manifest, refs):
    read = lambda name: pd.read_parquet(run_dir / name, engine="pyarrow")
    outer, test, selected = check_rows(
        read("outer_oof.parquet"), read("inner_oof.parquet"), read("alpha_selection.parquet"),
        read("alpha_selection_final.parquet"), read("test_predictions.parquet"), read("final_graph_base.parquet"),
        refs, manifest["base_mode"], manifest["run_id"], manifest.get("variant", "m0_full"),
    )
    summary, fold, county, bootstrap = recompute_metrics(outer)
    for filename, expected, keys in [
        ("cv_summary.csv", summary, ["horizon", "model"]),
        ("fold_metrics.csv", fold, ["horizon", "outer_fold", "model"]),
        ("county_metrics.csv", county, ["horizon", "fipsCode"]),
        ("county_bootstrap.csv", bootstrap, ["horizon"]),
    ]:
        dtype = {"fipsCode": str} if "fipsCode" in keys else None
        compare_table(pd.read_csv(run_dir / filename, dtype=dtype), expected, keys, filename)
    submission = pd.read_csv(run_dir / "submission_phase2_dem_gat.csv", dtype={"fipsCode": str})
    template = pd.read_csv(SUBMISSION_FILE, dtype={"fipsCode": str})
    check_submission(submission, template, test, refs)
    return selected
