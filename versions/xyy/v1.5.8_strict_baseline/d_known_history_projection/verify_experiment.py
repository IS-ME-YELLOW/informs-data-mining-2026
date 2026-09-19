"""Independent numerical reconstruction and leakage checks; never fits models."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from project_d import observed_history, known_bounds, project_components

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
START = pd.Timestamp("2026-03-11")
HOURS = (1, 6, 24, 48)
COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
MODES = ("C0_direct_osi", "C1_component_osi", "C2_equal_blend", "C3_aligned_component", "v18_rule")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


def read_csv(path):
    return pd.read_csv(path, dtype={"fipsCode": str}, float_precision="round_trip")


def brute_bounds(raw, meta):
    """Separate scalar implementation: iterate every window, never use future P."""
    lookup = {}
    for row in raw.itertuples(index=False):
        hour = int((pd.Timestamp(row.timestamp_et) - START) / pd.Timedelta(hours=1))
        if 0 <= hour <= 71:
            key = (str(row.fipsCode).zfill(5), hour)
            assert key not in lookup
            lookup[key] = float(row.P_t)
    answer = {}
    for h in HOURS:
        values = []
        for county, origin in zip(meta.fipsCode, meta.hour_idx):
            target = int(origin) + h
            if target > 215:
                values.append(np.nan)
            else:
                values.append(sum(lookup[(county, t)] for t in range(target-5, target+1) if 0 <= t <= 71)/6)
        answer[h] = np.array(values)
    return answer


def finish_osi(values):
    result = np.clip(values, 0, .65)
    result[result < .001] = 0
    return result


def compose(parts):
    return np.maximum(.40*parts["P_t"] + .35*parts["N_t"] + .25*parts["D_t"] - .10*parts["R_t"], 0)


def independent_controls(meta, raw_direct, parts):
    """Use scalar county/target-hour buckets, independent of protocol.py."""
    buckets = {}
    for h in HOURS:
        for i, (county, origin) in enumerate(zip(meta.fipsCode, meta.hour_idx)):
            target = int(origin)+h
            if target <= 215:
                key = (county, target)
                buckets.setdefault(key, []).append((h, [parts[h][c][i] for c in COMPONENTS]))
    means = {key: np.mean([v for _, v in values], axis=0) for key, values in buckets.items()}
    controls = {}
    for h in HOURS:
        aligned = np.array([means.get((f, int(t)+h), [np.nan]*4) for f, t in zip(meta.fipsCode, meta.hour_idx)])
        c1_raw = compose(parts[h])
        direct = raw_direct[f"raw_osi_target_t{h:02d}h"].to_numpy()
        c3_raw = compose(dict(zip(COMPONENTS, aligned.T)))
        values = [finish_osi(direct.copy()), finish_osi(c1_raw), finish_osi((direct+c1_raw)/2), finish_osi(c3_raw)]
        values.append(values[3 if h <= 6 else 1].copy())
        controls[h] = dict(zip(MODES, values))
    return controls, buckets, means


def compare_stats(y, b, a):
    be, ae = b-y, a-y
    br, ar = np.sqrt(np.mean(be**2)), np.sqrt(np.mean(ae**2))
    return {"n": len(y), "baseline_rmse": br, "candidate_rmse": ar,
            "baseline_mae": np.abs(be).mean(), "candidate_mae": np.abs(ae).mean(),
            "baseline_sse": np.dot(be, be), "candidate_sse": np.dot(ae, ae),
            "baseline_bias": be.mean(), "candidate_bias": ae.mean(),
            "rmse_delta": ar-br, "rmse_change_pct": 100*(ar/br-1) if br else np.nan,
            "sse_reduction": np.dot(be, be)-np.dot(ae, ae), "changed_predictions": int(np.sum(a != b))}


def assert_stats(row, y, b, a):
    for key, value in compare_stats(y, b, a).items():
        close(row[key], value)


def must_reject(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Invalid input was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE/"results_seed42")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out.is_relative_to(HERE) and out != HERE
    manifest = json.loads((out/"manifest.json").read_text())
    for name, digest in manifest["input_sha256"].items():
        assert sha(manifest["input_paths"][name]) == digest, name
    for name, digest in manifest["output_sha256"].items():
        assert sha(out/name) == digest, name
    identity = {k: manifest[k] for k in ("experiment", "baseline_identity_hash", "input_sha256")}
    assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == manifest["candidate_identity_hash"]
    protected = json.loads((HERE/"baseline_snapshot.json").read_text())
    for name, digest in protected.items():
        assert sha(BASE/name) == digest, name
    run = Path(manifest["baseline_run"])
    marker = json.loads((run/"CV_COMPLETE").read_text())
    assert marker["identity_hash"] == manifest["baseline_identity_hash"]
    for name, digest in marker["files"].items():
        assert sha(run/name) == digest, name
    frozen = pd.read_parquet(run/"oof_predictions.parquet")
    source_parts = pd.read_parquet(run/"oof_component_predictions.parquet")
    raw_direct = pd.read_parquet(run/"base_predictions_cv.parquet")
    oof = pd.read_parquet(out/"candidate_oof_predictions.parquet")
    part = pd.read_parquet(out/"projected_component_predictions.parquet")
    bound_frame = pd.read_parquet(out/"known_bounds_train.parquet")
    keys = ["fipsCode", "timestamp_et", "hour_idx", "stateAbbr", "fold", "split_id"]
    meta = frozen[keys]
    for frame in [oof, part, bound_frame, source_parts, raw_direct]:
        pd.testing.assert_frame_equal(frame[keys], meta)
    assert len(meta) == 239*144 and meta.fipsCode.nunique() == 239
    exact(meta.hour_idx, np.tile(np.arange(72, 216), 239))
    for frame in [oof, part]:
        assert frame.candidate_identity_hash.eq(manifest["candidate_identity_hash"]).all()
        assert frame.baseline_run_identity_hash.eq(manifest["baseline_identity_hash"]).all()
    raw = read_csv(manifest["input_paths"]["raw_train"])
    bounds = brute_bounds(raw, meta)
    parts = {}
    d_changed = 0
    projected_counties = set()
    for h in HOURS:
        horizon = f"osi_target_t{h:02d}h"
        mask = meta.hour_idx.to_numpy()+h <= 215
        early = mask & (meta.hour_idx.to_numpy()+h <= 76)
        for frame in [bound_frame, part]:
            close(frame[f"known_D_lower_bound_{horizon}"], bounds[h])
        parts[h] = {}
        for c in COMPONENTS:
            target = f"{c}_target_t{h:02d}h"
            old = source_parts[f"pred_{target}"].to_numpy()
            new = part[f"projected_pred_{target}"].to_numpy()
            exact(old, part[f"baseline_pred_{target}"])
            exact(source_parts[f"source_{target}"], part[f"baseline_model_id_{target}"])
            exact(np.isfinite(new), mask)
            close(new, np.maximum(old, bounds[h]) if c == "D_t" else old)
            if c != "D_t": exact(new, old)
            exact(new[~early], old[~early])
            parts[h][c] = new
        old = source_parts[f"pred_D_t_target_t{h:02d}h"].to_numpy()
        new = parts[h]["D_t"]
        changed = mask & (old != new)
        exact(part[f"D_projected_{horizon}"], changed)
        actual = source_parts[f"actual_D_t_target_t{h:02d}h"].to_numpy()
        assert (actual[mask]+1e-6 >= bounds[h][mask]).all()
        assert (np.abs(new[mask]-actual[mask]) <= np.abs(old[mask]-actual[mask])+1e-6).all()
        d_changed += int(changed.sum())
        projected_counties.update(meta.loc[changed, "fipsCode"])
    assert d_changed == manifest["exact_D_projection_rows"] == 92
    assert len(projected_counties) == manifest["projected_counties"] == 52
    controls, buckets, means = independent_controls(meta, raw_direct, parts)
    unique = pd.read_parquet(out/"candidate_aligned_unique_components.parquet")
    assert len(unique) == len(means)*4 and not unique.duplicated(["component", "fipsCode", "target_timestamp"]).any()
    for row in unique.itertuples(index=False):
        key = (row.fipsCode, int((row.target_timestamp-START)/pd.Timedelta(hours=1)))
        close(row.prediction, means[key][COMPONENTS.index(row.component)])
        assert row.candidate_count == len(buckets[key])
        assert row.source_horizons == ",".join(f"osi_target_t{h:02d}h" for h, _ in sorted(buckets[key]))
    for h in HOURS:
        horizon = f"osi_target_t{h:02d}h"
        mask = meta.hour_idx.to_numpy()+h <= 215
        early = mask & (meta.hour_idx.to_numpy()+h <= 76)
        exact(oof[f"actual_{horizon}"], frozen[f"actual_{horizon}"])
        exact(oof[f"is_scoreable_{horizon}"], mask)
        for mode in MODES:
            b = oof[f"baseline_{mode}_{horizon}"].to_numpy()
            a = oof[f"candidate_{mode}_{horizon}"].to_numpy()
            exact(b, frozen[f"pred_{mode}_{horizon}"])
            close(a, controls[h][mode])
            exact(np.isfinite(a), mask)
            exact(oof[f"changed_{mode}_{horizon}"], mask & (a != b))
            exact(a[~early], b[~early])
            if mode == "C0_direct_osi": exact(a, b)
    tables = {name: read_csv(out/f"{name}.csv") for name in ["metrics_comparison", "fold_metrics", "county_metrics", "paired_county_bootstrap", "D_component_metrics", "window_metrics", "primary_county_influence", "changed_rows"]}
    for name in ["metrics_comparison", "fold_metrics", "county_metrics", "primary_county_influence", "window_metrics"]:
        for _, row in tables[name].iterrows():
            horizon = row.get("horizon", "osi_target_t01h")
            y = oof[f"actual_{horizon}"].to_numpy()
            b = oof[f"baseline_{row.model}_{horizon}"].to_numpy()
            a = oof[f"candidate_{row.model}_{horizon}"].to_numpy()
            take = np.isfinite(y)
            if name == "fold_metrics": take &= meta.fold.to_numpy() == row.fold
            if name in ("county_metrics", "primary_county_influence"): take &= meta.fipsCode.to_numpy() == row.fipsCode
            if name == "window_metrics":
                take &= {"first_four_target_hours": meta.hour_idx.to_numpy() <= 75,
                         "remaining_scoreable_hours": meta.hour_idx.to_numpy() > 75,
                         "changed_final_predictions": a != b}[row.window]
            assert_stats(row, y[take], b[take], a[take])
    for _, row in tables["D_component_metrics"].iterrows():
        suffix = row.horizon.rsplit("_", 1)[-1]
        y = source_parts[f"actual_D_t_target_{suffix}"].to_numpy()
        b = part[f"baseline_pred_D_t_target_{suffix}"].to_numpy()
        a = part[f"projected_pred_D_t_target_{suffix}"].to_numpy()
        mask = np.isfinite(y)
        assert_stats(row, y[mask], b[mask], a[mask])
        assert row.exact_projection_rows == (mask & (a != b)).sum()
        assert row.material_projection_rows == (mask & (part[f"known_D_lower_bound_{row.horizon}"].to_numpy()-b > 1e-6)).sum()
    # Reconstruct every saved bootstrap using independently grouped SSE and sequential draws.
    for _, row in tables["paired_county_bootstrap"].iterrows():
        y = oof[f"actual_{row.horizon}"].to_numpy()
        b = oof[f"baseline_{row.model}_{row.horizon}"].to_numpy()
        a = oof[f"candidate_{row.model}_{row.horizon}"].to_numpy()
        take = np.isfinite(y)
        if row.population == "exclude_Morrow_diagnostic_only": take &= meta.fipsCode.to_numpy() != "39117"
        assert_stats(row, y[take], b[take], a[take])
        frame = pd.DataFrame({"fips": meta.fipsCode.to_numpy()[take], "b": (b[take]-y[take])**2, "a": (a[take]-y[take])**2})
        grouped = frame.groupby("fips", sort=True).agg(b=("b", "sum"), a=("a", "sum"), n=("b", "size"))
        assert len(grouped) == row.counties
        values = grouped.to_numpy()
        generator = np.random.default_rng(int(row.seed))
        deltas = []
        for _ in range(int(row.replicates)):
            bs, ass, ns = values[generator.integers(0, len(values), len(values))].sum(axis=0)
            deltas.append(np.sqrt(ass/ns)-np.sqrt(bs/ns))
        close([row.ci_low, row.ci_high], np.quantile(deltas, [.025, .975]))
        close(row.bootstrap_fraction_delta_lt0, np.mean(np.asarray(deltas) < 0))
        close(row.bootstrap_fraction_delta_eq0, np.mean(np.asarray(deltas) == 0))
    changes = tables["changed_rows"]
    assert len(changes) == d_changed and not changes.duplicated(["fipsCode", "origin_timestamp", "horizon"]).any()
    index = pd.MultiIndex.from_frame(meta[["fipsCode", "timestamp_et"]])
    changed_i = index.get_indexer(pd.MultiIndex.from_arrays([changes.fipsCode, pd.to_datetime(changes.origin_timestamp)]))
    assert (changed_i >= 0).all()
    exact(changes.target_hour, meta.hour_idx.to_numpy()[changed_i]+1)
    close(changes.baseline_D, part.baseline_pred_D_t_target_t01h.to_numpy()[changed_i])
    close(changes.projected_D, part.projected_pred_D_t_target_t01h.to_numpy()[changed_i])
    close(changes.known_D_lower_bound, bounds[1][changed_i])
    close(changes.actual_D, source_parts.actual_D_t_target_t01h.to_numpy()[changed_i])
    close(changes.D_sse_reduction, (changes.baseline_D-changes.actual_D)**2-(changes.projected_D-changes.actual_D)**2)
    for mode in MODES:
        b = oof[f"baseline_{mode}_osi_target_t01h"].to_numpy()[changed_i]
        a = oof[f"candidate_{mode}_osi_target_t01h"].to_numpy()[changed_i]
        y = oof.actual_osi_target_t01h.to_numpy()[changed_i]
        close(changes[f"baseline_{mode}"], b)
        close(changes[f"candidate_{mode}"], a)
        close(changes[f"sse_reduction_{mode}"], (b-y)**2-(a-y)**2)
    influence = tables["primary_county_influence"]
    close(influence.fraction_of_total_sse_reduction, influence.sse_reduction/influence.sse_reduction.sum())
    close(influence.sse_reduction.sum(), changes.sse_reduction_v18_rule.sum())
    # Prediction-only dependency tests. All training labels are absent from this frame.
    raw_only = raw[["fipsCode", "timestamp_et", "P_t"]].copy()
    history = observed_history(raw_only)
    reference = known_bounds(meta, history)
    future = pd.to_datetime(raw_only.timestamp_et) >= START+pd.Timedelta(hours=72)
    for replacement in [np.nan, 999., np.random.default_rng(7).normal(size=int(future.sum()))]:
        perturbed = raw_only.copy()
        perturbed.loc[future, "P_t"] = replacement
        changed_history = observed_history(perturbed)
        pd.testing.assert_frame_equal(history, changed_history)
        for name, values in known_bounds(meta, changed_history).items(): exact(values, reference[name])
    observed_only = raw_only.loc[~future].copy()
    pd.testing.assert_frame_equal(observed_history(observed_only), history)
    shuffled = meta.sample(frac=1, random_state=17)
    shuffled_bounds = known_bounds(shuffled, history.sample(frac=1, random_state=19))
    for name in reference: exact(shuffled_bounds[name], reference[name][shuffled.index])
    must_reject(lambda: observed_history(observed_only.drop(observed_only.index[0])))
    must_reject(lambda: observed_history(pd.concat([observed_only, observed_only.iloc[:1]])))
    must_reject(lambda: known_bounds(pd.concat([meta.iloc[:1], meta.iloc[:1]]), history))
    mismatch = meta.copy(); mismatch.loc[0, "timestamp_et"] += pd.Timedelta(hours=1)
    must_reject(lambda: known_bounds(mismatch, history))
    future_history = history.copy(); future_history.loc[0, "hour_idx"] = 72
    must_reject(lambda: known_bounds(meta, future_history))
    test_raw = read_csv(manifest["input_paths"]["raw_test"])
    test_saved = pd.read_parquet(out/"known_bounds_test.parquet")
    assert test_saved.fipsCode.nunique() == 63 and len(test_saved) == 63*144
    assert set(test_saved) == {"fipsCode", "timestamp_et", "hour_idx"} | {f"known_D_lower_bound_osi_target_t{h:02d}h" for h in HOURS}
    for h, values in brute_bounds(test_raw, test_saved).items(): close(test_saved[f"known_D_lower_bound_osi_target_t{h:02d}h"], values)
    test_future = pd.to_datetime(test_raw.timestamp_et) >= START+pd.Timedelta(hours=72)
    assert test_raw.loc[test_future, "P_t"].isna().all()
    test_history_only = test_raw.loc[~test_future, ["fipsCode", "timestamp_et", "P_t"]]
    test_bounds = known_bounds(test_saved, observed_history(test_history_only))
    for name, values in test_bounds.items(): exact(values, test_saved[f"known_D_lower_bound_{name}"])
    # Exercise the pure projector with no labels and prove input arrays are not mutated.
    base = {f"osi_target_t{h:02d}h": {c: source_parts[f"pred_{c}_target_t{h:02d}h"].to_numpy(copy=True) for c in COMPONENTS} for h in HOURS}
    before = {h: {c: a.copy() for c, a in values.items()} for h, values in base.items()}
    projected = project_components(base, reference)
    for h in HOURS:
        name = f"osi_target_t{h:02d}h"
        for c in COMPONENTS:
            exact(base[name][c], before[name][c])
            exact(projected[name][c], parts[h][c])
    for name, digest in protected.items(): assert sha(BASE/name) == digest
    result = {"status": "PASS", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_identity_hash": manifest["candidate_identity_hash"], "manifest_sha256": sha(out/"manifest.json"),
              "verifier_sha256": sha(__file__), "protected_baseline_files_unchanged": len(protected),
              "train_counties": 239, "test_counties": 63, "D_projected_rows": d_changed,
              "checks": ["all input/output hashes", "baseline identity and CV_COMPLETE", "independent scalar D bounds for train and test",
                         "independent C0/C1/C2/C3 and fixed primary recomposition", "all aligned component candidates and counts",
                         "all pooled/fold/county/window/component metrics", "all saved county bootstrap intervals",
                         "changed-row details and county influence", "exact unchanged P/N/R, C0, later rows and other horizons",
                         "future P perturbation and deletion invariance", "forecast row-order invariance and invalid-key rejection",
                         "test bounds without future P or labels", "pure projector input immutability"],
              "training_performed": False, "test_predictions_created": False}
    (out/"verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
