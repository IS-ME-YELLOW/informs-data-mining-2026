"""Verify saved matching artifacts against inputs, outcomes and distance rules."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def read(name):
    return pd.read_csv(OUT / name, dtype={"fipsCode": str, "query_fips": str, "match_fips": str},
                       float_precision="round_trip")


def main():
    manifest = json.loads((OUT / "analysis_manifest.json").read_text())
    for name, digest in manifest["input_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((OUT / "analyze_similar_counties.py").read_bytes()).hexdigest() == manifest["source_sha256"]
    x = read("observable_features.csv")
    r = read("rankings_without_outcomes.csv")
    top = read("focus_top10.csv")
    summaries = read("focus_match_summary.csv")
    support = read("support_diagnostics.csv")
    outcomes = read("county_window_outcomes.csv")
    assert len(x) == len(outcomes) == 956
    assert len(top) == 7 * 5 * 10
    assert len(summaries) == 7 * 5 * 2
    assert len(support) == 239 * 4 * 5
    assert not r.duplicated(["query_fips", "window", "variant", "rank"]).any()
    assert not r.duplicated(["query_fips", "window", "variant", "match_fips"]).any()
    assert (r.excluded_fold != r.match_fold).all()
    assert (r.query_fips != r.match_fips).all()
    assert not any("actual" in c or "prediction" in c or "outcome" in c for c in r.columns)
    groups = r.groupby(["query_fips", "window", "variant"], sort=False)
    assert len(groups) == 239 * 4 * 5
    for _, g in groups:
        assert np.array_equal(g["rank"], np.arange(1, len(g) + 1))
        assert np.all(np.diff(g.distance) >= -1e-14)
    max_error = 0.0
    for (q, window, variant), g in top.groupby(["query_fips", "window", "variant"]):
        query = x[(x.fipsCode == q) & (x.window == window)].iloc[0]
        pool = x[(x.window == window) & (x.fold != query.fold)].sort_values("fipsCode")
        distance_squared = np.zeros(len(pool))
        blocks = manifest["variants"][variant]
        for block in blocks:
            block_squared = np.zeros(len(pool))
            for name in manifest["blocks"][block]:
                p = pool[name].to_numpy(dtype=float)
                v = float(query[name])
                if variant.endswith("_rank"):
                    ordered = np.sort(p)
                    cp = (np.searchsorted(ordered, p, side="left") + np.searchsorted(ordered, p, side="right")) / (2 * len(p))
                    cv = (np.searchsorted(ordered, v, side="left") + np.searchsorted(ordered, v, side="right")) / (2 * len(p))
                else:
                    if name in manifest["blocks"]["history"]:
                        p, v = np.arcsinh(p / .01), np.arcsinh(v / .01)
                    elif name in manifest["log1p_features"]:
                        p, v = np.log1p(p), np.log1p(v)
                    scale = p.std(ddof=0)
                    if scale < 1e-12:
                        scale = 1.0
                    cp, cv = (p - p.mean()) / scale, (v - p.mean()) / scale
                block_squared += (cp - cv) ** 2 / len(manifest["blocks"][block])
            distance_squared += block_squared / len(blocks)
        d = np.sqrt(distance_squared)
        order = np.argsort(d, kind="stable")[:10]
        g = g.sort_values("rank")
        assert list(pool.iloc[order].fipsCode) == list(g.match_fips), (q, window, variant)
        max_error = max(max_error, float(np.max(np.abs(d[order] - g.distance.to_numpy()))))
        for k in (5, 10):
            t = g[g["rank"] <= k]
            s = summaries[(summaries.query_fips == q) & (summaries.window == window) &
                          (summaries.variant == variant) & (summaries.k == k)].iloc[0]
            assert (t.match_actual_peak < .01).sum() == s.neighbors_low_count
            assert (t.match_actual_peak >= .05).sum() == s.neighbors_high_count
            assert np.isclose(t.match_actual_peak.median(), s.neighbor_peak_median, atol=1e-14, rtol=0)
    assert max_error < 1e-10
    trajectories = read("county_window_trajectories.csv")
    assert not trajectories.duplicated(["window", "fipsCode", "target_hour"]).any()
    raw_path = next(Path(p) for p in manifest["input_sha256"] if p.endswith("data/DM_Train.csv"))
    raw = pd.read_csv(raw_path, dtype={"fipsCode": str})
    raw["target_hour"] = ((pd.to_datetime(raw.timestamp_et) - pd.Timestamp("2026-03-11")) / pd.Timedelta(hours=1)).astype(int)
    joined = trajectories.merge(raw[["fipsCode", "target_hour", "osi"]], on=["fipsCode", "target_hour"], validate="many_to_one")
    assert len(joined) == len(trajectories)
    assert np.array_equal(joined.actual.to_numpy(), joined.osi.to_numpy())
    for (window, fips), g in trajectories.groupby(["window", "fipsCode"]):
        spec = manifest["windows"][window]
        assert len(g) == spec["end"] - spec["start"] + 1
        o = outcomes[(outcomes.window == window) & (outcomes.fipsCode == fips)].iloc[0]
        assert np.isclose(g.actual.max(), o.actual_peak, atol=1e-14, rtol=0)
        assert np.isclose(g.actual.mean(), o.actual_mean, atol=1e-14, rtol=0)
        assert (g.actual >= .05).sum() == o.hours_ge005
    result = {
        "passed": True, "training_performed": False,
        "input_hashes_unchanged": True, "analysis_code_hash_matches": True,
        "county_window_rows": len(x), "focus_top10_rows": len(top),
        "all_county_variant_groups": len(groups), "focus_variant_recomputed_groups": 35,
        "entire_query_fold_excluded": True, "ranking_keys_unique_and_distances_sorted": True,
        "ranking_artifact_has_no_outcomes_or_predictions": True,
        "recomputed_focus_top10_identities_match": True,
        "distance_recompute_max_abs_difference": max_error,
        "summary_counts_and_medians_recomputed": True,
        "all_county_window_outcomes_recomputed": True,
        "trajectory_actuals_equal_raw_osi_exactly": True,
        "limits": "Checks numerical and data-flow consistency, not causal identification or completeness of available predictive information.",
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.csv"))},
    }
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "output_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
