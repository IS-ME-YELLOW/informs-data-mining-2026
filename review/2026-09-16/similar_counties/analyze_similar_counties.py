"""Outcome-blind county matching for error diagnosis; never trains a model.

Run with the project .venv Python. Writes only beside this script. Historical
OOF scores describe each county's own saved fold model, not a common refit.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
WENDY = ROOT.parent / "informs-data-mining-wendyxu"
WINDOWS = {
    "early": {"weather_start": 72, "start": 73, "end": 95, "horizon": 1},
    "renewed": {"weather_start": 96, "start": 96, "end": 119, "horizon": 24},
    "second_wave": {"weather_start": 120, "start": 120, "end": 167, "horizon": 48},
    "late": {"weather_start": 168, "start": 168, "end": 191, "horizon": 48},
}
CASES = [
    ("42053", "early"), ("39117", "early"), ("18013", "renewed"),
    ("54015", "second_wave"), ("54013", "second_wave"),
    ("54007", "second_wave"), ("39115", "late"),
]
BLOCKS = {
    "history": ["last_P_t", "last_D_t", "last_N_t", "last_R_t",
                "osi_max_72h", "osi_mean_72h", "osi_trend_last6h"],
    "weather": ["gust_max", "gust_mean", "gust_hours_gt30", "tp_sum",
                "soil_mean", "prior24_gust_max", "prior24_tp_sum"],
    "static": ["customers_at_cutoff", "pop_density", "pct_forest", "n_utilities"],
    "terrain": ["elevation_mean_m", "elevation_std_m", "slope_mean_deg"],
}
VARIANTS = {
    "context": ["history", "weather", "static"],
    "context_dem": ["history", "weather", "static", "terrain"],
    "weather_only": ["weather"],
    "context_rank": ["history", "weather", "static"],
    "full_cache_rank": ["cache_history", "cache_weather_time", "cache_static"],
}
SOURCES = {
    "raw": ROOT / "data/DM_Train.csv",
    "folds": ROOT / "cv/cv_assignments_balanced_v1_seed42.csv",
    "features": ROOT / "versions/xyy/v1.5.6/features_train_v1.5.6.parquet",
    "meta": ROOT / "versions/xyy/v1.5.6/meta_train_v1.5.6.parquet",
    "external": ROOT / "data/county_features_v1.5.6.csv",
    "terrain": WENDY / "data/geo/county_terrain.csv",
    "oof": ROOT / "versions/v1.12_tail_protected_ensemble/artifacts/oof_predictions.parquet",
}


def normalize_codes(frame):
    frame = frame.copy()
    frame["fipsCode"] = frame.fipsCode.astype(str).str.zfill(5)
    return frame


def build_inputs():
    raw = normalize_codes(pd.read_csv(SOURCES["raw"]))
    raw["time"] = pd.to_datetime(raw.timestamp_et)
    raw = raw.sort_values(["fipsCode", "time"]).reset_index(drop=True)
    raw["hour"] = ((raw.time - pd.Timestamp("2026-03-11")) / pd.Timedelta(hours=1)).astype(int)
    assert not raw.duplicated(["fipsCode", "hour"]).any()
    assert raw.groupby("fipsCode").size().eq(216).all()
    folds = normalize_codes(pd.read_csv(SOURCES["folds"]))
    assert len(folds) == 239 and not folds.fipsCode.duplicated().any()
    meta = normalize_codes(pd.read_parquet(SOURCES["meta"]))
    X = pd.read_parquet(SOURCES["features"])
    assert len(X) == len(meta) == 34416
    cache_static_names = {
        "log_customers", "rural_urban_code", "is_metro", "pop_density", "n_utilities",
        "pct_forest", "pct_developed", "pct_agriculture", "pct_water", "pct_wetland",
        "pct_unmapped", "pct_forest_classified", "tree_canopy_pct", "tree_canopy_std",
        "forest_x_customers",
    }
    cache_history_names = list(X.columns[:31])
    cache_groups = {
        "cache_history": cache_history_names,
        "cache_weather_time": [c for c in X if c not in cache_history_names and c not in cache_static_names],
        "cache_static": [c for c in X if c in cache_static_names],
    }
    assert sum(map(len, cache_groups.values())) == 163
    for group, names in cache_groups.items():
        BLOCKS[group] = ["cache__" + name for name in names]
    hist = X[BLOCKS["history"]].copy()
    hist["fipsCode"] = meta.fipsCode.to_numpy()
    assert hist.groupby("fipsCode").nunique(dropna=False).max().max() == 1
    hist = hist.groupby("fipsCode").first()
    cutoff = raw[raw.hour == 71].set_index("fipsCode")
    county = cutoff[["countyName", "stateAbbr", "customersTracked"]].rename(
        columns={"customersTracked": "customers_at_cutoff"})
    county = county.join(hist, validate="one_to_one")
    ext = normalize_codes(pd.read_csv(SOURCES["external"])).set_index("fipsCode")
    terrain = normalize_codes(pd.read_csv(SOURCES["terrain"])).set_index("fipsCode")
    county = county.join(ext[["pop_density", "pct_forest", "n_utilities"]], validate="one_to_one")
    county = county.join(terrain[BLOCKS["terrain"]], validate="one_to_one")
    county = county.join(folds.set_index("fipsCode")[["fold"]], validate="one_to_one")
    obs = raw[raw.hour < 72].groupby("fipsCode")
    county["observed_peak_P"] = obs.P_t.max()  # displayed only; not a matching column
    county["observed_mean_P"] = obs.P_t.mean()
    rows = []
    for window, spec in WINDOWS.items():
        weather = raw[raw.hour.between(spec["weather_start"], spec["end"])].copy()
        weather["gust_gt30"] = weather.gust > 30
        agg = weather.groupby("fipsCode").agg(
            gust_max=("gust", "max"), gust_mean=("gust", "mean"),
            gust_hours_gt30=("gust_gt30", "sum"), tp_sum=("tp", "sum"),
            soil_mean=("soil_moist", "mean"))
        prior = raw[raw.hour.between(spec["weather_start"] - 24, spec["weather_start"] - 1)]
        agg = agg.join(prior.groupby("fipsCode").agg(
            prior24_gust_max=("gust", "max"), prior24_tp_sum=("tp", "sum")))
        frame = county.join(agg, validate="one_to_one").reset_index()
        origin_mask = meta.hour_idx.between(spec["start"] - spec["horizon"], spec["end"] - spec["horizon"])
        cache_slice = X.loc[origin_mask].copy()
        cache_slice["fipsCode"] = meta.loc[origin_mask, "fipsCode"].to_numpy()
        cache_mean = cache_slice.groupby("fipsCode").mean().add_prefix("cache__")
        frame = frame.merge(cache_mean, on="fipsCode", validate="one_to_one")
        frame["window"] = window
        for key, value in spec.items():
            frame[key] = value
        rows.append(frame)
    inputs = pd.concat(rows, ignore_index=True)
    match_names = sum(BLOCKS.values(), [])
    assert np.isfinite(inputs[match_names].to_numpy(dtype=float)).all()
    assert not inputs.duplicated(["fipsCode", "window"]).any()
    return raw, inputs


def transform(inputs, names, rank_reference=None):
    values = inputs[names].to_numpy(dtype=float).copy()
    if rank_reference is not None:
        for j, name in enumerate(names):
            reference = np.sort(rank_reference[name].to_numpy(dtype=float))
            values[:, j] = (np.searchsorted(reference, values[:, j], side="left") +
                             np.searchsorted(reference, values[:, j], side="right")) / (2 * len(reference))
        return values
    for j, name in enumerate(names):
        if name in BLOCKS["history"]:
            values[:, j] = np.arcsinh(values[:, j] / 0.01)
        elif name in {"customers_at_cutoff", "pop_density", "n_utilities", "tp_sum", "prior24_tp_sum"}:
            values[:, j] = np.log1p(values[:, j])
    return values


def matching(inputs):
    """This function cannot access any future outcome or OOF prediction."""
    ranking, support, differences = [], [], []
    focus_set = set(CASES)
    for window, full in inputs.groupby("window", sort=False):
        full = full.sort_values("fipsCode").reset_index(drop=True)
        for excluded_fold in range(5):
            candidates = full[full.fold != excluded_fold].reset_index(drop=True)
            queries = full[full.fold == excluded_fold].reset_index(drop=True)
            for variant, blocks in VARIANTS.items():
                names = sum([BLOCKS[b] for b in blocks], [])
                ref = candidates if variant.endswith("_rank") else None
                c = transform(candidates, names, ref)
                q = transform(queries, names, ref)
                if ref is None:
                    center, scale = c.mean(axis=0), c.std(axis=0, ddof=0)
                    scale[scale < 1e-12] = 1
                    c, q = (c - center) / scale, (q - center) / scale
                # Blocks get equal total weight; features within a block get equal weight.
                weights = np.array([1 / (len(blocks) * len(BLOCKS[b])) for b in blocks for _ in BLOCKS[b]])
                distance = np.sqrt(np.sum((q[:, None, :] - c[None, :, :]) ** 2 * weights, axis=2))
                reference_distance = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2 * weights, axis=2))
                np.fill_diagonal(reference_distance, np.inf)
                reference_sorted = np.sort(reference_distance, axis=1)
                d1_ref, d5_ref = reference_sorted[:, 0], reference_sorted[:, 4]
                for i, query in queries.iterrows():
                    order = np.argsort(distance[i], kind="stable")
                    sorted_d = distance[i, order]
                    is_focus = (query.fipsCode, window) in focus_set
                    support.append({
                        "query_fips": query.fipsCode, "window": window, "variant": variant,
                        "excluded_fold": excluded_fold, "candidate_count": len(candidates),
                        "nearest_distance": sorted_d[0], "fifth_distance": sorted_d[4],
                        "nearest_distance_percentile": float(np.mean(d1_ref <= sorted_d[0])),
                        "fifth_distance_percentile": float(np.mean(d5_ref <= sorted_d[4])),
                        "reference_d5_q95": float(np.quantile(d5_ref, .95)),
                        "is_focus": is_focus,
                    })
                    # All 239 counties get top 10; focus cases retain their complete ranking.
                    keep = len(order) if is_focus else 10
                    for rank, idx in enumerate(order[:keep], 1):
                        candidate = candidates.iloc[idx]
                        row = {
                            "query_fips": query.fipsCode, "window": window, "variant": variant,
                            "excluded_fold": excluded_fold, "match_fips": candidate.fipsCode,
                            "match_fold": int(candidate.fold), "rank": rank,
                            "distance": float(distance[i, idx]), "is_focus": is_focus,
                        }
                        start = 0
                        for block in blocks:
                            end = start + len(BLOCKS[block])
                            row["distance_" + block] = float(np.sqrt(np.mean((q[i, start:end] - c[idx, start:end]) ** 2)))
                            start = end
                        ranking.append(row)
                        if is_focus and rank <= 10:
                            for j, name in enumerate(names):
                                differences.append({
                                    "query_fips": query.fipsCode, "window": window, "variant": variant,
                                    "match_fips": candidate.fipsCode, "rank": rank, "feature": name,
                                    "query_value": query[name], "match_value": candidate[name],
                                    "transformed_scaled_difference": float(q[i, j] - c[idx, j]),
                                })
    ranking = pd.DataFrame(ranking)
    assert (ranking.excluded_fold != ranking.match_fold).all()
    assert (ranking.query_fips != ranking.match_fips).all()
    return ranking, pd.DataFrame(support), pd.DataFrame(differences)


def build_outcomes(raw):
    oof = normalize_codes(pd.read_parquet(SOURCES["oof"]))
    oof["time"] = pd.to_datetime(oof.timestamp_et)
    lookup = raw.set_index(["fipsCode", "hour"])
    rows = []
    trajectories = []
    for window, spec in WINDOWS.items():
        h = spec["horizon"]
        work = oof.copy()
        work["target_hour"] = work.hour_idx + h
        work = work[work.target_hour.between(spec["start"], spec["end"])].copy()
        actual = lookup.loc[pd.MultiIndex.from_arrays([work.fipsCode, work.target_hour]), "osi"].to_numpy()
        assert np.array_equal(actual, work[f"actual_osi_target_t{h:02d}h"].to_numpy())
        work["actual"] = actual
        work["prediction"] = work[f"pred_tail_protected_osi_target_t{h:02d}h"]
        assert np.isfinite(work[["actual", "prediction"]]).all().all()
        work["sse"] = (work.actual - work.prediction) ** 2
        work["window"] = window
        work["target_time"] = work.time + pd.Timedelta(hours=h)
        trajectories.append(work[["window", "fipsCode", "target_hour", "target_time", "actual", "prediction"]])
        for fips, group in work.groupby("fipsCode"):
            assert len(group) == spec["end"] - spec["start"] + 1
            peak_row = group.loc[group.actual.idxmax()]
            outcome_peak = float(group.actual.max())
            rows.append({
                "fipsCode": fips, "window": window, "horizon": h, "hours": len(group),
                "actual_peak": outcome_peak, "actual_mean": float(group.actual.mean()),
                "actual_sum": float(group.actual.sum()), "actual_peak_hour": int(peak_row.target_hour),
                "hours_ge005": int((group.actual >= .05).sum()),
                "outcome_band": "low_lt001" if outcome_peak < .01 else ("high_ge005" if outcome_peak >= .05 else "moderate"),
                "oof_pred_peak": float(group.prediction.max()),
                "oof_pred_at_actual_peak": float(peak_row.prediction),
                "oof_pred_mean": float(group.prediction.mean()),
                "oof_rmse": float(np.sqrt(group.sse.mean())),
                "oof_bias": float((group.prediction - group.actual).mean()),
            })
    return pd.DataFrame(rows), pd.concat(trajectories, ignore_index=True)


def summarize(inputs, outcomes, ranking, support):
    display = inputs.merge(outcomes, on=["fipsCode", "window"], validate="one_to_one", suffixes=("", "_outcome"))
    rename_match = {c: "match_" + c for c in display.columns if c not in {"window", "fipsCode"}}
    match_display = display.rename(columns={"fipsCode": "match_fips", **rename_match})
    annotated = ranking.merge(match_display, on=["match_fips", "window"], validate="many_to_one", suffixes=("", "_display"))
    result = []
    for (q, window, variant), group in annotated.groupby(["query_fips", "window", "variant"], sort=False):
        query = display[(display.fipsCode == q) & (display.window == window)].iloc[0]
        pool = display[(display.window == window) & (display.fold != query.fold)]
        for k in (5, 10):
            g = group[group["rank"] <= k]
            assert len(g) == k
            result.append({
                "query_fips": q, "query_name": query.countyName, "query_state": query.stateAbbr,
                "window": window, "variant": variant, "k": k,
                "query_peak": query.actual_peak, "query_mean": query.actual_mean,
                "query_oof_peak": query.oof_pred_peak, "query_oof_mean": query.oof_pred_mean,
                "neighbors_low_count": int((g.match_actual_peak < .01).sum()),
                "neighbors_moderate_count": int(((g.match_actual_peak >= .01) & (g.match_actual_peak < .05)).sum()),
                "neighbors_high_count": int((g.match_actual_peak >= .05).sum()),
                "neighbors_ge010_count": int((g.match_actual_peak >= .10).sum()),
                "neighbor_peak_median": float(g.match_actual_peak.median()),
                "neighbor_peak_max": float(g.match_actual_peak.max()),
                "neighbor_mean_median": float(g.match_actual_mean.median()),
                "pool_high_fraction": float((pool.actual_peak >= .05).mean()),
                "pool_peak_max": float(pool.actual_peak.max()),
                "query_above_all_matched_peaks": bool(query.actual_peak > g.match_actual_peak.max()),
                "nearest_low_rank_in_full_ranking": int(group.loc[group.match_actual_peak < .01, "rank"].min()) if (group.match_actual_peak < .01).any() else None,
            })
    summary = pd.DataFrame(result).merge(support, on=["query_fips", "window", "variant"], validate="many_to_one")
    return display, annotated, summary


def supplementary_tables(inputs, display, annotated, summary):
    ranges = []
    names = sum([BLOCKS[b] for b in ["history", "weather", "static", "terrain"]], [])
    for fips, window in CASES:
        query = inputs[(inputs.fipsCode == fips) & (inputs.window == window)].iloc[0]
        pool = inputs[(inputs.window == window) & (inputs.fold != query.fold)]
        for name in names:
            ranges.append({
                "query_fips": fips, "window": window, "feature": name,
                "query_value": query[name], "candidate_min": pool[name].min(),
                "candidate_max": pool[name].max(), "percentile": float((pool[name] <= query[name]).mean()),
                "outside_range": bool(query[name] < pool[name].min() or query[name] > pool[name].max()),
            })
    pd.DataFrame(ranges).to_csv(OUT / "focus_feature_support.csv", index=False)
    # Select the nearest low/high outcome only AFTER the label-blind top10 is fixed.
    contrast = []
    focus = annotated[annotated.is_focus & (annotated["rank"] <= 10)]
    for keys, group in focus.groupby(["query_fips", "window", "variant"], sort=False):
        for band in ["low_lt001", "moderate", "high_ge005"]:
            eligible = group[group.match_outcome_band == band].sort_values("rank")
            if len(eligible):
                row = eligible.iloc[0].to_dict()
                row["selection_rule"] = "nearest in outcome band AFTER fixing label-blind top10; missing bands not backfilled"
                contrast.append(row)
    pd.DataFrame(contrast).to_csv(OUT / "nearest_outcome_contrasts.csv", index=False)
    sensitivity = []
    for keys, group in focus.groupby(["query_fips", "window", "variant"], sort=False):
        for k in (5, 10):
            g = group[group["rank"] <= k]
            sensitivity.append({
                "query_fips": keys[0], "window": keys[1], "variant": keys[2], "k": k,
                "peak_lt001": int((g.match_actual_peak < .01).sum()),
                "peak_lt002": int((g.match_actual_peak < .02).sum()),
                "peak_ge005": int((g.match_actual_peak >= .05).sum()),
                "peak_ge010": int((g.match_actual_peak >= .10).sum()),
                "median_hours_ge005": float(g.match_hours_ge005.median()),
                "median_window_mean": float(g.match_actual_mean.median()),
            })
    pd.DataFrame(sensitivity).to_csv(OUT / "outcome_threshold_sensitivity.csv", index=False)
    # Document support for all counties, so the focus cases are not the only
    # examples against which a large nearest-neighbor distance is judged.
    global_rows = []
    for (window, variant), group in summary[summary.k == 10].groupby(["window", "variant"]):
        global_rows.append({
            "window": window, "variant": variant, "counties": len(group),
            "nearest_percentile_ge095_count": int((group.nearest_distance_percentile >= .95).sum()),
            "fifth_percentile_ge095_count": int((group.fifth_distance_percentile >= .95).sum()),
        })
    pd.DataFrame(global_rows).to_csv(OUT / "all_county_support_counts.csv", index=False)


def main():
    raw, inputs = build_inputs()
    # Persist observable-only data and specifications before outcomes are attached.
    inputs.to_csv(OUT / "observable_features.csv", index=False)
    ranking, support, differences = matching(inputs)
    ranking.to_csv(OUT / "rankings_without_outcomes.csv", index=False)
    outcomes, trajectories = build_outcomes(raw)
    display, annotated, summary = summarize(inputs, outcomes, ranking, support)
    display.to_csv(OUT / "county_window_outcomes.csv", index=False)
    annotated[annotated.is_focus].to_csv(OUT / "focus_full_rankings.csv", index=False)
    annotated[annotated.is_focus & (annotated["rank"] <= 10)].to_csv(OUT / "focus_top10.csv", index=False)
    summary.to_csv(OUT / "all_county_match_summary.csv", index=False)
    summary[summary.is_focus].to_csv(OUT / "focus_match_summary.csv", index=False)
    support.to_csv(OUT / "support_diagnostics.csv", index=False)
    differences.to_csv(OUT / "focus_feature_differences.csv", index=False)
    trajectories.to_csv(OUT / "county_window_trajectories.csv", index=False)
    supplementary_tables(inputs, display, annotated, summary)
    manifest = {
        "analysis_date": "2026-09-16", "training_performed": False,
        "python": sys.executable, "numpy": np.__version__, "pandas": pd.__version__,
        "focus_cases": CASES, "windows": WINDOWS, "blocks": BLOCKS, "variants": VARIANTS,
        "design": "Exploratory follow-up of previously selected high-error counties. Matching uses no future outage labels or predictions; outcomes are attached afterward.",
        "candidate_rule": "239 training counties only; exclude the query's entire original balanced_v1 county fold. Never use test outcomes.",
        "scaling": "context/context_dem/weather_only: specified transforms, candidate-only mean/population std; equal block weights. context_rank: candidate mid-ECDF, equal block weights, no subsequent z-score.",
        "positive_history_transform": "asinh(value/0.01), including signed OSI trend",
        "log1p_features": ["customers_at_cutoff", "pop_density", "n_utilities", "tp_sum", "prior24_tp_sum"],
        "k": [5, 10], "outcome_bands": {"low": "window peak OSI<0.01", "high": ">=0.05", "moderate": "otherwise"},
        "outcome_threshold_sensitivity": "Also counts window peak<0.02 and >=0.10; no matching metric is selected by these counts.",
        "support_percentile_reference": "Leave-one-county-out nearest/fifth distance within the eligible candidate pool, using the same outcome-free scaling. Descriptive support diagnostic, not a calibrated OOD test.",
        "oof_limitation": "Each matched county's saved v1.12 prediction comes from its own historical fold model. Such models can have used focus county labels; these scores are descriptive and are not a common outer-model comparison or strict new nested evaluation.",
        "label_selection_limitation": "Focus counties and calendar windows follow earlier error diagnosis; this is not prospective validation, causal inference or a fitted classifier.",
        "full_cache_sensitivity": "Added after inspecting compact matching to check representation sensitivity, with no outcome tuning. All 163 frozen columns are averaged over each horizon's corresponding origin window, then candidate mid-ECDF transformed, grouped as 31 history / 117 weather-time / 15 static. Temporal averaging loses within-window order; this is not a test of all possible information in the original data.",
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES.values()},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": {"candidate_fold_exclusion": True, "future_osi_alignment_exact": True,
                   "matching_inputs_finite": True, "county_hour_unique": True,
                   "counties": int(inputs.fipsCode.nunique()), "windows": len(WINDOWS)},
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(summary[summary.is_focus & (summary.variant == "context") & (summary.k == 10)][[
        "query_fips", "query_name", "window", "query_peak", "neighbors_low_count",
        "neighbors_high_count", "neighbor_peak_median", "neighbor_peak_max",
        "nearest_distance_percentile", "fifth_distance_percentile"]].to_string(index=False))


if __name__ == "__main__":
    main()
