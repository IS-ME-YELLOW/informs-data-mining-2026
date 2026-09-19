"""Analyze the completed seed42 strict CV; never trains or changes predictions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE / "runs/v18_tree_nested_v2_split42_model42"
OUT = HERE / "analysis"
OLD_OOF = HERE / "reference/historical_v18_oof_predictions.parquet"
OLD_MODELS = HERE / "reference/historical_v18_model_manifest.csv"
HOURS = (1, 6, 24, 48)
FOCUS = ("42053", "39117", "18013", "54015", "54013", "54007", "39115", "39119")
WINDOWS = {"42053": (1,73,95), "39117": (1,73,95), "18013": (24,96,119),
           "54015": (48,120,167), "54013": (48,120,167), "54007": (48,120,167),
           "39115": (48,168,191), "39119": (24,96,119)}
MODES = ("C0_direct_osi", "C1_component_osi", "C2_equal_blend", "C3_aligned_component", "v18_rule")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype={"fipsCode":str}, float_precision="round_trip")


def keys(frame):
    frame = frame.copy()
    frame["fipsCode"] = frame.fipsCode.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    frame["timestamp_et"] = pd.to_datetime(frame.timestamp_et)
    if frame.duplicated(["fipsCode", "timestamp_et"]).any():
        raise ValueError("Duplicate county-time keys")
    return frame


def metric(truth, prediction):
    y, p = np.asarray(truth), np.asarray(prediction)
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Non-finite scored values")
    return {"n":len(y), "rmse":float(np.sqrt(np.mean((p-y)**2))),
            "mae":float(np.mean(np.abs(p-y))), "bias":float(np.mean(p-y))}


def table(frame, columns=None):
    frame = frame[columns] if columns else frame
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    def display(v):
        if isinstance(v, (float,np.floating)):
            return f"{v:.8f}" if np.isfinite(v) else "—"
        return str(v)
    lines += ["| " + " | ".join(display(v) for v in row) + " |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def main():
    if not (RUN / "CV_COMPLETE").is_file():
        raise RuntimeError("Only a completed, frozen CV can be analyzed")
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    marker = json.loads((RUN / "CV_COMPLETE").read_text())
    for name, digest in marker["files"].items():
        if sha(RUN / name) != digest:
            raise ValueError(f"Frozen CV artifact changed: {name}")
    settings = manifest["identity"]["settings"]
    assert settings["model_seed"] == 42 and settings["max_rounds"] == 2000
    assert settings["early_stopping_rounds"] == 100 and not settings["diagnostic"]
    assert not (RUN / "models/final").exists()
    oof = keys(pd.read_parquet(RUN / "oof_predictions.parquet"))
    parts = keys(pd.read_parquet(RUN / "oof_component_predictions.parquet"))
    old = keys(pd.read_parquet(OLD_OOF))
    pd.testing.assert_frame_equal(oof[["fipsCode","timestamp_et","hour_idx"]],old[["fipsCode","timestamp_et","hour_idx"]],check_dtype=False)
    pd.testing.assert_frame_equal(oof[["fipsCode","timestamp_et","hour_idx"]],parts[["fipsCode","timestamp_et","hour_idx"]],check_dtype=False)
    raw = keys(pd.read_csv(ROOT / "data/DM_Train.csv", dtype={"fipsCode":str}))
    raw["hour_idx"] = ((raw.timestamp_et-pd.Timestamp("2026-03-11"))/pd.Timedelta(hours=1)).astype(int)
    names = raw.groupby("fipsCode").first()[["countyName","stateAbbr"]]
    raw_lookup = raw.set_index(["fipsCode","timestamp_et"])
    summary = read_csv(RUN / "summary_metrics.csv")
    boot = read_csv(RUN / "paired_county_bootstrap.csv")
    fold_metrics = read_csv(RUN / "control_fold_metrics.csv")
    horizon_rows, county_rows, fold_rows, concentration, trajectory, peak_parts = [], [], [], [], [], []
    expected_counts = {1:34177,6:32982,24:28680,48:22944}
    for h in HOURS:
        target = f"osi_target_t{h:02d}h"
        valid = oof.hour_idx.to_numpy()+h<=215
        assert int(valid.sum()) == expected_counts[h]
        assert np.array_equal(np.isfinite(oof[f"actual_{target}"]),valid)
        assert np.array_equal(oof[f"actual_{target}"].to_numpy(),old[f"actual_{target}"].to_numpy(),equal_nan=True)
        for mode in MODES:
            pred = oof[f"pred_{mode}_{target}"].to_numpy()
            assert np.array_equal(np.isfinite(pred),valid)
            measured = metric(oof.loc[valid,f"actual_{target}"],pred[valid])
            row = summary[(summary.model==mode)&(summary.horizon==target)].iloc[0]
            assert abs(row.rmse-measured["rmse"])<1e-12 and row.n==valid.sum()
        work = oof.loc[valid,["fipsCode","timestamp_et","hour_idx","fold"]].copy()
        work["horizon"] = h
        work["target_timestamp"] = work.timestamp_et+pd.Timedelta(hours=h)
        work["target_hour"] = work.hour_idx+h
        work["actual"] = oof.loc[valid,f"actual_{target}"].to_numpy()
        work["prediction"] = oof.loc[valid,f"pred_v18_rule_{target}"].to_numpy()
        work["C1_prediction"] = oof.loc[valid,f"pred_C1_component_osi_{target}"].to_numpy()
        old_route = "C3_aligned_component" if h<=6 else "C1_component_osi"
        work["historical_prediction"] = old.loc[valid,f"pred_{old_route}_{target}"].to_numpy()
        truth = raw_lookup.osi.reindex(pd.MultiIndex.from_frame(work[["fipsCode","target_timestamp"]])).to_numpy()
        np.testing.assert_array_equal(truth,work.actual)
        work["sse"] = (work.prediction-work.actual)**2
        work["historical_sse"] = (work.historical_prediction-work.actual)**2
        total = work.sse.sum()
        current = metric(work.actual,work.prediction)
        legacy = metric(work.actual,work.historical_prediction)
        horizon_rows.append({"horizon":h,"n":len(work),"strict_rmse":current["rmse"],
                             "historical_rmse":legacy["rmse"],"rmse_change_pct":100*(current["rmse"]/legacy["rmse"]-1),
                             "strict_mae":current["mae"],"historical_mae":legacy["mae"],
                             "strict_bias":current["bias"],"historical_bias":legacy["bias"],
                             "comparison_type":"descriptive, different training/selection protocols"})
        for fips, group in work.groupby("fipsCode"):
            peak = group.loc[group.actual.idxmax()]
            prediction_peak = group.loc[group.prediction.idxmax()]
            county_rows.append({"horizon":h,"fipsCode":fips,**names.loc[fips].to_dict(),
                                "fold":int(group.fold.iloc[0]),**metric(group.actual,group.prediction),
                                "sse":float(group.sse.sum()),"sse_share":float(group.sse.sum()/total),
                                "historical_sse":float(group.historical_sse.sum()),
                                "sse_change":float(group.sse.sum()-group.historical_sse.sum()),
                                "actual_peak":float(peak.actual),"actual_peak_time":peak.target_timestamp,
                                "prediction_at_actual_peak":float(peak.prediction),
                                "prediction_peak":float(prediction_peak.prediction),"prediction_peak_time":prediction_peak.target_timestamp,
                                "actual_sum":float(group.actual.sum()),"prediction_sum":float(group.prediction.sum()),
                                "hours_actual_ge005":int((group.actual>=.05).sum())})
            if fips in FOCUS:
                at = parts[(parts.fipsCode==fips)&(parts.timestamp_et==peak.timestamp_et)].iloc[0]
                item = {"fipsCode":fips,"countyName":names.loc[fips,"countyName"],"horizon":h,
                        "target_timestamp":peak.target_timestamp,"actual_osi":float(peak.actual),
                        "primary_prediction":float(peak.prediction),"C1_prediction":float(peak.C1_prediction)}
                for c,weight in [("P_t",.4),("N_t",.35),("D_t",.25),("R_t",-.1)]:
                    stem=f"{c}_target_t{h:02d}h"
                    item[f"actual_{c}"]=float(at[f"actual_{stem}"])
                    item[f"pred_C1_{c}"]=float(at[f"pred_{stem}"])
                    item[f"weighted_error_{c}"]=weight*(item[f"pred_C1_{c}"]-item[f"actual_{c}"])
                peak_parts.append(item)
        for fold, group in work.groupby("fold"):
            train = work[work.fold!=fold]
            fold_rows.append({"horizon":h,"fold":int(fold),"counties":group.fipsCode.nunique(),
                              **metric(group.actual,group.prediction),"sse":float(group.sse.sum()),
                              "sse_share":float(group.sse.sum()/total),
                              "validation_max_osi":float(group.actual.max()),"training_max_osi":float(train.actual.max()),
                              "validation_rows_gt015":int((group.actual>.15).sum()),
                              "training_rows_gt015":int((train.actual>.15).sum())})
        county_sse = work.groupby("fipsCode").sse.sum().sort_values(ascending=False)
        high = work.actual>=work.actual.quantile(.95)
        concentration.append({"horizon":h,"top_county":county_sse.index[0],"top1_sse_share":county_sse.iloc[0]/total,
                              "top5_sse_share":county_sse.head(5).sum()/total,
                              "true_top5pct_sse_share":work.loc[high,"sse"].sum()/total,
                              "underprediction_sse_share":work.loc[work.prediction<work.actual,"sse"].sum()/total})
        trajectory.append(work[work.fipsCode.isin(FOCUS)])
    county = pd.DataFrame(county_rows)
    focus_trajectories = pd.concat(trajectory,ignore_index=True)
    period_rows = []
    for fips,(h,start,end) in WINDOWS.items():
        group=focus_trajectories[(focus_trajectories.fipsCode==fips)&(focus_trajectories.horizon==h)&focus_trajectories.target_hour.between(start,end)]
        assert len(group)==end-start+1
        period_rows.append({"fipsCode":fips,**names.loc[fips].to_dict(),"horizon":h,"target_hour_start":start,
                            "target_hour_end":end,**metric(group.actual,group.prediction),
                            "actual_peak":group.actual.max(),"prediction_peak":group.prediction.max(),
                            "actual_mean":group.actual.mean(),"prediction_mean":group.prediction.mean(),
                            "historical_prediction_mean":group.historical_prediction.mean()})
    within = []
    for row in boot[boot.metric=="rmse"].itertuples():
        left=fold_metrics[(fold_metrics.model==row.model)&(fold_metrics.horizon==row.horizon)].set_index("fold")
        right=fold_metrics[(fold_metrics.model==row.reference)&(fold_metrics.horizon==row.horizon)].set_index("fold")
        delta=left.rmse-right.rmse
        within.append({"model":row.model,"reference":row.reference,"horizon":row.horizon,
                       "rmse_difference":row.difference,"change_pct":row.change_pct,
                       "ci_low":row.ci_2_5,"ci_high":row.ci_97_5,"improved_folds":int((delta<0).sum()),
                       "tied_folds":int((delta==0).sum()),"replicates":row.replicates})
    models=json.loads((RUN/"model_manifest_cv.json").read_text())
    old_models=read_csv(OLD_MODELS)
    rounds=[]
    for record in models:
        spec,fit=record["spec"],record["fit"]
        fold=next(iter(set(range(5))-set(spec["scope"])))
        old_row=old_models[(old_models.role=="cv")&(old_models.target==spec["target"])&(old_models.fold==fold)].iloc[0]
        best=fit["best_iterations"]
        rounds.append({"target":spec["target"],"outer_fold":fold,"refit_rounds":fit["requested_rounds"],
                       "historical_rounds":int(old_row["rounds"]),"inner_min":min(best),"inner_max":max(best),
                       "inner_std":float(np.std(best)),"inner_cv":float(np.std(best)/np.mean(best)),
                       "inner_at_cap":sum(n==settings["max_rounds"] for n in best),
                       "inner_rounds":json.dumps(best)})
    # Deterministic-bound DIAGNOSIS only. Do not project D or change any prediction.
    bounds=[]
    raw_by_hour=raw.set_index(["fipsCode","hour_idx"])
    component_by_origin=parts.set_index(["fipsCode","hour_idx"])
    for fips in sorted(oof.fipsCode.unique()):
        for s in range(73,77):
            known=sum(float(raw_by_hour.loc[(fips,u),"P_t"]) for u in range(s-5,72))/6
            row=component_by_origin.loc[(fips,s-1)]
            actual=float(row.actual_D_t_target_t01h);pred=float(row.pred_D_t_target_t01h)
            assert actual+1e-6>=known
            bounds.append({"fipsCode":fips,"countyName":names.loc[fips,"countyName"],"target_hour":s,
                           "known_D_lower_bound":known,"actual_D":actual,"predicted_D":pred,
                           "gap":known-pred,"violates_lower_bound":bool(known-pred>1e-6)})
    frames={"horizon_comparison.csv":pd.DataFrame(horizon_rows),"within_protocol_comparisons.csv":pd.DataFrame(within),
            "county_error_summary.csv":county,"fold_error_summary.csv":pd.DataFrame(fold_rows),
            "error_concentration.csv":pd.DataFrame(concentration),"focus_trajectories.csv":focus_trajectories,
            "focus_period_metrics.csv":pd.DataFrame(period_rows),"focus_peak_components.csv":pd.DataFrame(peak_parts),
            "round_selection_comparison.csv":pd.DataFrame(rounds),"early_D_bound_diagnosis.csv":pd.DataFrame(bounds)}
    for name,frame in frames.items():frame.to_csv(OUT/name,index=False)
    change_rows=[]
    for h,group in county.groupby("horizon"):
        net=float(group.sse_change.sum())
        for row in group.nlargest(5,"sse_change").itertuples():
            change_rows.append({"horizon":h,"fipsCode":row.fipsCode,"countyName":row.countyName,
                                "fold":row.fold,"county_sse_change":row.sse_change,"net_horizon_sse_change":net,
                                "fraction_of_net_change":row.sse_change/net if net!=0 else np.nan})
    frames["sse_change_drivers.csv"]=pd.DataFrame(change_rows)
    frames["sse_change_drivers.csv"].to_csv(OUT/"sse_change_drivers.csv",index=False)
    panels=["# Strict CV quantitative summary", "All outputs use the fixed v18 rule unless stated. Historical comparisons use a different selection protocol.",
            "## Primary result vs historical v1.8",table(frames["horizon_comparison.csv"].drop(columns="comparison_type")),
            "## Controls within the strict protocol",table(summary.pivot(index="model",columns="horizon",values="rmse").reset_index()),
            "## Paired county bootstrap and fold directions",table(frames["within_protocol_comparisons.csv"]),
            "## Fold error distribution",table(frames["fold_error_summary.csv"]),
            "## Error concentration",table(frames["error_concentration.csv"]),
            "## Focus event windows",table(frames["focus_period_metrics.csv"])]
    (OUT/"Quantitative_Summary.md").write_text("\n\n".join(panels)+"\n")
    violations=frames["early_D_bound_diagnosis.csv"].query("violates_lower_bound")
    analysis={"scope":"Completed seed42 CV analysis; no fitting or prediction corrections", "run_identity_hash":manifest["identity_hash"],
              "frozen_cv_files_verified":len(marker["files"]),"n_outer_models":len(models),
              "n_inner_probes":sum(len(m["fit"]["best_iterations"]) for m in models),
              "training_valid_counts":expected_counts,"county_metric_rows":len(county),
              "known_D_bound_violating_rows":len(violations),"known_D_bound_violating_counties":violations.fipsCode.nunique(),
              "rounds_at_cap":int(frames["round_selection_comparison.csv"].inner_at_cap.sum()),
              "historical_comparison_caveat":"Early-stopping protocol and deterministic execution settings differ; differences are not an unbiased estimate of leakage optimism or feature gains.",
              "additional_posthoc_focus":"Muskingum 39119 added after identifying it as the largest positive SSE change under each horizon; its trajectory is diagnostic, not a preselected confirmation test.",
              "input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [RUN/"oof_predictions.parquet",RUN/"oof_component_predictions.parquet",OLD_OOF,OLD_MODELS,ROOT/"data/DM_Train.csv"]},
              "analysis_source_sha256":sha(Path(__file__)),"output_sha256":{name:sha(OUT/name) for name in frames}}
    (OUT/"analysis_manifest.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+"\n")
    print(frames["horizon_comparison.csv"].to_string(index=False))
    print(summary.pivot(index="model",columns="horizon",values="rmse").to_string())
    print("D lower-bound violations:",len(violations),"rows /",violations.fipsCode.nunique(),"counties")


if __name__ == "__main__":
    main()
