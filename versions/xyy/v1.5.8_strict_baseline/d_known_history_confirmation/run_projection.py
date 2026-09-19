"""Apply the unchanged D projection to one frozen confirmation baseline."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
from protocol import MODES, COMPONENTS, HORIZONS, HORIZON_HOURS, build_controls, expected_mask
from project_d import observed_history, known_bounds, project_components




def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(frame):
    frame = frame.copy()
    frame["fipsCode"] = frame.fipsCode.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    frame["timestamp_et"] = pd.to_datetime(frame.timestamp_et)
    if frame.duplicated(["fipsCode", "timestamp_et"]).any():
        raise ValueError("Duplicate county-time keys")
    return frame


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or not len(y):
        raise ValueError("Invalid score inputs")
    e = p-y
    return {"n":len(y), "rmse":float(np.sqrt(np.mean(e**2))), "mae":float(np.mean(np.abs(e))),
            "sse":float(np.sum(e**2)), "bias":float(e.mean())}


def comparison(y, before, after):
    b, a = metrics(y,before), metrics(y,after)
    return {"n":b["n"], **{f"baseline_{k}":v for k,v in b.items() if k!="n"},
            **{f"candidate_{k}":v for k,v in a.items() if k!="n"},
            "rmse_delta":a["rmse"]-b["rmse"], "rmse_change_pct":100*(a["rmse"]/b["rmse"]-1) if b["rmse"] else np.nan,
            "sse_reduction":b["sse"]-a["sse"], "changed_predictions":int(np.count_nonzero(np.asarray(before)!=np.asarray(after)))}


def bootstrap(y, before, after, codes, seed, replicates):
    counties, inverse = np.unique(np.asarray(codes), return_inverse=True)
    n = np.bincount(inverse)
    b = np.bincount(inverse, weights=(np.asarray(before)-np.asarray(y))**2)
    a = np.bincount(inverse, weights=(np.asarray(after)-np.asarray(y))**2)
    take = np.random.default_rng(seed).integers(0,len(counties),(replicates,len(counties)))
    delta = np.sqrt(a[take].sum(axis=1)/n[take].sum(axis=1))-np.sqrt(b[take].sum(axis=1)/n[take].sum(axis=1))
    low, high = np.quantile(delta,[.025,.975])
    return {"counties":len(counties),"replicates":replicates,"seed":seed,"ci_low":float(low),"ci_high":float(high),
            "bootstrap_fraction_delta_lt0":float(np.mean(delta<0)),"bootstrap_fraction_delta_eq0":float(np.mean(delta==0))}


def check_baseline(run):
    marker=json.loads((run/"CV_COMPLETE").read_text())
    for name,digest in marker["files"].items():
        if sha(run/name)!=digest:raise ValueError(f"Frozen baseline changed: {name}")
    manifest=json.loads((run/"run_manifest.json").read_text())
    if manifest["identity_hash"]!=marker["identity_hash"] or manifest["identity"]["settings"]["diagnostic"]:
        raise ValueError("Invalid or diagnostic baseline")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed",type=int,required=True,choices=(20260917,20260918))
    args=parser.parse_args(argv)
    confirmation=json.loads((HERE/"confirmation_plan.json").read_text())
    split=next(row for row in confirmation["splits"] if row["split_seed"]==args.split_seed)
    run,out=Path(split["baseline_dir"]).resolve(),Path(split["projection_dir"]).resolve()
    assert sha(HERE/"project_d.py")==confirmation["fixed_projection_source_sha256"]
    if not out.is_relative_to(HERE) or out==HERE:
        raise ValueError("Candidate output must be a child of the experiment directory")
    plan=json.loads((HERE/"experiment_plan.json").read_text())
    manifest=check_baseline(run)
    assert manifest["identity"]["inputs"]["cv"]==split["cv_sha256"]
    assert manifest["identity"]["settings"]==confirmation["settings"]
    assert manifest["identity"]["code_sha256"]==confirmation["baseline_code_sha256"]
    protected=json.loads((HERE/"numerical_reference_snapshot.json").read_text())
    if any(sha(BASE/name)!=digest for name,digest in protected.items()):
        raise ValueError("An existing strict-baseline artifact changed")
    paths={"baseline_oof":run/"oof_predictions.parquet","component_oof":run/"oof_component_predictions.parquet",
           "raw_predictions":run/"base_predictions_cv.parquet","raw_train":ROOT/"data/DM_Train.csv",
           "raw_test":ROOT/"data/DM_Test.csv","test_meta":ROOT/"versions/xyy/v1.5.6/meta_test_v1.5.6.parquet",
           "plan":HERE/"experiment_plan.json","projector":HERE/"project_d.py","evaluation_code":Path(__file__),
           "frozen_controls_code":BASE/"protocol.py","confirmation_plan":HERE/"confirmation_plan.json"}
    input_hashes={name:sha(path) for name,path in paths.items()}
    identity={"experiment":plan["experiment"],"baseline_identity_hash":manifest["identity_hash"],"input_sha256":input_hashes}
    candidate_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if (out/"manifest.json").exists():
        if json.loads((out/"manifest.json").read_text())["candidate_identity_hash"]!=candidate_hash:
            raise FileExistsError("Different candidate identity; use a new output directory")
    out.mkdir(parents=True,exist_ok=True)
    oof=normalized(pd.read_parquet(paths["baseline_oof"]))
    part=normalized(pd.read_parquet(paths["component_oof"]))
    raw_predictions=normalized(pd.read_parquet(paths["raw_predictions"]))
    id_columns=["fipsCode","timestamp_et","hour_idx","stateAbbr","fold","split_id"]
    meta=oof[id_columns].copy()
    for frame in [part,raw_predictions]:
        pd.testing.assert_frame_equal(frame[id_columns],meta,check_dtype=False)
    # No label column is read by the prediction-only projection functions.
    observed_raw=pd.read_csv(paths["raw_train"],usecols=["fipsCode","timestamp_et","P_t"],dtype={"fipsCode":str})
    history=observed_history(observed_raw)
    bounds=known_bounds(meta,history)
    base_components={h:{c:part[f"pred_{c}_target_{h.rsplit('_',1)[-1]}"].to_numpy(dtype=float,copy=True)
                        for c in COMPONENTS} for h in HORIZONS}
    direct={h:raw_predictions[f"raw_{h}"].to_numpy(dtype=float,copy=True) for h in HORIZONS}
    projected=project_components(base_components,bounds)
    base_controls,_=build_controls(meta,direct,base_components)
    candidate_controls,candidate_aux=build_controls(meta,direct,projected)
    baseline_replay_error=0.
    for h in HORIZONS:
        valid=expected_mask(meta,h)
        for mode in MODES:
            saved=oof[f"pred_{mode}_{h}"].to_numpy(dtype=float)
            np.testing.assert_allclose(base_controls[mode][h],saved,atol=1e-12,rtol=0)
            baseline_replay_error=max(baseline_replay_error,float(np.max(np.abs(base_controls[mode][h][valid]-saved[valid]))))
    base_ids=meta.copy()
    base_ids["baseline_run_identity_hash"]=manifest["identity_hash"]
    base_ids["candidate_identity_hash"]=candidate_hash
    predictions=base_ids.copy();components_out=base_ids.copy();bounds_out=meta.copy()
    metric_rows=[];fold_rows=[];county_rows=[];boot_rows=[];d_rows=[];change_rows=[];window_rows=[]
    raw_names=pd.read_csv(paths["raw_train"],usecols=["fipsCode","countyName"],dtype={"fipsCode":str}).drop_duplicates().set_index("fipsCode").countyName
    for h in HORIZONS:
        hours=HORIZON_HOURS[h];valid=expected_mask(meta,h);target_hour=meta.hour_idx.to_numpy()+hours
        known_window=valid & (target_hour>=73)&(target_hour<=76)
        y=oof[f"actual_{h}"].to_numpy(dtype=float)
        assert np.array_equal(np.isfinite(y),valid)
        bounds_out[f"known_D_lower_bound_{h}"]=bounds[h]
        predictions[f"actual_{h}"]=y
        predictions[f"is_scoreable_{h}"]=valid
        for c in COMPONENTS:
            target=f"{c}_target_{h.rsplit('_',1)[-1]}"
            components_out[f"baseline_pred_{target}"]=base_components[h][c]
            components_out[f"projected_pred_{target}"]=projected[h][c]
            components_out[f"baseline_model_id_{target}"]=part[f"source_{target}"].to_numpy()
            if c!="D_t":np.testing.assert_array_equal(base_components[h][c],projected[h][c])
        d_before,d_after=base_components[h]["D_t"],projected[h]["D_t"]
        actual_d=part[f"actual_D_t_target_{h.rsplit('_',1)[-1]}"].to_numpy(dtype=float)
        d_changed=valid & (d_before!=d_after)
        assert not d_changed[~known_window].any()
        assert (actual_d[valid]+1e-6>=bounds[h][valid]).all()
        assert (d_after[valid]>=bounds[h][valid]).all()
        # Published rounded D values can differ from exact P sums by tiny amounts.
        assert (np.abs(d_after[valid]-actual_d[valid])<=np.abs(d_before[valid]-actual_d[valid])+1e-6).all()
        components_out[f"known_D_lower_bound_{h}"]=bounds[h]
        components_out[f"D_projected_{h}"]=d_changed
        d_rows.append({"horizon":h,**comparison(actual_d[valid],d_before[valid],d_after[valid]),
                       "exact_projection_rows":int(d_changed.sum()),"material_projection_rows":int((valid & (bounds[h]-d_before>1e-6)).sum())})
        for mode in MODES:
            b,a=base_controls[mode][h],candidate_controls[mode][h]
            changed=valid & (b!=a)
            assert not changed[~known_window].any()
            if mode=="C0_direct_osi":assert not changed.any()
            predictions[f"baseline_{mode}_{h}"]=b
            predictions[f"candidate_{mode}_{h}"]=a
            predictions[f"changed_{mode}_{h}"]=changed
            metric_rows.append({"horizon":h,"model":mode,**comparison(y[valid],b[valid],a[valid])})
            for fold in range(5):
                take=valid & (meta.fold.to_numpy()==fold)
                fold_rows.append({"horizon":h,"model":mode,"fold":fold,**comparison(y[take],b[take],a[take])})
            for fips in meta.fipsCode.unique():
                take=valid & (meta.fipsCode.to_numpy()==fips)
                county_rows.append({"horizon":h,"model":mode,"fipsCode":fips,"countyName":raw_names.loc[fips],
                                    "fold":int(meta.loc[take,"fold"].iloc[0]),**comparison(y[take],b[take],a[take])})
            boot_rows.append({"horizon":h,"model":mode,"population":"all_counties",
                              **comparison(y[valid],b[valid],a[valid]),
                              **bootstrap(y[valid],b[valid],a[valid],meta.loc[valid,"fipsCode"],plan["bootstrap"]["seed"],plan["bootstrap"]["replicates"])})
            if hours==1:
                for label,take in [("first_four_target_hours",known_window),("remaining_scoreable_hours",valid&~known_window),("changed_final_predictions",changed)]:
                    if take.any():window_rows.append({"model":mode,"window":label,**comparison(y[take],b[take],a[take])})
            if mode in ("C1_component_osi","v18_rule") and hours==1:
                take=valid & (meta.fipsCode.to_numpy()!="39117")
                boot_rows.append({"horizon":h,"model":mode,"population":"exclude_Morrow_diagnostic_only",
                                  **comparison(y[take],b[take],a[take]),
                                  **bootstrap(y[take],b[take],a[take],meta.loc[take,"fipsCode"],plan["bootstrap"]["seed"],plan["bootstrap"]["replicates"])})
        for i in np.flatnonzero(d_changed):
            row={"fipsCode":meta.fipsCode.iloc[i],"countyName":raw_names.loc[meta.fipsCode.iloc[i]],"fold":int(meta.fold.iloc[i]),
                 "origin_timestamp":meta.timestamp_et.iloc[i],"horizon":h,"target_hour":int(target_hour[i]),
                 "target_timestamp":meta.timestamp_et.iloc[i]+pd.Timedelta(hours=hours),
                 "known_D_lower_bound":bounds[h][i],"baseline_D":d_before[i],"projected_D":d_after[i],"actual_D":actual_d[i],
                 "D_sse_reduction":(d_before[i]-actual_d[i])**2-(d_after[i]-actual_d[i])**2,"actual_osi":y[i]}
            for c in COMPONENTS:
                row[f"baseline_{c}"]=base_components[h][c][i]
                row[f"actual_{c}"]=float(part.iloc[i][f"actual_{c}_target_{h.rsplit('_',1)[-1]}"])
            for mode in MODES:
                b,a=base_controls[mode][h][i],candidate_controls[mode][h][i]
                row[f"baseline_{mode}"]=b;row[f"candidate_{mode}"]=a
                row[f"sse_reduction_{mode}"]=(b-y[i])**2-(a-y[i])**2
            change_rows.append(row)
    # Prepare the same label-free bounds for eventual test inference, without creating predictions.
    test_raw=pd.read_csv(paths["raw_test"],usecols=["fipsCode","timestamp_et","P_t"],dtype={"fipsCode":str})
    test_meta=normalized(pd.read_parquet(paths["test_meta"]))
    test_bounds=known_bounds(test_meta,observed_history(test_raw))
    test_out=test_meta[["fipsCode","timestamp_et","hour_idx"]].copy()
    for h in HORIZONS:test_out[f"known_D_lower_bound_{h}"]=test_bounds[h]
    frames={"candidate_oof_predictions.parquet":predictions,"projected_component_predictions.parquet":components_out,
            "known_bounds_train.parquet":bounds_out,"known_bounds_test.parquet":test_out,
            "candidate_aligned_unique_components.parquet":candidate_aux["unique_components"],
            "metrics_comparison.csv":pd.DataFrame(metric_rows),"fold_metrics.csv":pd.DataFrame(fold_rows),
            "county_metrics.csv":pd.DataFrame(county_rows),"paired_county_bootstrap.csv":pd.DataFrame(boot_rows),
            "D_component_metrics.csv":pd.DataFrame(d_rows),"changed_rows.csv":pd.DataFrame(change_rows),
            "window_metrics.csv":pd.DataFrame(window_rows)}
    primary=frames["county_metrics.csv"].query("model=='v18_rule' and horizon=='osi_target_t01h'").copy()
    total=primary.sse_reduction.sum()
    primary["fraction_of_total_sse_reduction"]=primary.sse_reduction/total if total!=0 else np.nan
    frames["primary_county_influence.csv"]=primary.sort_values("sse_reduction",ascending=False)
    for name,frame in frames.items():
        if name.endswith('.parquet'):frame.to_parquet(out/name,index=False)
        else:frame.to_csv(out/name,index=False)
    check_baseline(run)
    assert all(sha(BASE/name)==digest for name,digest in protected.items())
    final={**identity,"candidate_identity_hash":candidate_hash,"created_at_utc":datetime.now(timezone.utc).isoformat(),
           "baseline_run":str(run),"input_paths":{name:str(path) for name,path in paths.items()},
           "split_seed":args.split_seed,"training_performed":False,"fitted_coefficients":False,"baseline_replay_max_error":baseline_replay_error,
           "projection_target_hours":[73,74,75,76],"exact_D_projection_rows":len(change_rows),
           "projected_counties":len(set(row['fipsCode'] for row in change_rows)),
           "test_bounds_only_no_test_predictions":True,"baseline_unchanged":True,
           "output_sha256":{name:sha(out/name) for name in frames}}
    (out/"manifest.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
    print(frames['metrics_comparison.csv'].query("model=='v18_rule'").to_string(index=False))
    print(frames['paired_county_bootstrap.csv'].query("model=='v18_rule' and horizon=='osi_target_t01h'").to_string(index=False))
    print('D projected rows/counties:',len(change_rows),final['projected_counties'])


if __name__=='__main__':
    main()
