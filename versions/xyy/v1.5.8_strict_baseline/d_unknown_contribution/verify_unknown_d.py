"""Independent reload, U reconstruction, metrics and leakage-boundary verification."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from unknown_d_protocol import (HERE,CONFIG,KEYS,HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,H1,DT,UT,
    load_context,make_scoped,load_fit,output_path,experiment_identity,digest_object,json_read,
    array_sha,sha256_file,preflight_checks,make_source_manifest,expected_mask)
from run_artifacts import check_stage
from independent_numerics import (exact,close,brute_bounds,independent_controls,assert_stats)


def read_csv(path):
    return pd.read_csv(path,dtype={"fipsCode":str},float_precision="round_trip")


def verify_run(run,require_marker=True):
    run=output_path(run)
    ctx=load_context(42)
    manifest=json_read(run/"run_manifest.json")
    assert manifest["identity"]==experiment_identity(ctx)
    assert manifest["identity_hash"]==digest_object(manifest["identity"])
    identity=manifest["identity_hash"]
    marker=check_stage(run,"U_CV_COMPLETE",identity) if require_marker else None
    assert json_read(run/"base_source_manifest.json")==make_source_manifest(ctx)
    pd.testing.assert_frame_equal(read_csv(run/"cv_assignments.csv"),ctx.data.assignment,check_dtype=False)
    frames={name:pd.read_parquet(run/f"{name}.parquet") for name in
            ("target_audit","u_oof_predictions","component_predictions","control_predictions","primary_row_effects")}
    for frame in frames.values():
        pd.testing.assert_frame_equal(frame[KEYS],ctx.meta)
        assert frame.candidate_identity_hash.eq(identity).all()
        assert frame.baseline_identity_hash.eq(CONFIG["baseline_identity_hash"]).all()
    audit,u,parts_saved,oof,effects=[frames[name] for name in ("target_audit","u_oof_predictions","component_predictions","control_predictions","primary_row_effects")]
    independent_k=brute_bounds(ctx.raw_history,ctx.meta)[1]
    close(audit.K,independent_k);close(u.K,independent_k)
    exact(audit.official_D,ctx.actual_d)
    exact(audit.U_raw,ctx.actual_d-ctx.bounds[H1])
    close(audit.U_raw,ctx.actual_d-independent_k)
    exact(audit.unknown_terms,np.minimum(6,ctx.meta.hour_idx+1-71))
    exact(audit.rounding_negative_U,ctx.valid&(audit.U_raw.to_numpy()<0))
    exact(audit.scoreable,ctx.valid);exact(u.scoreable,ctx.valid)
    for frame in (audit,u):exact(frame.target_timestamp,ctx.meta.timestamp_et+pd.Timedelta(hours=1))
    records=json_read(run/"new_model_manifest.json")
    assert len(records)==5 and len(list(run.glob("models/outer*/*.txt")))==5
    raw_u=np.full(len(ctx.meta),np.nan);ids=np.full(len(ctx.meta),"",dtype=object)
    seen=set();round_rows=[]
    for record in records:
        assert record["spec"]["target"]==UT and record["reconstructed_target"]==DT
        assert record["target_semantics"]==CONFIG["definition"]
        scope=tuple(record["spec"]["scope"])
        assert len(scope)==4 and len(set(scope))==4 and set(scope).issubset(range(5))
        fold=next(iter(set(range(5))-set(scope)));assert fold not in seen;seen.add(fold)
        assert record["model_path"]==f"models/outer{fold}/{UT}.txt"
        assert record["receipt_path"]==f"models/outer{fold}/{UT}.json"
        assert json_read(run/record["receipt_path"])==record
        scoped=make_scoped(ctx,scope)
        assert record["U_labels_sha256"]==array_sha(scoped.y)
        assert record["negative_U_labels"]==int((scoped.y<0).sum())
        allowed=set(ctx.meta.loc[ctx.meta.fold.isin(scope),"fipsCode"])
        held=set(ctx.meta.loc[ctx.meta.fold==fold,"fipsCode"])
        assert set(record["fit"]["refit"]["counties"])==allowed and not allowed&held
        for probe,best in zip(record["fit"]["probes"],record["fit"]["best_iterations"]):
            tr,va=set(probe["train"]["counties"]),set(probe["early_stop"]["counties"])
            assert tr|va==allowed and not tr&va and not ((tr|va)&held)
            round_rows.append({"outer_fold":fold,"model_id":record["model_id"],"target":UT,"inner_fold":probe["inner_fold"],
                "train_counties":json.dumps(probe["train"]["counties"]),"early_stop_counties":json.dumps(probe["early_stop"]["counties"]),
                "best_iteration":best,"refit_rounds":record["fit"]["requested_rounds"]})
        model=load_fit(run,record,scoped,identity,CONFIG["settings"])
        take=np.flatnonzero(ctx.valid&(ctx.meta.fold.to_numpy()==fold))
        assert np.isnan(raw_u[take]).all()
        raw_u[take]=model.predict(ctx.data.X_train.iloc[take],num_iteration=record["fit"]["requested_rounds"],num_threads=1)
        ids[take]=record["model_id"]
    assert seen==set(range(5)) and len(round_rows)==20
    pd.testing.assert_frame_equal(read_csv(run/"round_selection.csv"),pd.DataFrame(round_rows),check_dtype=False)
    close(u.pred_U_raw,raw_u);exact(u.source_U_model_id,ids)
    exact(np.isfinite(raw_u),ctx.valid)
    positive=np.where(raw_u<0,0,raw_u)
    preclip=ctx.bounds[H1]+positive
    d=np.minimum(preclip,1.)
    close(u.pred_U_nonnegative,positive);close(u.pred_D_before_upper_clip,preclip);close(u.pred_D_reconstructed,d)
    exact(u.U_negative_clipped,ctx.valid&(raw_u<0));exact(u.D_upper_clipped,ctx.valid&(preclip>1))
    assert (d[ctx.valid]>=ctx.bounds[H1][ctx.valid]).all() and (d[ctx.valid]<=1).all()
    all_parts={}
    for case in ("A","B","C"):
        all_parts[case]={}
        for h in HORIZONS:
            all_parts[case][HORIZON_HOURS[h]]={}
            for comp in COMPONENTS:
                target=f"{comp}_target_{h.rsplit('_',1)[-1]}"
                reference=ctx.parts["B" if case=="B" else "A"][h][comp]
                if (case,h,comp)==("C",H1,"D_t"):reference=d
                close(parts_saved[f"pred_{case}_{target}"],reference)
                expected_source=ids if (case,h,comp)==("C",H1,"D_t") else ctx.component[f"source_{target}"].to_numpy()
                exact(parts_saved[f"source_{case}_{target}"],expected_source)
                if not (h==H1 and comp=="D_t"):exact(parts_saved[f"pred_{case}_{target}"],ctx.parts["A"][h][comp])
                all_parts[case][HORIZON_HOURS[h]][comp]=np.asarray(reference)
    for case in ("A","B","C"):
        independently,buckets,means=independent_controls(ctx.meta,ctx.raw,all_parts[case])
        for h in HORIZONS:
            valid=expected_mask(ctx.meta,h)
            exact(oof[f"actual_{h}"],ctx.data.y_train[h]);exact(oof[f"scoreable_{h}"],valid)
            exact(oof[f"raw_direct_{h}"],ctx.direct[h]);exact(oof[f"source_direct_{h}"],ctx.raw[f"source_{h}"])
            for mode in MODES:
                predicted=oof[f"pred_{case}_{mode}_{h}"].to_numpy()
                close(predicted,independently[HORIZON_HOURS[h]][mode]);exact(np.isfinite(predicted),valid)
                if case in ("A","B"):exact(predicted,ctx.controls[case][mode][h])
                if case=="C" and (mode=="C0_direct_osi" or (h!=H1 and mode in ("C1_component_osi","C2_equal_blend")) or (HORIZON_HOURS[h]>=24 and mode=="v18_rule")):
                    exact(predicted,ctx.controls["A"][mode][h])
        if case=="C":
            unique=pd.read_parquet(run/"aligned_unique_components.parquet")
            assert len(unique)==len(means)*4 and not unique.duplicated(["component","fipsCode","target_timestamp"]).any()
            for row in unique.itertuples(index=False):
                key=(row.fipsCode,int((row.target_timestamp-pd.Timestamp("2026-03-11"))/pd.Timedelta(hours=1)))
                close(row.prediction,means[key][COMPONENTS.index(row.component)])
                assert row.candidate_count==len(buckets[key])
                assert row.source_horizons==",".join(f"osi_target_t{h:02d}h" for h,_ in sorted(buckets[key]))
    pairs={"C_minus_B":("B","C"),"C_minus_A":("A","C"),"B_minus_A":("A","B")}
    for pair,(before,after) in pairs.items():
        for h in HORIZONS:
            for mode in MODES:
                b=oof[f"pred_{before}_{mode}_{h}"].to_numpy();a=oof[f"pred_{after}_{mode}_{h}"].to_numpy()
                exact(oof[f"changed_{pair}_{mode}_{h}"],expected_mask(ctx.meta,h)&(b!=a))
    for h in (H1,"osi_target_t06h"):
        y=oof[f"actual_{h}"].to_numpy();b=oof[f"pred_B_v18_rule_{h}"].to_numpy();c=oof[f"pred_C_v18_rule_{h}"].to_numpy()
        exact(effects[f"actual_{h}"],y);exact(effects[f"B_{h}"],b);exact(effects[f"C_{h}"],c)
        exact(effects[f"C_minus_B_{h}"],c-b);exact(effects[f"sse_reduction_{h}"],(b-y)**2-(c-y)**2)
    metric_tables={name:read_csv(run/f"metrics/{name}.csv") for name in ("pooled","fold","county","window","bootstrap","D_component","U_raw","primary_county_influence")}
    for name,n in (("pooled",60),("fold",300),("county",14340),("bootstrap",64),("D_component",12)):
        assert len(metric_tables[name])==n,(name,len(metric_tables[name]))
    for name in ("pooled","fold","county","window","primary_county_influence"):
        for _,row in metric_tables[name].iterrows():
            before,after=pairs[row.comparison]
            y=oof[f"actual_{row.horizon}"].to_numpy();b=oof[f"pred_{before}_{row.model}_{row.horizon}"].to_numpy();a=oof[f"pred_{after}_{row.model}_{row.horizon}"].to_numpy()
            take=np.isfinite(y)
            if name=="fold":take &= ctx.meta.fold.to_numpy()==row.fold
            if name in ("county","primary_county_influence"):take &= ctx.meta.fipsCode.to_numpy()==row.fipsCode
            if name=="window":
                take &= {"first_four_target_hours":ctx.meta.hour_idx.to_numpy()<=75,"later_target_hours":ctx.meta.hour_idx.to_numpy()>75,"changed_predictions":a!=b}[row.window]
            assert_stats(row,y[take],b[take],a[take])
    for _,row in metric_tables["D_component"].iterrows():
        before,after=pairs[row.comparison];target=f"D_t_target_{row.horizon.rsplit('_',1)[-1]}"
        y=ctx.data.component_targets[target].to_numpy();take=np.isfinite(y)
        assert_stats(row,y[take],parts_saved[f"pred_{before}_{target}"].to_numpy()[take],parts_saved[f"pred_{after}_{target}"].to_numpy()[take])
    for _,row in metric_tables["bootstrap"].iterrows():
        before,after=pairs[row.comparison]
        y=oof[f"actual_{row.horizon}"].to_numpy();b=oof[f"pred_{before}_{row.model}_{row.horizon}"].to_numpy();a=oof[f"pred_{after}_{row.model}_{row.horizon}"].to_numpy()
        take=np.isfinite(y)
        if row.population=="exclude_Morrow_diagnostic_only":take &= ctx.meta.fipsCode.to_numpy()!="39117"
        assert_stats(row,y[take],b[take],a[take])
        frame=pd.DataFrame({"county":ctx.meta.fipsCode.to_numpy()[take],"b":(b[take]-y[take])**2,"a":(a[take]-y[take])**2})
        grouped=frame.groupby("county",sort=True).agg(b=("b","sum"),a=("a","sum"),n=("b","size"))
        values=grouped.to_numpy();assert len(values)==row.counties
        generator=np.random.default_rng(int(row.seed));delta=[]
        for _ in range(int(row.replicates)):
            bs,ass,ns=values[generator.integers(0,len(values),len(values))].sum(axis=0)
            delta.append(np.sqrt(ass/ns)-np.sqrt(bs/ns))
        close([row.ci_low,row.ci_high],np.quantile(delta,[.025,.975]))
        close(row.bootstrap_fraction_delta_lt0,np.mean(np.asarray(delta)<0))
        close(row.bootstrap_fraction_delta_eq0,np.mean(np.asarray(delta)==0))
    from independent_numerics import compare_stats
    raw_stats=compare_stats(audit.U_raw.to_numpy()[ctx.valid],raw_u[ctx.valid],raw_u[ctx.valid])
    ur=metric_tables["U_raw"].iloc[0]
    for k in ("rmse","mae","sse","bias"):close(ur[k],raw_stats[f"baseline_{k}"])
    assert ur.n==ctx.valid.sum() and ur.negative_predictions==(raw_u[ctx.valid]<0).sum() and ur.D_lower_bound_violations==0
    assert ur.D_upper_clipped==int((preclip[ctx.valid]>1).sum())
    for _,pooled in metric_tables["pooled"].iterrows():
        counties=metric_tables["county"].query("comparison==@pooled.comparison and horizon==@pooled.horizon and model==@pooled.model")
        assert len(counties)==239 and counties.n.sum()==pooled.n
        close(counties.baseline_sse.sum(),pooled.baseline_sse);close(counties.candidate_sse.sum(),pooled.candidate_sse)
    influence=metric_tables["primary_county_influence"]
    close(influence.fraction_of_net_sse_reduction,influence.sse_reduction/influence.groupby("horizon").sse_reduction.transform("sum"))
    checks=preflight_checks(ctx)
    assert checks["status"]=="PASS"
    assert ctx.sources==json_read(run/"base_source_manifest.json")["source_sha256"]
    if require_marker:check_stage(run,"U_CV_COMPLETE",identity)
    # Marker must cover every immutable model, table and identity, and no missing output.
    required={str(f.relative_to(run)) for f in run.rglob('*') if f.is_file() and
              f.suffix in (".parquet",".csv",".txt",".json") and f.name not in ("execution.json",) and "logs" not in f.relative_to(run).parts}
    if marker is not None:assert set(marker["files"])==required
    return {"status":"PASS","protocol":CONFIG["protocol"],"identity_hash":identity,"split_seed":42,
        "new_models_reloaded":5,"inner_probes_audited":20,"referenced_baseline_models":95,
        "target_semantics":UT,"absolute_tolerance":1e-12,"verifier_sha256":sha256_file(__file__),
        "verified_at_utc":datetime.now(timezone.utc).isoformat(),"source_integrity":"PASS",
        "all_A_B_C_controls_independently_reconstructed":True,"all_metrics_and_bootstrap_recomputed":True,
        "future_history_and_outer_label_isolation":"PASS","unchanged_components_and_horizons":"PASS",
        "training_performed_by_verifier":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="v18_ud01_nested_v1_split42_model42",choices=("v18_ud01_nested_v1_split42_model42",))
    args=parser.parse_args()
    print(json.dumps(verify_run(HERE/"runs"/args.run_id),indent=2))


if __name__=="__main__":main()
