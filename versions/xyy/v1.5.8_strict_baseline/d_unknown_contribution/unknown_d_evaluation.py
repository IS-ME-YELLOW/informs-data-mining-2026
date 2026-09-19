"""Full-row A/B/C predictions and predeclared paired comparisons."""
import numpy as np
import pandas as pd
from unknown_d_protocol import (CONFIG, HORIZONS, HORIZON_HOURS, COMPONENTS, MODES, H1, DT, UT,
                                candidate, expected_mask, build_controls)
from metric_helpers import metrics, comparison, bootstrap

PAIRS = (("C_minus_B","B","C"),("C_minus_A","A","C"),("B_minus_A","A","B"))


def evaluate_frames(ctx,parts,controls):
    tables = {name:[] for name in ("pooled","fold","county","window","bootstrap","D_component")}
    raw_names = pd.read_csv(ctx.data.input_paths["raw_train"],usecols=["fipsCode","countyName"],dtype={"fipsCode":str})
    names = raw_names.drop_duplicates().set_index("fipsCode").countyName
    county_groups = {f:np.flatnonzero(ctx.meta.fipsCode.to_numpy()==f) for f in ctx.meta.fipsCode.unique()}
    for h in HORIZONS:
        mask = expected_mask(ctx.meta,h)
        y = ctx.data.y_train[h].to_numpy()
        for mode in MODES:
            for pair,before,after in PAIRS:
                b,a = controls[before][mode][h],controls[after][mode][h]
                ids = {"comparison":pair,"horizon":h,"model":mode}
                tables["pooled"].append({**ids,**comparison(y[mask],b[mask],a[mask])})
                for fold in range(5):
                    take = mask&(ctx.meta.fold.to_numpy()==fold)
                    tables["fold"].append({**ids,"fold":fold,**comparison(y[take],b[take],a[take])})
                for f,indices in county_groups.items():
                    take = indices[mask[indices]]
                    tables["county"].append({**ids,"fipsCode":f,"countyName":names.loc[f],
                                             "fold":int(ctx.meta.fold.iloc[take[0]]),**comparison(y[take],b[take],a[take])})
                tables["bootstrap"].append({**ids,"population":"all_counties",**comparison(y[mask],b[mask],a[mask]),
                    **bootstrap(y[mask],b[mask],a[mask],ctx.meta.loc[mask,"fipsCode"],CONFIG["settings"]["bootstrap_seed"],CONFIG["settings"]["bootstrap_replicates"])})
                if mode=="v18_rule" and HORIZON_HOURS[h] in (1,6) and pair!="B_minus_A":
                    take = mask&(ctx.meta.fipsCode.to_numpy()!="39117")
                    tables["bootstrap"].append({**ids,"population":"exclude_Morrow_diagnostic_only",**comparison(y[take],b[take],a[take]),
                        **bootstrap(y[take],b[take],a[take],ctx.meta.loc[take,"fipsCode"],CONFIG["settings"]["bootstrap_seed"],CONFIG["settings"]["bootstrap_replicates"])})
                if h==H1 and mode in ("v18_rule","C1_component_osi","C3_aligned_component"):
                    early = ctx.meta.hour_idx.to_numpy()<=75
                    for window,take in (("first_four_target_hours",mask&early),("later_target_hours",mask&~early),
                                        ("changed_predictions",mask&(b!=a))):
                        if take.any():tables["window"].append({**ids,"window":window,**comparison(y[take],b[take],a[take])})
        d = ctx.data.component_targets[f"D_t_target_{h.rsplit('_',1)[-1]}"].to_numpy()
        for pair,before,after in PAIRS:
            b,a = parts[before][h]["D_t"],parts[after][h]["D_t"]
            tables["D_component"].append({"comparison":pair,"horizon":h,**comparison(d[mask],b[mask],a[mask])})
    result = {f"metrics/{name}.csv":pd.DataFrame(rows) for name,rows in tables.items()}
    effect = result["metrics/county.csv"].query("model=='v18_rule' and comparison=='C_minus_B'").copy()
    effect["fraction_of_net_sse_reduction"] = effect.sse_reduction/effect.groupby("horizon").sse_reduction.transform("sum")
    result["metrics/primary_county_influence.csv"] = effect.sort_values(["horizon","sse_reduction"],ascending=[True,False])
    return result


def artifact_frames(ctx,raw_u,source_ids,identity_hash):
    cp,cc,caux = candidate(ctx,raw_u)
    parts = {**ctx.parts,"C":cp};controls = {**ctx.controls,"C":cc}
    ids = ctx.meta.copy();ids["baseline_identity_hash"]=CONFIG["baseline_identity_hash"];ids["candidate_identity_hash"]=identity_hash
    audit = ids.copy()
    audit["target_timestamp"] = audit.timestamp_et+pd.Timedelta(hours=1)
    audit["scoreable"] = ctx.valid
    audit["official_D"] = ctx.actual_d
    audit["K"] = ctx.bounds[H1]
    audit["U_raw"] = ctx.actual_d-ctx.bounds[H1]
    audit["unknown_terms"] = ctx.unknown_count
    audit["rounding_negative_U"] = ctx.valid&(audit.U_raw.to_numpy()<0)
    u = ids.copy();u["target_timestamp"]=audit.target_timestamp;u["scoreable"]=ctx.valid
    u["K"]=ctx.bounds[H1];u["pred_U_raw"]=raw_u;u["pred_U_nonnegative"]=np.maximum(raw_u,0)
    u["pred_D_before_upper_clip"]=ctx.bounds[H1]+np.maximum(raw_u,0)
    u["pred_D_reconstructed"]=cp[H1]["D_t"]
    u["U_negative_clipped"]=ctx.valid&(raw_u<0)
    u["D_upper_clipped"]=ctx.valid&(u.pred_D_before_upper_clip.to_numpy()>1)
    u["source_U_model_id"]=source_ids
    component = {k:ids[k] for k in ids};output={k:ids[k] for k in ids}
    for h in HORIZONS:
        output[f"actual_{h}"]=ctx.data.y_train[h].to_numpy()
        output[f"scoreable_{h}"]=expected_mask(ctx.meta,h)
        output[f"raw_direct_{h}"]=ctx.direct[h]
        output[f"source_direct_{h}"]=ctx.raw[f"source_{h}"].to_numpy()
        for case in ("A","B","C"):
            for mode in MODES:output[f"pred_{case}_{mode}_{h}"]=controls[case][mode][h]
            for c in COMPONENTS:
                target=f"{c}_target_{h.rsplit('_',1)[-1]}"
                component[f"pred_{case}_{target}"]=parts[case][h][c]
                component[f"source_{case}_{target}"] = source_ids if (case,h,c)==("C",H1,"D_t") else ctx.component[f"source_{target}"].to_numpy()
        for pair,before,after in PAIRS:
            for mode in MODES:
                valid=expected_mask(ctx.meta,h)
                output[f"changed_{pair}_{mode}_{h}"]=valid&(controls[before][mode][h]!=controls[after][mode][h])
    frames={"target_audit.parquet":audit,"u_oof_predictions.parquet":u,
            "component_predictions.parquet":pd.DataFrame(component),"control_predictions.parquet":pd.DataFrame(output),
            "aligned_unique_components.parquet":caux["unique_components"]}
    frames.update(evaluate_frames(ctx,parts,controls))
    y_u=audit.U_raw.to_numpy()[ctx.valid]
    frames["metrics/U_raw.csv"]=pd.DataFrame([{"target":UT,**metrics(y_u,raw_u[ctx.valid]),
        "negative_predictions":int((raw_u[ctx.valid]<0).sum()),"D_upper_clipped":int(u.D_upper_clipped.sum()),
        "D_lower_bound_violations":int((cp[H1]["D_t"][ctx.valid]<ctx.bounds[H1][ctx.valid]).sum())}])
    primary_rows=ids.copy()
    for h in (H1,"osi_target_t06h"):
        y=ctx.data.y_train[h].to_numpy();b=controls["B"]["v18_rule"][h];c=controls["C"]["v18_rule"][h]
        primary_rows[f"actual_{h}"]=y;primary_rows[f"B_{h}"]=b;primary_rows[f"C_{h}"]=c
        primary_rows[f"C_minus_B_{h}"]=c-b
        primary_rows[f"sse_reduction_{h}"]=(b-y)**2-(c-y)**2
    frames["primary_row_effects.parquet"]=primary_rows
    return frames
