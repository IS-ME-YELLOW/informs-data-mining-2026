"""Same evaluation definitions as the U experiment, for G versus B/A/C."""
import numpy as np
import pandas as pd
from switch_protocol import CONFIG,HORIZONS,HORIZON_HOURS,MODES,H1,expected_mask
from metric_helpers import comparison,bootstrap
PAIRS=(("G_minus_B","B","G"),("G_minus_A","A","G"),("G_minus_C","C","G"))

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
                    **bootstrap(y[mask],b[mask],a[mask],ctx.meta.loc[mask,"fipsCode"],CONFIG["bootstrap"]["seed"],CONFIG["bootstrap"]["replicates"])})
                if mode=="v18_rule" and HORIZON_HOURS[h] in (1,6) and True:
                    take = mask&(ctx.meta.fipsCode.to_numpy()!="39117")
                    tables["bootstrap"].append({**ids,"population":"exclude_Morrow_diagnostic_only",**comparison(y[take],b[take],a[take]),
                        **bootstrap(y[take],b[take],a[take],ctx.meta.loc[take,"fipsCode"],CONFIG["bootstrap"]["seed"],CONFIG["bootstrap"]["replicates"])})
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
    effect = result["metrics/county.csv"].query("model=='v18_rule' and comparison=='G_minus_B'").copy()
    effect["fraction_of_net_sse_reduction"] = effect.sse_reduction/effect.groupby("horizon").sse_reduction.transform("sum")
    result["metrics/primary_county_influence.csv"] = effect.sort_values(["horizon","sse_reduction"],ascending=[True,False])
    return result
