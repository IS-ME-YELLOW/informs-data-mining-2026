"""Independent numeric consistency checks for the strict-baseline analysis."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE / "runs/v18_tree_nested_v2_split42_model42"
OUT = HERE / "analysis"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path,dtype={"fipsCode":str},float_precision="round_trip")


def main():
    record=json.loads((OUT/"analysis_manifest.json").read_text())
    for name,digest in record["output_sha256"].items():assert sha(OUT/name)==digest,name
    for name,digest in record["input_sha256"].items():assert sha(ROOT/name)==digest,name
    assert sha(HERE/"analyze_cv_results.py")==record["analysis_source_sha256"]
    oof=pd.read_parquet(RUN/"oof_predictions.parquet")
    county=read(OUT/"county_error_summary.csv")
    folds=read(OUT/"fold_error_summary.csv")
    summary=read(OUT/"horizon_comparison.csv").set_index("horizon")
    concentration=read(OUT/"error_concentration.csv").set_index("horizon")
    assert len(county)==239*4 and len(folds)==5*4
    assert not county.duplicated(["fipsCode","horizon"]).any()
    for h in [1,6,24,48]:
        name=f"osi_target_t{h:02d}h";mask=oof.hour_idx+h<=215
        y=oof.loc[mask,f"actual_{name}"].to_numpy()
        p=oof.loc[mask,f"pred_v18_rule_{name}"].to_numpy()
        assert np.isfinite(y).all() and np.isfinite(p).all()
        sse=float(np.sum((p-y)**2));rmse=float(np.sqrt(sse/len(y)))
        assert abs(rmse-summary.loc[h,"strict_rmse"])<1e-12
        c=county[county.horizon==h].sort_values("sse",ascending=False)
        f=folds[folds.horizon==h]
        assert c.n.sum()==len(y)==f.n.sum()
        assert abs(c.sse.sum()-sse)<1e-12 and abs(f.sse.sum()-sse)<1e-12
        assert abs(c.sse_share.sum()-1)<1e-12 and abs(f.sse_share.sum()-1)<1e-12
        assert abs(c.head(5).sse.sum()/sse-concentration.loc[h,"top5_sse_share"])<1e-12
        for fold in range(5):
            row=f[f.fold==fold].iloc[0];v=mask & (oof.fold==fold)
            errors=oof.loc[v,f"pred_v18_rule_{name}"]-oof.loc[v,f"actual_{name}"]
            assert abs(np.sqrt(np.mean(errors**2))-row.rmse)<1e-12
    bounds=read(OUT/"early_D_bound_diagnosis.csv")
    assert len(bounds)==239*4
    assert np.array_equal(bounds.violates_lower_bound,bounds.gap>1e-6)
    assert (bounds.actual_D+1e-6>=bounds.known_D_lower_bound).all()
    assert int(bounds.violates_lower_bound.sum())==record["known_D_bound_violating_rows"]
    assert bounds.loc[bounds.violates_lower_bound,"fipsCode"].nunique()==record["known_D_bound_violating_counties"]
    protected=json.loads((HERE/"reference/protected_original_assets.json").read_text())
    assert all(sha(ROOT/p)==digest for p,digest in protected.items())
    snapshot=json.loads((HERE/"source_snapshot_manifest.json").read_text())
    assert all(sha(HERE/item["snapshot"])==item["sha256"] for item in snapshot["files"])
    assert not (RUN/"models/final").exists()
    assert not list(RUN.glob("submission_*.csv"))
    output={"passed":True,"horizons":4,"county_metric_rows":len(county),"fold_metric_rows":len(folds),
            "paired_comparison_inputs":"within the same strict CV for control comparisons; historical comparisons descriptive only",
            "rmse_sse_shares_recomputed":True,"D_bound_diagnosis_only_no_projection":True,
            "protected_original_assets_unchanged":len(protected),"training_source_snapshot_unchanged":True,
            "no_final_models_or_submission":True}
    (OUT/"verification.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
