"""Confirmation-split summaries from verified frozen A/B/C OOF artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/"experiment_config.json").read_text())
RUN=HERE/f"runs/v18_ud01_nested_v1_split{CONFIG['split_seed']}_model42"
HORIZONS=[f"osi_target_t{h:02d}h" for h in (1,6,24,48)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):return pd.read_csv(path,dtype={"fipsCode":str},float_precision="round_trip")


def main():
    marker=json.loads((RUN/"U_CV_COMPLETE").read_text())
    for name,digest in marker["files"].items():assert sha(RUN/name)==digest,name
    verify=json.loads((RUN/"logs/independent_verification.json").read_text())
    assert verify["status"]=="PASS" and verify["identity_hash"]==marker["identity_hash"]
    metrics={name:read_csv(RUN/f"metrics/{name}.csv") for name in ("pooled","county","fold","window","bootstrap","D_component","U_raw")}
    oof=pd.read_parquet(RUN/"control_predictions.parquet")
    parts=pd.read_parquet(RUN/"component_predictions.parquet")
    u=pd.read_parquet(RUN/"u_oof_predictions.parquet")
    audit=pd.read_parquet(RUN/"target_audit.parquet")
    frames={};primary=[];diagnostics=[];focus=[]
    for h in HORIZONS:
        p=metrics["pooled"].query("model=='v18_rule' and horizon==@h").set_index("comparison")
        primary.append({"horizon":h,"n":int(p.loc['C_minus_B','n']),
            "A_rmse":p.loc['C_minus_A','baseline_rmse'],"B_rmse":p.loc['C_minus_B','baseline_rmse'],"C_rmse":p.loc['C_minus_B','candidate_rmse'],
            "A_mae":p.loc['C_minus_A','baseline_mae'],"B_mae":p.loc['C_minus_B','baseline_mae'],"C_mae":p.loc['C_minus_B','candidate_mae'],
            "C_minus_B_rmse_delta":p.loc['C_minus_B','rmse_delta'],"C_minus_B_rmse_change_pct":p.loc['C_minus_B','rmse_change_pct'],
            "C_minus_A_rmse_change_pct":p.loc['C_minus_A','rmse_change_pct'],"C_minus_B_sse_reduction":p.loc['C_minus_B','sse_reduction'],
            "changed_predictions_C_vs_B":int(p.loc['C_minus_B','changed_predictions'])})
        y=oof[f"actual_{h}"].to_numpy();valid=np.isfinite(y)
        for pair,before in (("C_minus_B","B"),("C_minus_A","A")):
            c=metrics["county"].query("model=='v18_rule' and horizon==@h and comparison==@pair").copy()
            b=oof[f"pred_{before}_v18_rule_{h}"].to_numpy();after=oof[f"pred_C_v18_rule_{h}"].to_numpy()
            row_sse=(b[valid]-y[valid])**2-(after[valid]-y[valid])**2
            total=float(c.sse_reduction.sum());morrow=c.loc[c.fipsCode=='39117'].iloc[0]
            diagnostics.append({"comparison":pair,"horizon":h,"improved_counties":int((c.sse_reduction>0).sum()),
                "worsened_counties":int((c.sse_reduction<0).sum()),"unchanged_counties":int((c.sse_reduction==0).sum()),
                "improved_rows":int((row_sse>0).sum()),"worsened_rows":int((row_sse<0).sum()),"unchanged_rows":int((row_sse==0).sum()),
                "sse_reduction":total,"morrow_sse_reduction":float(morrow.sse_reduction),
                "morrow_fraction_of_net_sse_reduction":float(morrow.sse_reduction/total) if total else np.nan})
            frames[f"county_{pair}_{h}.csv"]=c.sort_values('sse_reduction',ascending=False)
            if pair=='C_minus_B' and h in HORIZONS[:2]:
                selected=c.loc[c.fipsCode.isin(['39117','42053','18013','54015','54013','39119','39115'])]
                focus.append(selected)
    frames['primary_metrics.csv']=pd.DataFrame(primary)
    frames['diagnostic_summary.csv']=pd.DataFrame(diagnostics)
    frames['focus_counties.csv']=pd.concat(focus,ignore_index=True)
    frames['primary_bootstrap.csv']=metrics['bootstrap'].query("model=='v18_rule' and comparison!='B_minus_A'")
    frames['primary_fold_metrics.csv']=metrics['fold'].query("model=='v18_rule' and comparison!='B_minus_A'")
    frames['primary_window_metrics.csv']=metrics['window'].query("model=='v18_rule' and comparison!='B_minus_A'")
    frames['D_component_metrics.csv']=metrics['D_component']
    frames['U_raw_metrics.csv']=metrics['U_raw']
    selected=(oof.fipsCode.isin(['39117','18013','42053']))&(oof.hour_idx<=75)
    early=oof.loc[selected,['fipsCode','timestamp_et','hour_idx','fold']].copy()
    early['target_hour']=early.hour_idx+1
    for key in ('official_D','K','U_raw'):early[key]=audit.loc[selected,key].to_numpy()
    early['pred_U_raw']=u.loc[selected,'pred_U_raw'].to_numpy()
    for case in ('A','B','C'):
        early[f'{case}_D']=parts.loc[selected,f'pred_{case}_D_t_target_t01h'].to_numpy()
        early[f'{case}_osi']=oof.loc[selected,f'pred_{case}_v18_rule_osi_target_t01h'].to_numpy()
    early['actual_osi']=oof.loc[selected,'actual_osi_target_t01h'].to_numpy()
    frames['focus_first_four_hours.csv']=early
    output=HERE/f'summary/split{CONFIG["split_seed"]}';output.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items():frame.to_csv(output/name,index=False)
    saved={'status':'PASS','created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':f'confirmation split {CONFIG["split_seed"]}', 'split_seed':CONFIG['split_seed'],
           'source_sha256':sha(__file__),'run_identity_hash':marker['identity_hash'],
           'input_sha256':{str(RUN/name):sha(RUN/name) for name in marker['files']},
           'independent_verification_sha256':sha(RUN/'logs/independent_verification.json'),
           'output_sha256':{name:sha(output/name) for name in frames}}
    (output/'summary_manifest.json').write_text(json.dumps(saved,indent=2)+'\n')
    print(frames['primary_metrics.csv'].to_string(index=False))
    print(frames['diagnostic_summary.csv'].to_string(index=False))


if __name__=='__main__':main()
