"""Three-split gated replay summary; no pooled or ensembled repeated OOF score."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv(path):return pd.read_csv(path,dtype={'fipsCode':str},float_precision='round_trip')


def main():
    metric_names=('pooled','fold','county','window','bootstrap','D_component','D_spaces','D_bootstrap','gate_windows')
    collected={name:[] for name in metric_names};primary=[];diagnostics=[];routing=[];examples=[];inputs={}
    for seed in (42,20260917,20260918):
        run=HERE/'runs'/f'split{seed}';marker=json.loads((run/'G_COMPLETE').read_text())
        for name,digest in marker['files'].items():assert sha(run/name)==digest,name
        independent=json.loads((run/'logs/independent_verification.json').read_text())
        assert independent['status']=='PASS' and independent['identity_hash']==marker['identity_hash']
        local={}
        for name in metric_names:
            f=run/f'metrics/{name}.csv';d=csv(f);d.insert(0,'split_seed',seed);local[name]=d;collected[name].append(d);inputs[str(f)]=sha(f)
        route=json.loads((run/'routing_stats.json').read_text());routing.append({'split_seed':seed,**route})
        details=csv(run/'eligible_rows.csv');details.insert(0,'split_seed',seed);examples.append(details)
        for h in (f'osi_target_t{n:02d}h' for n in (1,6,24,48)):
            metric=local['pooled'].query("horizon==@h and model=='v18_rule'").set_index('comparison')
            row={'split_seed':seed,'horizon':h,'n':int(metric.loc['G_minus_B','n']),
                 'A_rmse':metric.loc['G_minus_A','baseline_rmse'],'B_rmse':metric.loc['G_minus_B','baseline_rmse'],
                 'C_full_U_rmse':metric.loc['G_minus_C','baseline_rmse'],'G_gated_rmse':metric.loc['G_minus_B','candidate_rmse'],
                 'G_vs_B_pct':metric.loc['G_minus_B','rmse_change_pct'],'G_vs_A_pct':metric.loc['G_minus_A','rmse_change_pct'],
                 'G_vs_C_pct':metric.loc['G_minus_C','rmse_change_pct'],
                 'sse_reduction_G_vs_B':metric.loc['G_minus_B','sse_reduction'],'G_vs_B_changed_rows':int(metric.loc['G_minus_B','changed_predictions'])}
            primary.append(row)
            if h!='osi_target_t01h':assert row['G_vs_B_changed_rows']==0 and row['G_vs_B_pct']==0
            county=local['county'].query("horizon==@h and model=='v18_rule' and comparison=='G_minus_B'")
            total=float(county.sse_reduction.sum());morrow=county.loc[county.fipsCode=='39117'].iloc[0]
            np.testing.assert_allclose(total,row['sse_reduction_G_vs_B'],atol=1e-12,rtol=0)
            diagnostics.append({'split_seed':seed,'horizon':h,'improved_counties':int((county.sse_reduction>0).sum()),
                'worsened_counties':int((county.sse_reduction<0).sum()),'unchanged_counties':int((county.sse_reduction==0).sum()),
                'total_sse_reduction':total,'Morrow_sse_reduction':float(morrow.sse_reduction),
                'Morrow_fraction_of_net_reduction':float(morrow.sse_reduction/total) if total else np.nan})
        for f in (run/'G_COMPLETE',run/'manifest.json',run/'routing_stats.json',run/'eligible_rows.csv',run/'logs/independent_verification.json'):inputs[str(f)]=sha(f)
    out=HERE/'summary';out.mkdir(exist_ok=True);outputs={}
    def save(name,frame):frame.to_csv(out/name,index=False);outputs[name]=sha(out/name)
    tables={name:pd.concat(frames,ignore_index=True) for name,frames in collected.items()}
    for name,frame in tables.items():save(name+'_all_splits.csv',frame)
    primary=pd.DataFrame(primary);save('primary_metrics.csv',primary)
    save('county_diagnostics.csv',pd.DataFrame(diagnostics));save('routing_summary.csv',pd.DataFrame(routing))
    all_details=pd.concat(examples,ignore_index=True);save('eligible_rows_all_splits.csv',all_details)
    focus_fips=['39117','42053','18013','54015','54013','39119','39115']
    save('focus_eligible_rows.csv',all_details.loc[all_details.fipsCode.isin(focus_fips)])
    save('focus_counties.csv',tables['county'].query("horizon=='osi_target_t01h' and model=='v18_rule' and comparison=='G_minus_B'").loc[lambda f:f.fipsCode.isin(focus_fips)])
    save('primary_bootstrap.csv',tables['bootstrap'].query("model=='v18_rule'"))
    save('primary_windows.csv',tables['window'].query("model=='v18_rule'"))
    save('primary_folds.csv',tables['fold'].query("model=='v18_rule'"))
    for path,digest in inputs.items():assert sha(path)==digest,path
    manifest={'status':'PASS','created_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
              'input_sha256':inputs,'output_sha256':outputs,'training_performed':False,
              'interpretation':'All splits previously inspected; fixed-gate development replay, not new untouched confirmation',
              'no_averaged_or_pooled_repeated_predictions':True}
    (out/'summary_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(primary.to_string(index=False));print(pd.DataFrame(diagnostics).to_string(index=False));print(pd.DataFrame(routing).to_string(index=False))


if __name__=='__main__':main()
