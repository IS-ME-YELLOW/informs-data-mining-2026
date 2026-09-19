"""Compare all three U-supervision splits without pooling repeated OOF predictions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OLD=HERE.parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv(path):return pd.read_csv(path,dtype={'fipsCode':str},float_precision='round_trip')


def main():
    entries=[(42,OLD,OLD/'summary/seed42')]+[(s,HERE/f'split{s}',HERE/f'split{s}/summary/split{s}') for s in (20260917,20260918)]
    metric_names=('pooled','fold','county','window','bootstrap','D_component','U_raw')
    summary_names=('primary_metrics','diagnostic_summary','focus_counties','focus_first_four_hours')
    collected={name:[] for name in (*metric_names,*summary_names)};input_hashes={};index=[]
    for seed,folder,summary in entries:
        run=folder/'runs'/f'v18_ud01_nested_v1_split{seed}_model42'
        marker=json.loads((run/'U_CV_COMPLETE').read_text())
        for name,digest in marker['files'].items():assert sha(run/name)==digest,name
        manifest=json.loads((run/'run_manifest.json').read_text())
        assert manifest['identity_hash']==marker['identity_hash']
        verification=json.loads((run/'logs/independent_verification.json').read_text())
        assert verification['status']=='PASS' and verification['identity_hash']==marker['identity_hash']
        if seed!=42:
            completed=json.loads((folder/'completion.json').read_text());assert completed['status']=='COMPLETE'
            for name,digest in completed['files'].items():assert sha(folder/name)==digest,name
        sm=json.loads((summary/'summary_manifest.json').read_text());assert sm['status']=='PASS'
        for name,digest in sm['output_sha256'].items():assert sha(summary/name)==digest,name
        for name in metric_names:
            path=run/f'metrics/{name}.csv';frame=csv(path);frame.insert(0,'split_seed',seed)
            collected[name].append(frame);input_hashes[str(path)]=sha(path)
        for name in summary_names:
            path=summary/f'{name}.csv';frame=csv(path);frame.insert(0,'split_seed',seed)
            collected[name].append(frame);input_hashes[str(path)]=sha(path)
        for path in (run/'U_CV_COMPLETE',run/'run_manifest.json',run/'logs/independent_verification.json',summary/'summary_manifest.json'):
            input_hashes[str(path)]=sha(path)
        index.append({'split_seed':seed,'role':'development_reference' if seed==42 else 'confirmation',
                      'run':str(run),'identity_hash':marker['identity_hash'],'baseline_identity_hash':manifest['identity']['baseline_identity_hash'],
                      'new_models':5,'formal_fits':25})
    tables={name:pd.concat(frames,ignore_index=True) for name,frames in collected.items()}
    output=HERE/'summary';output.mkdir(exist_ok=True);hashes={}
    def save(name,frame):
        frame.to_csv(output/name,index=False);hashes[name]=sha(output/name)
    for name,frame in tables.items():save(name+'_all_splits.csv',frame)
    save('run_index.csv',pd.DataFrame(index))
    assert len(tables['primary_metrics'])==12
    assert tables['primary_metrics'].groupby('horizon').n.nunique().eq(1).all()
    primary_boot=tables['bootstrap'].query("model=='v18_rule' and comparison in ['C_minus_B','C_minus_A']")
    save('primary_bootstrap.csv',primary_boot)
    save('primary_fold_metrics.csv',tables['fold'].query("model=='v18_rule' and comparison!='B_minus_A'"))
    save('primary_window_metrics.csv',tables['window'].query("model=='v18_rule' and comparison!='B_minus_A'"))
    stats=[]
    for _,row in tables['primary_metrics'].iterrows():
        seed,h=row.split_seed,row.horizon
        d=tables['diagnostic_summary'].query("split_seed==@seed and horizon==@h and comparison=='C_minus_B'").iloc[0]
        boot=primary_boot.query("split_seed==@seed and horizon==@h and comparison=='C_minus_B' and population=='all_counties'").iloc[0]
        item={**row.to_dict(),**{k:d[k] for k in ('improved_counties','worsened_counties','morrow_sse_reduction','morrow_fraction_of_net_sse_reduction')},
              'ci_low':boot.ci_low,'ci_high':boot.ci_high,'bootstrap_fraction_delta_lt0':boot.bootstrap_fraction_delta_lt0}
        exclude=primary_boot.query("split_seed==@seed and horizon==@h and comparison=='C_minus_B' and population=='exclude_Morrow_diagnostic_only'")
        if len(exclude):
            r=exclude.iloc[0];item.update(exclude_Morrow_rmse_change_pct=r.rmse_change_pct,exclude_Morrow_ci_low=r.ci_low,exclude_Morrow_ci_high=r.ci_high)
        stats.append(item)
        county=tables['county'].query("split_seed==@seed and horizon==@h and model=='v18_rule' and comparison=='C_minus_B'")
        assert len(county)==239 and county.n.sum()==row.n
        np.testing.assert_allclose(county.sse_reduction.sum(),row.C_minus_B_sse_reduction,atol=1e-12,rtol=0)
        if h in ('osi_target_t24h','osi_target_t48h'):assert row.C_minus_B_rmse_delta==0 and row.changed_predictions_C_vs_B==0
    save('primary_diagnostic_overview.csv',pd.DataFrame(stats))
    stability=[]
    for h in ('osi_target_t01h','osi_target_t06h'):
        data=tables['county'].query("horizon==@h and model=='v18_rule' and comparison=='C_minus_B'")
        wide=data.pivot(index=['fipsCode','countyName'],columns='split_seed',values='sse_reduction')
        wide.columns=[f'sse_reduction_split{s}' for s in wide.columns]
        columns=[f'sse_reduction_split{s}' for s in (20260917,20260918)]
        wide['confirmation_improved_splits']=(wide[columns]>0).sum(axis=1)
        wide['confirmation_worsened_splits']=(wide[columns]<0).sum(axis=1)
        wide=wide.reset_index();wide.insert(0,'horizon',h);stability.append(wide)
    save('county_effect_stability.csv',pd.concat(stability,ignore_index=True))
    descriptive=[]
    for h,group in tables['primary_metrics'].query('split_seed!=42').groupby('horizon',sort=True):
        descriptive.append({'horizon':h,'confirmation_splits':2,
            'mean_within_split_rmse_delta_C_vs_B':group.C_minus_B_rmse_delta.mean(),
            'mean_within_split_rmse_change_pct_C_vs_B':group.C_minus_B_rmse_change_pct.mean(),
            'min_within_split_rmse_change_pct_C_vs_B':group.C_minus_B_rmse_change_pct.min(),
            'max_within_split_rmse_change_pct_C_vs_B':group.C_minus_B_rmse_change_pct.max()})
    save('confirmation_descriptive_summary.csv',pd.DataFrame(descriptive))
    for path,digest in input_hashes.items():assert sha(path)==digest,path
    manifest={'status':'PASS','created_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
              'input_sha256':input_hashes,'output_sha256':hashes,'no_pooled_repeated_OOF_score':True,
              'role_note':'seed42 development; both prespecified confirmation splits reported without selection'}
    (output/'summary_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(stats).to_string(index=False))


if __name__=='__main__':main()
