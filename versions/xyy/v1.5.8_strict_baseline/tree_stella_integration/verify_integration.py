"""Independent keyed-source/scalar combination verification; no production numerics imported."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]
SEEDS=(42,20260917,20260918);HS=(1,6,24,48);CASES=('B0','I1','I2','I3')
PAIRS=(('I3','I1'),('I3','I2'),('I2','I1'),('I1','B0'),('I3','B0'))


def read_json(p):return json.loads(Path(p).read_text())
def sha(p):
    hh=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):hh.update(b)
    return hh.hexdigest()
def save(p,v):
    p=Path(p).resolve();assert p.is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def close(a,b):np.testing.assert_allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),atol=1e-12,rtol=0,equal_nan=True)
def tag(h):return f'osi_target_t{h:02d}h'


def keyed(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    assert not d.duplicated(['fipsCode','timestamp_et']).any()
    return d.set_index(['fipsCode','timestamp_et']).sort_index()


def scores(y,p):
    y=np.asarray(y);p=np.asarray(p);assert np.isfinite(y).all() and np.isfinite(p).all()
    n=len(y)
    if not n:return dict(n=0,rmse=np.nan,mae=np.nan,sse=0.,sae=0.,bias=np.nan)
    e=p-y;sse=math.fsum(float(v)*float(v) for v in e);sae=math.fsum(abs(float(v)) for v in e)
    return dict(n=n,rmse=math.sqrt(sse/n),mae=sae/n,sse=sse,sae=sae,bias=math.fsum(map(float,e))/n)


def difference(y,b,p):
    a,c=scores(y,b),scores(y,p)
    return dict(**c,baseline_rmse=a['rmse'],rmse_delta=c['rmse']-a['rmse'],
        rmse_change_pct=100*(c['rmse']/a['rmse']-1) if a['rmse'] else np.nan,sse_reduction=a['sse']-c['sse'])


def verify_seed(seed):
    run=OUT/f'runs/split{seed}';m=read_json(run/'run_manifest.json');marker=read_json(run/'EVALUATION_COMPLETE')
    assert m['identity_hash']==marker['identity_hash']
    assert hashlib.sha256(json.dumps(m['identity'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==m['identity_hash']
    for name,h in marker['files'].items():assert sha(run/name)==h,name
    for name,h in m['identity']['evaluation_code_sha256'].items():assert sha(OUT/name)==h,name
    assert sha(OUT/'integration_config.json')==m['identity']['config_sha256']
    item=next(x for x in read_json(BASE/'p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json')['splits'] if x['split_seed']==seed)
    tr=Path(item['run_directory']);sr=ROOT/f'versions/stella_v112_strict/runs/stella_v112_nested_v1_split{seed}_model42'
    a=keyed(pd.read_parquet(tr/'control_predictions.parquet'));s=keyed(pd.read_parquet(sr/'candidate_predictions.parquet'))
    out=keyed(pd.read_parquet(run/'integrated_predictions.parquet'));source=keyed(pd.read_parquet(run/'source_predictions.parquet'))
    assert a.index.equals(s.index) and a.index.equals(out.index) and a.index.equals(source.index)
    index=a.index;n=len(a);assert n==34416 and index.get_level_values(0).nunique()==239
    cv=pd.read_csv(ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv',dtype={'fipsCode':str}).set_index('fipsCode')
    fold=index.get_level_values(0).map(cv.fold).to_numpy(int)
    for d in (a,s,out,source):np.testing.assert_array_equal(d.fold,fold)
    meta=pd.read_parquet(ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet')
    official=pd.read_parquet(ROOT/'versions/xyy/v1.5.6/targets_train_v1.5.6.parquet')
    assert len(official)==len(meta)==n
    truth=keyed(pd.concat([meta[['fipsCode','timestamp_et','hour_idx']].reset_index(drop=True),official.reset_index(drop=True)],axis=1))
    assert truth.index.equals(index)
    hours=a.hour_idx.to_numpy();expected={case:{} for case in CASES};ys={};masks={};gate24=s['anchor_gt_theta_'+tag(24)].to_numpy(bool)
    assert np.array_equal(hours,np.tile(np.arange(72,216),239))
    pd.testing.assert_index_equal(index.get_level_values(1),pd.Index(pd.Timestamp('2026-03-11')+pd.to_timedelta(hours,unit='h'),name='timestamp_et'),exact=False)
    for h in HS:
        t=tag(h);valid=hours+h<=215;y=truth[t].to_numpy();np.testing.assert_array_equal(np.isfinite(y),valid)
        np.testing.assert_array_equal(a['true_'+t],y);np.testing.assert_array_equal(s['truth_'+t],y)
        T=a['pred_NB_P24_v18_rule_'+t].to_numpy();assert np.isfinite(T[valid]).all() and np.isnan(T[~valid]).all()
        np.testing.assert_array_equal(source['T_'+t],T);np.testing.assert_array_equal(source['truth_'+t],y)
        np.testing.assert_array_equal(source['scoreable_'+t],valid);np.testing.assert_array_equal(source['target_hour_'+t],hours+h)
        for case in CASES:expected[case][h]=T.copy()
        if h>=24:
            S=s['pred_L2_'+t].to_numpy();np.testing.assert_array_equal(source['S_'+t],S)
            for prefix,column in [('S_L0','pred_L0'),('S_L1','pred_L1'),('S_theta','theta'),('S_gate','anchor_gt_theta')]:
                np.testing.assert_array_equal(source[prefix+'_'+t],s[column+'_'+t])
            g=s['anchor_gt_theta_'+t].to_numpy(bool)
            np.testing.assert_array_equal(g,valid&(s['pred_L0_'+t].to_numpy()>s['theta_'+t].to_numpy()))
            np.testing.assert_array_equal(S,np.where(g,s['pred_L0_'+t],s['pred_L1_'+t]))
        if h==48:
            for case in ('I1','I2','I3'):expected[case][h]=S.copy()
        elif h==24:
            expected['I2'][h]=S.copy()
            mixed=np.full(n,np.nan)
            for j in np.flatnonzero(valid):
                value=.5*float(T[j])+.5*float(S[j]);value=min(.65,max(0.,value));mixed[j]=value if value>=.001 else 0.
            expected['I3'][h]=mixed
        for case in CASES:
            stored=out[f'pred_{case}_{t}'].to_numpy()
            if case=='I3' and h==24:close(stored,expected[case][h])
            else:np.testing.assert_array_equal(stored,expected[case][h])
            assert np.isnan(stored[~valid]).all() and np.isfinite(stored[valid]).all()
            assert ((stored[valid]>=0)&(stored[valid]<=.65)&((stored[valid]==0)|(stored[valid]>=.001))).all()
        masks[h]=valid;ys[h]=y
    assert 'S_'+tag(1) not in out and 'S_'+tag(6) not in out
    lineage=read_json(run/'source_lineage.json');assert lineage['new_training']==0 and not lineage['source_models_modified']
    for r in lineage['rows']:
        case,h=r['candidate'],r['horizon']
        role='T' if h<=6 or case=='B0' or (case=='I1' and h==24) else ('T_and_S' if case=='I3' and h==24 else 'S')
        assert r['source']==role and r['apply_postprocess']==(role=='T_and_S')
        assert r['T_weight']==(.5 if role=='T_and_S' else float(role=='T'))
        assert r['S_weight']==(.5 if role=='T_and_S' else float(role=='S'))
        assert r['T_identity']==m['identity']['T_identity'] and r['S_identity']==m['identity']['S_identity']
    base_metrics=pd.read_csv(run/'metrics/primary_scores.csv',float_precision='round_trip')
    for r in base_metrics.itertuples(index=False):
        mask=masks[r.horizon]
        for k,v in scores(ys[r.horizon][mask],expected[r.candidate][r.horizon][mask]).items():close(v,getattr(r,k))
    table_names=('comparisons','paired_county_bootstrap','county_metrics','fold_metrics','target_window_metrics','target_day_metrics','severity_metrics','original_gate_metrics')
    tables={name:pd.read_csv(run/f'metrics/{name}.csv',float_precision='round_trip',dtype={'fipsCode':str} if name=='county_metrics' else {}) for name in table_names}
    sampled=np.random.default_rng(20260910).integers(0,239,(2000,239))
    multiplicity=np.stack([np.bincount(row,minlength=239) for row in sampled])
    metric_count=0;interval_count=0
    for name,d in tables.items():
        for r in d.itertuples(index=False):
            h=r.horizon;mask=masks[h].copy();y=ys[h];b=expected[r.reference][h];p=expected[r.candidate][h]
            assert (r.candidate,r.reference) in PAIRS and r.pair==f'{r.candidate}_minus_{r.reference}' and r.cv_seed==seed
            if name=='county_metrics':mask&=index.get_level_values(0).to_numpy()==r.fipsCode
            elif name=='fold_metrics':mask&=fold==r.fold
            elif name=='target_window_metrics':mask&=(hours+h>=r.start)&(hours+h<=r.end)
            elif name=='target_day_metrics':mask&=(index.get_level_values(1)+pd.to_timedelta(h,unit='h')).strftime('%Y-%m-%d').to_numpy()==r.target_day
            elif name=='severity_metrics':
                mask&={'zero':y==0,'0_to_.01':(y>0)&(y<=.01),'.01_to_.05':(y>.01)&(y<=.05),'above_.05':y>.05}[r.severity]
            elif name=='original_gate_metrics':assert h==24;mask&=gate24==r.original_gate
            vals=difference(y[mask],b[mask],p[mask])
            for key,val in vals.items():close(val,getattr(r,key))
            if name=='original_gate_metrics':
                eb=b[mask]-y[mask];ep=p[mask]-y[mask]
                close(np.sum(eb[eb<0]**2),r.baseline_underprediction_sse);close(np.sum(ep[ep<0]**2),r.candidate_underprediction_sse)
                assert r.baseline_underprediction_rows==int((eb<0).sum()) and r.candidate_underprediction_rows==int((ep<0).sum())
            if name=='paired_county_bootstrap':
                if np.array_equal(b[mask],p[mask]):assert r.ci_low==r.ci_high==0 and r.replicates==0
                else:
                    yy=y[mask].reshape(239,-1);bb=b[mask].reshape(239,-1);pp=p[mask].reshape(239,-1)
                    sb=np.sum((bb-yy)**2,axis=1);sp=np.sum((pp-yy)**2,axis=1)
                    delta=np.sqrt(np.sum(multiplicity*sp,axis=1)/yy.size)-np.sqrt(np.sum(multiplicity*sb,axis=1)/yy.size)
                    close(np.quantile(delta,[.025,.975]),[r.ci_low,r.ci_high]);assert r.replicates==2000;interval_count+=1
            metric_count+=1
    # Check every partition's counts and SSEs, including unchanged pre-96 short rows.
    for name in ('county_metrics','fold_metrics','target_window_metrics','target_day_metrics','severity_metrics','original_gate_metrics'):
        d=tables[name]
        if name=='target_window_metrics':d=d[~d.common_target]
        for (pair,h),g in d.groupby(['pair','horizon']):
            ref=tables['comparisons'][(tables['comparisons'].pair==pair)&(tables['comparisons'].horizon==h)].iloc[0]
            assert int(g.n.sum())==int(ref.n);close(g.sse.sum(),ref.sse);close(g.sae.sum(),ref.sae);close(g.sse_reduction.sum(),ref.sse_reduction)
    changes=pd.read_parquet(run/'row_changes.parquet');checked_rows=0
    for (pair,h),g in changes.groupby(['pair','horizon'],sort=True):
        g=keyed(g);mask=masks[h];assert g.index.equals(index[mask])
        candidate,reference=pair.split('_minus_');b=expected[reference][h][mask];p=expected[candidate][h][mask];y=ys[h][mask]
        close(g.truth,y);close(g.baseline_prediction,b);close(g.candidate_prediction,p);close(g.sse_reduction,(b-y)**2-(p-y)**2)
        np.testing.assert_array_equal(g.target_hour,hours[mask]+h);np.testing.assert_array_equal(g.fold,fold[mask])
        original=s['anchor_gt_theta_'+tag(h)].to_numpy()[mask] if h>=24 else np.zeros(mask.sum(),bool)
        np.testing.assert_array_equal(g.original_stella_gate,original);checked_rows+=len(g)
    audit=keyed(pd.read_parquet(run/'blend_postprocessing_audit.parquet'));assert audit.index.equals(index)
    a=expected['I1'][24];b=expected['I2'][24];y=ys[24];mask=masks[24]
    average=.5*a+.5*b;final=expected['I3'][24]
    close(audit.mean_before_post,average);close(audit.I3_after_post,final)
    lhs=(average-y)**2;rhs=.5*(a-y)**2+.5*(b-y)**2-.25*(a-b)**2
    close(audit.mean_squared_error,lhs);close(audit.mean_mse_identity_rhs,rhs);close(lhs,rhs)
    close(audit.postprocessing_sse_increase,(final-y)**2-lhs)
    np.testing.assert_array_equal(audit.post_changed,mask&(average!=final));np.testing.assert_array_equal(audit.original_gate,gate24)
    comp=pd.read_csv(run/'metrics/complementarity.csv',float_precision='round_trip')
    for r in comp.itertuples(index=False):
        select=mask if r.group=='all' else mask&(gate24 if r.group=='gate' else ~gate24)
        yy=y[select];aa=a[select];bb=b[select];mm=average[select];ff=final[select]
        assert r.n==int(select.sum())
        close(r.T_sse,scores(yy,aa)['sse']);close(r.S_sse,scores(yy,bb)['sse'])
        close(r.mean_before_post_sse,scores(yy,mm)['sse']);close(r.I3_sse,scores(yy,ff)['sse'])
        close(r.diversity_sse_reduction,.25*np.sum((aa-bb)**2));close(r.postprocessing_sse_increase,scores(yy,ff)['sse']-scores(yy,mm)['sse'])
        assert r.post_changed_rows==int(np.sum(select&(average!=final)))
        def corr(x,z):
            x=x-x.mean();z=z-z.mean();denom=np.linalg.norm(x)*np.linalg.norm(z)
            return float(np.dot(x,z)/denom) if denom else np.nan
        close(r.error_correlation,corr(aa-yy,bb-yy))
        anchor=s['pred_L0_'+tag(24)].to_numpy()[select]
        close(r.correction_correlation,corr(aa-anchor,bb-anchor))
    result=dict(status='PASS',cv_seed=seed,source_models_reloaded=0,new_fits=0,source_join='independent keyed pandas',
        blend_formula='independent scalar half-weight calculation and threshold branch',
        exact_copy_invariants=True,metrics_checked=metric_count,nonconstant_bootstrap_intervals_checked=interval_count,
        row_change_records_checked=checked_rows,postprocessing_identity_checked=True,group_partitions_checked=True,
        source_gate_unchanged=True,verifier_sha256=sha(Path(__file__)),absolute_tolerance=1e-12)
    save(run/'verification.json',result)
    print(f'split{seed}: independent PASS, {metric_count} grouped scores, {interval_count} nonconstant intervals, {checked_rows} row changes',flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split-seeds',nargs='+',type=int,default=list(SEEDS));args=ap.parse_args();assert tuple(args.split_seeds)==SEEDS
    sources=read_json(OUT/'reference/initial_input_hashes.json')
    for p,h in sources.items():assert sha(p)==h,p
    reports=[verify_seed(s) for s in SEEDS]
    for p,h in sources.items():assert sha(p)==h,p
    save(OUT/'final_integrity_check.json',dict(status='PASS',source_files_unchanged=len(sources),all_three_CVs_pass=True,
        source_models_reloaded=0,new_fits=0,old_files_modified=False,verifier_sha256=sha(Path(__file__))))
    print('All integration checks PASS; old inputs unchanged; no fitting or source-model replay',flush=True)


if __name__=='__main__':main()
