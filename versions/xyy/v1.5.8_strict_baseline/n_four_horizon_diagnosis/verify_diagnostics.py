"""Independent pandas-keyed reconstruction; no production alignment/scoring functions."""
import argparse
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
BASE=OUT.parent;ROOT=BASE.parents[2]
HS=(1,6,24,48);COMP=('P','N','D','R');WEIGHT=np.array([.4,.35,.25,-.1])


def hashfile(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def readjson(p):return json.loads(Path(p).read_text())


def readcsv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)


def savejson(p,v):
    assert p.resolve().is_relative_to(OUT)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def close(a,b,atol=1e-12):
    np.testing.assert_allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=1e-10,atol=atol,equal_nan=True)


def numeric_score(y,p):
    e=np.asarray(p)-np.asarray(y)
    assert np.isfinite(e).all()
    return dict(n=e.size,rmse=float(np.sqrt(np.average(np.square(e)))),mae=float(np.average(abs(e))),
                sse=float(np.square(e).sum()),bias=float(np.average(e)))


def process(comp):
    z=comp.to_numpy() @ WEIGHT
    z=np.maximum(0,np.minimum(.65,z));z[z<.001]=0
    return z


def keyed(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    return d


def output_views(sources):
    average=pd.concat([v.reset_index() for v in sources.values()]).groupby(['fipsCode','target_hour'],sort=True)[list(COMP)].mean()
    result={}
    for h,d in sources.items():
        same=average.reindex(d.index)
        assert same.notna().all().all()
        for mode,part in [('C1',d),('C3',same),('main',same if h<=6 else d)]:result[mode,h]=(part,process(part))
    return result


def bootstrap_independent(y,b,p):
    sample=np.random.default_rng(20260910).integers(0,239,size=(2000,239))
    counts=np.stack([np.bincount(row,minlength=239) for row in sample])
    sb=np.square(b-y).sum(axis=1);sp=np.square(p-y).sum(axis=1)
    delta=np.sqrt((counts*sp[None,:]).sum(axis=1)/y.size)-np.sqrt((counts*sb[None,:]).sum(axis=1)/y.size)
    return np.quantile(delta,[.025,.975])


def context(seed):
    r=BASE/f'p_information_increment/runs/v18_pinfo_v1_split{seed}_model42'
    d=keyed(pd.read_parquet(r/'component_predictions.parquet')).sort_values(['fipsCode','hour_idx'])
    raw=keyed(readcsv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str}))
    raw['target_hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta(hours=1)).astype(int)
    raw=raw.sort_values(['fipsCode','target_hour'])
    truth=raw.set_index(['fipsCode','target_hour'])[[c+'_t' for c in COMP]].rename(columns={c+'_t':c for c in COMP})
    y=raw.set_index(['fipsCode','target_hour']).osi
    sources={}
    for h in HS:
        v=d.loc[d.hour_idx+h<=215].copy();v['target_hour']=v.hour_idx+h
        v=v.set_index(['fipsCode','target_hour'])
        src=v[[f'pred_F2_{c}_t_target_t{h:02d}h' for c in COMP]].copy()
        src.columns=list(COMP);sources[h]=src.clip(0,1)
    return raw,truth,y,sources,output_views(sources)


def verify_split(seed):
    run=OUT/f'runs/split{seed}';m=readjson(run/'manifest.json')
    for rel,digest in m['files'].items():assert hashfile(run/rel)==digest,rel
    for name,digest in m['code_sha256'].items():assert hashfile(OUT/name)==digest,name
    raw,truth,y,sources,views=context(seed)
    d=keyed(pd.read_parquet(run/'rows/component_diagnostics.parquet'))
    assert len(d)==4*34416 and not d.duplicated(['horizon','fipsCode','hour_idx']).any()
    metrics=readcsv(run/'tables/component_metrics.csv')
    bins=readcsv(run/'tables/distribution_and_bins.csv')
    boot=readcsv(run/'tables/bootstrap_intervals.csv')
    bootstrap_checked=0;metric_checked=0
    old=BASE/f'runs/v18_tree_nested_v2_split{seed}_model42'
    original=keyed(pd.read_parquet(old/'base_predictions_cv.parquet')).sort_values(['fipsCode','hour_idx'])
    for h in HS:
        allh=d.loc[d.horizon==h].sort_values(['fipsCode','target_hour'])
        assert np.array_equal(allh.scoreable,allh.target_hour<=215)
        z=allh.loc[allh.scoreable].set_index(['fipsCode','target_hour'])
        assert z.index.equals(sources[h].index) and len(z)==239*(144-h)
        close(z.true_N,truth.reindex(z.index).N)
        close(z.true_OSI,y.reindex(z.index));close(z.baseline_OSI,views['main',h][1])
        close(z.N_component_clipped,sources[h].N)
        close(z.N_aligned,views['C3',h][0].N)
        close(z.N_model_raw,original.loc[original.hour_idx+h<=215,f'raw_N_t_target_t{h:02d}h'])
        assert allh.loc[~allh.scoreable,['true_N','N_component_clipped','N_aligned','N_model_raw']].isna().all().all()
        ss=z.index.get_level_values('target_hour').to_numpy();tt=z.true_N.to_numpy()
        last=raw.loc[raw.target_hour==71].set_index('fipsCode')
        lp=z.index.get_level_values(0).map(last.P_t).to_numpy();ln=z.index.get_level_values(0).map(last.N_t).to_numpy()
        for row in bins.loc[bins.horizon==h].itertuples(index=False):
            kind,name=row.group_type,row.group
            if kind=='all':mask=np.ones(len(z),bool)
            elif kind=='common_target':mask=ss>=120
            elif kind=='target_window':
                lo,hi=map(int,name.split('_'));mask=(ss>=lo)&(ss<=hi)
            elif kind=='candidate_count':mask=z.candidate_count.to_numpy()==int(name)
            elif kind=='fold':mask=z.fold.to_numpy()==int(name)
            elif kind=='truth_N':mask={'N=0':tt==0,'0<N<=1e-4':(tt>0)&(tt<=1e-4),'1e-4<N<=1e-3':(tt>1e-4)&(tt<=1e-3),'1e-3<N<=1e-2':(tt>1e-3)&(tt<=1e-2),'N>1e-2':tt>1e-2}[name]
            elif kind=='P71':mask={'p=0':lp==0,'0<p<.01':(lp>0)&(lp<.01),'.01<=p<.1':(lp>=.01)&(lp<.1),'.1<=p<.5':(lp>=.1)&(lp<.5),'.5<=p<=1':lp>=.5}[name]
            elif kind=='N71':mask={'n=0':ln==0,'0<n<=.001':(ln>0)&(ln<=.001),'n>.001':ln>.001}[name]
            else:raise AssertionError(kind)
            assert row.n==int(mask.sum())
            if mask.any():
                pr=z['N_'+row.space].to_numpy();vals=numeric_score(tt[mask],pr[mask])
                for k,v in vals.items():close(v,getattr(row,k))
                close(row.sse_fraction_of_full,vals['sse']/np.square(pr-tt).sum())
                event=tt[mask]>.001;pred=pr[mask]>.001
                for k,v in {'tp':(event&pred).sum(),'fp':(~event&pred).sum(),'fn':(event&~pred).sum(),'tn':(~event&~pred).sum()}.items():close(v,getattr(row,k))
            else:assert row.sse==0 and np.isnan(row.rmse)
            metric_checked+=1
        bb=boot.loc[(boot.horizon==h)&(boot.scenario=='N_aligned_minus_clipped')].iloc[0]
        limits=bootstrap_independent(tt.reshape(239,-1),z.N_component_clipped.to_numpy().reshape(239,-1),z.N_aligned.to_numpy().reshape(239,-1))
        close(limits,[bb.ci_low,bb.ci_high]);bootstrap_checked+=1
    # Independent decomposition using source-indexed predictions, not saved error terms.
    dec=keyed(pd.read_parquet(run/'rows/osi_error_decomposition.parquet'))
    terms=readcsv(run/'tables/osi_decomposition.csv')
    for (h,mode),group in dec.groupby(['horizon','mode']):
        q=group.set_index(['fipsCode','target_hour']).sort_index();predcomp,pred=views[mode,h]
        tc=truth.reindex(q.index).to_numpy();pc=predcomp.reindex(q.index).to_numpy();yy=y.reindex(q.index).to_numpy()
        err=(pc-tc)*WEIGHT;adjust=pred-pc@WEIGHT+tc@WEIGHT-yy
        close(q[[f'e_{c}' for c in COMP]],err);close(q.b,adjust);close(q.OSI_error,pred-yy)
        close(q.N_signed_allocation,err[:,1]*(pred-yy));close(err.sum(axis=1)+adjust,pred-yy)
        allerr=np.column_stack([err,adjust]);names=list(COMP)+['b']
        these=terms.loc[(terms.horizon==h)&(terms['mode']==mode)]
        for row in these.itertuples(index=False):
            i,j=names.index(row.term_a),names.index(row.term_b)
            close(row.sse_term,np.sum(allerr[:,i]*allerr[:,j])*(1 if i==j else 2))
        close(these.sse_term.sum(),np.square(pred-yy).sum())
    scenarios=readcsv(run/'tables/oracle_metrics.csv');verified_oracle_rows=0
    for scenario in scenarios.loc[scenarios.reference=='B_current','scenario'].unique():
        modified={h:v.copy() for h,v in sources.items()}
        is_anchor=scenario.startswith('A_')
        for h,v in modified.items():
            if scenario.startswith('O_N_source_'):
                if h==int(scenario.rsplit('_',1)[-1]):v['N']=truth.reindex(v.index).N
            elif scenario.startswith('O_N_partial_'):
                alpha=float(scenario.rsplit('_',1)[-1]);v['N']=v.N*(1-alpha)+truth.reindex(v.index).N*alpha
            elif scenario=='A_N_zero':v['N']=0.
            elif scenario=='A_N71':
                lookup=raw.loc[raw.target_hour==71].set_index('fipsCode').N_t
                v['N']=v.index.get_level_values(0).map(lookup).to_numpy()
            else:
                cols=COMP if scenario=='O_all_components' else (scenario.split('_')[1],)
                for c in cols:v[c]=truth.reindex(v.index)[c]
        new=output_views(modified)
        path=run/'rows'/('anchor_predictions.parquet' if is_anchor else 'oracle_predictions.parquet')
        saved=keyed(pd.read_parquet(path,filters=[('scenario','=',scenario)]))
        assert saved.uses_future_truth.eq(not is_anchor).all() and not saved.deployable.any()
        for h,g in saved.groupby('horizon'):
            g=g.set_index(['fipsCode','target_hour']).sort_index();pc,p=new['main',h]
            yy=y.reindex(g.index).to_numpy();b=views['main',h][1]
            assert g.index.equals(pc.index)
            close(g.true_OSI,yy);close(g.baseline_OSI,b);close(g.diagnostic_OSI,p)
            close(g[[f'diagnostic_{c}' for c in COMP]],pc.to_numpy())
            close(g.sse_reduction,np.square(b-yy)-np.square(p-yy))
            row=scenarios.loc[(scenarios.horizon==h)&(scenarios.reference=='B_current')&(scenarios.scenario==scenario)].iloc[0]
            vals=numeric_score(yy,p)
            for key,val in vals.items():close(row[key],val)
            close(row.rmse_change_pct,100*(vals['rmse']/np.sqrt(np.mean((b-yy)**2))-1))
            if scenario.startswith(('O_N_','A_')):
                bb=boot.loc[(boot.horizon==h)&(boot.scenario==scenario)].iloc[0]
                limits=bootstrap_independent(yy.reshape(239,-1),b.reshape(239,-1),p.reshape(239,-1))
                close(limits,[bb.ci_low,bb.ci_high]);bootstrap_checked+=1
            verified_oracle_rows+=len(g)
    # Full matching reconstruction with independent per-hour arrays and exact tie order.
    matches=readcsv(run/'tables/matched_support.csv',dtype={'fipsCode':str,'donor_fipsCode':str})
    rankings=keyed(pd.read_parquet(run/'tables/matching_rankings_without_outcomes.parquet'))
    identity=readjson(run/'tables/matching_ranking_identity.json')
    assert hashfile(run/'tables/matching_rankings_without_outcomes.parquet')==identity['sha256']
    cols=['split_seed','fipsCode','fold','horizon','target_hour','donor_fipsCode','donor_fold','rank','distance']
    pd.testing.assert_frame_equal(matches[cols],rankings[cols],check_dtype=False,check_exact=True)
    cv=readcsv(ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv',dtype={'fipsCode':str}).set_index('fipsCode')
    meta=keyed(pd.read_parquet(ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet'))
    x=pd.read_parquet(ROOT/'versions/xyy/v1.5.6/features_train_v1.5.6.parquet')
    x.index=pd.MultiIndex.from_arrays([meta.fipsCode,meta.hour_idx])
    fcols=['last_P_t','last_N_t','last_R_t','N_t_mean_72h','N_t_max_72h','gust_at_t{h}h','log_customers','pct_forest']
    queries=0
    for (f,h,s),g in matches.groupby(['fipsCode','horizon','target_hour'],sort=True):
        g=g.sort_values('rank');held=int(cv.loc[f,'fold']);donors=sorted(cv.index[cv.fold!=held]);columns=[v.format(h=h) for v in fcols]
        xx=x.loc[[(j,s-h) for j in donors],columns].to_numpy();q=x.loc[(f,s-h),columns].to_numpy()
        st=xx.std(axis=0);active=st>0
        dd=np.sqrt(np.sum(((q[active]-xx[:,active])/st[active])**2,axis=1))
        order=np.lexsort((np.array(donors),dd))[:5]
        assert g.donor_fipsCode.tolist()==[donors[k] for k in order]
        close(g.distance,dd[order]);assert (g.donor_fold!=held).all() and g['rank'].tolist()==list(range(1,6))
        dy=truth.loc[[(j,s) for j in g.donor_fipsCode],'N'].to_numpy()
        close(g.donor_true_N,dy);close(g.query_true_N,truth.loc[(f,s),'N'])
        close(g.donor_true_OSI,y.loc[[(j,s) for j in g.donor_fipsCode]])
        close(g.query_true_OSI,y.loc[(f,s)])
        queries+=1
    # Formula sanity independently from vectorized customer count differences.
    arr=raw[['outageCount','customersTracked','N_t','R_t']].to_numpy().reshape(239,216,4)
    signed=np.diff(arr[:,:,0],axis=1)/arr[:,1:,1]
    pos=np.maximum(signed,0);neg=np.maximum(-signed,0)
    nr=np.stack([(pos[:,:-2]+pos[:,1:-1]+pos[:,2:])/3,(neg[:,:-2]+neg[:,1:-1]+neg[:,2:])/3],axis=2)
    close(nr,arr[:,2:215,2:],atol=1e-6)
    result=dict(status='PASS',split_seed=seed,method='independent keyed pandas reconstruction; no production numerical helpers',
        metrics_checked=metric_checked,bootstrap_intervals_checked=bootstrap_checked,
        oracle_and_anchor_rows_checked=verified_oracle_rows,matched_queries_checked=queries,
        current_reconstruction=True,decomposition_verified=True,N_R_formula_verified=True,
        floating_tolerances=dict(atol=1e-12,rtol=1e-10,formula_atol=1e-6),training_performed=False,
        verifier_sha256=hashfile(Path(__file__)))
    savejson(run/'verification.json',result)
    print(f'split{seed}: independent verification PASS ({verified_oracle_rows} scenario rows; {queries} matched queries)',flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split-seeds',nargs='+',type=int,default=[42,20260917,20260918]);args=ap.parse_args()
    assert args.split_seeds==[42,20260917,20260918]
    inputs=readjson(OUT/'reference/initial_input_hashes.json')
    for p,h in inputs.items():assert hashfile(p)==h,p
    results=[verify_split(seed) for seed in args.split_seeds]
    for p,h in inputs.items():assert hashfile(p)==h,p
    savejson(OUT/'final_integrity_check.json',dict(status='PASS',input_files_checked=len(inputs),all_inputs_unchanged=True,
        all_split_verifications_passed=True,training_performed=False,
        output_directory=str(OUT),verifier_sha256=hashfile(Path(__file__))))
    print('All independent checks passed; frozen inputs unchanged',flush=True)


if __name__=='__main__':main()
