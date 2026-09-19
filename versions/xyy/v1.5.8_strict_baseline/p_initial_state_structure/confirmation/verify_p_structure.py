"""New-process model reload and independent numeric verification; no fitting."""
from datetime import datetime,timezone
import argparse
import json
import numpy as np
import pandas as pd
from p_structure_protocol import (HERE,CONFIG,PT,H1,HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,KEYS,KINDS,
    load_context,experiment_identity,output_path,read_json,digest_object,array_sha,sha256_file,write_json,expected_mask)
from weighted_scoped_training import specification,load_model
from independent_numerics import exact,close,independent_controls,assert_stats
from run_artifacts import check_stage

from _runtime import SEED, RUN_ID

CASES=('E0','E1','E2')
PAIRS={'E2_minus_E1':('E1','E2'),'E2_minus_E0':('E0','E2'),'E1_minus_E0':('E0','E1')}


def csv(path):
    return pd.read_csv(path,dtype={'fipsCode':str,'slice_value':str},float_precision='round_trip',keep_default_na=True)


def independent_labels(y,p,kind):
    if kind=='direct':return y.copy(),np.ones(len(y))
    if kind=='a':return np.where(y<=p,y/p,1.),p*p
    return np.where(y>p,(y-p)/(1-p),0.),(1-p)*(1-p)


def verify(run,unfrozen=False):
    run=output_path(run);ctx=load_context(SEED);m=read_json(run/'run_manifest.json')
    assert m['identity']==experiment_identity(ctx)
    identity=digest_object(m['identity']);assert identity==m['identity_hash']
    if not unfrozen:check_stage(run,'P_CV_COMPLETE',identity)
    elif (run/'P_CV_COMPLETE').exists():raise ValueError('Cannot use unfrozen verification on completed run')
    # p is read again using a separate timestamp lookup; no production history builder.
    cutoff=pd.Timestamp('2026-03-13 23:00')
    lookup={r.fipsCode:float(r.P_t) for r in ctx.raw.itertuples(index=False) if pd.Timestamp(r.timestamp_et)==cutoff}
    p=np.array([lookup[f] for f in ctx.meta.fipsCode]);exact(p,ctx.p)
    valid=ctx.meta.hour_idx.to_numpy()+1<=215;exact(valid,ctx.valid)
    codes=ctx.meta.fipsCode.to_numpy();folds=ctx.meta.fold.to_numpy();origins=ctx.meta.hour_idx.to_numpy()
    y=ctx.data.component_targets[PT].to_numpy();records=read_json(run/'new_model_manifest.json')
    assert len(records)==15 and len({r['model_id'] for r in records})==15
    exact(read_json(run/'fit_provenance.json'),[r['spec'] for r in records])
    raw={k:np.full(len(p),np.nan) for k in KINDS};ids={k:np.full(len(p),'',object) for k in KINDS}
    round_rows=[]
    for record in records:
        spec=record['spec'];outer=spec['outer_fold'];kind=spec['kind']
        expected=specification(ctx,outer,kind,identity)
        model=load_model(run,record,expected)
        take=valid&(folds==outer);active=take&({'direct':np.ones(len(p),bool),'a':p>0,'b':p<1}[kind])
        raw[kind][active]=model.predict(ctx.data.X_train.loc[active],num_threads=1)
        ids[kind][take]=record['model_id']
        probe=pd.read_parquet(run/record['probe_predictions_path'])
        assert probe.kind.eq(kind).all() and probe.outer_fold.eq(outer).all()
        for saved,curve,best in zip(spec['probes'],record['fit']['curves'],record['fit']['best_iterations']):
            inner=saved['inner_fold'];train_folds=sorted(set(range(5))-{outer,inner})
            assert saved['train']['folds']==train_folds and saved['validation']['folds']==[inner]
            for role,fs in [('train',train_folds),('validation',[inner])]:
                all_take=valid&np.isin(folds,fs)
                support=all_take&({'direct':np.ones(len(p),bool),'a':p>0,'b':p<1}[kind])
                ind_y,ind_w=independent_labels(y[support],p[support],kind)
                desc=saved[role]
                assert desc['rows']==support.sum() and desc['total_valid_rows']==all_take.sum()
                assert desc['counties']==sorted(set(codes[support]))
                assert desc['label_sha256']==array_sha(ind_y)
                assert desc['weight_sha256']==array_sha(ind_w)
                close(desc['weight_mean'],ind_w.mean());close(desc['normalized_mean'],1.)
                # ESS is a large, dimensionless diagnostic; independent summation
                # can differ at 1e-11 while all prediction checks remain 1e-12 absolute.
                np.testing.assert_allclose(desc['row_weight_ess'],np.sum(ind_w)**2/np.dot(ind_w,ind_w),rtol=1e-12,atol=1e-12)
            frame=probe.loc[probe.inner_fold==inner].reset_index(drop=True)
            va=valid&(folds==inner)&({'direct':np.ones(len(p),bool),'a':p>0,'b':p<1}[kind])
            pd.testing.assert_frame_equal(frame[KEYS],ctx.meta.loc[va,KEYS].reset_index(drop=True))
            label,weight=independent_labels(y[va],p[va],kind)
            close(frame.official_transformed_label,label);close(frame.raw_weight,weight)
            metric_label=label.astype(np.float32).astype(float) if kind=='direct' else label
            close(frame.metric_label,metric_label)
            n_all=int((valid&(folds==inner)).sum());exact(frame.all_validation_rows,np.full(len(frame),n_all))
            pred=frame.raw_prediction_at_best.to_numpy()
            score=np.sqrt(np.dot(weight,(pred-metric_label)**2)/n_all)
            close(score,curve['best_value']);close(curve['values'][best-1],score)
            assert best==int(np.argmin(curve['values']))+1
            assert frame.best_iteration.eq(best).all()
            close(curve['validation_weight_scale_training_mean'],saved['train']['weight_mean'])
            round_rows.append((outer,kind,inner,best,record['fit']['requested_rounds'],record['fit']['actual_trees']))
        assert record['fit']['formal_fit_count']==5
    assert len(round_rows)==60
    rounds=csv(run/'round_selection.csv')
    assert list(rounds[['outer_fold','kind','inner_fold','best_iteration','refit_rounds','actual_trees']].itertuples(index=False,name=None))==round_rows
    # Independent scalar branch clipping/reconstruction, with explicit endpoints.
    rebuilt=np.full(len(p),np.nan);applied={k:np.full(len(p),np.nan) for k in ('a','b')}
    for i in np.flatnonzero(valid):
        applied['a'][i]=0. if p[i]==0 else max(0.,min(1.,raw['a'][i]))
        applied['b'][i]=0. if p[i]==1 else max(0.,min(1.,raw['b'][i]))
        rebuilt[i]=p[i]*applied['a'][i]+(1-p[i])*applied['b'][i]
    newp={'E0':ctx.parts[H1]['P_t'],'E1':np.clip(raw['direct'],0,1),'E2':rebuilt}
    branch=pd.read_parquet(run/'p_branch_oof_predictions.parquet')
    audit=pd.read_parquet(run/'target_audit.parquet')
    component=pd.read_parquet(run/'component_predictions.parquet')
    oof=pd.read_parquet(run/'control_predictions.parquet')
    secondary=pd.read_parquet(run/'D_B_control_predictions.parquet')
    for frame in (branch,audit,component,oof,secondary):
        pd.testing.assert_frame_equal(frame[KEYS],ctx.meta)
        assert frame.candidate_identity_hash.eq(identity).all()
        assert frame.baseline_identity_hash.eq(CONFIG['baseline_identity_hash']).all()
        assert frame.D_G_identity_hash.eq(CONFIG['g_identity_hash']).all()
    for frame in (branch,audit):
        exact(frame.p71,p);exact(frame.scoreable,valid)
        exact(frame.target_timestamp,ctx.meta.timestamp_et+pd.Timedelta(hours=1))
    exact(audit.official_P,y)
    for kind in KINDS:
        close(branch[f'pred_{kind}_raw'],raw[kind]);exact(branch[f'source_{kind}_model_id'],ids[kind])
        exact(branch[f'{kind}_clipped_low'],np.isfinite(raw[kind])&(raw[kind]<0))
        exact(branch[f'{kind}_clipped_high'],np.isfinite(raw[kind])&(raw[kind]>1))
    for kind in ('a','b'):
        take=valid&((p>0) if kind=='a' else (p<1))
        exact(audit[f'{kind}_supported'],take);exact(branch[f'{kind}_supported'],take)
        labels=np.full(len(p),np.nan);weights=labels.copy()
        labels[take],weights[take]=independent_labels(y[take],p[take],kind)
        close(audit[f'{kind}_label'],labels);close(audit[f'{kind}_raw_weight'],weights)
        close(branch[f'{kind}_applied'],applied[kind])
        close(branch[f'{kind}_P_contribution'],(p if kind=='a' else 1-p)*applied[kind])
    close(branch.pred_E1_P,newp['E1']);close(branch.pred_E2_P,rebuilt)
    assert not branch.reconstruction_numeric_clipped.any()
    assert branch.p71_source_sha256.eq(ctx.data.loaded_hashes['raw_train']).all()
    reconstruction_ids=np.full(len(p),'',object)
    for outer in range(5):
        mask=valid&(folds==outer)
        spec={'run':identity,'outer_fold':outer,'formula':CONFIG['reconstruction'],
            'a_model':ids['a'][mask][0],'b_model':ids['b'][mask][0],'p71_source':ctx.data.loaded_hashes['raw_train']}
        reconstruction_ids[mask]='reconstruction:'+digest_object(spec)
    exact(branch.reconstruction_id,reconstruction_ids)
    raw_frame=pd.DataFrame({f'raw_{h}':ctx.direct[h] for h in HORIZONS})
    pred={};aligned={};parts={};means_all={};bpred={}
    for case in CASES:
        ip={};ib={}
        for h in HORIZONS:
            hour=HORIZON_HOURS[h];ip[hour]={};ib[hour]={}
            for c in COMPONENTS:
                target=f'{c}_target_{h.rsplit("_",1)[-1]}'
                values=newp[case] if (h,c)==(H1,'P_t') else ctx.parts[h][c]
                close(component[f'pred_{case}_{target}'],values)
                source=ctx.component[f'source_G_{target}'].to_numpy()
                if (h,c)==(H1,'P_t'):
                    if case=='E1':source=ids['direct']
                    if case=='E2':source=reconstruction_ids
                exact(component[f'source_{case}_{target}'],source)
                if (h,c)!=(H1,'P_t'):exact(component[f'pred_{case}_{target}'],values)
                ip[hour][c]=values;ib[hour][c]=ctx.b_d[h] if c=='D_t' else values
        got,buckets,means=independent_controls(ctx.meta,raw_frame,ip)
        bgot,_,_=independent_controls(ctx.meta,raw_frame,ib)
        pred[case]=got;bpred[case]=bgot;parts[case]=ip;means_all[case]=(buckets,means)
        aligned[case]={}
        for h in HORIZONS:
            hour=HORIZON_HOURS[h];mask=origins+hour<=215
            aligned[case][hour]=np.array([means.get((f,int(t)+hour),[np.nan]*4) for f,t in zip(codes,origins)])
            for mode in MODES:
                close(oof[f'pred_{case}_{mode}_{h}'],got[hour][mode])
                close(secondary[f'pred_{case}_{mode}_{h}'],bgot[hour][mode])
                close(oof[f'error_{case}_{mode}_{h}'],got[hour][mode]-ctx.data.y_train[h].to_numpy())
                if case=='E0' or mode=='C0_direct_osi' or (h!=H1 and mode in ('C1_component_osi','C2_equal_blend')) or (hour>=24 and mode=='v18_rule'):
                    exact(oof[f'pred_{case}_{mode}_{h}'],ctx.controls[mode][h])
                exact(np.isfinite(oof[f'pred_{case}_{mode}_{h}']),mask)
            exact(oof[f'actual_{h}'],ctx.data.y_train[h]);exact(oof[f'scoreable_{h}'],mask)
            exact(oof[f'raw_direct_{h}'],ctx.direct[h]);exact(oof[f'source_direct_{h}'],ctx.saved[f'source_direct_{h}'])
    unique=pd.read_parquet(run/'aligned_unique_components.parquet')
    assert len(unique)==3*4*34177 and not unique.duplicated(['case','component','fipsCode','target_timestamp']).any()
    for (case,c),group in unique.groupby(['case','component'],sort=False):
        buckets,means=means_all[case];ci=COMPONENTS.index(c)
        keys=[(f,int((ts-pd.Timestamp('2026-03-11'))/pd.Timedelta(hours=1))) for f,ts in zip(group.fipsCode,group.target_timestamp)]
        close(group.prediction,[means[k][ci] for k in keys])
        exact(group.candidate_count,[len(buckets[k]) for k in keys])
        exact(group.source_horizons,[','.join(f'osi_target_t{v:02d}h' for v,_ in sorted(buckets[k])) for k in keys])
        assert group.candidate_identity_hash.eq(identity).all()
    for comparison,(before,after) in PAIRS.items():
        for h in HORIZONS:
            hour=HORIZON_HOURS[h];truth=ctx.data.y_train[h].to_numpy()
            close(oof[f'sse_reduction_{comparison}_{h}'],(pred[before][hour]['v18_rule']-truth)**2-(pred[after][hour]['v18_rule']-truth)**2)
    def arrays(row,secondary=False):
        before,after=PAIRS[row.comparison];hour=HORIZON_HOURS[row.horizon]
        if row.domain=='OSI':
            source=bpred if secondary else pred
            return ctx.data.y_train[row.horizon].to_numpy(),source[before][hour][row.mode],source[after][hour][row.mode]
        truth=ctx.data.component_targets[f'P_t_target_{row.horizon.rsplit("_",1)[-1]}'].to_numpy()
        if row.mode=='raw':return truth,parts[before][hour]['P_t'],parts[after][hour]['P_t']
        return truth,aligned[before][hour][:,0],aligned[after][hour][:,0]
    tables={n:csv(run/f'metrics/{n}.csv') for n in ('pooled','slices','bootstrap','d_interaction','branch_diagnostics','branch_loss_decomposition','error_interactions')}
    assert len(tables['pooled'])==84 and len(tables['bootstrap'])==24 and len(tables['d_interaction'])==60
    bin_names=np.array(['p=0' if v==0 else '0<p<0.01' if v<.01 else '0.01<=p<0.1' if v<.1 else '0.1<=p<0.5' if v<.5 else '0.5<=p<=1' for v in p])
    for name in ('pooled','slices','d_interaction','bootstrap'):
        for row in tables[name].itertuples(index=False):
            truth,b,a=arrays(row,secondary=name=='d_interaction');mask=np.isfinite(truth)
            if name=='slices':
                typ,v=row.slice_type,row.slice_value;th=origins+HORIZON_HOURS[row.horizon]
                if typ=='fold':mask&=folds==int(v)
                elif typ=='county':mask&=codes==v
                elif typ=='window':lo,hi=map(int,v.split('-'));mask&=(th>=lo)&(th<=hi)
                elif typ=='p_bin':mask&=bin_names==v
                elif typ=='target_hour':mask&=th==int(v)
                elif typ=='population':assert v=='exclude_Morrow';mask&=codes!='39117'
                else:raise AssertionError('Unexpected slice')
                assert row.counties==len(set(codes[mask]))
                if typ=='county':assert row.countyName==ctx.names.loc[v]
            if name=='bootstrap' and row.population=='exclude_Morrow':mask&=codes!='39117'
            assert_stats(row._asdict(),truth[mask],b[mask],a[mask])
            if name=='bootstrap':
                frame=pd.DataFrame({'county':codes[mask],'b':(b[mask]-truth[mask])**2,'a':(a[mask]-truth[mask])**2})
                grouped=frame.groupby('county',sort=True).agg(b=('b','sum'),a=('a','sum'),n=('b','size')).to_numpy()
                rng=np.random.default_rng(int(row.seed));delta=[]
                assert row.counties==len(grouped) and row.replicates==2000 and row.seed==20260910
                for _ in range(2000):
                    bs,ass,n=grouped[rng.integers(0,len(grouped),len(grouped))].sum(axis=0)
                    delta.append(np.sqrt(ass/n)-np.sqrt(bs/n))
                close([row.ci_low,row.ci_high],np.quantile(delta,[.025,.975]))
                close(row.bootstrap_fraction_delta_lt0,np.mean(np.array(delta)<0))
                close(row.bootstrap_fraction_delta_eq0,np.mean(np.array(delta)==0))
    assert len(tables['branch_diagnostics'])==18 and len(tables['branch_loss_decomposition'])==6
    for row in tables['branch_diagnostics'].itertuples(index=False):
        take=valid.copy()
        if row.population!='all':take&=folds==int(row.population[4:])
        mask=take&({'direct':np.ones(len(p),bool),'a':p>0,'b':p<1}[row.kind])
        label,w=independent_labels(y[mask],p[mask],row.kind);pr=raw[row.kind][mask]
        assert row.support_rows==mask.sum() and row.all_rows==take.sum() and row.support_counties==len(set(codes[mask]))
        close(row.raw_label_rmse,np.sqrt(np.mean((pr-label)**2)))
        close(row.clipped_label_rmse,np.sqrt(np.mean((np.clip(pr,0,1)-label)**2)))
        close(row.raw_P_contribution_rmse,np.sqrt(np.dot(w,(pr-label)**2)/take.sum()))
        close(row.clipped_P_contribution_rmse,np.sqrt(np.dot(w,(np.clip(pr,0,1)-label)**2)/take.sum()))
        assert row.clipped_low==(pr<0).sum() and row.clipped_high==(pr>1).sum()
    for row in tables['branch_loss_decomposition'].itertuples(index=False):
        take=valid if row.population=='all' else valid&(folds==int(row.population[4:]))
        ea=p[take]*applied['a'][take]-np.minimum(p[take],y[take])
        eb=(1-p[take])*applied['b'][take]-np.maximum(y[take]-p[take],0)
        close([row.a_mse,row.b_mse,row.twice_cross_mean,row.P_mse],
              [np.mean(ea**2),np.mean(eb**2),2*np.mean(ea*eb),np.mean((rebuilt[take]-y[take])**2)])
        close(row.P_mse,row.a_mse+row.b_mse+row.twice_cross_mean)
    for row in tables['error_interactions'].itertuples(index=False):
        h=row.horizon;hour=HORIZON_HOURS[h];mask=origins+hour<=215;truth=ctx.data.y_train[h].to_numpy()[mask]
        target=np.column_stack([ctx.data.component_targets[f'{c}_target_{h.rsplit("_",1)[-1]}'].to_numpy()[mask] for c in COMPONENTS])
        estimate=np.column_stack([parts[row.case][hour][c][mask] for c in COMPONENTS]) if row.space=='raw' else aligned[row.case][hour][mask]
        weights=np.array([.4,.35,.25,-.1]);e=(estimate-target)*weights
        ep=e[:,0];other=e[:,1:].sum(axis=1);rounding=target@weights-truth
        linear=estimate@weights-truth;mode='C1_component_osi' if row.space=='raw' else 'C3_aligned_component'
        final=pred[row.case][hour][mode][mask]-truth
        close([row.P_weighted_rmse,row.other_weighted_rmse,row.two_mean_P_other_product,row.P_other_covariance,
               row.official_truth_rounding_rmse,row.linear_osi_rmse,row.postprocessed_osi_rmse,row.postprocessing_mse_change],
              [np.sqrt(np.mean(ep**2)),np.sqrt(np.mean(other**2)),2*np.mean(ep*other),np.cov(ep,other,ddof=0)[0,1],
               np.sqrt(np.mean(rounding**2)),np.sqrt(np.mean(linear**2)),np.sqrt(np.mean(final**2)),np.mean(final**2)-np.mean(linear**2)])
    # Sum county SSE independently to the pooled rows for all detailed domains.
    counties=tables['slices'].query("slice_type=='county'")
    for (domain,mode,h,pair),group in counties.groupby(['domain','mode','horizon','comparison']):
        row=tables['pooled'].query('domain==@domain and mode==@mode and horizon==@h and comparison==@pair').iloc[0]
        assert len(group)==239 and group.n.sum()==row.n
        close(group.baseline_sse.sum(),row.baseline_sse);close(group.candidate_sse.sum(),row.candidate_sse)
    if not unfrozen:check_stage(run,'P_CV_COMPLETE',identity)
    return {'status':'PASS','identity_hash':identity,'split_seed':SEED,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'verifier_sha256':sha256_file(__file__),'process':'independent model reload, no training',
        'new_outer_models':15,'inner_probes':60,'formal_fits':75,'absolute_tolerance':1e-12,
        'source_hashes_and_lineage':'PASS','scoped_labels_weights_and_rounds':'PASS',
        'independent_P_reconstruction_and_all_controls':'PASS','all_metrics_and_bootstrap_recomputed':'PASS',
        'long_horizon_and_unchanged_component_invariants':'PASS','valid_rows':34177,'counties':239}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-seed',type=int,choices=(20260917,20260918),required=True)
    parser.add_argument('--unfrozen',action='store_true')
    args=parser.parse_args();run=HERE/'runs'/RUN_ID
    result=verify(run,args.unfrozen)
    if args.unfrozen:write_json(run/'logs/independent_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
