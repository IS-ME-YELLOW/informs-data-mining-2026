"""Independent feature reconstruction, saved-model reload and all-metric audit."""
from datetime import datetime,timezone
import argparse
import json
import numpy as np
import pandas as pd
from information_protocol import (HERE,CONFIG,PT,H1,HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,KEYS,KINDS,CASES,NEW_CASES,
    load_reference,attach_features,experiment_identity,output_path,read_json,digest_object,array_sha,sha256_file,write_json)
from information_training import specification,load_model
from independent_features import verify_features
from independent_numerics import exact,close,independent_controls,assert_stats
from run_artifacts import check_stage

PAIRS={'F1_minus_F0':('F0','F1'),'F2_minus_F0':('F0','F2'),'F2_minus_F1':('F1','F2')}


def csv(path):return pd.read_csv(path,dtype={'fipsCode':str,'slice_value':str},float_precision='round_trip')


def labels(y,p,kind):
    return (np.where(y<=p,y/p,1.),p*p) if kind=='a' else (np.where(y>p,(y-p)/(1-p),0.),(1-p)*(1-p))


def verify(run,unfrozen=False):
    run=output_path(run);ctx=attach_features(load_reference(42));m=read_json(run/'run_manifest.json')
    assert m['identity']==experiment_identity(ctx);identity=digest_object(m['identity']);assert identity==m['identity_hash']
    if not unfrozen:check_stage(run,'INFO_CV_COMPLETE',identity)
    elif (run/'INFO_CV_COMPLETE').exists():raise ValueError('Unfrozen verification forbidden on frozen run')
    feature_report,rebuilt_features=verify_features(ctx,return_features=True)
    p=ctx.p;valid=ctx.valid;codes=ctx.meta.fipsCode.to_numpy();folds=ctx.meta.fold.to_numpy();origins=ctx.meta.hour_idx.to_numpy()
    y=ctx.data.component_targets[PT].to_numpy()
    records=read_json(run/'new_model_manifest.json');assert len(records)==20 and len({r['model_id'] for r in records})==20
    assert read_json(run/'fit_provenance.json')==[r['spec'] for r in records]
    raw={'F0':{k:ctx.f0_branch[f'pred_{k}_raw'].to_numpy(copy=True) for k in KINDS}}
    ids={'F0':{k:ctx.f0_branch[f'source_{k}_model_id'].to_numpy(copy=True) for k in KINDS}}
    raw.update({c:{k:np.full(len(p),np.nan) for k in KINDS} for c in NEW_CASES})
    ids.update({c:{k:np.full(len(p),'',object) for k in KINDS} for c in NEW_CASES})
    rounds=[]
    for rec in records:
        spec=rec['spec'];case=spec['candidate'];outer=spec['outer_fold'];kind=spec['kind'];assert case in NEW_CASES and kind in KINDS
        assert spec['feature_package_identity']==ctx.feature_bundle['manifest']['identity_hash']
        expected=specification(ctx.cases[case],outer,kind,identity);model=load_model(run,rec,expected)
        all_rows=valid&(folds==outer);active=all_rows&((p>0) if kind=='a' else (p<1))
        # Predict on independently reconstructed augmented features, not saved augmented values.
        raw[case][kind][active]=model.predict(rebuilt_features[case]['train'].loc[active],num_threads=1)
        ids[case][kind][all_rows]=rec['model_id']
        probe=pd.read_parquet(run/rec['probe_predictions_path'])
        assert probe.candidate.eq(case).all() and probe.kind.eq(kind).all() and probe.outer_fold.eq(outer).all()
        for detail,curve,best in zip(spec['probes'],rec['fit']['curves'],rec['fit']['best_iterations']):
            q=detail['inner_fold'];trfolds=sorted(set(range(5))-{outer,q})
            assert detail['train']['folds']==trfolds and detail['validation']['folds']==[q]
            for role,fs in [('train',trfolds),('validation',[q])]:
                all_take=valid&np.isin(folds,fs);take=all_take&((p>0) if kind=='a' else (p<1))
                target,w=labels(y[take],p[take],kind);desc=detail[role]
                assert desc['rows']==take.sum() and desc['total_valid_rows']==all_take.sum()
                assert desc['counties']==sorted(set(codes[take]))
                assert desc['label_sha256']==array_sha(target) and desc['weight_sha256']==array_sha(w)
                close(desc['weight_mean'],w.mean());close(desc['normalized_mean'],1.)
                np.testing.assert_allclose(desc['row_weight_ess'],w.sum()**2/np.dot(w,w),rtol=1e-12,atol=1e-12)
            take=valid&(folds==q)&((p>0) if kind=='a' else (p<1))
            frame=probe.loc[probe.inner_fold==q].reset_index(drop=True)
            pd.testing.assert_frame_equal(frame[KEYS],ctx.meta.loc[take,KEYS].reset_index(drop=True))
            target,w=labels(y[take],p[take],kind);close(frame.official_transformed_label,target);close(frame.metric_label,target);close(frame.raw_weight,w)
            n_all=int((valid&(folds==q)).sum());assert frame.all_validation_rows.eq(n_all).all()
            score=np.sqrt(np.dot(w,(frame.raw_prediction_at_best.to_numpy()-target)**2)/n_all)
            close(score,curve['best_value']);close(score,curve['values'][best-1]);assert best==int(np.argmin(curve['values']))+1
            assert frame.best_iteration.eq(best).all();close(curve['validation_weight_scale_training_mean'],detail['train']['weight_mean'])
            rounds.append((case,outer,kind,q,best,rec['fit']['requested_rounds'],rec['fit']['actual_trees']))
        assert rec['fit']['formal_fit_count']==5
        close(rec['fit']['refit_weight_mean'],spec['refit']['weight_mean'])
    assert len(rounds)==80
    table=csv(run/'round_selection.csv')
    assert list(table[['candidate','outer_fold','kind','inner_fold','best_iteration','refit_rounds','actual_trees']].itertuples(index=False,name=None))==rounds
    # Reject wrong candidate/schema/split identity without writing or fitting.
    for change in ('candidate','feature_names','identity_hash'):
        bad=dict(records[0]['spec'])
        bad[change]=list(reversed(bad[change])) if change=='feature_names' else 'invalid'
        try:load_model(run,records[0],bad)
        except ValueError:pass
        else:raise AssertionError('Invalid receipt reuse accepted')
    branch=pd.read_parquet(run/'branch_oof_predictions.parquet');audit=pd.read_parquet(run/'target_audit.parquet')
    component=pd.read_parquet(run/'component_predictions.parquet');oof=pd.read_parquet(run/'control_predictions.parquet')
    assert len(branch)==3*34416 and not branch.duplicated(['case','fipsCode','timestamp_et']).any()
    for frame in (audit,component,oof):pd.testing.assert_frame_equal(frame[KEYS],ctx.meta)
    for frame in (audit,component,oof,branch):
        assert frame.candidate_identity_hash.eq(identity).all() and frame.reference_E2_identity_hash.eq(CONFIG['reference_identity_hash']).all()
    exact(audit.p71,p);exact(audit.official_P,y);exact(audit.scoreable,valid)
    exact(audit.target_timestamp,ctx.meta.timestamp_et+pd.Timedelta(hours=1))
    for k in KINDS:
        take=valid&((p>0) if k=='a' else (p<1));t=np.full(len(p),np.nan);w=t.copy();t[take],w[take]=labels(y[take],p[take],k)
        exact(audit[k+'_supported'],take);close(audit[k+'_label'],t);close(audit[k+'_raw_weight'],w)
    p_hat={};applied={};rid={}
    for case in CASES:
        applied[case]={k:np.full(len(p),np.nan) for k in KINDS};p_hat[case]=np.full(len(p),np.nan)
        for i in np.flatnonzero(valid):
            applied[case]['a'][i]=0. if p[i]==0 else min(1.,max(0.,raw[case]['a'][i]))
            applied[case]['b'][i]=0. if p[i]==1 else min(1.,max(0.,raw[case]['b'][i]))
            p_hat[case][i]=p[i]*applied[case]['a'][i]+(1-p[i])*applied[case]['b'][i]
        if case=='F0':rid[case]=ctx.f0_branch.reconstruction_id.to_numpy()
        else:
            rid[case]=np.full(len(p),'',object)
            for outer in range(5):
                mask=valid&(folds==outer)
                s={'run':identity,'candidate':case,'outer_fold':outer,'feature_package':ctx.feature_bundle['manifest']['identity_hash'],
                    'a_model':ids[case]['a'][mask][0],'b_model':ids[case]['b'][mask][0],
                    'p_source':ctx.data.loaded_hashes['raw_train'],'formula':CONFIG['reconstruction']}
                rid[case][mask]='reconstruction:'+digest_object(s)
        frame=branch.loc[branch['case']==case].reset_index(drop=True);pd.testing.assert_frame_equal(frame[KEYS],ctx.meta)
        exact(frame.p71,p);exact(frame.scoreable,valid);exact(frame.reconstruction_id,rid[case])
        exact(frame.target_timestamp,ctx.meta.timestamp_et+pd.Timedelta(hours=1))
        assert frame.p71_source_sha256.eq(ctx.data.loaded_hashes['raw_train']).all()
        assert frame.feature_package_identity.eq('frozen_163_E2' if case=='F0' else ctx.feature_bundle['manifest']['identity_hash']).all()
        for k in KINDS:
            active=valid&((p>0) if k=='a' else (p<1));exact(frame[k+'_supported'],active)
            close(frame[f'pred_{k}_raw'],raw[case][k]);exact(frame[f'source_{k}_model_id'],ids[case][k])
            close(frame[k+'_applied'],applied[case][k]);close(frame[k+'_P_contribution'],(p if k=='a' else 1-p)*applied[case][k])
            exact(frame[k+'_clipped_low'],np.isfinite(raw[case][k])&(raw[case][k]<0));exact(frame[k+'_clipped_high'],np.isfinite(raw[case][k])&(raw[case][k]>1))
        close(frame.pred_P,p_hat[case]);assert not frame.reconstruction_numeric_clipped.any()
    exact(p_hat['F0'],ctx.parts[H1]['P_t'])
    pred={};parts={};aligned={};mean_data={};df=pd.DataFrame({f'raw_{h}':ctx.direct[h] for h in HORIZONS})
    for case in CASES:
        ip={}
        for h in HORIZONS:
            hour=HORIZON_HOURS[h];ip[hour]={}
            for c in COMPONENTS:
                target=f'{c}_target_{h.rsplit("_",1)[-1]}'
                values=p_hat[case] if (h,c)==(H1,'P_t') else ctx.parts[h][c]
                close(component[f'pred_{case}_{target}'],values);ip[hour][c]=values
                source=rid[case] if (h,c)==(H1,'P_t') else ctx.f0_component[f'source_E2_{target}'].to_numpy()
                exact(component[f'source_{case}_{target}'],source)
                if (h,c)!=(H1,'P_t'):exact(component[f'pred_{case}_{target}'],ctx.parts[h][c])
        controls,buckets,means=independent_controls(ctx.meta,df,ip)
        pred[case]=controls;parts[case]=ip;mean_data[case]=(buckets,means);aligned[case]={}
        for h in HORIZONS:
            hour=HORIZON_HOURS[h];mask=origins+hour<=215;truth=ctx.data.y_train[h].to_numpy()
            aligned[case][hour]=np.array([means.get((f,int(t)+hour),[np.nan]*4) for f,t in zip(codes,origins)])
            exact(oof[f'actual_{h}'],truth);exact(oof[f'scoreable_{h}'],mask)
            exact(oof[f'raw_direct_{h}'],ctx.direct[h]);exact(oof[f'source_direct_{h}'],ctx.f0_saved[f'source_direct_{h}'])
            for mode in MODES:
                close(oof[f'pred_{case}_{mode}_{h}'],controls[hour][mode]);close(oof[f'error_{case}_{mode}_{h}'],controls[hour][mode]-truth)
                exact(np.isfinite(oof[f'pred_{case}_{mode}_{h}']),mask)
                if case=='F0' or mode=='C0_direct_osi' or (h!=H1 and mode in ('C1_component_osi','C2_equal_blend')) or (hour>=24 and mode=='v18_rule'):
                    exact(oof[f'pred_{case}_{mode}_{h}'],ctx.controls[mode][h])
    unique=pd.read_parquet(run/'aligned_unique_components.parquet')
    assert len(unique)==3*4*34177 and not unique.duplicated(['case','component','fipsCode','target_timestamp']).any()
    for (case,c),g in unique.groupby(['case','component'],sort=False):
        buckets,means=mean_data[case];ci=COMPONENTS.index(c)
        keys=[(f,int((ts-pd.Timestamp('2026-03-11'))/pd.Timedelta(hours=1))) for f,ts in zip(g.fipsCode,g.target_timestamp)]
        close(g.prediction,[means[k][ci] for k in keys]);exact(g.candidate_count,[len(buckets[k]) for k in keys])
        exact(g.source_horizons,[','.join(f'osi_target_t{v:02d}h' for v,_ in sorted(buckets[k])) for k in keys])
        assert g.candidate_identity_hash.eq(identity).all()
    for pair,(b,a) in PAIRS.items():
        for h in HORIZONS:
            hour=HORIZON_HOURS[h];truth=ctx.data.y_train[h].to_numpy()
            close(oof[f'sse_reduction_{pair}_{h}'],(pred[b][hour]['v18_rule']-truth)**2-(pred[a][hour]['v18_rule']-truth)**2)
    def arrays(row):
        b,a=PAIRS[row.comparison];hour=HORIZON_HOURS[row.horizon]
        if row.domain=='OSI':return ctx.data.y_train[row.horizon].to_numpy(),pred[b][hour][row.mode],pred[a][hour][row.mode]
        truth=ctx.data.component_targets[f'P_t_target_{row.horizon.rsplit("_",1)[-1]}'].to_numpy()
        if row.mode=='raw':return truth,parts[b][hour]['P_t'],parts[a][hour]['P_t']
        return truth,aligned[b][hour][:,0],aligned[a][hour][:,0]
    tables={n:csv(run/f'metrics/{n}.csv') for n in ('pooled','slices','bootstrap','branch_diagnostics','branch_loss_decomposition','error_interactions')}
    assert len(tables['pooled'])==84 and len(tables['bootstrap'])==16
    bins=np.array(['p=0' if v==0 else '0<p<0.01' if v<.01 else '0.01<=p<0.1' if v<.1 else '0.1<=p<0.5' if v<.5 else '0.5<=p<=1' for v in p])
    for name in ('pooled','slices','bootstrap'):
        for row in tables[name].itertuples(index=False):
            truth,b,a=arrays(row);mask=np.isfinite(truth)
            if name=='slices':
                typ,v=row.slice_type,row.slice_value;th=origins+HORIZON_HOURS[row.horizon]
                if typ=='fold':mask&=folds==int(v)
                elif typ=='county':mask&=codes==v
                elif typ=='window':lo,hi=map(int,v.split('-'));mask&=(th>=lo)&(th<=hi)
                elif typ=='p_bin':mask&=bins==v
                elif typ=='target_hour':mask&=th==int(v)
                elif typ=='population':assert v=='exclude_Morrow';mask&=codes!='39117'
                else:raise AssertionError('Unknown slice')
                assert row.counties==len(set(codes[mask]))
            if name=='bootstrap' and row.population=='exclude_Morrow':mask&=codes!='39117'
            assert_stats(row._asdict(),truth[mask],b[mask],a[mask])
            if name=='bootstrap':
                frame=pd.DataFrame({'county':codes[mask],'b':(b[mask]-truth[mask])**2,'a':(a[mask]-truth[mask])**2})
                grouped=frame.groupby('county',sort=True).agg(b=('b','sum'),a=('a','sum'),n=('b','size')).to_numpy()
                rng=np.random.default_rng(20260910);deltas=[]
                assert row.counties==len(grouped) and row.replicates==2000 and row.seed==20260910
                for _ in range(2000):
                    bs,ass,n=grouped[rng.integers(0,len(grouped),len(grouped))].sum(axis=0);deltas.append(np.sqrt(ass/n)-np.sqrt(bs/n))
                close([row.ci_low,row.ci_high],np.quantile(deltas,[.025,.975]));close(row.bootstrap_fraction_delta_lt0,np.mean(np.array(deltas)<0));close(row.bootstrap_fraction_delta_eq0,np.mean(np.array(deltas)==0))
    assert len(tables['branch_diagnostics'])==36 and len(tables['branch_loss_decomposition'])==18
    for row in tables['branch_diagnostics'].itertuples(index=False):
        take=valid if row.population=='all' else valid&(folds==int(row.population[4:]));mask=take&((p>0) if row.kind=='a' else (p<1))
        target,w=labels(y[mask],p[mask],row.kind);pr=raw[row.case][row.kind][mask]
        assert row.support_rows==mask.sum() and row.all_rows==take.sum() and row.support_counties==len(set(codes[mask]))
        close(row.raw_label_rmse,np.sqrt(np.mean((pr-target)**2)));close(row.clipped_label_rmse,np.sqrt(np.mean((np.clip(pr,0,1)-target)**2)))
        close(row.raw_P_contribution_rmse,np.sqrt(np.dot(w,(pr-target)**2)/take.sum()));close(row.clipped_P_contribution_rmse,np.sqrt(np.dot(w,(np.clip(pr,0,1)-target)**2)/take.sum()))
        assert row.clipped_low==(pr<0).sum() and row.clipped_high==(pr>1).sum()
    for row in tables['branch_loss_decomposition'].itertuples(index=False):
        take=valid if row.population=='all' else valid&(folds==int(row.population[4:]))
        ea=p[take]*applied[row.case]['a'][take]-np.minimum(y[take],p[take]);eb=(1-p[take])*applied[row.case]['b'][take]-np.maximum(y[take]-p[take],0)
        close([row.a_mse,row.b_mse,row.twice_cross_mean,row.P_mse],[np.mean(ea**2),np.mean(eb**2),2*np.mean(ea*eb),np.mean((p_hat[row.case][take]-y[take])**2)])
        close(row.P_mse,row.a_mse+row.b_mse+row.twice_cross_mean)
    for row in tables['error_interactions'].itertuples(index=False):
        h=row.horizon;hour=HORIZON_HOURS[h];take=origins+hour<=215;truth=ctx.data.y_train[h].to_numpy()[take]
        target=np.column_stack([ctx.data.component_targets[f'{c}_target_{h.rsplit("_",1)[-1]}'].to_numpy()[take] for c in COMPONENTS])
        estimate=np.column_stack([parts[row.case][hour][c][take] for c in COMPONENTS]) if row.space=='raw' else aligned[row.case][hour][take]
        weights=np.array([.4,.35,.25,-.1]);err=(estimate-target)*weights;ep=err[:,0];other=err[:,1:].sum(axis=1)
        rounding=target@weights-truth;linear=estimate@weights-truth;mode='C1_component_osi' if row.space=='raw' else 'C3_aligned_component';final=pred[row.case][hour][mode][take]-truth
        close([row.P_weighted_rmse,row.other_weighted_rmse,row.two_mean_P_other_product,row.P_other_covariance,row.official_truth_rounding_rmse,row.linear_osi_rmse,row.postprocessed_osi_rmse,row.postprocessing_mse_change],
            [np.sqrt(np.mean(ep**2)),np.sqrt(np.mean(other**2)),2*np.mean(ep*other),np.cov(ep,other,ddof=0)[0,1],np.sqrt(np.mean(rounding**2)),np.sqrt(np.mean(linear**2)),np.sqrt(np.mean(final**2)),np.mean(final**2)-np.mean(linear**2)])
    for key,g in tables['slices'].query("slice_type=='county'").groupby(['domain','mode','horizon','comparison']):
        domain,mode,h,pair=key;po=tables['pooled'];r=po.loc[(po.domain==domain)&(po['mode']==mode)&(po.horizon==h)&(po.comparison==pair)].iloc[0]
        assert len(g)==239 and g.n.sum()==r.n;close(g.baseline_sse.sum(),r.baseline_sse);close(g.candidate_sse.sum(),r.candidate_sse)
    if not unfrozen:check_stage(run,'INFO_CV_COMPLETE',identity)
    return {'status':'PASS','identity_hash':identity,'split_seed':42,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'verifier_sha256':sha256_file(__file__),'new_outer_models':20,'inner_probes':80,'formal_fits':100,
        'independently_rebuilt_features_used_for_model_predictions':True,'feature_verification':feature_report,
        'scoped_labels_weights_rounds_and_receipts':'PASS','P_reconstruction_and_all_controls':'PASS',
        'all_metrics_and_bootstrap_recomputed':'PASS','long_horizon_and_other_component_invariants':'PASS',
        'reference_and_feature_identities':'PASS','prediction_metric_absolute_tolerance':1e-12,'training_performed_by_verifier':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--split-seed',type=int,choices=(42,),default=42)
    parser.add_argument('--unfrozen',action='store_true');args=parser.parse_args()
    run=HERE/'runs/v18_pinfo_v1_split42_model42';result=verify(run,args.unfrozen)
    if args.unfrozen:write_json(run/'logs/independent_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
