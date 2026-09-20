"""Independent weight/scope audit, real-model reload and pandas OSI reconstruction."""
import lightgbm as lgb
from transfer_protocol import OUT,BASE,ROOT,RUN,REF,F2_ID,TRANSFER,KINDS,CASES,KEYS,HORIZONS,HORIZON_HOURS,COMPONENTS
from transfer_protocol import read_json,write_json,sha256_file,digest_object,array_sha,load_data,normalize,params
import numpy as np
import pandas as pd
from pathlib import Path


def close(a,b):np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=1e-12,rtol=0,equal_nan=True)


def independent_scope(data,p,folds,h,kind):
    eligible=np.isin(data.row_folds,folds)&(data.meta_train.hour_idx.to_numpy()+HORIZON_HOURS[h]<=215)
    supported=(p!=0) if kind=='a' else (p!=1)
    idx=np.flatnonzero(eligible&supported);y=data.component_targets[f'P_t_target_{h.rsplit("_",1)[-1]}'].iloc[idx].to_numpy(float)
    initial=p[idx]
    if kind=='a':labels=np.where(y<initial,y/initial,1.);weights=np.square(initial)
    else:labels=np.where(y>initial,(y-initial)/(1-initial),0.);weights=np.square(1-initial)
    meta=data.meta_train.iloc[idx];omitted=np.flatnonzero(eligible&~supported)
    record=dict(folds=sorted(folds),counties=sorted(meta.fipsCode.unique()),rows=len(idx),total_valid_rows=int(eligible.sum()),
        omitted_rows=len(omitted),row_indices_sha256=array_sha(idx),
        row_keys_sha256=digest_object((meta.fipsCode+'|'+meta.timestamp_et.astype(str)).tolist()),
        omitted_indices_sha256=array_sha(omitted),label_sha256=array_sha(labels),
        feature_sha256=array_sha(data.X_train.iloc[idx].to_numpy()),p_sha256=array_sha(initial),weight_sha256=array_sha(weights),
        weight_mean=float(weights.mean()),weight_min=float(weights.min()),weight_max=float(weights.max()),
        normalized_weight_sha256=array_sha(weights/weights.mean()),row_weight_ess=float(weights.sum()**2/np.sum(weights**2)))
    return record,idx,labels,weights


def score(y,p):
    mask=np.isfinite(y);assert np.isfinite(p[mask]).all()
    e=p[mask]-y[mask]
    if not len(e):return dict(n=0,rmse=np.nan,mae=np.nan,sse=0.,bias=np.nan)
    return dict(n=len(e),rmse=float(np.sqrt(np.average(np.square(e)))),mae=float(np.average(abs(e))),sse=float(np.square(e).sum()),bias=float(np.average(e)))


def comparison(y,b,p):
    bm,pm=score(y,b),score(y,p)
    return dict(**pm,baseline_rmse=bm['rmse'],rmse_delta=pm['rmse']-bm['rmse'],
        rmse_change_pct=100*(pm['rmse']/bm['rmse']-1) if bm['rmse'] else np.nan,sse_reduction=bm['sse']-pm['sse'])


def finish(z):
    z=np.maximum(0,np.minimum(.65,z));z[z<.001]=0;return z


def independent_controls(meta,parts,direct):
    all_sources=[]
    for h in HORIZONS:
        hit=meta.hour_idx.to_numpy()+HORIZON_HOURS[h]<=215
        d=meta.loc[hit,['fipsCode','timestamp_et']].copy();d['timestamp_et']+=pd.to_timedelta(HORIZON_HOURS[h],unit='h')
        for c in COMPONENTS:d[c]=np.clip(parts[h][c][hit],0,1)
        all_sources.append(d)
    avg=pd.concat(all_sources).groupby(['fipsCode','timestamp_et'],sort=True)[list(COMPONENTS)].mean()
    controls={};aligned={};w=np.array([.4,.35,.25,-.1])
    for h in HORIZONS:
        hh=HORIZON_HOURS[h];valid=meta.hour_idx.to_numpy()+hh<=215
        key=pd.MultiIndex.from_arrays([meta.fipsCode,meta.timestamp_et+pd.to_timedelta(hh,unit='h')])
        a=avg.reindex(key).to_numpy();a[~valid]=np.nan;aligned[h]=a
        raw=np.column_stack([np.clip(parts[h][c],0,1) for c in COMPONENTS]);same=np.maximum(raw@w,0)
        controls[h]={'C0_direct_osi':finish(direct[h].copy()),'C1_component_osi':finish(same),
            'C2_equal_blend':finish((direct[h]+same)/2),'C3_aligned_component':finish(np.maximum(a@w,0))}
        controls[h]['v18_rule']=controls[h]['C3_aligned_component' if hh<=6 else 'C1_component_osi'].copy()
        for v in controls[h].values():v[~valid]=np.nan
    return controls,aligned


def independent_interval(y,b,p):
    mask=np.isfinite(y);yy=y[mask].reshape(239,-1);bb=b[mask].reshape(239,-1);pp=p[mask].reshape(239,-1)
    sampled=np.random.default_rng(20260910).integers(0,239,(2000,239))
    counts=np.stack([np.bincount(r,minlength=239) for r in sampled])
    sb=np.sum((bb-yy)**2,axis=1);sp=np.sum((pp-yy)**2,axis=1)
    deltas=np.sqrt(np.sum(counts*sp,axis=1)/yy.size)-np.sqrt(np.sum(counts*sb,axis=1)/yy.size)
    return np.quantile(deltas,[.025,.975])


def verify():
    data=load_data();meta=data.meta_train.copy();meta['fold']=data.row_folds;p=data.X_train.last_P_t.to_numpy()
    raw=pd.read_csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str},float_precision='round_trip')
    hist=raw.loc[pd.to_datetime(raw.timestamp_et)==pd.Timestamp('2026-03-13 23:00')].set_index('fipsCode')
    np.testing.assert_array_equal(p,meta.fipsCode.map(hist.P_t))
    run=read_json(RUN/'run_manifest.json');identity=run['identity_hash'];assert digest_object(run['identity'])==identity
    for name,digest in run['identity']['training_code_sha256'].items():assert sha256_file(OUT/name)==digest,name
    for path,digest in run['identity']['source_sha256'].items():assert sha256_file(path)==digest,path
    for stage in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']:
        marker=read_json(RUN/stage);assert marker['identity_hash']==identity
        for path,digest in marker['files'].items():assert sha256_file(RUN/path)==digest,path
    assert sha256_file(OUT/'evaluate_transfer.py')==read_json(RUN/'EVALUATION_COMPLETE')['evaluation_code_sha256']
    branch=pd.read_parquet(RUN/'branch_oof_predictions.parquet');byh={h:normalize(branch.loc[branch.horizon==h]) for h in TRANSFER}
    registry={r['reconstruction_id']:r['definition'] for r in read_json(RUN/'reconstruction_registry.json')}
    rebuilt={};model_checks=[];probe_count=0
    for h in TRANSFER:
        hh=HORIZON_HOURS[h];valid=meta.hour_idx.to_numpy()+hh<=215;d=byh[h]
        np.testing.assert_array_equal(d[KEYS].to_numpy(),meta[KEYS].to_numpy());close(d.P71,p)
        np.testing.assert_array_equal(d.scoreable,valid)
        predicted={k:np.full(len(meta),np.nan) for k in KINDS}
        for outer in range(5):
            allowed=[f for f in range(5) if f!=outer];pair={}
            for kind in KINDS:
                target=('P_anchor_a_' if kind=='a' else 'P_excess_b_')+h.rsplit('_',1)[-1]
                receipt=read_json(RUN/f'models/outer{outer}/{target}.json');spec=receipt['spec'];fit=receipt['fit']
                assert digest_object(spec)==receipt['model_id'] and spec['identity_hash']==identity
                assert spec['outer_fold']==outer and spec['horizon']==h and spec['kind']==kind and spec['target']==target
                assert spec['feature_names']==list(data.X_train) and len(spec['feature_names'])==163
                assert spec['params']==fit['params']==params() and spec['allowed_folds']==allowed
                record,_,_,_=independent_scope(data,p,allowed,h,kind);assert record==spec['refit']
                assert fit['refit_weight_mean']==record['weight_mean']
                assert sha256_file(RUN/receipt['probe_path'])==receipt['probe_sha256']
                probes=pd.read_parquet(RUN/receipt['probe_path']);best=[]
                for q,rec,curve in zip(allowed,spec['probes'],fit['curves']):
                    tr,_,_,_=independent_scope(data,p,[f for f in allowed if f!=q],h,kind)
                    va,idx,labels,weights=independent_scope(data,p,[q],h,kind)
                    assert rec['inner_fold']==curve['inner_fold']==q and tr==rec['train'] and va==rec['validation']
                    assert not set(tr['counties']).intersection(va['counties'])
                    held=set(meta.loc[meta.fold==outer,'fipsCode']);assert not held.intersection(tr['counties']+va['counties'])
                    z=probes.loc[probes.inner_fold==q].sort_values('row_index')
                    np.testing.assert_array_equal(z.row_index,idx);np.testing.assert_array_equal(z.branch_label,labels)
                    np.testing.assert_array_equal(z.raw_weight,weights);np.testing.assert_array_equal(z.P71,p[idx])
                    close(z.normalized_weight,weights/tr['weight_mean'])
                    assert z.training_weight_mean.eq(tr['weight_mean']).all() and z.all_validation_rows.eq(va['total_valid_rows']).all()
                    assert curve['training_weight_mean']==tr['weight_mean'] and curve['metric']=='raw_P_contribution_rmse'
                    iteration=int(np.argmin(curve['values'])+1);assert z.best_iteration.eq(iteration).all();best.append(iteration)
                    value=float(np.sqrt(np.dot(weights,np.square(z.prediction_at_best.to_numpy()-labels))/va['total_valid_rows']))
                    close(value,curve['best_value']);close(value,curve['values'][iteration-1]);probe_count+=1
                assert best==fit['best_iterations'] and fit['requested_rounds']==max(1,int(np.mean(best))) and fit['fits']==5
                assert sha256_file(RUN/receipt['model_path'])==receipt['model_sha256']
                model=lgb.Booster(model_file=str(RUN/receipt['model_path']))
                assert model.feature_name()==list(data.X_train) and model.current_iteration()==fit['trees_saved']
                assert model.params['objective']=='regression'
                active=valid&(meta.fold.to_numpy()==outer)&((p>0) if kind=='a' else (p<1))
                predicted[kind][active]=model.predict(data.X_train.loc[active],num_threads=1)
                assert d.loc[active,'source_'+kind].eq(receipt['model_id']).all();pair[kind]=receipt['model_id']
                model_checks.append(dict(horizon=h,outer_fold=outer,kind=kind,trees_saved=fit['trees_saved'],model_id=receipt['model_id']))
            mask=valid&(meta.fold.to_numpy()==outer);ids=d.loc[mask,'reconstruction_id'].unique();assert len(ids)==1
            definition=registry[ids[0]];assert digest_object(definition)==ids[0]
            assert definition['branch_model_ids']==pair and definition['horizon']==h and definition['outer_fold']==outer
            assert definition['identity_hash']==identity and definition['p71_sha256']==array_sha(p)
            assert definition['p71_source_sha256']==data.loaded_hashes['raw_train']
        applied={}
        for kind in KINDS:
            active=valid&((p>0) if kind=='a' else (p<1));close(predicted[kind],d['raw_'+kind])
            assert d['support_'+kind].to_numpy().tolist()==active.tolist()
            assert d.loc[~active,'source_'+kind].eq('').all() and np.isnan(predicted[kind][~active]).all()
            result=np.zeros(len(meta));result[~valid]=np.nan;result[active]=np.minimum(1,np.maximum(0,predicted[kind][active]));applied[kind]=result
            close(result,d['applied_'+kind])
        ca=p*applied['a'];cb=(1-p)*applied['b'];v=np.minimum(1,np.maximum(0,ca+cb));rebuilt[h]=v
        close(ca,d.contribution_a);close(cb,d.contribution_b);close(v,d.pred_P)
        assert np.isnan(v[~valid]).all() and np.isfinite(v[valid]).all()
        close(d.true_P,data.component_targets[f'P_t_target_t{hh:02d}h'])
    original=normalize(pd.read_parquet(REF/'component_predictions.parquet'));refcontrols=normalize(pd.read_parquet(REF/'control_predictions.parquet'))
    compsaved=normalize(pd.read_parquet(RUN/'component_predictions.parquet'));saved=normalize(pd.read_parquet(RUN/'control_predictions.parquet'))
    direct={h:refcontrols['raw_direct_'+h].to_numpy() for h in HORIZONS}
    result={};aligned={};n_metrics=0
    allm=pd.read_csv(RUN/'metrics/all_control_metrics.csv',float_precision='round_trip')
    primary=pd.read_csv(RUN/'metrics/primary_scores.csv',float_precision='round_trip')
    boots=pd.read_csv(RUN/'metrics/bootstrap_intervals.csv',float_precision='round_trip')
    pm=pd.read_csv(RUN/'metrics/P_component_metrics.csv',float_precision='round_trip')
    bins=pd.read_csv(RUN/'metrics/P71_bins.csv',float_precision='round_trip')
    for case,changed in CASES.items():
        parts={}
        for h in HORIZONS:
            parts[h]={}
            for c in COMPONENTS:
                tag=f'{c}_target_{h.rsplit("_",1)[-1]}';v=original[f'pred_F2_{tag}'].to_numpy()
                if h==changed and c=='P_t':v=rebuilt[h]
                parts[h][c]=v;close(v,compsaved[f'pred_{case}_{tag}'])
                source=byh[h].reconstruction_id if h==changed and c=='P_t' else original['source_F2_'+tag]
                np.testing.assert_array_equal(source,compsaved[f'source_{case}_{tag}'])
        result[case],aligned[case]=independent_controls(meta,parts,direct)
        for h in HORIZONS:
            y=data.y_train[h].to_numpy();s=meta.hour_idx.to_numpy()+HORIZON_HOURS[h]
            for mode,v in result[case][h].items():
                close(v,saved[f'pred_{case}_{mode}_{h}'])
                b=result['B0'][h][mode]
                if case=='B0':close(v,refcontrols[f'pred_F2_{mode}_{h}'])
                row=allm.loc[(allm['case']==case)&(allm.horizon==h)&(allm['mode']==mode)].iloc[0]
                for k,value in comparison(y,b,v).items():close(value,row[k])
                n_metrics+=1
            v=result[case][h]['v18_rule'];b=result['B0'][h]['v18_rule']
            row=primary.loc[(primary['case']==case)&(primary.horizon==h)].iloc[0]
            for k,value in comparison(y,b,v).items():close(value,row[k])
            if case!='B0':
                row=boots.loc[(boots.domain=='OSI')&(boots['case']==case)&(boots.horizon==h)].iloc[0]
                close(independent_interval(y,b,v),[row.ci_low,row.ci_high])
                unaffected=(s<72+HORIZON_HOURS[changed])&np.isfinite(y);np.testing.assert_array_equal(v[unaffected],b[unaffected])
                if HORIZON_HOURS[h]>=24 and h!=changed:np.testing.assert_array_equal(v,b)
            yp=data.component_targets[f'P_t_target_{h.rsplit("_",1)[-1]}'].to_numpy()
            for space,v,bp in [('component_clipped',parts[h]['P_t'],original[f'pred_F2_P_t_target_{h.rsplit("_",1)[-1]}'].to_numpy()),
                ('aligned',aligned[case][h][:,0],aligned['B0'][h][:,0])]:
                for group,mask in [('full',np.isfinite(yp)),('common_120_215',(s>=120)&(s<=215))]:
                    row=pm.loc[(pm['case']==case)&(pm.horizon==h)&(pm.space==space)&(pm['group']==group)].iloc[0]
                    for k,value in comparison(yp[mask],bp[mask],v[mask]).items():close(value,row[k])
                masks={'p=0':p==0,'0<p<.01':(p>0)&(p<.01),'.01<=p<.1':(p>=.01)&(p<.1),'.1<=p<.5':(p>=.1)&(p<.5),'.5<=p<=1':p>=.5}
                for group,mask in masks.items():
                    row=bins.loc[(bins['case']==case)&(bins.horizon==h)&(bins.space==space)&(bins.P71_group==group)].iloc[0]
                    for k,value in comparison(yp[mask],bp[mask],v[mask]).items():close(value,row[k])
                if h==changed:
                    row=boots.loc[(boots.domain=='P')&(boots['case']==case)&(boots.horizon==h)&(boots.space==space)].iloc[0]
                    close(independent_interval(yp,bp,v),[row.ci_low,row.ci_high])
    for filename,groupcols in [('fold_metrics',['fold']),('county_metrics',['fipsCode']),('window_metrics',['start','end'])]:
        df=pd.read_csv(RUN/f'metrics/{filename}.csv',dtype={'fipsCode':str},float_precision='round_trip')
        for row in df.itertuples(index=False):
            h=row.horizon;s=meta.hour_idx.to_numpy()+HORIZON_HOURS[h]
            if filename=='fold_metrics':mask=meta.fold.to_numpy()==row.fold
            elif filename=='county_metrics':mask=meta.fipsCode.to_numpy()==row.fipsCode
            else:mask=(s>=row.start)&(s<=row.end)
            y=data.y_train[h].to_numpy();b=result['B0'][h]['v18_rule'];v=result[row.case][h]['v18_rule']
            for k,value in comparison(y[mask],b[mask],v[mask]).items():close(value,getattr(row,k))
    for path,digest in run['identity']['source_sha256'].items():assert sha256_file(path)==digest,path
    report=dict(status='PASS',identity_hash=identity,new_models_reloaded=30,inner_probes_checked=probe_count,
        full_control_metrics_checked=n_metrics,bootstrap_intervals_checked=len(boots),
        scoped_weights_and_labels_verified=True,independent_P_reconstruction=True,independent_pandas_C3=True,
        unchanged_sources_and_output_propagation_verified=True,all_county_fold_window_metrics_checked=True,
        training_performed_by_verifier=False,verifier_sha256=sha256_file(Path(__file__)),model_checks=model_checks)
    write_json(RUN/'logs/independent_verification.json',report)
    write_json(RUN/'VERIFIED_COMPLETE',dict(identity_hash=identity,verification_sha256=sha256_file(RUN/'logs/independent_verification.json'),verifier_sha256=sha256_file(Path(__file__))))
    print('PASS: 30 real models, 120 probes, reconstruction, 80 control scores, 18 intervals and county/fold/window tables verified',flush=True)


if __name__=='__main__':verify()
