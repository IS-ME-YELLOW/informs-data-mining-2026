"""Independent model reload, label-scope, pandas alignment and metric verification."""
import lightgbm as lgb
from loss_protocol import *


def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)


def scoped_record(data,folds,target):
    hh=int(target.rsplit('_',1)[-1][1:3]);mask=np.isin(data.row_folds,folds)&(data.meta_train.hour_idx.to_numpy()+hh<=215)
    indices=np.flatnonzero(mask);m=data.meta_train.iloc[indices]
    return dict(folds=sorted(folds),counties=sorted(m.fipsCode.unique()),rows=len(indices),
        row_indices_sha256=array_sha(indices),row_keys_sha256=digest_object((m.fipsCode+'|'+m.timestamp_et.astype(str)).tolist()),
        feature_sha256=array_sha(data.X_train.iloc[indices].to_numpy()),
        label_sha256=array_sha(data.component_targets[target].iloc[indices].to_numpy(dtype=float)))


def score(y,p):
    y,p=np.asarray(y),np.asarray(p);hit=~np.isnan(y)
    assert np.isfinite(p[hit]).all()
    e=p[hit]-y[hit]
    return dict(n=len(e),rmse=float(np.sqrt(np.mean(np.square(e)))),mae=float(np.mean(abs(e))),sse=float(np.sum(np.square(e))),bias=float(np.mean(e)))


def finish(z):
    v=np.maximum(np.minimum(z,.65),0);v[v<.001]=0;return v


def weighted(comp):
    return comp @ np.array([.4,.35,.25,-.1])


def independent_controls(meta,parts,direct):
    views=[]
    for h in HORIZONS:
        hh=HORIZON_HOURS[h];hit=meta.hour_idx.to_numpy()+hh<=215
        f=meta.loc[hit,['fipsCode','timestamp_et']].copy();f['timestamp_et']+=pd.to_timedelta(hh,unit='h')
        for c in COMPONENTS:f[c]=np.clip(parts[h][c][hit],0,1)
        views.append(f)
    aligned=pd.concat(views).groupby(['fipsCode','timestamp_et'],sort=True)[list(COMPONENTS)].mean()
    result={};aligned_arrays={}
    for h in HORIZONS:
        hh=HORIZON_HOURS[h];valid=meta.hour_idx.to_numpy()+hh<=215
        keys=pd.MultiIndex.from_arrays([meta.fipsCode,meta.timestamp_et+pd.to_timedelta(hh,unit='h')])
        aa=aligned.reindex(keys).to_numpy();aa[~valid]=np.nan;aligned_arrays[h]=aa
        same=np.column_stack([np.clip(parts[h][c],0,1) for c in COMPONENTS])
        c1=np.maximum(weighted(same),0)
        result[h]={'C0_direct_osi':finish(direct[h].copy()),'C1_component_osi':finish(c1),
            'C2_equal_blend':finish((direct[h]+c1)/2),'C3_aligned_component':finish(np.maximum(weighted(aa),0))}
        result[h]['v18_rule']=result[h]['C3_aligned_component' if hh<=6 else 'C1_component_osi'].copy()
        for value in result[h].values():value[~valid]=np.nan
    return result,aligned_arrays


def paired_interval(data,h,y,b,p):
    hit=~np.isnan(y);nc=239;yy=y[hit].reshape(nc,-1);bb=b[hit].reshape(nc,-1);pp=p[hit].reshape(nc,-1)
    sampled=np.random.default_rng(20260910).integers(0,nc,(2000,nc))
    multiplicity=np.stack([np.bincount(a,minlength=nc) for a in sampled])
    ssb=np.sum((bb-yy)**2,axis=1);ssp=np.sum((pp-yy)**2,axis=1)
    diff=np.sqrt(np.sum(multiplicity*ssp,axis=1)/yy.size)-np.sqrt(np.sum(multiplicity*ssb,axis=1)/yy.size)
    return np.quantile(diff,[.025,.975])


def verify():
    data=load_data();meta=data.meta_train.copy();meta['fold']=data.row_folds
    run=read_json(RUN/'run_manifest.json');identity=run['identity_hash']
    assert digest_object(run['identity'])==identity
    for path,digest in run['identity']['source_sha256'].items():assert sha256_file(path)==digest,path
    for stage in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']:
        marker=read_json(RUN/stage);assert marker['identity_hash']==identity
        for p,digest in marker['files'].items():assert sha256_file(RUN/p)==digest,p
    for name,digest in run['identity']['training_code_sha256'].items():assert sha256_file(OUT/name)==digest,name
    assert sha256_file(OUT/'evaluate_n_loss.py')==read_json(RUN/'EVALUATION_COMPLETE')['evaluation_code_sha256']
    saved=normalize(pd.read_parquet(RUN/'n_l2_oof_predictions.parquet'))
    oldraw=normalize(pd.read_parquet(HUBER/'base_predictions_cv.parquet'))
    np.testing.assert_array_equal(saved[KEYS].to_numpy(),meta[KEYS].to_numpy())
    rebuilt={};old_rebuilt={};model_checks=[];probe_count=0
    for h in HORIZONS:
        hh=HORIZON_HOURS[h];target=f'N_t_target_t{hh:02d}h';valid=meta.hour_idx.to_numpy()+hh<=215
        newpred=np.full(len(meta),np.nan);oldpred=newpred.copy()
        for f in range(5):
            receipt=read_json(RUN/f'models/outer{f}/{target}.json');spec=receipt['spec'];fit=receipt['fit']
            assert digest_object(spec)==receipt['model_id'] and spec['identity_hash']==identity
            assert spec['outer_fold']==f and spec['target']==target and spec['feature_names']==list(data.X_train)
            allowed=[q for q in range(5) if q!=f]
            assert spec['allowed_folds']==allowed and spec['params']==l2_params()
            assert spec['refit']==scoped_record(data,allowed,target)
            probes=pd.read_parquet(RUN/receipt['probe_path'])
            assert sha256_file(RUN/receipt['probe_path'])==receipt['probe_sha256']
            assert len(spec['probes'])==len(fit['curves'])==4
            best=[]
            for q,record,curve in zip(allowed,spec['probes'],fit['curves']):
                assert record['inner_fold']==curve['inner_fold']==q
                assert record['train']==scoped_record(data,[k for k in allowed if k!=q],target)
                assert record['validation']==scoped_record(data,[q],target)
                rec=probes.loc[probes.inner_fold==q].sort_values('row_index');idx=rec.row_index.to_numpy(int)
                required=np.flatnonzero(valid&(data.row_folds==q));np.testing.assert_array_equal(idx,required)
                np.testing.assert_array_equal(rec.fipsCode.to_numpy(),data.meta_train.iloc[idx].fipsCode.to_numpy())
                np.testing.assert_array_equal(rec.truth,data.component_targets[target].iloc[idx])
                np.testing.assert_array_equal(rec.metric_label,data.component_targets[target].iloc[idx].to_numpy().astype(np.float32).astype(float))
                iteration=int(np.argmin(curve['values'])+1);best.append(iteration)
                assert rec.best_iteration.eq(iteration).all()
                rmse=np.sqrt(np.mean((rec.prediction_at_best-rec.metric_label)**2))
                close(rmse,curve['best_value']);close(curve['values'][iteration-1],curve['best_value'])
                probe_count+=1
            assert best==fit['best_iterations'] and fit['requested_rounds']==max(1,int(np.mean(best)))
            assert fit['params']==l2_params() and fit['fits']==5
            assert sha256_file(RUN/receipt['model_path'])==receipt['model_sha256']
            model=lgb.Booster(model_file=str(RUN/receipt['model_path']))
            assert model.feature_name()==list(data.X_train) and model.current_iteration()==fit['trees_saved']
            assert model.params['objective']=='regression'
            hit=valid&(data.row_folds==f)
            newpred[hit]=model.predict(data.X_train.loc[hit],num_threads=1)
            assert saved.loc[hit,'source_'+target].eq(receipt['model_id']).all()
            oldreceipt=read_json(HUBER/f'models/outer{f}/{target}.json')
            assert oldreceipt['fit']['params']==make_lgbm_params(42)
            assert oldreceipt['spec']['scope']==allowed
            assert oldreceipt['fit']['refit']['counties']==spec['refit']['counties']
            assert sha256_file(HUBER/oldreceipt['model_path'])==oldreceipt['model_sha256']
            oldmodel=lgb.Booster(model_file=str(HUBER/oldreceipt['model_path']))
            assert oldmodel.feature_name()==list(data.X_train) and oldmodel.params['objective']=='huber'
            oldpred[hit]=oldmodel.predict(data.X_train.loc[hit],num_threads=1)
            close(oldpred[hit],oldraw.loc[hit,'raw_'+target])
            model_checks.append(dict(horizon=h,outer_fold=f,requested_rounds=fit['requested_rounds'],trees_saved=fit['trees_saved'],model_id=receipt['model_id']))
        close(newpred,saved['raw_'+target]);assert np.isnan(newpred[~valid]).all()
        rebuilt[h]=newpred;old_rebuilt[h]=oldpred
    reference=normalize(pd.read_parquet(REF/'component_predictions.parquet'))
    refcontrols=normalize(pd.read_parquet(REF/'control_predictions.parquet'))
    comp=normalize(pd.read_parquet(RUN/'component_predictions.parquet'))
    ctrl=normalize(pd.read_parquet(RUN/'control_predictions.parquet'))
    direct={h:refcontrols['raw_direct_'+h].to_numpy() for h in HORIZONS}
    metrics=pd.read_csv(RUN/'metrics/all_control_metrics.csv',float_precision='round_trip')
    boots=pd.read_csv(RUN/'metrics/bootstrap_intervals.csv',float_precision='round_trip')
    nmetrics=pd.read_csv(RUN/'metrics/N_component_metrics.csv',float_precision='round_trip')
    nbins=pd.read_csv(RUN/'metrics/N_bins.csv',float_precision='round_trip')
    results={};naligned={};count=0
    for case in CASES:
        pp={}
        for h in HORIZONS:
            hh=HORIZON_HOURS[h];pp[h]={}
            for c in COMPONENTS:
                tag=component_target_name(c,h);p=reference[f'pred_F2_{tag}'].to_numpy()
                changed=c=='N_t' and (case=='L2_all' or case==f'L2_N{hh:02d}')
                if changed:p=np.clip(rebuilt[h],0,1)
                pp[h][c]=p;close(p,comp[f'pred_{case}_{tag}'])
                src=saved['source_'+tag] if changed else reference['source_F2_'+tag]
                np.testing.assert_array_equal(src,comp[f'source_{case}_{tag}'])
        results[case],naligned[case]=independent_controls(meta,pp,direct)
        for h in HORIZONS:
            y=data.y_train[h].to_numpy()
            for mode,p in results[case][h].items():
                close(p,ctrl[f'pred_{case}_{mode}_{h}'])
                if case=='H0_Huber':close(p,refcontrols[f'pred_F2_{mode}_{h}'])
                row=metrics.loc[(metrics['case']==case)&(metrics.horizon==h)&(metrics['mode']==mode)].iloc[0]
                for k,v in score(y,p).items():close(row[k],v)
                count+=1
            if case!='H0_Huber':
                b=results['H0_Huber'][h]['v18_rule'];p=results[case][h]['v18_rule']
                row=boots.loc[(boots.domain=='OSI')&(boots['case']==case)&(boots.horizon==h)].iloc[0]
                close(paired_interval(data,h,y,b,p),[row.ci_low,row.ci_high])
            yn=data.component_targets[component_target_name('N_t',h)].to_numpy();s=meta.hour_idx.to_numpy()+HORIZON_HOURS[h]
            changed=case=='L2_all' or case==f'L2_N{HORIZON_HOURS[h]:02d}'
            npreds={'model_raw':rebuilt[h] if changed else old_rebuilt[h],
                'component_clipped':pp[h]['N_t'],'aligned':naligned[case][h][:,1]}
            base_n={'model_raw':old_rebuilt[h],'component_clipped':np.clip(old_rebuilt[h],0,1),'aligned':naligned['H0_Huber'][h][:,1]}
            for space,p in npreds.items():
                for group,hit in [('full',np.isfinite(yn)),('common_120_215',(s>=120)&(s<=215))]:
                    row=nmetrics.loc[(nmetrics['case']==case)&(nmetrics.horizon==h)&(nmetrics.space==space)&(nmetrics['group']==group)].iloc[0]
                    for k,v in score(yn[hit],p[hit]).items():close(row[k],v)
                if case=='L2_all':
                    row=boots.loc[(boots.domain=='N')&(boots.horizon==h)&(boots.space==space)].iloc[0]
                    close(paired_interval(data,h,yn,base_n[space],p),[row.ci_low,row.ci_high])
                masks={'N=0':yn==0,'0<N<=1e-4':(yn>0)&(yn<=1e-4),'1e-4<N<=1e-3':(yn>1e-4)&(yn<=1e-3),
                       '1e-3<N<=1e-2':(yn>1e-3)&(yn<=1e-2),'N>1e-2':yn>1e-2}
                for group,hit in masks.items():
                    row=nbins.loc[(nbins['case']==case)&(nbins.horizon==h)&(nbins.space==space)&(nbins['group']==group)].iloc[0]
                    for k,v in score(yn[hit],p[hit]).items():close(row[k],v)
    for path,digest in run['identity']['source_sha256'].items():assert sha256_file(path)==digest,path
    result=dict(status='PASS',identity_hash=identity,L2_models_reloaded=20,Huber_models_reloaded=20,
        inner_probes_checked=probe_count,all_control_metrics_checked=count,bootstrap_intervals_checked=len(boots),
        independent_pandas_alignment=True,full_horizon_masks_checked=True,all_component_sources_checked=True,
        training_performed_by_verifier=False,verifier_sha256=sha256_file(Path(__file__)),model_checks=model_checks)
    write_json(RUN/'logs/independent_verification.json',result)
    write_json(RUN/'VERIFIED_COMPLETE',dict(identity_hash=identity,verification_sha256=sha256_file(RUN/'logs/independent_verification.json'),
        verifier_sha256=sha256_file(Path(__file__))))
    print('PASS: 20 L2 + 20 Huber models reloaded; 80 probes, all predictions, metrics and intervals independently checked')


if __name__=='__main__':verify()
