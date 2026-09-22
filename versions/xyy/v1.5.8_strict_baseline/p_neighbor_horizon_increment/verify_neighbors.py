"""Independent feature reconstruction, model replay, scope and scoring verification."""
import importlib.util
import lightgbm as lgb
from neighbor_protocol import *
from independent_features import verify_features


def close(a,b):np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=1e-12,rtol=0,equal_nan=True)


def independent_scope(data,X,p,folds,h,kind):
    eligible=np.isin(data.row_folds,folds)&(data.meta_train.hour_idx.to_numpy()+HORIZON_HOURS[h]<=215)
    hit=np.ones(len(p),bool) if kind=='direct' else (p>0 if kind=='a' else p<1)
    idx=np.flatnonzero(eligible&hit);y=data.component_targets[component_target_name('P_t',h)].iloc[idx].to_numpy(float)
    pp=p[idx]
    if kind=='direct':label=y;weight=np.ones(len(idx))
    elif kind=='a':label=np.where(y<pp,y/pp,1.);weight=np.square(pp)
    else:label=np.where(y>pp,(y-pp)/(1-pp),0.);weight=np.square(1-pp)
    meta=data.meta_train.iloc[idx]
    rec=dict(folds=sorted(folds),counties=sorted(meta.fipsCode.unique()),rows=len(idx),total_valid=int(eligible.sum()),
        row_sha=array_sha(idx),feature_sha=array_sha(X.iloc[idx].to_numpy()),label_sha=array_sha(label),p_sha=array_sha(pp),
        weight_sha=array_sha(weight),weight_mean=float(weight.mean()),normalized_weight_sha=array_sha(weight/weight.mean()))
    return rec,idx,label,weight


def independent_math():
    sys.path.insert(0,str(PTRANSFER))
    spec=importlib.util.spec_from_file_location('independent_neighbor_scoring',PTRANSFER/'verify_transfer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify():
    data=load_data();meta=data.meta_train.copy();meta['fold']=data.row_folds;p=data.X_train.last_P_t.to_numpy()
    manifest=read_json(RUN/'run_manifest.json');identity=manifest['identity_hash'];assert digest_object(manifest['identity'])==identity
    for path,digest in manifest['identity']['source_sha256'].items():assert sha256_file(path)==digest,path
    for name,digest in manifest['identity']['training_code_sha256'].items():assert sha256_file(OUT/name)==digest,name
    for stage in ('MODEL_CV_COMPLETE','EVALUATION_COMPLETE'):
        marker=read_json(RUN/stage);assert marker['identity_hash']==identity
        for path,digest in marker['files'].items():assert sha256_file(RUN/path)==digest,path
    assert sha256_file(OUT/'evaluate_neighbors.py')==read_json(RUN/'EVALUATION_COMPLETE')['evaluation_code_sha256']
    fm=read_json(FEATURES/'feature_manifest.json');assert digest_object(fm['identity'])==fm['identity_hash']==manifest['identity']['feature_identity']
    assert sha256_file(FEATURES/'feature_manifest.json')==manifest['identity']['feature_manifest_sha256']
    for name,digest in fm['files'].items():assert sha256_file(FEATURES/name)==digest,name
    feature_report,X=verify_features(data,causality=False)
    branch=pd.read_parquet(RUN/'new_P_oof_predictions.parquet');byh={h:normalize(branch[branch.horizon==h]) for h in CHANGED}
    registry={r['source_id']:r['definition'] for r in read_json(RUN/'source_registry.json')}
    rebuilt={};models=[];nprobe=0
    for h in CHANGED:
        hh=HORIZON_HOURS[h];valid=meta.hour_idx.to_numpy()+hh<=215;kinds=['a','b'] if hh==6 else ['direct']
        newx=X[hh,'train'];d=byh[h]
        pd.testing.assert_frame_equal(d[KEYS],meta[KEYS],check_dtype=False);close(d.P71,p)
        assert d.candidate_identity_hash.eq(identity).all();np.testing.assert_array_equal(d.scoreable,valid)
        pred={k:np.full(len(meta),np.nan) for k in kinds}
        for outer in range(5):
            allowed=[f for f in range(5) if f!=outer];pair={}
            for kind in kinds:
                r=read_json(RUN/f'models/outer{outer}/P_{kind}_t{hh:02d}h.json');s=r['spec'];fit=r['fit']
                assert digest_object(s)==r['model_id'] and s['identity_hash']==identity
                assert s['outer_fold']==outer and s['horizon']==h and s['kind']==kind and s['allowed_folds']==allowed
                assert s['feature_names']==list(newx) and len(newx.columns)==183
                assert s['params']==fit['params']==params(kind)
                record,_,_,_=independent_scope(data,newx,p,allowed,h,kind);assert s['refit']==record
                assert fit['refit_weight_mean']==record['weight_mean']
                probe=pd.read_parquet(RUN/r['probe_path']);assert sha256_file(RUN/r['probe_path'])==r['probe_sha256']
                best=[]
                for q,record,curve in zip(allowed,s['probes'],fit['curves']):
                    tr,_,_,_=independent_scope(data,newx,p,[f for f in allowed if f!=q],h,kind)
                    va,idx,y,w=independent_scope(data,newx,p,[q],h,kind)
                    assert record['inner_fold']==curve['inner_fold']==q and record['train']==tr and record['validation']==va
                    held=set(meta.loc[meta.fold==outer,'fipsCode']);assert not held.intersection(tr['counties']+va['counties'])
                    z=probe[probe.inner_fold==q].sort_values('row_index')
                    np.testing.assert_array_equal(z.row_index,idx);np.testing.assert_array_equal(z.label,y);np.testing.assert_array_equal(z.raw_weight,w)
                    close(z.normalized_weight,w/tr['weight_mean']);assert z.training_weight_mean.eq(tr['weight_mean']).all()
                    assert z.all_validation_rows.eq(va['total_valid']).all()
                    yl=y.astype(np.float32).astype(float) if kind=='direct' else y
                    np.testing.assert_array_equal(z.metric_label,yl)
                    err=z.prediction_at_best.to_numpy()-yl
                    metric=float(np.sqrt(np.mean(err**2))) if kind=='direct' else float(np.sqrt(np.dot(w,err**2)/va['total_valid']))
                    iteration=int(np.argmin(curve['values'])+1);assert z.best_iteration.eq(iteration).all()
                    close(curve['best_value'],metric);close(curve['values'][iteration-1],metric)
                    best.append(iteration);nprobe+=1
                assert best==fit['best_iterations'] and fit['requested_rounds']==max(1,int(np.mean(best))) and fit['fits']==5
                assert sha256_file(RUN/r['model_path'])==r['model_sha256']
                model=lgb.Booster(model_file=str(RUN/r['model_path']))
                assert model.feature_name()==list(newx) and model.current_iteration()==fit['trees_saved']
                assert model.params['objective']==('huber' if kind=='direct' else 'regression')
                active=valid&(meta.fold.to_numpy()==outer)&(np.ones(len(meta),bool) if kind=='direct' else (p>0 if kind=='a' else p<1))
                pred[kind][active]=model.predict(newx.loc[active],num_threads=1);assert d.loc[active,'source_'+kind].eq(r['model_id']).all()
                pair[kind]=r['model_id'];models.append(dict(horizon=h,kind=kind,outer_fold=outer,model_id=r['model_id']))
            ids=d.loc[valid&(meta.fold.to_numpy()==outer),'P_source_id'].unique();assert len(ids)==1
            definition=registry[ids[0]]
            assert definition['models']==pair and definition['horizon']==h and definition['outer_fold']==outer
            assert definition['feature_identity']==fm['identity_hash'] and definition['p71_sha']==array_sha(p)
            assert ids[0]==(digest_object(definition) if hh==6 else pair['direct'])
        for kind in kinds:close(pred[kind],d['raw_'+kind])
        if hh==6:
            applied={}
            for kind in kinds:
                active=valid&(p>0 if kind=='a' else p<1);values=np.where(valid,0.,np.nan)
                values[active]=np.minimum(1,np.maximum(0,pred[kind][active]));applied[kind]=values
                assert d.loc[~active,'source_'+kind].eq('').all();close(values,d['applied_'+kind])
            ca=p*applied['a'];cb=(1-p)*applied['b'];out=np.clip(ca+cb,0,1)
            close(ca,d.contribution_a);close(cb,d.contribution_b)
        else:out=np.clip(pred['direct'],0,1)
        close(out,d.pred_P);assert np.isnan(out[~valid]).all() and np.isfinite(out[valid]).all()
        close(d.true_P,data.component_targets[component_target_name('P_t',h)]);rebuilt[h]=out
    calc=independent_math()
    old=normalize(pd.read_parquet(CURRENT/'component_predictions.parquet'));oldctl=normalize(pd.read_parquet(CURRENT/'control_predictions.parquet'))
    rawdir=normalize(pd.read_parquet(F2/'control_predictions.parquet'));direct={h:rawdir['raw_direct_'+h].to_numpy() for h in HORIZONS}
    comp=normalize(pd.read_parquet(RUN/'component_predictions.parquet'));saved=normalize(pd.read_parquet(RUN/'control_predictions.parquet'))
    allm=pd.read_csv(RUN/'metrics/all_control_metrics.csv',float_precision='round_trip')
    primary=pd.read_csv(RUN/'metrics/primary_scores.csv',float_precision='round_trip')
    pm=pd.read_csv(RUN/'metrics/P_component_metrics.csv',float_precision='round_trip')
    bins=pd.read_csv(RUN/'metrics/P71_bins.csv',float_precision='round_trip')
    boot=pd.read_csv(RUN/'metrics/bootstrap_intervals.csv',float_precision='round_trip');ctls={};aligned={};count=0
    for case,changed in CASES.items():
        parts={}
        for h in HORIZONS:
            parts[h]={}
            for c in COMPONENTS:
                tag=component_target_name(c,h);v=old[f'pred_AB_P06_{tag}'].to_numpy()
                if h==changed and c=='P_t':v=rebuilt[h]
                parts[h][c]=v;close(v,comp[f'pred_{case}_{tag}'])
                src=byh[h].P_source_id if h==changed and c=='P_t' else old['source_AB_P06_'+tag]
                np.testing.assert_array_equal(src,comp[f'source_{case}_{tag}'])
        ctls[case],aligned[case]=calc.independent_controls(meta,parts,direct)
        for h in HORIZONS:
            y=data.y_train[h].to_numpy();ss=meta.hour_idx.to_numpy()+HORIZON_HOURS[h]
            for mode,pred in ctls[case][h].items():
                close(pred,saved[f'pred_{case}_{mode}_{h}']);base=ctls['B0'][h][mode]
                if case=='B0':close(pred,oldctl[f'pred_AB_P06_{mode}_{h}'])
                row=allm[(allm['case']==case)&(allm.horizon==h)&(allm['mode']==mode)].iloc[0]
                for k,v in calc.comparison(y,base,pred).items():close(v,row[k])
                count+=1
            pred=ctls[case][h]['v18_rule'];base=ctls['B0'][h]['v18_rule']
            row=primary[(primary['case']==case)&(primary.horizon==h)].iloc[0]
            for k,v in calc.comparison(y,base,pred).items():close(v,row[k])
            if changed:
                row=boot[(boot['case']==case)&(boot.horizon==h)&(boot.domain=='OSI')].iloc[0]
                close(calc.independent_interval(y,base,pred),[row.ci_low,row.ci_high])
                unaffected=(ss<72+HORIZON_HOURS[changed])&np.isfinite(y);np.testing.assert_array_equal(pred[unaffected],base[unaffected])
                if HORIZON_HOURS[h]>=24 and h!=changed:np.testing.assert_array_equal(pred,base)
            yp=data.component_targets[component_target_name('P_t',h)].to_numpy()
            for space,v,b in [('component_clipped',parts[h]['P_t'],old[f'pred_AB_P06_{component_target_name("P_t",h)}'].to_numpy()),
                ('aligned',aligned[case][h][:,0],aligned['B0'][h][:,0])]:
                for group,mask in [('full',np.isfinite(yp)),('common_120_215',(ss>=120)&(ss<=215))]:
                    row=pm[(pm['case']==case)&(pm.horizon==h)&(pm.space==space)&(pm['group']==group)].iloc[0]
                    for k,val in calc.comparison(yp[mask],b[mask],v[mask]).items():close(val,row[k])
                masks={'p=0':p==0,'0<p<.01':(p>0)&(p<.01),'.01<=p<.1':(p>=.01)&(p<.1),'.1<=p<.5':(p>=.1)&(p<.5),'.5<=p<=1':p>=.5}
                for group,mask in masks.items():
                    row=bins[(bins['case']==case)&(bins.horizon==h)&(bins.space==space)&(bins.P71_group==group)].iloc[0]
                    for k,val in calc.comparison(yp[mask],b[mask],v[mask]).items():close(val,row[k])
                if h==changed:
                    row=boot[(boot['case']==case)&(boot.horizon==h)&(boot.domain=='P')&(boot.space==space)].iloc[0]
                    close(calc.independent_interval(yp,b,v),[row.ci_low,row.ci_high])
    for name in ('county_metrics','fold_metrics','window_metrics'):
        df=pd.read_csv(RUN/f'metrics/{name}.csv',dtype={'fipsCode':str},float_precision='round_trip')
        for row in df.itertuples(index=False):
            h=row.horizon;s=meta.hour_idx.to_numpy()+HORIZON_HOURS[h]
            if name=='county_metrics':mask=meta.fipsCode.to_numpy()==row.fipsCode
            elif name=='fold_metrics':mask=meta.fold.to_numpy()==row.fold
            else:mask=(s>=row.start)&(s<=row.end)
            y=data.y_train[h].to_numpy();b=ctls['B0'][h]['v18_rule'];v=ctls[row.case][h]['v18_rule']
            for k,val in calc.comparison(y[mask],b[mask],v[mask]).items():close(val,getattr(row,k))
    for path,digest in manifest['identity']['source_sha256'].items():assert sha256_file(path)==digest,path
    report=dict(status='PASS',identity_hash=identity,feature_verification=feature_report,models_reloaded=len(models),
        probe_count=nprobe,control_metrics_checked=count,bootstrap_intervals_checked=len(boot),
        independent_feature_arrays_used_for_model_reload=True,independent_math_source=str(PTRANSFER/'verify_transfer.py'),
        unchanged_sources_verified=True,all_county_fold_window_metrics_verified=True,training_performed=False,
        verifier_sha256=sha256_file(Path(__file__)),models=models)
    write_json(RUN/'logs/independent_verification.json',report)
    write_json(RUN/'VERIFIED_COMPLETE',dict(identity_hash=identity,verifier_sha256=sha256_file(Path(__file__)),
        verification_sha256=sha256_file(RUN/'logs/independent_verification.json')))
    print('PASS: exact features, 20 reloaded models, 80 probes, 80 control scores, 18 intervals and all county/fold/window tables',flush=True)


if __name__=='__main__':verify()
