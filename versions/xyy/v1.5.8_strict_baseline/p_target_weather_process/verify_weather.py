"""Independent feature/checkpoint replay and target-keyed metric verification."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
import lightgbm as lgb
from verify_features import rebuild,verify_bundle,COLS

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=OUT.parents[3];RUN=OUT/'runs/v18_pweather_v1_split42_model42'
TREE=BASE/'p_neighbor_horizon_increment/runs/v18_pneighbor_v1_split42_model42';F2=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42'
HS=[1,6,24,48];CS=['P_t','N_t','D_t','R_t'];WEIGHTS=np.array([.4,.35,.25,-.1])
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ah(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def norm(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.zfill(5);d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    return d.sort_values(['fipsCode','hour_idx']).reset_index(drop=True)
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)
def post(a):
    a=np.clip(a,0,.65);return np.where(a<.001,0,a)
def metric(y,p):
    mask=np.isfinite(y);e=np.asarray(p)[mask]-np.asarray(y)[mask];assert np.isfinite(e).all();n=len(e)
    s=math.fsum(float(x)*float(x) for x in e)
    return dict(n=n,sse=s,rmse=math.sqrt(s/n) if n else np.nan,mae=math.fsum(abs(float(x)) for x in e)/n if n else np.nan,bias=math.fsum(map(float,e))/n if n else np.nan)
def difference(y,b,p):
    a,c=metric(y,b),metric(y,p)
    return dict(**c,baseline_rmse=a['rmse'],rmse_delta=c['rmse']-a['rmse'],rmse_change_pct=100*(c['rmse']/a['rmse']-1) if a['rmse']>0 else np.nan,sse_reduction=a['sse']-c['sse'])

def verify():
    source=read(OUT/'reference/source_hashes.json');manifest=read(RUN/'run_manifest.json');identity=manifest['identity_hash'];cfg=read(OUT/'experiment_config.json')
    assert digest(manifest['identity'])==identity
    for p,h in source.items():assert sha(p)==h,p
    for p,h in manifest['identity']['training_code_sha256'].items():assert sha(OUT/p)==h,p
    assert manifest['identity']['config_sha256']==sha(OUT/'experiment_config.json')
    for marker in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']:
        rec=read(RUN/marker);assert rec['identity_hash']==identity
        for p,h in rec['files'].items():assert sha(RUN/p)==h,p
    assert read(RUN/'EVALUATION_COMPLETE')['evaluation_code_sha256']==sha(OUT/'evaluate_weather.py')
    feature_check=verify_bundle();X=rebuild('train')
    meta=norm(pd.read_parquet(ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet'))
    pd.testing.assert_frame_equal(meta[['fipsCode','timestamp_et','hour_idx']],pd.read_parquet(OUT/'features/train_meta.parquet'),check_dtype=False)
    cv=csv(ROOT/'cv/cv_assignments_balanced_v1_seed42.csv',dtype={'fipsCode':str}).set_index('fipsCode').fold
    meta['fold']=meta.fipsCode.map(cv);fold=meta.fold.to_numpy();hours=meta.hour_idx.to_numpy();fips=meta.fipsCode.to_numpy();n=len(meta)
    assert n==34416 and X.shape==(n,191)
    raw=csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str});raw['target_hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
    raw=raw.set_index(['fipsCode','target_hour']).sort_index();y=raw.P_t.reindex(pd.MultiIndex.from_arrays([fips,hours+24])).to_numpy()
    valid=hours+24<=215;assert valid.sum()==28680 and np.array_equal(np.isfinite(y),valid)
    saved_new=norm(pd.read_parquet(RUN/'P24_oof_predictions.parquet'));close(saved_new.true_P24,y)
    records=read(RUN/'model_manifest.json');assert len(records)==25;inner_count=outer_count=0;max_prediction_diff=0.;new_raw=np.full(n,np.nan)
    def check_desc(desc):
        ids=np.flatnonzero(valid&np.isin(fold,desc['folds']));labels=y[ids]
        assert desc['rows']==len(ids) and desc['counties']==sorted(set(fips[ids]))
        # Training X was verified exact to the independent raw-keyed reconstruction.
        for key,a in [('row_sha256',ids),('feature_sha256',X.iloc[ids].to_numpy()),('label_sha256',labels),('metric_label_sha256',labels.astype(np.float32).astype(float))]:
            assert desc[key]==ah(a),key
        return ids,labels
    for rec in records:
        spec=rec['spec'];outer=spec['outer_fold'];inner=spec['inner_fold'];excluded={outer} if inner is None else {outer,inner}
        assert rec['model_id']==digest(spec) and spec['identity_hash']==identity and spec['params']==cfg['params'] and spec['feature_names']==list(X)
        assert set(spec['train']['folds'])==set(range(5))-excluded
        tr,ty=check_desc(spec['train']);assert not set(fold[tr])&excluded
        path=RUN/rec['model_path'];assert sha(path)==rec['model_sha256'];model=lgb.Booster(model_file=str(path));assert model.feature_name()==list(X)
        assert model.current_iteration()==rec['actual_trees']
        if inner is not None:
            inner_count+=1;assert spec['validation']['folds']==[inner]
            va,vy=check_desc(spec['validation']);assert outer not in set(fold[va]) and not set(tr)&set(va)
            predicted=model.predict(X.iloc[va],num_threads=1);stored=pd.read_parquet(RUN/rec['validation_path'])
            assert sha(RUN/rec['validation_path'])==rec['validation_sha256'];np.testing.assert_array_equal(stored.row_index,va)
            close(stored.official_P,vy);np.testing.assert_array_equal(stored.metric_P,vy.astype(np.float32).astype(float));close(stored.prediction,predicted)
            max_prediction_diff=max(max_prediction_diff,float(np.max(abs(stored.prediction.to_numpy()-predicted))))
            curve=np.array(rec['curve']);best=rec['best_iteration'];assert 1<=best<=2000 and best==int(np.argmin(curve))+1
            close(curve[best-1],metric(vy.astype(np.float32).astype(float),predicted)['rmse'])
        else:
            outer_count+=1;assert spec['validation'] is None
            inner_records=[r for r in records if r['spec']['outer_fold']==outer and r['spec']['inner_fold'] is not None];assert len(inner_records)==4
            rounds=max(1,sum(r['best_iteration'] for r in inner_records)//4);assert spec['requested_rounds']==rounds and 1<=rec['actual_trees']<=rounds
            ids=np.flatnonzero(valid&(fold==outer));predicted=model.predict(X.iloc[ids],num_threads=1);close(saved_new.loc[ids,'W1_raw_P24'],predicted)
            max_prediction_diff=max(max_prediction_diff,float(np.max(abs(saved_new.loc[ids,'W1_raw_P24'].to_numpy()-predicted))))
            new_raw[ids]=predicted;assert (saved_new.loc[ids,'model_id']==rec['model_id']).all()
    assert (inner_count,outer_count)==(20,5);close(np.clip(new_raw,0,1),saved_new.W1_P24)
    print('25 models replayed with independently reconstructed 191 columns',flush=True)
    reference=norm(pd.read_parquet(TREE/'component_predictions.parquet'));old_final=norm(pd.read_parquet(TREE/'control_predictions.parquet'))
    direct=norm(pd.read_parquet(F2/'control_predictions.parquet'));stored=norm(pd.read_parquet(RUN/'control_predictions.parquet'));stored_comp=norm(pd.read_parquet(RUN/'component_predictions.parquet'))
    for outer in range(5):
        rec=read(TREE/f'models/outer{outer}/P_direct_t24h.json');assert sha(TREE/rec['model_path'])==rec['model_sha256']
        model=lgb.Booster(model_file=str(TREE/rec['model_path']));hit=valid&(fold==outer)
        predicted=np.clip(model.predict(X.iloc[np.flatnonzero(hit),:183],num_threads=1),0,1)
        np.testing.assert_array_equal(predicted,reference.loc[hit,'pred_NB_P24_P_t_target_t24h'])
    values={};parts={};aligned={};ys={};pys={}
    for case in ['W0','W1']:
        parts[case]={};frames=[]
        for h in HS:
            pp=np.column_stack([reference[f'pred_NB_P24_{c}_target_t{h:02d}h'].to_numpy() for c in CS])
            if case=='W1' and h==24:pp[:,0]=np.clip(new_raw,0,1)
            for i,c in enumerate(CS):
                column=f'pred_{case}_{c}_target_t{h:02d}h';close(stored_comp[column],pp[:,i])
                if case=='W0' or (h,c)!=(24,'P_t'):
                    np.testing.assert_array_equal(stored_comp[column],reference[f'pred_NB_P24_{c}_target_t{h:02d}h'])
                    np.testing.assert_array_equal(stored_comp[f'source_{case}_{c}_target_t{h:02d}h'],reference[f'source_NB_P24_{c}_target_t{h:02d}h'])
                else:np.testing.assert_array_equal(stored_comp[f'source_{case}_{c}_target_t{h:02d}h'],saved_new.model_id)
            clipped=np.clip(pp,0,1);parts[case][h]=clipped;hit=hours+h<=215
            frame=pd.DataFrame(clipped[hit],columns=CS);frame['fipsCode']=fips[hit];frame['target_hour']=hours[hit]+h;frames.append(frame)
        unique=pd.concat(frames,ignore_index=True).groupby(['fipsCode','target_hour'])[CS].mean();values[case]={};aligned[case]={}
        for h in HS:
            idx=pd.MultiIndex.from_arrays([fips,hours+h]);hit=hours+h<=215
            truth=raw.osi.reindex(idx).to_numpy();ys[h]=truth;pys[h]=raw.P_t.reindex(idx).to_numpy();assert np.array_equal(np.isfinite(truth),hit)
            np.testing.assert_array_equal(stored[f'true_osi_target_t{h:02d}h'],truth)
            al=unique.reindex(idx).to_numpy();aligned[case][h]=al
            c1=np.maximum(parts[case][h]@WEIGHTS,0);c3=np.maximum(al@WEIGHTS,0);dr=direct[f'raw_direct_osi_target_t{h:02d}h'].to_numpy()
            results=dict(C0_direct_osi=post(dr),C1_component_osi=post(c1),C2_equal_blend=post((dr+c1)/2),C3_aligned_component=post(c3))
            results['v18_rule']=results['C3_aligned_component' if h<=6 else 'C1_component_osi'].copy()
            for mode,predicted in results.items():predicted[~hit]=np.nan;close(stored[f'pred_{case}_{mode}_osi_target_t{h:02d}h'],predicted)
            values[case][h]=results
            invariant=hit&((hours+h<96)|(h==48))
            np.testing.assert_array_equal(stored.loc[invariant,f'pred_{case}_v18_rule_osi_target_t{h:02d}h'],old_final.loc[invariant,f'pred_NB_P24_v18_rule_osi_target_t{h:02d}h'])
    member=pd.read_parquet(OUT/'reference/test_like_membership.parquet').set_index(['horizon','target_day','fipsCode']).test_like
    def group_mask(h,name):
        s=hours+h;ix=pd.MultiIndex.from_arrays([np.full(n,h),s//24,fips]);near=member.reindex(ix,fill_value=False).to_numpy(bool)
        return dict(all=np.ones(n,bool),pre96=s<96,from96=s>=96,test_like=near,not_test_like=~near,all_except_Forest=fips!='42053',test_like_except_Forest=near&(fips!='42053'))[name]
    metric_count=0
    for name in ['primary_scores','all_controls','comparisons','groups','county_metrics','fold_metrics','day_metrics','bootstrap','P_metrics']:
        table=csv(RUN/f'metrics/{name}.csv',dtype={'fipsCode':str} if name=='county_metrics' else None)
        for r in table.itertuples():
            h=r.horizon;truth=ys[h].copy();b=values['W0'][h]['v18_rule'];p=values['W1'][h]['v18_rule']
            if name in ['primary_scores','all_controls']:calc=metric(truth,values[r.case][h][r.mode if name=='all_controls' else 'v18_rule'])
            else:
                if name in ['groups','bootstrap']:truth[~group_mask(h,r.group)]=np.nan
                elif name=='county_metrics':truth[fips!=r.fipsCode]=np.nan
                elif name=='fold_metrics':truth[fold!=r.fold]=np.nan
                elif name=='day_metrics':truth[(hours+h)//24!=r.target_day]=np.nan
                elif name=='P_metrics':
                    truth=pys[h];space=parts if r.space=='source' else aligned;b=space['W0'][h][:,0];p=space['W1'][h][:,0]
                calc=difference(truth,b,p)
            for k,v in calc.items():close(getattr(r,k),v)
            metric_count+=1
    process=csv(RUN/'metrics/process_groups_24h.csv');has=X.has_high_gust_48h.to_numpy();age=X.hours_since_last_high_gust.to_numpy();segments=X.high_gust_episode_count_48h.to_numpy()
    masks=dict(no_high=has==0,active_high=(has==1)&(age==0),post_high_1_6=(has==1)&(age>=1)&(age<=6),post_high_7plus=(has==1)&(age>=7),multiple_segments=segments>=2)
    assert np.array_equal(sum(masks[k].astype(int) for k in ['no_high','active_high','post_high_1_6','post_high_7plus']),valid.astype(int))
    for r in process.itertuples():
        hit=masks[r.group]&valid;yy=np.where(hit,ys[24],np.nan);calc=difference(yy,values['W0'][24]['v18_rule'],values['W1'][24]['v18_rule'])
        for k,v in calc.items():close(getattr(r,k),v)
        close([r.truth_mean,r.W0_mean,r.W1_mean],[ys[24][hit].mean(),values['W0'][24]['v18_rule'][hit].mean(),values['W1'][24]['v18_rule'][hit].mean()])
    boots=csv(RUN/'metrics/bootstrap.csv');draw=np.random.default_rng(20260910).integers(0,239,(2000,239));multiplicity=np.stack([np.bincount(d,minlength=239) for d in draw]);codes=pd.Categorical(fips,categories=sorted(set(fips))).codes
    for r in boots.itertuples():
        h=r.horizon;hit=np.isfinite(ys[h])&group_mask(h,r.group);b=values['W0'][h]['v18_rule'];p=values['W1'][h]['v18_rule'];truth=ys[h]
        if np.array_equal(b[hit],p[hit]):close([r.ci_low,r.ci_high],[0,0]);continue
        count=np.bincount(codes[hit],minlength=239);ss0=np.bincount(codes[hit],weights=(b[hit]-truth[hit])**2,minlength=239);ss1=np.bincount(codes[hit],weights=(p[hit]-truth[hit])**2,minlength=239)
        delta=np.sqrt((multiplicity@ss1)/(multiplicity@count))-np.sqrt((multiplicity@ss0)/(multiplicity@count));close([r.ci_low,r.ci_high],np.quantile(delta,[.025,.975]))
    changes=pd.read_parquet(RUN/'row_changes.parquet')
    for h,r in changes.groupby('horizon'):
        hit=np.isfinite(ys[h]);assert len(r)==hit.sum();np.testing.assert_array_equal(r.fipsCode,fips[hit]);close(r.truth,ys[h][hit])
        b=values['W0'][h]['v18_rule'][hit];p=values['W1'][h]['v18_rule'][hit];close(r.W0,b);close(r.W1,p);close(r.sse_reduction,(b-ys[h][hit])**2-(p-ys[h][hit])**2)
    daily=csv(RUN/'metrics/county_day_24h.csv',dtype={'fipsCode':str})
    for r in daily.itertuples():
        hit=(fips==r.fipsCode)&valid&((hours+24)//24==r.target_day)
        for name,v in [('true_P',pys[24]),('W0_P',parts['W0'][24][:,0]),('W1_P',parts['W1'][24][:,0]),('true_OSI',ys[24]),('W0_OSI',values['W0'][24]['v18_rule']),('W1_OSI',values['W1'][24]['v18_rule'])]:
            a=v[hit];close(getattr(r,name+'_mean'),a.mean());close(getattr(r,name+'_max'),a.max());assert getattr(r,name+'_peak_hour')==(hours[hit]+24)[np.argmax(a)]
        calc=difference(ys[24][hit],values['W0'][24]['v18_rule'][hit],values['W1'][24]['v18_rule'][hit])
        for k,v in calc.items():close(getattr(r,k),v)
    decomposition=csv(RUN/'metrics/P_change_decomposition.csv')
    close(decomposition.P_squared_reduction+decomposition.P_other_cross_reduction+decomposition.postprocess_precision_remainder,decomposition.OSI_sse_reduction)
    membership_raw=pd.read_parquet(BASE/'train_test_feature_support/data/nearest_five.parquet');membership_raw=membership_raw[(membership_raw.scope=='test_to_train')&(membership_raw.view=='context')]
    sets=membership_raw.groupby(['horizon','target_day']).matched_fips.agg(set)
    for key,r in member.reset_index().groupby(['horizon','target_day']):assert set(r.loc[r.test_like,'fipsCode'])==sets.loc[key]
    for p,h in source.items():assert sha(p)==h,p
    result=dict(status='PASS',identity_hash=identity,independently_rebuilt_features=feature_check,new_inner_models_reloaded=20,new_outer_models_reloaded=5,
        reference_P24_models_reloaded=5,max_checkpoint_prediction_difference=max_prediction_diff,metric_rows_checked=metric_count,process_group_rows_checked=len(process),
        county_day_rows_checked=len(daily),bootstrap_rows_checked=len(boots),row_changes_checked=len(changes),unmodified_components_exact=True,
        final48_and_short_pre96_exact=True,independent_target_keyed_C3=True,aux_membership_exact=True,source_files_unchanged=len(source),absolute_tolerance=1e-12,verifier_sha256=sha(__file__))
    (RUN/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':verify()
