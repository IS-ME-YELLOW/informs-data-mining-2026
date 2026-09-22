"""Independent original-feature/raw-label checkpoint replay and target-keyed C3 audit."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
import lightgbm as lgb
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=OUT.parents[3]
RUN=OUT/'runs/v18_nr2stage_v1_split42_model42';TREE=BASE/'p_neighbor_horizon_increment/runs/v18_pneighbor_v1_split42_model42'
F2=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42';OLD=BASE/'runs/v18_tree_nested_v2_split42_model42'
HS=[1,6,24,48];CS=['P_t','N_t','D_t','R_t'];CASES=['B0','N_only','R_only','NR_both'];KEYS=['fipsCode','timestamp_et','hour_idx','fold']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ah(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def norm(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.zfill(5);d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    assert not d.duplicated(['fipsCode','hour_idx']).any();return d.sort_values(['fipsCode','hour_idx']).reset_index(drop=True)
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)
def post(a):
    a=np.clip(a,0,.65);return np.where(a<.001,0,a)
def metric(y,p):
    y=np.asarray(y);hit=np.isfinite(y);e=np.asarray(p)[hit]-y[hit];assert np.isfinite(e).all();n=len(e)
    s=math.fsum(float(x)*float(x) for x in e)
    return dict(n=n,sse=s,rmse=math.sqrt(s/n) if n else np.nan,mae=math.fsum(abs(float(x)) for x in e)/n if n else np.nan,bias=math.fsum(map(float,e))/n if n else np.nan)
def difference(y,b,p):
    a,c=metric(y,b),metric(y,p)
    return dict(**c,baseline_rmse=a['rmse'],rmse_delta=c['rmse']-a['rmse'],rmse_change_pct=100*(c['rmse']/a['rmse']-1) if a['rmse']>0 else np.nan,sse_reduction=a['sse']-c['sse'])
def cmask(y,g):return dict(all=np.ones(len(y),bool),zero=y==0,small_positive=(y>0)&(y<=.01),tail_gt_001=y>.01,positive=y>0)[g]
def binary_loss(y,q):
    z=(np.asarray(y)>0).astype(float);p=np.clip(q,1e-15,1-1e-15)
    return float(-np.mean(z*np.log(p)+(1-z)*np.log1p(-p)))
def verify():
    sources=read(OUT/'reference/source_hashes.json');manifest=read(RUN/'run_manifest.json');identity=manifest['identity_hash'];cfg=read(OUT/'experiment_config.json')
    assert digest(manifest['identity'])==identity and manifest['identity']['source_hashes']==sources
    for p,h in sources.items():assert sha(p)==h,p
    for p,h in manifest['identity']['training_code_sha256'].items():assert sha(OUT/p)==h,p
    assert manifest['identity']['config_sha256']==sha(OUT/'experiment_config.json')
    assert cfg['protocol_sha256']==sha(OUT/'Experiment_Protocol_2026-09-21.md')
    for marker in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']:
        rec=read(RUN/marker);assert rec['identity_hash']==identity
        for p,h in rec['files'].items():assert sha(RUN/p)==h,p
    assert read(RUN/'EVALUATION_COMPLETE')['evaluation_code_sha256']==sha(OUT/'evaluate_nr.py')
    execution=read(RUN/'execution.json');assert execution['status']=='complete' and execution['fits_started']==execution['fits_completed']==100
    X=pd.read_parquet(ROOT/'versions/xyy/v1.5.6/features_train_v1.5.6.parquet');meta=norm(pd.read_parquet(ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet'))
    assert list(X)==manifest['identity']['feature_columns'] and X.shape==(34416,163)
    cv=csv(ROOT/'cv/cv_assignments_balanced_v1_seed42.csv',dtype={'fipsCode':str}).set_index('fipsCode').fold
    meta['fold']=meta.fipsCode.map(cv);fold=meta.fold.to_numpy();hours=meta.hour_idx.to_numpy();fips=meta.fipsCode.to_numpy();n=len(meta)
    raw=pd.read_csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str});raw['target_hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
    raw=raw.set_index(['fipsCode','target_hour']).sort_index();ys={};cys={}
    for h in HS:
        idx=pd.MultiIndex.from_arrays([fips,hours+h]);ys[h]=raw.osi.reindex(idx).to_numpy();cys[h]={c:raw[c].reindex(idx).to_numpy() for c in CS}
        assert np.array_equal(np.isfinite(ys[h]),hours+h<=215)
    valid=hours+1<=215;assert valid.sum()==34177
    oof=norm(pd.read_parquet(RUN/'NR1_oof_predictions.parquet'));pd.testing.assert_frame_equal(oof[KEYS],meta[KEYS],check_dtype=False)
    records=read(RUN/'model_manifest.json');assert len(records)==100;by_id={r['model_id']:r for r in records};assert len(by_id)==100
    replay={c:{head:np.full(n,np.nan) for head in ['q','m']} for c in ['N_t','R_t']};inner_count=outer_count=0;maxdiff=0.;dep_checks=0
    def check_desc(desc,c,expected_folds,positive):
        assert desc['folds']==sorted(expected_folds) and desc['positive_only']==positive
        hit=valid&np.isin(fold,list(expected_folds))
        if positive:hit&=cys[1][c]>0
        ids=np.flatnonzero(hit);labels=cys[1][c][ids]
        assert desc['rows']==len(ids) and desc['positive_rows']==int((labels>0).sum()) and desc['counties']==sorted(set(fips[ids]))
        for key,a in [('row_sha256',ids),('feature_sha256',X.iloc[ids].to_numpy()),('label_sha256',labels)]:assert desc[key]==ah(a),key
        return ids,labels
    for rec in records:
        spec=rec['spec'];c=spec['component'];head=spec['head'];outer=spec['outer_fold'];inner=spec['inner_fold'];excluded={outer} if inner is None else {outer,inner}
        assert rec['model_id']==digest(spec) and spec['identity_hash']==identity and spec['params']==cfg['params'][head] and spec['feature_names']==list(X)
        assert c in ['N_t','R_t'] and head in ['q','m']
        tr,ty=check_desc(spec['train'],c,set(range(5))-excluded,head=='m');assert not set(fold[tr])&excluded
        assert spec['training_label_sha256']==ah((ty>0).astype(float) if head=='q' else ty)
        path=RUN/rec['model_path'];assert sha(path)==rec['model_sha256'];model=lgb.Booster(model_file=str(path));assert model.feature_name()==list(X)
        assert model.current_iteration()==rec['actual_trees']
        if inner is not None:
            inner_count+=1;va,vy=check_desc(spec['validation'],c,{inner},False);assert outer not in set(fold[va]) and not set(tr)&set(va)
            assert spec['requested_rounds']==2000
            pred=model.predict(X.iloc[va],num_threads=1);saved=pd.read_parquet(RUN/rec['validation_path']);assert sha(RUN/rec['validation_path'])==rec['validation_sha256']
            np.testing.assert_array_equal(saved.row_index,va);pd.testing.assert_frame_equal(saved[KEYS],meta.iloc[va][KEYS].reset_index(drop=True),check_dtype=False)
            np.testing.assert_array_equal(saved.official_Z,vy);np.testing.assert_array_equal(saved.occurrence,(vy>0).astype(int));close(saved.prediction,pred)
            maxdiff=max(maxdiff,float(np.max(abs(pred-saved.prediction.to_numpy()))));curve=np.asarray(rec['curve']);best=rec['best_iteration']
            assert 1<=best<=2000 and best==int(np.argmin(curve))+1 and best<=len(curve)<=2000 and len(curve)==min(2000,best+100)
            assert rec['actual_trees']<=best
            if head=='q':
                assert spec['q_dependency'] is None;close(curve[best-1],binary_loss(vy,pred))
            else:
                dep=spec['q_dependency'];qr=by_id[dep['model_id']];qs=qr['spec'];assert (qs['component'],qs['head'],qs['outer_fold'],qs['inner_fold'])==(c,'q',outer,inner)
                for k in ['model_path','model_sha256','validation_path','validation_sha256']:assert dep[k]==qr[k]
                qframe=pd.read_parquet(RUN/qr['validation_path']);np.testing.assert_array_equal(qframe.row_index,va);q=qframe.prediction.to_numpy()
                assert dep['q_sha256']==ah(q);close(saved.q,q);assert (saved.q_model_id==qr['model_id']).all();close(saved.clipped_m,np.clip(pred,0,1));close(saved['product'],q*np.clip(pred,0,1))
                close(curve[best-1],metric(vy,q*np.clip(pred,0,1))['rmse']);dep_checks+=1
        else:
            outer_count+=1;assert spec['validation'] is None and spec['q_dependency'] is None
            probes=[r for r in records if r['spec']['component']==c and r['spec']['head']==head and r['spec']['outer_fold']==outer and r['spec']['inner_fold'] is not None]
            assert len(probes)==4 and {r['spec']['inner_fold'] for r in probes}==set(range(5))-{outer}
            rounds=max(1,sum(r['best_iteration'] for r in probes)//4);assert spec['requested_rounds']==rounds and 1<=rec['actual_trees']<=rounds
            rows=np.flatnonzero(valid&(fold==outer));pred=model.predict(X.iloc[rows],num_threads=1);close(oof.loc[rows,c+'_'+head],pred)
            maxdiff=max(maxdiff,float(np.max(abs(pred-oof.loc[rows,c+'_'+head].to_numpy()))));replay[c][head][rows]=pred
            assert (oof.loc[rows,c+'_'+head+'_model_id']==rec['model_id']).all()
            if head=='q':close(oof.loc[rows,c+'_prior_q'],float((ty>0).mean()))
    assert (inner_count,outer_count,dep_checks)==(80,20,40)
    for c in ['N_t','R_t']:
        close(oof['true_'+c],cys[1][c]);q=replay[c]['q'];m=replay[c]['m'];close(oof[c+'_clipped_m'],np.clip(m,0,1));close(oof[c+'_product'],q*np.clip(m,0,1))
        expected=[digest(dict(q=qid,m=mid,formula='q*clip(m,0,1)')) if qid else '' for qid,mid in zip(oof[c+'_q_model_id'],oof[c+'_m_model_id'])]
        np.testing.assert_array_equal(oof[c+'_product_id'],expected)
    print('100 checkpoints and 40 inner q-to-m dependencies independently replayed',flush=True)
    reference=norm(pd.read_parquet(TREE/'component_predictions.parquet'));old_final=norm(pd.read_parquet(TREE/'control_predictions.parquet'))
    direct=norm(pd.read_parquet(F2/'control_predictions.parquet'));stored=norm(pd.read_parquet(RUN/'control_predictions.parquet'));stored_comp=norm(pd.read_parquet(RUN/'component_predictions.parquet'))
    stored_al=pd.read_parquet(RUN/'aligned_components.parquet')
    for d in [reference,old_final,direct,stored,stored_comp]:pd.testing.assert_frame_equal(d[KEYS],meta[KEYS],check_dtype=False)
    for c in ['N_t','R_t']:
        for outer in range(5):
            rec=read(OLD/f'models/outer{outer}/{c}_target_t01h.json');assert sha(OLD/rec['model_path'])==rec['model_sha256']
            model=lgb.Booster(model_file=str(OLD/rec['model_path']));hit=valid&(fold==outer)
            p=np.clip(model.predict(X.loc[hit],num_threads=1),0,1);np.testing.assert_array_equal(p,reference.loc[hit,f'pred_NB_P24_{c}_target_t01h'])
    values={};parts={};aligned={}
    for case in CASES:
        parts[case]={};frames=[]
        for h in HS:
            pp=np.column_stack([reference[f'pred_NB_P24_{c}_target_t{h:02d}h'].to_numpy() for c in CS])
            for i,c in enumerate(CS):
                changed=h==1 and c in ['N_t','R_t'] and case in [c[0]+'_only','NR_both']
                if changed:pp[:,i]=replay[c]['q']*np.clip(replay[c]['m'],0,1)
                column=f'pred_{case}_{c}_target_t{h:02d}h';close(stored_comp[column],pp[:,i])
                if not changed:np.testing.assert_array_equal(stored_comp[column],reference[f'pred_NB_P24_{c}_target_t{h:02d}h'])
                expected=oof[c+'_product_id'] if changed else reference[f'source_NB_P24_{c}_target_t{h:02d}h']
                np.testing.assert_array_equal(stored_comp[f'source_{case}_{c}_target_t{h:02d}h'],expected)
            parts[case][h]=np.clip(pp,0,1);hit=hours+h<=215;f=pd.DataFrame(parts[case][h][hit],columns=CS);f['fipsCode']=fips[hit];f['target_hour']=hours[hit]+h;frames.append(f)
        unique=pd.concat(frames,ignore_index=True).groupby(['fipsCode','target_hour'])[CS].mean();values[case]={};aligned[case]={}
        for h in HS:
            idx=pd.MultiIndex.from_arrays([fips,hours+h]);hit=hours+h<=215;al=unique.reindex(idx).to_numpy();aligned[case][h]=al
            np.testing.assert_array_equal(stored[f'true_osi_target_t{h:02d}h'],ys[h]);ar=norm(stored_al[(stored_al.case==case)&(stored_al.horizon==h)])
            for i,c in enumerate(CS):close(ar[c],al[:,i])
            def combine(a):return np.maximum(.4*a[:,0]+.35*a[:,1]+.25*a[:,2]-.1*a[:,3],0)
            c1=combine(parts[case][h]);c3=combine(al);dr=direct[f'raw_direct_osi_target_t{h:02d}h'].to_numpy()
            result=dict(C0_direct_osi=post(dr),C1_component_osi=post(c1),C2_equal_blend=post((dr+c1)/2),C3_aligned_component=post(c3))
            result['v18_rule']=result['C3_aligned_component' if h<=6 else 'C1_component_osi'].copy()
            for mode,p in result.items():p[~hit]=np.nan;close(stored[f'pred_{case}_{mode}_osi_target_t{h:02d}h'],p)
            values[case][h]=result
            if h>=24 or case=='B0':np.testing.assert_array_equal(stored[f'pred_{case}_v18_rule_osi_target_t{h:02d}h'],old_final[f'pred_NB_P24_v18_rule_osi_target_t{h:02d}h'])
    member=pd.read_parquet(OUT/'reference/test_like_membership.parquet').set_index(['horizon','target_day','fipsCode']).test_like
    assert sha(OUT/'reference/test_like_membership.parquet')==cfg['membership_sha256']
    def gmask(h,name):
        s=hours+h;ix=pd.MultiIndex.from_arrays([np.full(n,h),s//24,fips]);near=member.reindex(ix,fill_value=False).to_numpy(bool)
        return dict(all=np.ones(n,bool),early_73_95=s<=95,late_96_215=s>=96,test_like=near,not_test_like=~near,all_except_Forest=fips!='42053',test_like_except_Forest=near&(fips!='42053'))[name]
    votes=pd.read_parquet(BASE/'train_test_feature_support/data/nearest_five.parquet');votes=votes[(votes.scope=='test_to_train')&(votes.view=='context')]
    sets=votes.groupby(['horizon','target_day']).matched_fips.agg(set)
    for key,r in member.reset_index().groupby(['horizon','target_day']):assert set(r.loc[r.test_like,'fipsCode'])==sets.loc[key]
    count=0
    for name in ['primary_scores','all_controls','comparisons','group_metrics','county_metrics','fold_metrics','day_metrics','bootstrap','component_metrics']:
        df=csv(RUN/f'metrics/{name}.csv',dtype={'fipsCode':str} if name=='county_metrics' else None)
        for r in df.itertuples():
            h=r.horizon;y=ys[h].copy()
            if name in ['primary_scores','all_controls']:calc=metric(y,values[r.case][h][r.mode if name=='all_controls' else 'v18_rule'])
            else:
                b=values[r.reference][h]['v18_rule'];p=values[r.case][h]['v18_rule']
                if name in ['group_metrics','bootstrap']:y[~gmask(h,r.group)]=np.nan
                elif name=='county_metrics':y[fips!=r.fipsCode]=np.nan
                elif name=='fold_metrics':y[fold!=r.fold]=np.nan
                elif name=='day_metrics':y[(hours+h)//24!=r.target_day]=np.nan
                elif name=='component_metrics':
                    y=cys[h][r.component].copy();domain=parts if r.space=='source' else aligned;i=CS.index(r.component);b=domain[r.reference][h][:,i];p=domain[r.case][h][:,i]
                    total=metric(y,b)['sse'];y[~cmask(y,r.group)]=np.nan;sse=metric(y,b)['sse'];close(r.baseline_group_sse,sse);close(r.baseline_group_sse_share,sse/total if total else np.nan)
                calc=difference(y,b,p)
            for k,v in calc.items():close(getattr(r,k),v)
            count+=1
    boots=csv(RUN/'metrics/bootstrap.csv');draw=np.random.default_rng(20260910).integers(0,239,(2000,239));multiplicity=np.stack([np.bincount(d,minlength=239) for d in draw]);codes=pd.Categorical(fips,categories=sorted(set(fips))).codes
    for r in boots.itertuples():
        h=r.horizon;hit=np.isfinite(ys[h])&gmask(h,r.group);b=values[r.reference][h]['v18_rule'];p=values[r.case][h]['v18_rule'];y=ys[h]
        if np.array_equal(b[hit],p[hit]):close([r.ci_low,r.ci_high],[0,0]);continue
        nn=np.bincount(codes[hit],minlength=239);sb=np.bincount(codes[hit],weights=(b[hit]-y[hit])**2,minlength=239);sp=np.bincount(codes[hit],weights=(p[hit]-y[hit])**2,minlength=239)
        delta=np.sqrt((multiplicity@sp)/(multiplicity@nn))-np.sqrt((multiplicity@sb)/(multiplicity@nn));close([r.ci_low,r.ci_high],np.quantile(delta,[.025,.975]))
    changes=pd.read_parquet(RUN/'row_changes.parquet')
    for (pair,h),r in changes.groupby(['pair','horizon']):
        case,ref=pair.split('_minus_');hit=np.isfinite(ys[h]);assert len(r)==hit.sum();np.testing.assert_array_equal(r.fipsCode,fips[hit]);close(r.truth,ys[h][hit])
        b=values[ref][h]['v18_rule'][hit];p=values[case][h]['v18_rule'][hit];close(r.baseline,b);close(r.candidate,p);close(r.sse_reduction,(b-ys[h][hit])**2-(p-ys[h][hit])**2)
        np.testing.assert_array_equal(r.test_like,gmask(h,'test_like')[hit])
    cls=csv(RUN/'metrics/classification.csv',dtype={'fold':str})
    for r in cls.itertuples():
        hit=valid if r.fold=='all' else valid&(fold==int(r.fold));y=cys[1][r.component][hit];q=oof.loc[hit,r.component+('_q' if r.variant=='q_model' else '_prior_q')].to_numpy()
        assert r.n==len(y);close([r.logloss,r.brier,r.observed_positive_fraction,r.mean_q],[binary_loss(y,q),np.mean((q-(y>0))**2),np.mean(y>0),q.mean()])
    calibration=csv(RUN/'metrics/calibration.csv')
    for r in calibration.itertuples():
        q=replay[r.component]['q'];hit=valid&(q>=r.bin_low)&((q<r.bin_high) if r.bin_low<.9 else q<=1);assert r.n==int(hit.sum())
        close([r.mean_q,r.observed_positive_fraction],[q[hit].mean() if hit.any() else np.nan,(cys[1][r.component][hit]>0).mean() if hit.any() else np.nan])
    amps=csv(RUN/'metrics/amplitude_diagnostics.csv')
    for r in amps.itertuples():
        y=cys[1][r.component];hit=valid&cmask(y,r.group);q=replay[r.component]['q'];m=replay[r.component]['m'];mc=np.clip(m,0,1)
        assert r.n==int(hit.sum()) and r.raw_m_below_zero==int((m[hit]<0).sum()) and r.raw_m_above_one==int((m[hit]>1).sum())
        close([r.true_mean,r.q_mean,r.raw_m_mean,r.clipped_m_mean,r.product_mean,r.conditional_m_rmse],
              [y[hit].mean(),q[hit].mean(),m[hit].mean(),mc[hit].mean(),(q[hit]*mc[hit]).mean(),metric(y[hit],mc[hit])['rmse']])
    decomposition=csv(RUN/'metrics/OSI_change_decomposition.csv')
    for r in decomposition.itertuples():
        h=r.horizon;hit=np.isfinite(ys[h]);ba=aligned[r.reference][h];ca=aligned[r.case][h];oldsum=.4*ba[:,0]+.35*ba[:,1]+.25*ba[:,2]-.1*ba[:,3]
        dn=(.35*(ca[:,1]-ba[:,1]))[hit];dr=(-.1*(ca[:,3]-ba[:,3]))[hit];error=(oldsum-ys[h])[hit];delta=dn+dr
        cross=-2*np.sum(error*delta);sq=-np.sum(delta**2);actual=difference(ys[h],values[r.reference][h]['v18_rule'],values[r.case][h]['v18_rule'])['sse_reduction']
        close([r.old_error_cross_reduction,r.change_squared_reduction,r.N_R_change_interaction,r.latent_sse_reduction,r.postprocess_remainder,r.OSI_sse_reduction],[cross,sq,-2*np.sum(dn*dr),cross+sq,actual-cross-sq,actual])
    for p,h in sources.items():assert sha(p)==h,p
    result=dict(status='PASS',identity_hash=identity,formal_fits=100,inner_models_reloaded=80,outer_models_reloaded=20,reference_models_reloaded=10,
        inner_q_m_dependencies_verified=dep_checks,maximum_prediction_difference=maxdiff,original_163_columns=True,raw_official_labels=True,
        county_fold_exclusion=True,m_positive_train_all_validation=True,full_validation_float64_product_metric=True,metric_rows_checked=count,
        bootstrap_rows_checked=len(boots),row_comparisons_checked=len(changes),classification_rows_checked=len(cls),amplitude_rows_checked=len(amps),
        final24_48_exact=True,unmodified_component_sources_exact=True,independent_target_keyed_C3=True,auxiliary_membership_exact=True,
        source_files_unchanged=len(sources),absolute_tolerance=1e-12,verifier_sha256=sha(__file__))
    (RUN/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':verify()
