"""Independent 50-checkpoint replay and scalar/target-keyed evaluation verification."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
import lightgbm as lgb

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=OUT.parents[3]
RUN=OUT/'runs/v18_pgrowth_v1_split42_model42';F2=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42'
TREE=BASE/'p_neighbor_horizon_increment/runs/v18_pneighbor_v1_split42_model42'
HS=[1,6,24,48];CS=['P_t','N_t','D_t','R_t'];W=[.4,.35,.25,-.1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ah(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def norm(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.zfill(5);d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    return d.sort_values(['fipsCode','hour_idx']).reset_index(drop=True)
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)
def post(a):
    a=np.maximum(0,np.minimum(.65,a));return np.where(a<.001,0,a)
def metrics(y,p):
    hit=np.isfinite(y);e=np.asarray(p)[hit]-np.asarray(y)[hit];assert np.isfinite(e).all();n=len(e)
    sse=math.fsum(float(v)*float(v) for v in e)
    return dict(n=n,sse=sse,rmse=math.sqrt(sse/n) if n else np.nan,mae=math.fsum(abs(float(v)) for v in e)/n if n else np.nan,
        bias=math.fsum(map(float,e))/n if n else np.nan)
def difference(y,b,p):
    b,c=metrics(y,b),metrics(y,p)
    return dict(**c,baseline_rmse=b['rmse'],rmse_delta=c['rmse']-b['rmse'],rmse_change_pct=100*(c['rmse']/b['rmse']-1) if b['rmse']>0 else np.nan,sse_reduction=b['sse']-c['sse'])

def verify():
    source_hashes=read(OUT/'reference/input_hashes.json');m=read(RUN/'run_manifest.json');identity=m['identity_hash']
    assert digest(m['identity'])==identity
    for p,h in source_hashes.items():assert sha(p)==h,p
    for code,h in m['identity']['training_code_sha256'].items():assert sha(OUT/code)==h
    for marker in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']:
        rec=read(RUN/marker);assert rec['identity_hash']==identity
        for p,h in rec['files'].items():assert sha(RUN/p)==h,p
    assert read(RUN/'EVALUATION_COMPLETE')['evaluation_code_sha256']==sha(OUT/'evaluate_growth.py')
    cfg=read(OUT/'experiment_config.json');assert m['identity']['config_sha256']==sha(OUT/'experiment_config.json')
    meta=norm(pd.read_parquet(ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet'))
    cv=csv(ROOT/'cv/cv_assignments_balanced_v1_seed42.csv',dtype={'fipsCode':str}).set_index('fipsCode').fold
    meta['fold']=meta.fipsCode.map(cv);folds=meta.fold.to_numpy();n=len(meta);assert n==34416
    X=pd.read_parquet(BASE/'p_information_increment/features/v1/F2_train.parquet');assert X.shape==(n,183)
    raw=csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str});raw['hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
    raw=raw.set_index(['fipsCode','hour']).sort_index();cut=raw.xs(71,level='hour')
    p=meta.fipsCode.map(cut.P_t).to_numpy(float);g=meta.fipsCode.map(cut.N_t).to_numpy(float)
    y1=raw.P_t.reindex(pd.MultiIndex.from_arrays([meta.fipsCode,meta.hour_idx+1])).to_numpy()
    early=(meta.hour_idx.to_numpy()+1<=95);assert early.sum()==5497
    branches=norm(pd.read_parquet(RUN/'branch_predictions.parquet'))
    np.testing.assert_array_equal(branches.p71,p);np.testing.assert_array_equal(branches.N71,g)
    model_records=read(RUN/'model_manifest.json');assert len(model_records)==50
    max_reload=0.;new_raw={c:np.full(n,np.nan) for c in ['G1','G2']};inner_count=outer_count=0
    def scoped(fs,case):
        allowed=early&np.isin(folds,fs);indices=np.flatnonzero(allowed&(p<1))
        scale=np.array([(1-p[j])*(1+g[j]/.01) if case=='G2' else 1-p[j] for j in indices])
        labels=np.array([max(float(y1[j])-float(p[j]),0)/scale[k] for k,j in enumerate(indices)])
        weights=scale*scale
        return indices,scale,labels,weights,int(allowed.sum())
    def check_description(desc,case):
        ids,scale,labels,weights,total=scoped(desc['folds'],case)
        assert desc['counties']==sorted(meta.iloc[ids].fipsCode.unique()) and desc['rows']==len(ids) and desc['total_valid']==total
        for key,arr in [('row_sha256',ids),('feature_sha256',X.iloc[ids].to_numpy()),('label_sha256',labels),('scale_sha256',scale),('weight_sha256',weights),('normalized_weight_sha256',weights/weights.mean())]:
            assert desc[key]==ah(arr),key
        close(desc['weight_mean'],weights.mean())
        return ids,scale,labels,weights,total
    for rec in model_records:
        spec=rec['spec'];case=spec['case'];outer=spec['outer_fold'];inner=spec['inner_fold']
        assert rec['model_id']==digest(spec) and spec['identity_hash']==identity and spec['params']==cfg['params'] and spec['feature_names']==list(X)
        excluded={outer} if inner is None else {outer,inner}
        assert set(spec['train']['folds'])==set(range(5))-excluded
        tr,_,_,weights,_=check_description(spec['train'],case)
        assert not set(folds[tr])&excluded;close(rec['training_weight_mean'],weights.mean())
        path=RUN/rec['model_path'];assert sha(path)==rec['model_sha256'];model=lgb.Booster(model_file=str(path))
        assert model.feature_name()==list(X) and model.current_iteration()==rec['actual_trees']
        if inner is not None:
            inner_count+=1;assert spec['validation']['folds']==[inner]
            va,scale,labels,ww,total=check_description(spec['validation'],case);assert outer not in set(folds[va]) and not set(va)&set(tr)
            pred=model.predict(X.iloc[va],num_threads=1);saved=pd.read_parquet(RUN/rec['validation_path']);assert sha(RUN/rec['validation_path'])==rec['validation_sha256']
            np.testing.assert_array_equal(saved.row_index,va);close(saved.label_u,labels);close(saved.scale,scale);close(saved.raw_weight,ww);close(saved.prediction,pred)
            max_reload=max(max_reload,float(np.max(abs(saved.prediction.to_numpy()-pred))))
            curve=np.asarray(rec['curve']);best=rec['best_iteration'];assert 1<=best<=2000 and best==int(np.argmin(curve))+1
            close(curve[best-1],math.sqrt(math.fsum(float(w)*float(a-b)**2 for w,a,b in zip(ww,pred,labels))/total))
            close(rec['validation_weight_scale_training_mean'],weights.mean())
        else:
            outer_count+=1;assert spec['validation'] is None
            inner_records=[x for x in model_records if x['spec']['case']==case and x['spec']['outer_fold']==outer and x['spec']['inner_fold'] is not None]
            assert len(inner_records)==4;rounds=max(1,sum(x['best_iteration'] for x in inner_records)//4)
            assert spec['requested_rounds']==rounds and 1<=rec['actual_trees']<=rounds
            hit=np.flatnonzero(early&(p<1)&(folds==outer));pred=model.predict(X.iloc[hit],num_threads=1)
            close(branches.loc[hit,case+'_raw_b'],pred);new_raw[case][hit]=pred
            max_reload=max(max_reload,float(np.max(abs(branches.loc[hit,case+'_raw_b'].to_numpy()-pred))))
            assert (branches.loc[hit,case+'_model_id']==rec['model_id']).all()
    assert (inner_count,outer_count)==(40,10)
    reference_a=np.full(n,np.nan);valid1=meta.hour_idx.to_numpy()+1<=215;reference_a[valid1&(p==0)]=0.
    for outer in range(5):
        rec=read(F2/f'models/F2/outer{outer}/P_anchor_a_t01h.json');spec=rec['spec']
        assert outer not in spec['allowed_folds'] and rec['model_id']==digest(spec) and sha(F2/rec['model_path'])==rec['model_sha256']
        mask=valid1&(p>0)&(folds==outer);model=lgb.Booster(model_file=str(F2/rec['model_path']))
        reference_a[mask]=np.clip(model.predict(X.loc[mask],num_threads=1),0,1)
    close(reference_a,branches.reference_a_applied)
    print('All 50 new checkpoints and 5 reference-a checkpoints PASS',flush=True)
    original=norm(pd.read_parquet(TREE/'component_predictions.parquet'));old_final=norm(pd.read_parquet(TREE/'control_predictions.parquet'))
    direct=norm(pd.read_parquet(F2/'control_predictions.parquet'));saved=norm(pd.read_parquet(RUN/'control_predictions.parquet'))
    stored_parts=norm(pd.read_parquet(RUN/'component_predictions.parquet'))
    expected={};component={};aligned={};truth={}
    for case in ['G0','G1','G2']:
        component[case]={};expanded=[]
        for h in HS:
            pp=np.column_stack([original[f'pred_NB_P24_{c}_target_t{h:02d}h'].to_numpy() for c in CS])
            if case!='G0' and h==1:
                for j in np.flatnonzero(early):
                    if p[j]==1:delta=0.
                    else:
                        scale=(1-p[j])*(1+g[j]/.01) if case=='G2' else 1-p[j]
                        delta=min(1-p[j],max(0.,scale*min(1.,max(0.,new_raw[case][j]))))
                    pp[j,0]=min(1.,max(0.,p[j]*reference_a[j]+delta))
                close(pp[:,0],branches[case+'_P1'])
            for k,c in enumerate(CS):
                val=stored_parts[f'pred_{case}_{c}_target_t{h:02d}h'].to_numpy();close(val,pp[:,k])
                if case=='G0' or (h,c)!=(1,'P_t'):np.testing.assert_array_equal(val,original[f'pred_NB_P24_{c}_target_t{h:02d}h'])
            if case!='G0' and h==1:np.testing.assert_array_equal(pp[~early,0],original.pred_NB_P24_P_t_target_t01h.to_numpy()[~early])
            clipped=np.clip(pp,0,1);component[case][h]=clipped
            hit=meta.hour_idx.to_numpy()+h<=215
            ex=pd.DataFrame(clipped[hit],columns=CS);ex['fipsCode']=meta.fipsCode.to_numpy()[hit];ex['target_hour']=meta.hour_idx.to_numpy()[hit]+h;expanded.append(ex)
        unique=pd.concat(expanded,ignore_index=True).groupby(['fipsCode','target_hour'])[CS].mean()
        expected[case]={};aligned[case]={}
        for h in HS:
            hit=meta.hour_idx.to_numpy()+h<=215;idx=pd.MultiIndex.from_arrays([meta.fipsCode,meta.hour_idx+h]);yy=raw.osi.reindex(idx).to_numpy();truth[h]=yy
            assert np.array_equal(np.isfinite(yy),hit);close(saved[f'true_osi_target_t{h:02d}h'],yy)
            aa=unique.reindex(idx).to_numpy();aligned[case][h]=aa
            source=np.maximum(component[case][h]@np.array(W),0);align=np.maximum(aa@np.array(W),0);dr=direct[f'raw_direct_osi_target_t{h:02d}h'].to_numpy()
            values={'C0_direct_osi':post(dr),'C1_component_osi':post(source),'C2_equal_blend':post((dr+source)/2),'C3_aligned_component':post(align)}
            values['v18_rule']=values['C3_aligned_component' if h<=6 else 'C1_component_osi'].copy()
            for mode,a in values.items():
                a[~hit]=np.nan;close(a,saved[f'pred_{case}_{mode}_osi_target_t{h:02d}h'])
            expected[case][h]=values['v18_rule']
            unchanged=hit&((meta.hour_idx.to_numpy()+h>=96)|(h>=24))
            if case!='G0':np.testing.assert_array_equal(saved.loc[unchanged,f'pred_{case}_v18_rule_osi_target_t{h:02d}h'],old_final.loc[unchanged,f'pred_NB_P24_v18_rule_osi_target_t{h:02d}h'])
    definitions=read(RUN/'group_definitions.json');member=pd.read_parquet(OUT/'reference/test_like_membership.parquet').set_index(['horizon','target_day','fipsCode']).test_like
    yearly=pd.DataFrame({'fipsCode':meta.fipsCode.to_numpy()[early],'y':truth[1][early]}).groupby('fipsCode').y.max()
    low=meta.fipsCode.map(yearly).to_numpy()<=.01;fips=meta.fipsCode.to_numpy();metric_count=0
    def group_mask(h,name):
        s=meta.hour_idx.to_numpy()+h;ix=pd.MultiIndex.from_arrays([np.full(n,h),s//24,fips]);near=member.reindex(ix,fill_value=False).to_numpy(bool)
        return {'all':np.ones(n,bool),'early_73_95':s<=95,'late_96_215':s>=96,'test_like':near,'not_test_like':~near,'all_except_Forest':fips!='42053',
            'test_like_except_Forest':near&(fips!='42053'),'g_zero':g==0,'g_positive':g>0,'g_positive_low_early_damage':(g>0)&low}[name]
    for name in ['primary_scores','all_controls','comparisons','county_metrics','fold_metrics','group_metrics','bootstrap']:
        table=csv(RUN/f'metrics/{name}.csv',dtype={'fipsCode':str} if name=='county_metrics' else None)
        for r in table.itertuples():
            h=r.horizon;yy=truth[h].copy();candidate=expected[r.case][h]
            if name=='primary_scores':calc=metrics(yy,candidate)
            elif name=='all_controls':calc=metrics(yy,saved[f'pred_{r.case}_{r.mode}_osi_target_t{h:02d}h'].to_numpy())
            else:
                baseline=expected[r.reference][h]
                if name=='county_metrics':yy[fips!=r.fipsCode]=np.nan
                elif name=='fold_metrics':yy[folds!=r.fold]=np.nan
                elif name in ['group_metrics','bootstrap']:yy[~group_mask(h,r.group)]=np.nan
                calc=difference(yy,baseline,candidate)
            for key,value in calc.items():close(getattr(r,key),value)
            metric_count+=1
    boots=csv(RUN/'metrics/bootstrap.csv');draw=np.random.default_rng(20260910).integers(0,239,(2000,239));counts=np.stack([np.bincount(v,minlength=239) for v in draw]);county_indices=pd.Categorical(fips,categories=sorted(set(fips))).codes
    for r in boots.itertuples():
        h=r.horizon;mask=np.isfinite(truth[h])&group_mask(h,r.group);b=expected[r.reference][h];c=expected[r.case][h];yy=truth[h]
        if np.array_equal(b[mask],c[mask]):close([r.ci_low,r.ci_high],[0,0]);continue
        nn=np.bincount(county_indices[mask],minlength=239);sb=np.bincount(county_indices[mask],weights=(b[mask]-yy[mask])**2,minlength=239);sc=np.bincount(county_indices[mask],weights=(c[mask]-yy[mask])**2,minlength=239)
        delta=np.sqrt((counts@sc)/(counts@nn))-np.sqrt((counts@sb)/(counts@nn));close([r.ci_low,r.ci_high],np.quantile(delta,[.025,.975]))
    ptable=csv(RUN/'metrics/P_metrics.csv')
    for r in ptable.itertuples():
        h=r.horizon;idx=pd.MultiIndex.from_arrays([meta.fipsCode,meta.hour_idx+h]);y=raw.P_t.reindex(idx).to_numpy(copy=True)
        if r.group=='early_73_95':y[meta.hour_idx.to_numpy()+h>95]=np.nan
        data=component if r.space=='source' else aligned
        calc=difference(y,data[r.reference][h][:,0],data[r.case][h][:,0])
        for k,v in calc.items():close(getattr(r,k),v)
    rows=pd.read_parquet(RUN/'row_changes.parquet')
    for (pair,h),r in rows.groupby(['pair','horizon']):
        case,ref=pair.split('_minus_');hit=np.isfinite(truth[h]);assert len(r)==hit.sum()
        np.testing.assert_array_equal(r.fipsCode,fips[hit]);close(r.truth,truth[h][hit]);close(r.baseline,expected[ref][h][hit]);close(r.candidate,expected[case][h][hit])
        close(r.sse_reduction,(expected[ref][h][hit]-truth[h][hit])**2-(expected[case][h][hit]-truth[h][hit])**2)
    boundary=csv(RUN/'metrics/boundary_95_96.csv',dtype={'fipsCode':str})
    for r in boundary.itertuples():
        h=6 if r.domain=='OSI6' else 1;before=np.flatnonzero((fips==r.fipsCode)&(meta.hour_idx.to_numpy()+h==95))[0];after=np.flatnonzero((fips==r.fipsCode)&(meta.hour_idx.to_numpy()+h==96))[0]
        a=component[r.case][1][:,0] if r.domain=='P1_source' else expected[r.case][h];b=component['G0'][1][:,0] if r.domain=='P1_source' else expected['G0'][h]
        close([r.value95,r.value96,r.additional_jump],[a[before],a[after],(a[after]-a[before])-(b[after]-b[before])])
    oldbranch=pd.read_parquet(F2/'branch_oof_predictions.parquet');oldbranch=norm(oldbranch[oldbranch['case']=='F2'])
    branch_metrics=csv(RUN/'metrics/branch_contribution.csv')
    for r in branch_metrics.itertuples():
        raw_b=oldbranch.pred_b_raw.to_numpy() if r.case=='G0' else new_raw[r.case]
        scale=(1-p)*(1+g/.01) if r.case=='G2' else 1-p
        inc=np.clip(scale[early]*np.clip(raw_b[early],0,1),0,1-p[early])
        ea=p[early]*reference_a[early]-np.minimum(y1[early],p[early]);eb=inc-np.maximum(y1[early]-p[early],0)
        close([r.b_increment_rmse,r.P1_early_rmse,r.a_squared_term,r.b_squared_term,r.two_ab_cross_term],
            [np.sqrt(np.mean(eb**2)),np.sqrt(np.mean((component[r.case][1][early,0]-y1[early])**2)),np.mean(ea**2),np.mean(eb**2),2*np.mean(ea*eb)])
        assert r.raw_b_below_zero==int((raw_b[early]<0).sum()) and r.raw_b_above_one==int((raw_b[early]>1).sum())
        assert r.increment_capped_at_remaining==int((scale[early]*np.clip(raw_b[early],0,1)>1-p[early]).sum())
    # Auxiliary members must exactly reproduce the pre-existing, outcome-blind vote union.
    votes=pd.read_parquet(BASE/'train_test_feature_support/data/nearest_five.parquet');votes=votes[(votes.scope=='test_to_train')&(votes.view=='context')]
    sets=votes.groupby(['horizon','target_day']).matched_fips.agg(set)
    for (h,day),r in member.reset_index().groupby(['horizon','target_day']):assert set(r.loc[r.test_like,'fipsCode'])==sets.loc[(h,day)]
    for p,h in source_hashes.items():assert sha(p)==h,p
    result=dict(status='PASS',identity_hash=identity,new_inner_models_reloaded=40,new_outer_models_reloaded=10,reference_a_models_reloaded=5,
        checkpoint_prediction_max_abs_diff=max_reload,metric_groups_checked=metric_count,P_metric_groups_checked=len(ptable),bootstrap_rows_checked=len(boots),
        row_changes_checked=len(rows),late_and_long_invariants_exact=True,unmodified_components_exact=True,independent_target_keyed_C3=True,
        auxiliary_membership_exact=True,source_files_unchanged=len(source_hashes),absolute_tolerance=1e-12,verifier_sha256=sha(__file__))
    (RUN/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':verify()
