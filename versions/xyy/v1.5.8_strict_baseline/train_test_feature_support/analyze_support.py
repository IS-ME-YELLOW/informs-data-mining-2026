"""Unsupervised train/test feature support audit. Never fits or predicts a model."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=OUT.parents[3]
SEEDS=(42,20260917,20260918);HS=(1,6,24,48)
HIST=['last_P_t','last_D_t','last_N_t','last_R_t','osi_max_72h','osi_mean_72h','osi_trend_last6h']
WEATHER=['gust_max','gust_mean','gust_hours_gt30','tp_sum','soil_mean','prior24_gust_max','prior24_tp_sum']
STATIC=['customers_at_cutoff','pop_density','pct_forest','n_utilities']
CONTEXT={'history':HIST,'weather':WEATHER,'static':STATIC}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def norm(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5);return d
def safe(p):
    p=Path(p).resolve();assert p.is_relative_to(OUT) and p!=OUT;p.parent.mkdir(parents=True,exist_ok=True);return p
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,Path):return str(v)
    return v
def save(p,v):safe(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def frame(p,d):
    p=safe(p);d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)
    if p.suffix=='.parquet':d.to_parquet(p,index=False)
    else:d.to_csv(p,index=False)

def inputs():
    source=[]
    for split in ['train','test']:
        source.extend([ROOT/f'data/DM_{split.title()}.csv',ROOT/f'versions/xyy/v1.5.6/features_{split}_v1.5.6.parquet',ROOT/f'versions/xyy/v1.5.6/meta_{split}_v1.5.6.parquet',
            BASE/f'p_information_increment/features/v1/F2_{split}.parquet',BASE/f'p_neighbor_horizon_increment/features/v1/h24_{split}.parquet'])
    source.extend([BASE/'p_information_increment/features/v1/coordinates.csv',BASE/'common_error_diagnosis/matching/inputs.parquet',
        BASE/'common_error_diagnosis/matching/cache_blocks.json',BASE/'common_error_diagnosis/matching/support.csv',BASE/'common_error_diagnosis/summary/case_selection.csv',
        BASE/'common_error_diagnosis/Results_2026-09-20.md'])
    source.extend(ROOT/f'cv/cv_assignments_balanced_v1_seed{s}.csv' for s in SEEDS)
    hashes={str(p):sha(p) for p in source}
    if (OUT/'reference/source_hashes.json').exists():assert read(OUT/'reference/source_hashes.json')==hashes
    else:save(OUT/'reference/source_hashes.json',hashes)
    rows=[];names=[];blocks=read(BASE/'common_error_diagnosis/matching/cache_blocks.json');quality=[]
    for split in ['train','test']:
        path=ROOT/f'data/DM_{split.title()}.csv'
        # Read only identities, allowed weather, and outage history needed at the cutoff.
        raw=norm(csv(path,usecols=['timestamp_et','fipsCode','countyName','stateAbbr','customersTracked','P_t','N_t','D_t','R_t','gust','tp','soil_moist']))
        raw['hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
        expected=239 if split=='train' else 63
        assert len(raw)==216*expected and not raw.duplicated(['fipsCode','hour']).any()
        if split=='test':assert raw[raw.hour>=72][['P_t','N_t','D_t','R_t']].isna().all().all()
        cut=raw[raw.hour==71].set_index('fipsCode').sort_index()
        names.append(cut[['countyName','stateAbbr']].assign(dataset=split).reset_index())
        history=raw[raw.hour<=71].copy()
        # Future outage columns are not passed to any feature builder.
        weather=raw[['fipsCode','hour','gust','tp','soil_moist']].copy();del raw
        meta=norm(pd.read_parquet(ROOT/f'versions/xyy/v1.5.6/meta_{split}_v1.5.6.parquet'))
        x=pd.read_parquet(ROOT/f'versions/xyy/v1.5.6/features_{split}_v1.5.6.parquet')
        assert len(x)==len(meta)==expected*144
        hist=x[HIST+STATIC[1:]].assign(fipsCode=meta.fipsCode.to_numpy())
        assert hist.groupby('fipsCode').nunique(dropna=False).max().max()==1
        hist=hist.groupby('fipsCode').first()
        hist['customers_at_cutoff']=cut.customersTracked
        for c in ['P','N','D','R']:np.testing.assert_array_equal(hist['last_'+c+'_t'],cut.loc[hist.index,c+'_t'])
        for h in HS:
            cache=pd.read_parquet(BASE/f'p_information_increment/features/v1/F2_{split}.parquet') if h==1 else (pd.read_parquet(BASE/f'p_neighbor_horizon_increment/features/v1/h24_{split}.parquet') if h==24 else x)
            assert len(cache)==len(meta)
            for day in range((72+h)//24,9):
                lo=max(72+h,day*24);hi=min(215,day*24+23)
                w=weather[weather.hour.between(lo,hi)].copy();w['gt30']=w.gust>30
                a=w.groupby('fipsCode').agg(gust_max=('gust','max'),gust_mean=('gust','mean'),gust_hours_gt30=('gt30','sum'),tp_sum=('tp','sum'),soil_mean=('soil_moist','mean'))
                b=weather[weather.hour.between(lo-24,lo-1)].groupby('fipsCode').agg(prior24_gust_max=('gust','max'),prior24_tp_sum=('tp','sum'))
                d=hist.join(a).join(b)
                mask=(meta.hour_idx+h).between(lo,hi)
                ca=cache.loc[mask].assign(fipsCode=meta.loc[mask,'fipsCode'].to_numpy()).groupby('fipsCode').mean().add_prefix('cache__')
                d=d.join(ca);d['horizon']=h;d['target_day']=day;d['start']=lo;d['end']=hi;d['dataset']=split
                rows.append(d.reset_index())
        quality.append(dict(dataset=split,counties=expected,history_rows=len(history),future_weather_rows=expected*144,future_outages_used=False))
    x=pd.concat(rows,ignore_index=True);name=pd.concat(names,ignore_index=True)
    assert len(name)==302 and name.fipsCode.nunique()==302
    train=x[x.dataset=='train'].drop(columns='dataset')
    old=pd.read_parquet(BASE/'common_error_diagnosis/matching/inputs.parquet')
    ix=['horizon','target_day','fipsCode']
    pd.testing.assert_frame_equal(train.set_index(ix)[old.columns.difference(ix)].sort_index(),old.set_index(ix)[old.columns.difference(ix)].sort_index(),check_exact=True)
    frame(OUT/'data/allowed_inputs.parquet',x);frame(OUT/'data/county_names.csv',name);save(OUT/'data/input_checks.json',quality)
    save(OUT/'data/cache_blocks.json',blocks)
    return x,blocks,name,hashes

def transform(full,ref,groups,rank):
    arrays=[];group_slices={};off=0
    for group,cols in groups.items():
        a=full[cols].to_numpy(float,copy=True);r=ref[cols].to_numpy(float,copy=True)
        for j,c in enumerate(cols):
            if rank:
                if np.isnan(a[:,j]).all() and np.isnan(r[:,j]).all():a[:,j]=r[:,j]=0;continue
                missing=np.isnan(a[:,j])
                if missing.any():assert c=='cache__hours_since_peak' and np.isfinite(r[:,j]).all(),c
                assert not np.isinf(a[:,j]).any() and np.isfinite(r[:,j]).all(),c
                z=np.sort(r[:,j].copy());a[:,j]=(np.searchsorted(z,a[:,j],'left')+np.searchsorted(z,a[:,j],'right'))/(2*len(z))
                a[missing,j]=-1.0
                r[:,j]=(np.searchsorted(z,r[:,j],'left')+np.searchsorted(z,r[:,j],'right'))/(2*len(z))
            else:
                if c in HIST:a[:,j]=np.arcsinh(a[:,j]/.01);r[:,j]=np.arcsinh(r[:,j]/.01)
                elif c in ['customers_at_cutoff','pop_density','n_utilities','tp_sum','prior24_tp_sum']:a[:,j]=np.log1p(a[:,j]);r[:,j]=np.log1p(r[:,j])
        if not rank:
            mu=r.mean(axis=0);sd=r.std(axis=0);sd=np.where(sd>1e-12,sd,1);a=(a-mu)/sd
        assert np.isfinite(a).all()
        arrays.append(a/np.sqrt(len(cols)*len(groups)));group_slices[group]=(off,off+len(cols));off+=len(cols)
    return np.column_stack(arrays),group_slices

def distances(a,b):return np.sqrt(np.maximum(((a[:,None,:]-b[None,:,:])**2).sum(axis=2),0))

def evaluate(full,z,slices,ref_indices,query_indices,candidate_indices,scope,view,seed,fold,h,day,metrics,pairs,thresholds,ref_cache=None):
    ref_indices=np.asarray(ref_indices);query_indices=np.asarray(query_indices);candidate_indices=np.asarray(candidate_indices)
    if ref_cache is None:
        rr=distances(z[ref_indices],z[ref_indices]);np.fill_diagonal(rr,np.inf)
        sorted_ref=np.sort(rr,axis=1);reference1=sorted_ref[:,0];reference5=sorted_ref[:,4]
        q1,q5=np.quantile(reference1,.95),np.quantile(reference5,.95)
    else:reference1,reference5,q1,q5=ref_cache
    querydist=distances(z[query_indices],z[candidate_indices]);fips=full.fipsCode.to_numpy();splits=full.dataset.to_numpy()
    q_codes=fips[query_indices];candidate_codes=fips[candidate_indices]
    querydist[q_codes[:,None]==candidate_codes[None,:]]=np.inf
    if scope in ['train_loo','oof_before']:
        thresholds.append(dict(scope=scope,cv_seed=seed,fold=fold,horizon=h,target_day=day,view=view,fit_counties=len(ref_indices),q95_d1=q1,q95_d5=q5))
    for i,k in enumerate(query_indices):
        order=np.lexsort((candidate_codes,querydist[i]))[:5];selected=candidate_indices[order]
        vals=querydist[i,order];d1,d5=vals[0],vals[-1]
        key=dict(scope=scope,cv_seed=seed,fold=fold,horizon=h,target_day=day,view=view,fipsCode=fips[k],dataset=splits[k])
        row=dict(**key,candidate_count=len(candidate_indices)-int(k in candidate_indices),fit_counties=len(ref_indices),
            d1=d1,d5=d5,reference_d1_q95=q1,reference_d5_q95=q5,d1_percentile=100*(reference1<=d1).mean(),d5_percentile=100*(reference5<=d5).mean(),
            flag_d1=bool(d1>q1+1e-12),flag_d5=bool(d5>q5+1e-12),nearest_fips=fips[selected[0]],nearest_split=splits[selected[0]],
            test_count_in_top5=int((splits[selected]=='test').sum()),d1_over_q95=d1/q1 if q1>0 else np.nan)
        for group,(a,b) in slices.items():row[group+'_nearest_squared_distance']=float(((z[k,a:b]-z[selected[0],a:b])**2).sum())
        metrics.append(row)
        if view in ['context','source_cache_rank']:
            for rank,idx,value in zip(range(1,6),selected,vals):pairs.append(dict(**key,rank=rank,matched_fips=fips[idx],matched_split=splits[idx],distance=value))
    return reference1,reference5,q1,q5

def feature_support(x,blocks):
    metrics=[];pairs=[];thresholds=[];ranges=[]
    for (h,day),full in x.groupby(['horizon','target_day']):
        full=full.sort_values('fipsCode').reset_index(drop=True);tr=np.flatnonzero(full.dataset=='train');te=np.flatnonzero(full.dataset=='test');allidx=np.arange(len(full))
        views={'context':CONTEXT,'history_only':{'history':HIST},'weather_only':{'weather':WEATHER},'static_only':{'static':STATIC},'source_cache_rank':blocks[str(h)]}
        for view,groups in views.items():
            z,sl=transform(full,full.iloc[tr],groups,view=='source_cache_rank')
            cache=evaluate(full,z,sl,tr,tr,tr,'train_loo',view,0,-1,h,day,metrics,pairs,thresholds)
            evaluate(full,z,sl,tr,te,tr,'test_to_train',view,0,-1,h,day,metrics,pairs,thresholds,cache)
            evaluate(full,z,sl,tr,tr,allidx,'train_plus_test',view,0,-1,h,day,metrics,pairs,thresholds,cache)
        for c in HIST+WEATHER+STATIC:
            a=full.iloc[tr][c];lo,hi=a.min(),a.max()
            for r in full.itertuples():
                v=getattr(r,c)
                ranges.append(dict(horizon=h,target_day=day,fipsCode=r.fipsCode,dataset=r.dataset,feature=c,value=v,train_min=lo,train_max=hi,outside_train_range=bool(v<lo or v>hi),
                    joint_min=full[c].min(),joint_max=full[c].max()))
    print('Full-train/test support finished',flush=True)
    for seed in SEEDS:
        cv=norm(csv(ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv')).set_index('fipsCode').fold
        for (h,day),full in x.groupby(['horizon','target_day']):
            full=full.sort_values('fipsCode').reset_index(drop=True);folds=full.fipsCode.map(cv);is_train=full.dataset=='train';is_test=full.dataset=='test'
            for fold in range(5):
                ref=np.flatnonzero(is_train&(folds!=fold));query=np.flatnonzero(is_train&(folds==fold));aug=np.flatnonzero((is_train&(folds!=fold))|is_test)
                for view,groups in [('context',CONTEXT),('source_cache_rank',blocks[str(h)])]:
                    z,sl=transform(full,full.iloc[ref],groups,view=='source_cache_rank')
                    cache=evaluate(full,z,sl,ref,query,ref,'oof_before',view,seed,fold,h,day,metrics,pairs,thresholds)
                    evaluate(full,z,sl,ref,query,aug,'oof_plus_test',view,seed,fold,h,day,metrics,pairs,thresholds,cache)
        print('OOF addition finished',seed,flush=True)
    d=pd.DataFrame(metrics);p=pd.DataFrame(pairs);t=pd.DataFrame(thresholds)
    frame(OUT/'data/support_metrics.parquet',d);frame(OUT/'data/nearest_five.parquet',p);frame(OUT/'data/reference_thresholds.csv',t);frame(OUT/'data/original_scale_ranges.parquet',ranges)
    key=['cv_seed','horizon','target_day','view','fipsCode']
    changes=[]
    for before,after in [('train_loo','train_plus_test'),('oof_before','oof_plus_test')]:
        a=d[d.scope==before];b=d[d.scope==after];g=a.merge(b,on=key,suffixes=('_before','_after'),validate='one_to_one')
        assert (g.d1_after<=g.d1_before+1e-12).all() and (g.d5_after<=g.d5_before+1e-12).all()
        np.testing.assert_array_equal(g.reference_d1_q95_before,g.reference_d1_q95_after)
        np.testing.assert_array_equal(g.reference_d5_q95_before,g.reference_d5_q95_after)
        z=g[key].copy();z['comparison']=before+'_vs_'+after
        for c in ['d1','d5','flag_d1','flag_d5','d1_percentile','d5_percentile','nearest_fips','nearest_split','test_count_in_top5']:
            z[c+'_before']=g[c+'_before'];z[c+'_after']=g[c+'_after']
        z['relative_d1_reduction_pct']=np.divide(100*(g.d1_before-g.d1_after),g.d1_before,out=np.zeros(len(g)),where=g.d1_before>0)
        z['cleared_d1_flag']=g.flag_d1_before&~g.flag_d1_after;z['cleared_d5_flag']=g.flag_d5_before&~g.flag_d5_after;changes.append(z)
    changes=pd.concat(changes,ignore_index=True);frame(OUT/'data/addition_changes.parquet',changes)
    # Reproduce old OOF distances, not old scores or future outcomes.
    old=csv(BASE/'common_error_diagnosis/matching/support.csv',dtype={'fipsCode':str})
    old=old.rename(columns={'variant':'view'});new=d[d.scope=='oof_before']
    g=new.merge(old,on=key,validate='one_to_one')
    assert len(g)==30114
    np.testing.assert_allclose(g.d1,g.nearest_distance,atol=1e-12,rtol=0)
    np.testing.assert_allclose(g.d5,g.fifth_distance,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(g.d1_percentile,g.nearest_percentile)
    return d,p,changes

def geography(names):
    coords=norm(csv(BASE/'p_information_increment/features/v1/coordinates.csv'));d=names.merge(coords,on='fipsCode',validate='one_to_one').sort_values('fipsCode').reset_index(drop=True)
    la=np.deg2rad(d.latitude.to_numpy());lo=np.deg2rad(d.longitude.to_numpy())
    a=np.sin((la[:,None]-la)/2)**2+np.cos(la[:,None])*np.cos(la)*np.sin((lo[:,None]-lo)/2)**2
    dist=2*6371.0088*np.arcsin(np.sqrt(np.clip(a,0,1)));np.fill_diagonal(dist,np.inf)
    tr=np.flatnonzero(d.dataset=='train');te=np.flatnonzero(d.dataset=='test');allidx=np.arange(len(d));own=np.sort(dist[np.ix_(tr,tr)],axis=1)
    q1,q5=np.quantile(own[:,0],.95),np.quantile(own[:,4],.95);records=[]
    for scope,queries,candidates in [('train_loo',tr,tr),('test_to_train',te,tr),('train_plus_test',tr,allidx)]:
        for i in queries:
            ix=np.lexsort((d.fipsCode.to_numpy()[candidates],dist[i,candidates]))[:5];idx=candidates[ix];v=dist[i,idx]
            records.append(dict(scope=scope,fipsCode=d.fipsCode.iloc[i],dataset=d.dataset.iloc[i],countyName=d.countyName.iloc[i],stateAbbr=d.stateAbbr.iloc[i],
                geo_d1_km=v[0],geo_d5_km=v[4],geo_d1_q95_km=q1,geo_d5_q95_km=q5,geo_flag_d1=v[0]>q1+1e-12,
                geo_d1_percentile=100*(own[:,0]<=v[0]).mean(),nearest_fips=d.fipsCode.iloc[idx[0]],nearest_split=d.dataset.iloc[idx[0]],
                test_count_in_top5=int((d.dataset.iloc[idx]=='test').sum())))
    frame(OUT/'data/geographic_support.csv',records)

def summaries(d,changes,names):
    grouped=d.groupby(['scope','cv_seed','view','fipsCode','dataset'])
    summary=grouped.agg(windows=('flag_d1','size'),flagged_d1_windows=('flag_d1','sum'),flagged_d5_windows=('flag_d5','sum'),max_d1_ratio=('d1_over_q95','max'),
        min_d1_percentile=('d1_percentile','min'),max_d1_percentile=('d1_percentile','max'),mean_d1=('d1','mean'),mean_d5=('d5','mean')).reset_index()
    summary['flagged_d1_fraction']=summary.flagged_d1_windows/summary.windows;summary['flagged_d5_fraction']=summary.flagged_d5_windows/summary.windows
    summary=summary.merge(names,on=['fipsCode','dataset'],validate='many_to_one');frame(OUT/'summary/all_county_support.csv',summary)
    tests=summary[summary.scope=='test_to_train'].sort_values(['view','flagged_d1_windows','max_d1_ratio','fipsCode'],ascending=[True,False,False,True])
    frame(OUT/'summary/test_counties.csv',tests)
    by_h=d.groupby(['scope','cv_seed','view','horizon','fipsCode']).agg(windows=('flag_d1','size'),flagged_d1_windows=('flag_d1','sum'),flagged_d5_windows=('flag_d5','sum'),max_d1_ratio=('d1_over_q95','max')).reset_index()
    frame(OUT/'summary/by_horizon.csv',by_h)
    agg=changes.groupby(['comparison','cv_seed','view','fipsCode']).agg(windows=('cleared_d1_flag','size'),before_flagged=('flag_d1_before','sum'),after_flagged=('flag_d1_after','sum'),
        before_flagged_d5=('flag_d5_before','sum'),after_flagged_d5=('flag_d5_after','sum'),cleared=('cleared_d1_flag','sum'),mean_distance_reduction_pct=('relative_d1_reduction_pct','mean'),
        test_nearest_windows=('nearest_split_after',lambda a:int((a=='test').sum()))).reset_index()
    agg=agg.merge(names,on='fipsCode',validate='many_to_one');frame(OUT/'summary/train_addition.csv',agg)
    focus=set(norm(csv(BASE/'common_error_diagnosis/summary/case_selection.csv')).fipsCode)
    frame(OUT/'summary/historical_focus_addition.csv',agg[agg.fipsCode.isin(focus)])
    return summary

def main():
    assert not (OUT/'ANALYSIS_COMPLETE.json').exists(),'Completed outputs will not be overwritten.'
    save(OUT/'config.json',dict(protocol='train_test_feature_support_v1',horizons=HS,cv_seeds=SEEDS,quantile=.95,quantile_method='linear',
        flag_tolerance=1e-12,views=['context','history_only','weather_only','static_only','source_cache_rank'],new_fits=0,draw_figures=False,
        reference_scaling='training_only_frozen_before_test_addition',history_end=71,plan_sha256=sha(OUT/'Analysis_Plan_2026-09-20.md')))
    x,b,n,source=inputs();d,p,c=feature_support(x,b);geography(n);summaries(d,c,n)
    for path,h in source.items():assert sha(path)==h,path
    save(OUT/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',input_rows=len(x),support_rows=len(d),nearest_pairs=len(p),addition_comparisons=len(c),
         source_files_unchanged=len(source),old_oof_queries_reproduced=30114,new_fits=0,figures_created=0,code_sha256=sha(__file__),
         files={str(p.relative_to(OUT)):sha(p) for folder in ['data','summary'] for p in sorted((OUT/folder).glob('*')) if p.is_file()}))
    print('Support analysis complete',len(d),'rows',flush=True)

if __name__=='__main__':main()
