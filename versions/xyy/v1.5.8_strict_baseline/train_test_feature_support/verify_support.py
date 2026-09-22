"""Independent metric and whitelist verification, using SciPy distances."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=OUT.parents[3]
H=['last_P_t','last_D_t','last_N_t','last_R_t','osi_max_72h','osi_mean_72h','osi_trend_last6h']
W=['gust_max','gust_mean','gust_hours_gt30','tp_sum','soil_mean','prior24_gust_max','prior24_tp_sum']
S=['customers_at_cutoff','pop_density','pct_forest','n_utilities']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def norm(d):d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.zfill(5);return d
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)

def independent_coordinates(full,fit,groups,rank):
    pieces=[]
    for names in groups:
        columns=[]
        for name in names:
            f=fit[name].to_numpy();v=full[name].to_numpy()
            if rank:
                if np.isnan(v).all():z=np.zeros(len(v))
                else:
                    assert np.isfinite(f).all()
                    z=np.array([-1. if np.isnan(a) else (np.count_nonzero(f<a)+.5*np.count_nonzero(f==a))/len(f) for a in v])
            else:
                if name in H:f=np.arcsinh(f/.01);v=np.arcsinh(v/.01)
                elif name in ['customers_at_cutoff','pop_density','n_utilities','tp_sum','prior24_tp_sum']:f=np.log1p(f);v=np.log1p(v)
                sd=np.std(f);z=(v-np.mean(f))/(sd if sd>1e-12 else 1)
            columns.append(z)
        pieces.append(np.column_stack(columns)/np.sqrt(len(names)*len(groups)))
    return np.column_stack(pieces)

def rebuild_inputs(x,blocks):
    checks=0
    for split in ['train','test']:
        d=norm(csv(ROOT/f'data/DM_{split.title()}.csv',usecols=['fipsCode','timestamp_et','customersTracked','P_t','N_t','D_t','R_t','gust','tp','soil_moist']))
        d['hour']=((pd.to_datetime(d.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
        if split=='test':assert d[d.hour>=72][['P_t','N_t','D_t','R_t']].isna().all().all()
        cut=d[d.hour==71].set_index('fipsCode')
        meta=norm(pd.read_parquet(ROOT/f'versions/xyy/v1.5.6/meta_{split}_v1.5.6.parquet'))
        base=pd.read_parquet(ROOT/f'versions/xyy/v1.5.6/features_{split}_v1.5.6.parquet')
        hist=base[H+S[1:]].assign(fipsCode=meta.fipsCode.to_numpy()).groupby('fipsCode').first()
        for c in ['P','N','D','R']:close(hist['last_'+c+'_t'],cut.loc[hist.index,c+'_t'])
        for h in [1,6,24,48]:
            cache=pd.read_parquet(BASE/f'p_information_increment/features/v1/F2_{split}.parquet') if h==1 else (pd.read_parquet(BASE/f'p_neighbor_horizon_increment/features/v1/h24_{split}.parquet') if h==24 else base)
            for day in range((72+h)//24,9):
                lo=max(72+h,day*24);hi=min(215,day*24+23)
                saved=x[(x.dataset==split)&(x.horizon==h)&(x.target_day==day)].set_index('fipsCode').sort_index()
                for c in hist:close(saved[c],hist.loc[saved.index,c])
                close(saved.customers_at_cutoff,cut.loc[saved.index,'customersTracked'])
                a=d[d.hour.between(lo,hi)];b=d[d.hour.between(lo-24,lo-1)];ag=a.groupby('fipsCode');bg=b.groupby('fipsCode')
                weather={'gust_max':ag.gust.max(),'gust_mean':ag.gust.mean(),'gust_hours_gt30':a.assign(v=a.gust>30).groupby('fipsCode').v.sum(),
                    'tp_sum':ag.tp.sum(),'soil_mean':ag.soil_moist.mean(),'prior24_gust_max':bg.gust.max(),'prior24_tp_sum':bg.tp.sum()}
                for c,v in weather.items():close(saved[c],v.loc[saved.index])
                mask=(meta.hour_idx+h).between(lo,hi);values=cache.loc[mask].assign(fipsCode=meta.loc[mask,'fipsCode'].to_numpy()).groupby('fipsCode').mean()
                for c in sum(blocks[str(h)].values(),[]):close(saved[c],values.loc[saved.index,c.removeprefix('cache__')])
                checks+=len(saved)
    return checks

def main():
    comp=read(OUT/'ANALYSIS_COMPLETE.json');source=read(OUT/'reference/source_hashes.json')
    for p,h in source.items():assert sha(p)==h,p
    for p,h in comp['files'].items():assert sha(OUT/p)==h,p
    assert comp['code_sha256']==sha(OUT/'analyze_support.py')
    x=pd.read_parquet(OUT/'data/allowed_inputs.parquet');blocks=read(OUT/'data/cache_blocks.json')
    rebuilt=rebuild_inputs(x,blocks);assert rebuilt==6342
    print('Allowed inputs independently rebuilt',rebuilt,flush=True)
    metrics=pd.read_parquet(OUT/'data/support_metrics.parquet');pairs=pd.read_parquet(OUT/'data/nearest_five.parquet')
    keys=['scope','cv_seed','horizon','target_day','view','fipsCode']
    assert not metrics.duplicated(keys).any()
    saved=metrics.set_index(keys).sort_index();pair=saved_pairs=pairs.set_index(keys).sort_index()
    checked=0;pair_checks=0
    for seed in [0,42,20260917,20260918]:
        cv=None if seed==0 else norm(csv(ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv')).set_index('fipsCode').fold
        for (h,day),full in x.groupby(['horizon','target_day']):
            full=full.sort_values('fipsCode').reset_index(drop=True);codes=full.fipsCode.to_numpy();which=full.dataset.to_numpy();tr=np.flatnonzero(which=='train');te=np.flatnonzero(which=='test')
            views={'context':[H,W,S],'history_only':[H],'weather_only':[W],'static_only':[S],'source_cache_rank':list(blocks[str(h)].values())}
            if seed:views={k:v for k,v in views.items() if k in ['context','source_cache_rank']}
            for fold in ([-1] if seed==0 else range(5)):
                fitidx=tr if not seed else np.flatnonzero((which=='train')&(full.fipsCode.map(cv)!=fold))
                queries=tr if not seed else np.flatnonzero((which=='train')&(full.fipsCode.map(cv)==fold))
                for view,groups in views.items():
                    z=independent_coordinates(full,full.iloc[fitidx],groups,view=='source_cache_rank')
                    dist=cdist(z,z,'euclidean');np.fill_diagonal(dist,np.inf)
                    ref=np.sort(dist[np.ix_(fitidx,fitidx)],axis=1);r1,r5=ref[:,0],ref[:,4];q1,q5=np.quantile(r1,.95),np.quantile(r5,.95)
                    scopes=[('train_loo',tr,tr),('test_to_train',te,tr),('train_plus_test',tr,np.arange(302))] if not seed else [('oof_before',queries,fitidx),('oof_plus_test',queries,np.concatenate([fitidx,te]))]
                    for scope,queryidx,candidateidx in scopes:
                        for i in queryidx:
                            key=(scope,seed,h,day,view,codes[i]);s=saved.loc[key]
                            order=np.lexsort((codes[candidateidx],dist[i,candidateidx]))[:5];idx=candidateidx[order];vals=dist[i,idx]
                            close([s.d1,s.d5,s.reference_d1_q95,s.reference_d5_q95],[vals[0],vals[4],q1,q5])
                            assert s.flag_d1==bool(vals[0]>q1+1e-12) and s.flag_d5==bool(vals[4]>q5+1e-12)
                            # C and numpy summation may differ at exact near ties; bound the empirical rank with the declared tolerance.
                            for arr,v,pct in [(r1,vals[0],s.d1_percentile),(r5,vals[4],s.d5_percentile)]:
                                assert 100*(arr<v-1e-12).mean()-1e-10<=pct<=100*(arr<=v+1e-12).mean()+1e-10
                            assert s.fit_counties==len(fitidx) and s.candidate_count==len(candidateidx)-int(i in candidateidx)
                            nearest_index=np.flatnonzero(codes==s.nearest_fips)[0];assert nearest_index in candidateidx and nearest_index!=i
                            close(dist[i,nearest_index],vals[0]);assert s.nearest_split==which[nearest_index]
                            if view in ['context','source_cache_rank']:
                                pp=pair.loc[key].sort_values('rank');positions=np.array([np.flatnonzero(codes==f)[0] for f in pp.matched_fips])
                                assert len(pp)==5 and len(set(positions))==5 and i not in positions
                                assert set(positions).issubset(set(candidateidx))
                                close(pp.distance,dist[i,positions]);close(np.sort(pp.distance),vals)
                                assert (pp.matched_split.to_numpy()==which[positions]).all()
                                assert s.test_count_in_top5==(which[positions]=='test').sum()
                                pair_checks+=5
                            checked+=1
        print('Distances and references PASS',seed,flush=True)
    assert checked==len(metrics)==117033 and pair_checks==len(pairs)==414750
    changes=pd.read_parquet(OUT/'data/addition_changes.parquet')
    for (name,seed),g in changes.groupby(['comparison','cv_seed']):
        assert (g.d1_after<=g.d1_before+1e-12).all() and (g.d5_after<=g.d5_before+1e-12).all()
        assert np.array_equal(g.cleared_d1_flag,g.flag_d1_before&~g.flag_d1_after)
        close(g.relative_d1_reduction_pct,np.divide(100*(g.d1_before-g.d1_after),g.d1_before,out=np.zeros(len(g)),where=g.d1_before>0))
    summary=csv(OUT/'summary/all_county_support.csv',dtype={'fipsCode':str}).set_index(['scope','cv_seed','view','fipsCode','dataset'])
    group=metrics.groupby(['scope','cv_seed','view','fipsCode','dataset'])
    for col,outcol in [('flag_d1','flagged_d1_windows'),('flag_d5','flagged_d5_windows')]:
        v=group[col].sum();close(summary.loc[v.index,outcol],v)
    for p,h in source.items():assert sha(p)==h,p
    result=dict(status='PASS',new_model_fits=0,figures_created=0,source_files_unchanged=len(source),input_windows_rebuilt=rebuilt,
        distance_rows_checked=checked,nearest_pairs_checked=pair_checks,preprocessing_training_only=True,
        self_and_outer_fold_exclusion=True,fixed_threshold_before_after=True,addition_monotonicity=True,
        verifier_sha256=sha(__file__),absolute_tolerance=1e-12)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':main()
