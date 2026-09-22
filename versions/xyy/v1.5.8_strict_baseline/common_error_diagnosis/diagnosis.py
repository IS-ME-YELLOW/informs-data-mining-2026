"""Read-only county diagnostics; no fitting, prediction changes or model imports."""
from pathlib import Path
import hashlib
import json
import itertools
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]
SEEDS=(42,20260917,20260918);HS=(1,6,24,48);CS=('P','N','D','R');WEIGHTS=(.4,.35,.25,-.1)
HIST=['last_P_t','last_D_t','last_N_t','last_R_t','osi_max_72h','osi_mean_72h','osi_trend_last6h']
WEATHER=['gust_max','gust_mean','gust_hours_gt30','tp_sum','soil_mean','prior24_gust_max','prior24_tp_sum']
STATIC=['customers_at_cutoff','pop_density','pct_forest','n_utilities']
HISTORICAL=('42053','39117','18013','54015','54013','39115','42131')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def safe(p):
    p=Path(p).resolve();assert p.is_relative_to(OUT) and p!=OUT
    p.parent.mkdir(parents=True,exist_ok=True);return p
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def save(p,v):safe(p).write_text(json.dumps(clean(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def frame(p,d):
    d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d);p=safe(p)
    if p.suffix=='.parquet':d.to_parquet(p,index=False)
    else:d.to_csv(p,index=False)
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def norm(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    return d
def tag(h):return f'osi_target_t{h:02d}h'
def tree_path(seed):
    m=read(BASE/'p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json')
    return Path(next(r['run_directory'] for r in m['splits'] if r['split_seed']==seed))
def stella_path(seed):return ROOT/f'versions/stella_v112_strict/runs/stella_v112_nested_v1_split{seed}_model42'

def initialize():
    assert not (OUT/'ANALYSIS_COMPLETE.json').exists(),'Completed analysis is immutable; use a new directory.'
    source=[]
    def add(p):source.append(Path(p).resolve())
    integration=BASE/'tree_stella_integration'
    for name,h in read(integration/'completion.json')['artifacts_sha256'].items():
        assert sha(integration/name)==h, name
    add(integration/'completion.json');add(integration/'summary/route_candidates.json')
    add(BASE/'p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json')
    for s in SEEDS:
        for n in ['source_predictions.parquet','run_manifest.json','verification.json','EVALUATION_COMPLETE']:
            add(integration/f'runs/split{s}'/n)
        for n in ['component_predictions.parquet','aligned_unique_components.parquet','VERIFIED_COMPLETE','run_manifest.json','EVALUATION_COMPLETE']:
            add(tree_path(s)/n)
        for n in ['candidate_predictions.parquet','CV_COMPLETE','run_manifest.json','verification.json']:
            add(stella_path(s)/n)
        add(ROOT/f'cv/cv_assignments_balanced_v1_seed{s}.csv')
        # Check saved source receipts without invoking old scripts or writing to sources.
        for p,marker in [(tree_path(s),'EVALUATION_COMPLETE'),(stella_path(s),'CV_COMPLETE')]:
            m=read(p/marker)
            for n in ['component_predictions.parquet','aligned_unique_components.parquet','candidate_predictions.parquet']:
                if n in m.get('files',{}):assert sha(p/n)==m['files'][n]
    for p in [ROOT/'data/DM_Train.csv',ROOT/'versions/xyy/v1.5.6/features_train_v1.5.6.parquet',
              ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet',
              BASE/'p_information_increment/features/v1/F2_train.parquet',
              BASE/'p_neighbor_horizon_increment/features/v1/h24_train.parquet']:
        add(p)
    manifest={str(p):sha(p) for p in sorted(set(source))}
    path=OUT/'reference/source_hashes.json'
    if path.exists():assert read(path)==manifest
    else:save(path,manifest)
    save(OUT/'diagnostic_config.json',dict(protocol='common_error_diagnosis_v1',seeds=SEEDS,horizons=HS,
         tree_case='NB_P24',stella_case='L2',common_horizons=[24,48],common_top_counties=5,short_top_counties=3,
         common_score='same_sign * min(eT^2,eS^2)',history_end=71,matching_k=5,
         matching_variants=['context','source_cache_rank'],whole_outer_fold_excluded=True,
         history_columns=HIST,weather_columns=WEATHER,static_columns=STATIC,
         low_loss_peak_max=.01,high_loss_peak_min=.05,new_fits=0,model_reloads=0,
         plan_sha256=sha(OUT/'Diagnosis_Plan_2026-09-20.md')))
    return manifest

def raw_data():
    d=norm(csv(ROOT/'data/DM_Train.csv'));d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    d['target_hour']=((d.timestamp_et-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
    assert len(d)==239*216 and not d.duplicated(['fipsCode','target_hour']).any()
    return d.sort_values(['fipsCode','target_hour']).reset_index(drop=True)

def build_rows(raw):
    rows=[];sixrows=[]
    truth=raw[['fipsCode','target_hour','osi','P_t','N_t','D_t','R_t','outageCount','customersTracked','gust','tp','soil_moist']]
    for s in SEEDS:
        source=norm(pd.read_parquet(BASE/f'tree_stella_integration/runs/split{s}/source_predictions.parquet'))
        parts=norm(pd.read_parquet(tree_path(s)/'component_predictions.parquet')).set_index(['fipsCode','hour_idx'])
        aligned=norm(pd.read_parquet(tree_path(s)/'aligned_unique_components.parquet'))
        aligned['component']=aligned.component.str.removesuffix('_t')
        aligned=aligned[aligned['case']=='NB_P24'].pivot(index=['fipsCode','target_timestamp'],columns='component',values='prediction').reset_index()
        aligned['target_hour']=((pd.to_datetime(aligned.target_timestamp)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
        aligned=aligned.set_index(['fipsCode','target_hour'])
        stella=norm(pd.read_parquet(stella_path(s)/'candidate_predictions.parquet')).set_index(['fipsCode','hour_idx'])
        for h in HS:
            t=tag(h);d=source[source['scoreable_'+t]].copy()
            g=d[['cv_seed','fipsCode','countyName','stateAbbr','fold','hour_idx']].copy()
            g['horizon']=h;g['target_hour']=g.hour_idx+h;g['target_day']=g.target_hour//24
            g['truth']=d['truth_'+t].to_numpy();g['T']=d['T_'+t].to_numpy()
            g['S']=d['S_'+t].to_numpy() if h>=24 else np.nan
            g=g.merge(truth,on=['fipsCode','target_hour'],validate='one_to_one')
            np.testing.assert_allclose(g.truth,g.osi,atol=1e-12,rtol=0)
            g['eT']=g['T']-g.truth
            g['eS']=g.S-g.truth;g['T_sse']=g.eT**2;g['S_sse']=g.eS**2
            g['direction']=np.select([(g.eT<0)&(g.eS<0),(g.eT>0)&(g.eS>0),g.eT*g.eS<0],['both_under','both_over','opposite'],default='zero_error') if h>=24 else 'short_T_only'
            g['common_score']=np.where(g.eT*g.eS>0,np.minimum(g.T_sse,g.S_sse),0) if h>=24 else np.nan
            pred=[]
            for c,w in zip(CS,WEIGHTS):
                if h<=6:
                    p=aligned[c].reindex(pd.MultiIndex.from_frame(g[['fipsCode','target_hour']])).to_numpy()
                else:
                    p=parts[f'pred_NB_P24_{c}_t_target_t{h:02d}h'].reindex(pd.MultiIndex.from_frame(g[['fipsCode','hour_idx']])).to_numpy()
                    p=np.clip(p,0,1)
                assert np.isfinite(p).all();g[c+'_pred']=p;g[c+'_true']=g[c+'_t'];g['e_'+c]=w*(p-g[c+'_t']);pred.append(w*p)
            z=sum(pred);post=np.where(np.clip(z,0,.65)<.001,0,np.clip(z,0,.65))
            np.testing.assert_allclose(post,g['T'],atol=1e-12,rtol=0)
            g['e_remainder']=g.eT-g[['e_'+c for c in CS]].sum(axis=1)
            rows.append(g)
            if h>=24:
                st=stella.reindex(pd.MultiIndex.from_frame(g[['fipsCode','hour_idx']]))
                q=g[['cv_seed','fipsCode','hour_idx','horizon','target_hour','target_day','truth','T','S','eT','eS','common_score','direction']].copy()
                source_names=['lgbm_direct','lgbm_component','xgboost_direct','catboost_direct','xgboost_component','catboost_component']
                for n in source_names:q[n]=st['processed_'+n+'_'+t].to_numpy()
                q['L0']=st['pred_L0_'+t].to_numpy();q['L1']=st['pred_L1_'+t].to_numpy()
                q['theta']=st['theta_'+t].to_numpy();q['gate']=st['anchor_gt_theta_'+t].to_numpy()
                q['sources_under']=(q[source_names].to_numpy()<q.truth.to_numpy()[:,None]).sum(axis=1)
                q['sources_over']=(q[source_names].to_numpy()>q.truth.to_numpy()[:,None]).sum(axis=1)
                sixrows.append(q)
    rows=pd.concat(rows,ignore_index=True);six=pd.concat(sixrows,ignore_index=True)
    assert len(rows)==3*(34177+32982+28680+22944)
    frame(OUT/'data/route_rows.parquet',rows);frame(OUT/'data/stella_source_rows.parquet',six)
    return rows,six

def summarize(g):
    n=len(g);r=dict(n=n,T_sse=g.T_sse.sum(),T_rmse=np.sqrt(g.T_sse.mean()),T_bias=g.eT.mean(),truth_mean=g.truth.mean(),truth_max=g.truth.max())
    if g.horizon.iloc[0]>=24:
        r.update(S_sse=g.S_sse.sum(),S_rmse=np.sqrt(g.S_sse.mean()),S_bias=g.eS.mean(),common_score=g.common_score.sum(),
            under_common_score=g.loc[g.direction=='both_under','common_score'].sum(),over_common_score=g.loc[g.direction=='both_over','common_score'].sum())
        for direction in ['both_under','both_over','opposite','zero_error']:
            sel=g.direction==direction;r[direction+'_n']=sel.sum()
            for source in ['T','S']:r[direction+'_'+source+'_sse']=g.loc[sel,source+'_sse'].sum()
    return r

def aggregate(rows,cols):
    return pd.DataFrame([dict(zip(cols,key if isinstance(key,tuple) else (key,)),**summarize(g)) for key,g in rows.groupby(cols,sort=True)])

def summaries(rows,six):
    keys=['cv_seed','horizon']
    names={r.fipsCode:(r.countyName,r.stateAbbr) for r in rows.drop_duplicates('fipsCode').itertuples()}
    tables={}
    for name,extra in [('overall',[]),('county',['fipsCode']),('day',['target_day']),('county_day',['fipsCode','target_day'])]:
        d=aggregate(rows,keys+extra)
        if 'fipsCode' in d:
            d['countyName']=d.fipsCode.map(lambda x:names[x][0]);d['stateAbbr']=d.fipsCode.map(lambda x:names[x][1])
        if name=='county':
            d['rank']=0
            for (s,h),g in d.groupby(keys):
                rank=g.sort_values(['common_score' if h>=24 else 'T_sse','fipsCode'],ascending=[False,True]).index
                d.loc[rank,'rank']=np.arange(1,len(g)+1)
        frame(OUT/f'summary/{name}_metrics.csv',d);tables[name]=d
    selected=[]
    for h in HS:
        g=tables['county'].query('cv_seed==42 and horizon==@h').sort_values('rank').head(5 if h>=24 else 3)
        for r in g.itertuples():selected.append(dict(fipsCode=r.fipsCode,countyName=r.countyName,stateAbbr=r.stateAbbr,horizon=h,rank=r.rank,reason='common_top5' if h>=24 else 'short_top3'))
    for f in HISTORICAL:selected.append(dict(fipsCode=f,countyName=names[f][0],stateAbbr=names[f][1],horizon=0,rank=0,reason='historical_only'))
    selection=pd.DataFrame(selected);frame(OUT/'summary/case_selection.csv',selection)
    focus=set(selection.fipsCode)
    frame(OUT/'summary/case_stability.csv',tables['county'][tables['county'].fipsCode.isin(focus)])
    frame(OUT/'data/case_trajectories.parquet',rows[rows.fipsCode.isin(focus)])
    st=[]
    for key,g in six.groupby(['cv_seed','horizon','fipsCode']):
        st.append(dict(zip(['cv_seed','horizon','fipsCode'],key),n=len(g),gate_n=g.gate.sum(),
            common_score=g.common_score.sum(),common_score_all6_under=g.loc[g.sources_under==6,'common_score'].sum(),
            common_score_all6_over=g.loc[g.sources_over==6,'common_score'].sum(),
            common_score_gate=g.loc[g.gate,'common_score'].sum(),
            truth_peak=g.truth.max(),source_max_at_truth_peak=g.loc[g.truth.idxmax(),['lgbm_direct','lgbm_component','xgboost_direct','catboost_direct','xgboost_component','catboost_component']].max()))
    frame(OUT/'summary/stella_county_diagnostics.csv',st)
    decomp=[];terms=['e_'+c for c in CS]+['e_remainder']
    for key,g in rows.groupby(['cv_seed','horizon','fipsCode','target_day']):
        r=dict(zip(['cv_seed','horizon','fipsCode','target_day'],key),n=len(g),T_sse=g.T_sse.sum())
        for a in terms:r[a+'_squared']=float((g[a]**2).sum())
        for a,b in itertools.combinations(terms,2):r[a+'_x_'+b]=float((2*g[a]*g[b]).sum())
        calc=sum(v for k,v in r.items() if k.endswith('_squared') or '_x_' in k)
        assert abs(calc-r['T_sse'])<1e-12
        decomp.append(r)
    frame(OUT/'summary/component_error_decomposition.csv',decomp)
    shapes=[]
    for key,g in rows[rows.fipsCode.isin(focus)].groupby(['cv_seed','horizon','fipsCode','target_day']):
        g=g.sort_values('target_hour');r=dict(zip(['cv_seed','horizon','fipsCode','target_day'],key),n=len(g))
        for c in ['truth','T','S','P_true','P_pred','N_true','N_pred','D_true','D_pred','R_true','R_pred']:
            a=g[c];r[c+'_mean']=a.mean();r[c+'_max']=a.max()
            r[c+'_peak_hour']=g.loc[a.idxmax(),'target_hour'] if a.notna().any() else np.nan
            r[c+'_net_change']=a.iloc[-1]-a.iloc[0]
        r['P_true_hours_gt05']=(g.P_true>.05).sum();r['P_pred_hours_gt05']=(g.P_pred>.05).sum()
        r['gust_max']=g.gust.max();r['customers_min']=g.customersTracked.min();r['customers_max']=g.customersTracked.max()
        shapes.append(r)
    frame(OUT/'summary/case_daily_shapes.csv',shapes)
    return selection,tables

def build_match_inputs(raw):
    meta=norm(pd.read_parquet(ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet'))
    x=pd.read_parquet(ROOT/'versions/xyy/v1.5.6/features_train_v1.5.6.parquet')
    assert len(x)==len(meta)==34416
    history=x[HIST+['pop_density','pct_forest','n_utilities']].copy();history['fipsCode']=meta.fipsCode.to_numpy()
    assert history.groupby('fipsCode').nunique().max().max()==1
    history=history.groupby('fipsCode').first()
    history['customers_at_cutoff']=raw[raw.target_hour==71].set_index('fipsCode').customersTracked
    static_names=['log_customers','rural_urban_code','is_metro','pop_density','n_utilities','pct_forest','pct_developed','pct_agriculture','pct_water','pct_wetland','pct_unmapped','pct_forest_classified','tree_canopy_pct','tree_canopy_std','forest_x_customers']
    blocks={};allrows=[]
    for h in HS:
        cache=pd.read_parquet(BASE/'p_information_increment/features/v1/F2_train.parquet') if h==1 else (pd.read_parquet(BASE/'p_neighbor_horizon_increment/features/v1/h24_train.parquet') if h==24 else x.copy())
        assert len(cache)==len(meta) and set(x.columns).issubset(cache.columns)
        group={'cache_history':list(x.columns[:31]),'cache_weather_time':[c for c in x if c not in list(x.columns[:31])+static_names],'cache_static':static_names}
        extra=[c for c in cache if c not in x]
        if extra:group['cache_neighbors']=extra
        blocks[str(h)]={k:['cache__'+c for c in cols] for k,cols in group.items()}
        for day in range((72+h)//24,9):
            start=max(72+h,day*24);end=min(215,day*24+23)
            weather=raw[raw.target_hour.between(start,end)].copy();weather['gt30']=weather.gust>30
            a=weather.groupby('fipsCode').agg(gust_max=('gust','max'),gust_mean=('gust','mean'),gust_hours_gt30=('gt30','sum'),tp_sum=('tp','sum'),soil_mean=('soil_moist','mean'))
            prior=raw[raw.target_hour.between(start-24,start-1)].groupby('fipsCode').agg(prior24_gust_max=('gust','max'),prior24_tp_sum=('tp','sum'))
            d=history.join(a).join(prior)
            mask=(meta.hour_idx+h).between(start,end)
            xx=cache.loc[mask].copy();xx['fipsCode']=meta.loc[mask,'fipsCode'].to_numpy()
            d=d.join(xx.groupby('fipsCode').mean().add_prefix('cache__'))
            d['horizon']=h;d['target_day']=day;d['start']=start;d['end']=end
            allrows.append(d.reset_index())
    inputs=pd.concat(allrows,ignore_index=True)
    assert np.isfinite(inputs[HIST+WEATHER+STATIC]).all().all()
    frame(OUT/'matching/inputs.parquet',inputs);save(OUT/'matching/cache_blocks.json',blocks)
    return inputs,blocks

def match_all(inputs,blocks):
    rankings=[];supports=[];range_rows=[]
    focus=set(csv(OUT/'summary/case_selection.csv',dtype={'fipsCode':str}).fipsCode)
    for s in SEEDS:
        folds=norm(csv(ROOT/f'cv/cv_assignments_balanced_v1_seed{s}.csv')).set_index('fipsCode').fold
        for (h,day),d in inputs.groupby(['horizon','target_day']):
            d=d.sort_values('fipsCode').reset_index(drop=True);d['fold']=d.fipsCode.map(folds)
            for fold in range(5):
                ref=d[d.fold!=fold].reset_index(drop=True);query=d[d.fold==fold].reset_index(drop=True)
                for variant,groups in [('context',{'history':HIST,'weather':WEATHER,'static':STATIC}),('source_cache_rank',blocks[str(h)])]:
                    cn=[];qn=[]
                    for names in groups.values():
                        c=ref[names].to_numpy(dtype=float,copy=True);q=query[names].to_numpy(dtype=float,copy=True)
                        if variant=='source_cache_rank':
                            for j in range(len(names)):
                                if np.isnan(c[:,j]).all() and np.isnan(q[:,j]).all():
                                    c[:,j]=0.;q[:,j]=0.;continue
                                assert np.isfinite(c[:,j]).all() and np.isfinite(q[:,j]).all(),names[j]
                                z=np.sort(c[:,j].copy())
                                c[:,j]=(np.searchsorted(z,c[:,j],'left')+np.searchsorted(z,c[:,j],'right'))/(2*len(z))
                                q[:,j]=(np.searchsorted(z,q[:,j],'left')+np.searchsorted(z,q[:,j],'right'))/(2*len(z))
                        else:
                            assert np.isfinite(c).all() and np.isfinite(q).all()
                            for j,name in enumerate(names):
                                if name in HIST:c[:,j]=np.arcsinh(c[:,j]/.01);q[:,j]=np.arcsinh(q[:,j]/.01)
                                elif name in ['customers_at_cutoff','pop_density','n_utilities','tp_sum','prior24_tp_sum']:c[:,j]=np.log1p(c[:,j]);q[:,j]=np.log1p(q[:,j])
                            mu=c.mean(axis=0);sd=c.std(axis=0);sd=np.where(sd>1e-12,sd,1)
                            c=(c-mu)/sd;q=(q-mu)/sd
                        scale=np.sqrt(len(names)*len(groups));cn.append(c/scale);qn.append(q/scale)
                    c=np.concatenate(cn,axis=1);q=np.concatenate(qn,axis=1)
                    dist=np.sqrt(np.maximum(((q[:,None,:]-c[None,:,:])**2).sum(axis=2),0))
                    cc=np.sqrt(np.maximum(((c[:,None,:]-c[None,:,:])**2).sum(axis=2),0));np.fill_diagonal(cc,np.inf);loo=cc.min(axis=1)
                    for i,r in enumerate(query.itertuples()):
                        idx=np.lexsort((ref.fipsCode.to_numpy(),dist[i]))[:5]
                        key=dict(cv_seed=s,horizon=h,target_day=day,fipsCode=r.fipsCode,excluded_fold=fold,variant=variant)
                        supports.append(dict(**key,pool_n=len(ref),nearest_distance=dist[i,idx[0]],nearest_percentile=100*(loo<=dist[i,idx[0]]).mean(),fifth_distance=dist[i,idx[-1]]))
                        for rank,j in enumerate(idx,1):rankings.append(dict(**key,rank=rank,matched_fips=ref.fipsCode.iloc[j],matched_fold=ref.fold.iloc[j],distance=dist[i,j]))
                        if variant=='context' and r.fipsCode in focus:
                            for name in HIST+WEATHER+STATIC:
                                lo,hi=ref[name].min(),ref[name].max();value=getattr(r,name)
                                range_rows.append(dict(**key,feature=name,value=value,train_min=lo,train_max=hi,outside=bool(value<lo or value>hi)))
        print('Matching finished',s,flush=True)
    ranks=pd.DataFrame(rankings);support=pd.DataFrame(supports)
    assert (ranks.matched_fold!=ranks.excluded_fold).all()
    frame(OUT/'matching/rankings.parquet',ranks);frame(OUT/'matching/support.csv',support);frame(OUT/'matching/focus_feature_ranges.csv',range_rows)
    save(OUT/'matching/RANKINGS_FROZEN.json',dict(status='FROZEN_BEFORE_OUTCOME_JOIN',files={n:sha(OUT/'matching'/n) for n in ['inputs.parquet','cache_blocks.json','rankings.parquet','support.csv','focus_feature_ranges.csv']},outcomes_used_for_ranking=False))
    return ranks,support

def connect_outcomes(rows,ranks):
    # Separate stage: receives frozen ranks and future outcomes only after rank hashes exist.
    marker=read(OUT/'matching/RANKINGS_FROZEN.json')
    for n,h in marker['files'].items():assert sha(OUT/'matching'/n)==h
    groups=['cv_seed','horizon','target_day','fipsCode']
    outcomes=[]
    for key,g in rows.groupby(groups):
        outcomes.append(dict(zip(groups,key),osi_peak=g.truth.max(),osi_mean=g.truth.mean(),P_peak=g.P_true.max(),P_mean=g.P_true.mean(),
            N_peak=g.N_true.max(),R_mean=g.R_true.mean(),D_mean=g.D_true.mean(),P_hours_gt05=(g.P_true>.05).sum(),
            T_rmse=np.sqrt(g.T_sse.mean()),S_rmse=np.sqrt(g.S_sse.mean()) if g.horizon.iloc[0]>=24 else np.nan))
    outcome=pd.DataFrame(outcomes);frame(OUT/'matching/window_outcomes.csv',outcome)
    o=outcome.rename(columns={'fipsCode':'matched_fips',**{c:'control_'+c for c in outcome if c not in groups}})
    joined=ranks.merge(o,on=['cv_seed','horizon','target_day','matched_fips'],validate='many_to_one')
    o=outcome.rename(columns={c:'query_'+c for c in outcome if c not in groups})
    joined=joined.merge(o,on=groups,validate='many_to_one')
    joined['control_low_loss']=joined.control_osi_peak<=.01;joined['control_high_loss']=joined.control_osi_peak>=.05
    frame(OUT/'matching/ranked_controls_with_outcomes.parquet',joined)
    summary=joined.groupby(groups+['variant']).agg(low_loss_controls=('control_low_loss','sum'),high_loss_controls=('control_high_loss','sum'),
       control_max_osi_peak=('control_osi_peak','max'),control_mean_osi_peak=('control_osi_peak','mean'),control_max_osi_mean=('control_osi_mean','max'),
       query_osi_peak=('query_osi_peak','first'),query_osi_mean=('query_osi_mean','first'),query_P_mean=('query_P_mean','first'),
       control_mean_P_mean=('control_P_mean','mean'),control_max_P_mean=('control_P_mean','max')).reset_index()
    frame(OUT/'matching/control_summary.csv',summary)

def main():
    manifest=initialize();raw=raw_data();rows,six=build_rows(raw);selection,tables=summaries(rows,six)
    print('Frozen selected cases:',selection.to_dict('records'),flush=True)
    inputs,blocks=build_match_inputs(raw);ranks,support=match_all(inputs,blocks);connect_outcomes(rows,ranks)
    for p,h in manifest.items():assert sha(p)==h,p
    save(OUT/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',new_fits=0,source_files_unchanged=len(manifest),
         row_count=len(rows),stella_row_count=len(six),ranking_rows=len(ranks),support_rows=len(support),
         code_sha256=sha(__file__),files={str(p.relative_to(OUT)):sha(p) for d in ['data','summary','matching'] for p in sorted((OUT/d).glob('*')) if p.is_file()}))
    print('Analysis complete',len(rows),len(ranks),flush=True)

if __name__=='__main__':main()
