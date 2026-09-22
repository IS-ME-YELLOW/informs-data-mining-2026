"""Independent source-keyed, arithmetic and outcome-blind matching verification."""
from pathlib import Path
import hashlib,json,math,itertools
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def codes(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.zfill(5);return d
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)

def verify():
    manifest=read(OUT/'reference/source_hashes.json');complete=read(OUT/'ANALYSIS_COMPLETE.json');cfg=read(OUT/'diagnostic_config.json')
    for p,h in manifest.items():assert sha(p)==h,p
    for p,h in complete['files'].items():assert sha(OUT/p)==h,p
    assert complete['code_sha256']==sha(OUT/'diagnosis.py')
    assert cfg['plan_sha256']==sha(OUT/'Diagnosis_Plan_2026-09-20.md')
    d=pd.read_parquet(OUT/'data/route_rows.parquet');six=pd.read_parquet(OUT/'data/stella_source_rows.parquet')
    keys=['cv_seed','horizon','fipsCode','hour_idx'];assert not d.duplicated(keys).any()
    raw=codes(csv(ROOT/'data/DM_Train.csv'));raw['target_hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
    raw=raw.set_index(['fipsCode','target_hour']).sort_index()
    official=raw.reindex(pd.MultiIndex.from_frame(d[['fipsCode','target_hour']]))
    close(d.truth,official.osi)
    for c in ['P','N','D','R']:close(d[c+'_true'],official[c+'_t'])
    close(d.eT,d['T']-d.truth);close(d.eS,d['S']-d.truth);close(d.T_sse,d.eT**2);close(d.S_sse,d.eS**2)
    mask=d.horizon>=24
    close(d.loc[mask,'common_score'],np.where(d.loc[mask,'eT']*d.loc[mask,'eS']>0,np.minimum(d.loc[mask,'T_sse'],d.loc[mask,'S_sse']),0))
    expected_dir=np.select([(d.eT<0)&(d.eS<0),(d.eT>0)&(d.eS>0),d.eT*d.eS<0],['both_under','both_over','opposite'],default='zero_error')
    assert (d.loc[mask,'direction'].to_numpy()==expected_dir[mask]).all()
    assert d.loc[~mask,'S'].isna().all() and (d.loc[~mask,'direction']=='short_T_only').all()
    ti=read(BASE/'p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json')
    for s in cfg['seeds']:
        cv=codes(csv(ROOT/f'cv/cv_assignments_balanced_v1_seed{s}.csv')).set_index('fipsCode').fold
        run=Path(next(x['run_directory'] for x in ti['splits'] if x['split_seed']==s))
        comp=codes(pd.read_parquet(run/'component_predictions.parquet'))
        expanded=[]
        for h in [1,6,24,48]:
            g=comp[comp.hour_idx+h<=215].copy();g['target_hour']=g.hour_idx+h
            z=g[['fipsCode','target_hour']].copy()
            for c in ['P','N','D','R']:z[c]=np.clip(g[f'pred_NB_P24_{c}_t_target_t{h:02d}h'],0,1)
            z['source_horizon']=h;expanded.append(z)
        expanded=pd.concat(expanded,ignore_index=True);aligned=expanded.groupby(['fipsCode','target_hour'])[['P','N','D','R']].mean()
        source=codes(pd.read_parquet(BASE/f'tree_stella_integration/runs/split{s}/source_predictions.parquet')).set_index(['fipsCode','hour_idx'])
        for h,n in [(1,34177),(6,32982),(24,28680),(48,22944)]:
            g=d[(d.cv_seed==s)&(d.horizon==h)];assert len(g)==n and g.fipsCode.nunique()==239
            assert g.target_hour.between(72+h,215).all() and (g.target_hour==g.hour_idx+h).all()
            assert (g.fold.to_numpy()==g.fipsCode.map(cv).to_numpy()).all()
            ss=source.reindex(pd.MultiIndex.from_frame(g[['fipsCode','hour_idx']]))
            np.testing.assert_array_equal(g['T'],ss[f'T_osi_target_t{h:02d}h'])
            if h>=24:np.testing.assert_array_equal(g.S,ss[f'S_osi_target_t{h:02d}h'])
            parts=aligned if h<=6 else expanded[expanded.source_horizon==h].set_index(['fipsCode','target_hour'])
            pp=parts.reindex(pd.MultiIndex.from_frame(g[['fipsCode','target_hour']]))
            for c,w in zip(['P','N','D','R'],[.4,.35,.25,-.1]):
                close(g[c+'_pred'],pp[c]);close(g['e_'+c],w*(g[c+'_pred']-g[c+'_true']))
            z=.4*pp.P+.35*pp.N+.25*pp.D-.1*pp.R
            close(g['T'],np.where(np.clip(z,0,.65)<.001,0,np.clip(z,0,.65)))
    selected=csv(OUT/'summary/case_selection.csv',dtype={'fipsCode':str})
    county=csv(OUT/'summary/county_metrics.csv',dtype={'fipsCode':str})
    for h,k in [(1,3),(6,3),(24,5),(48,5)]:
        g=d[(d.cv_seed==42)&(d.horizon==h)].groupby('fipsCode')['common_score' if h>=24 else 'T_sse'].sum().reset_index()
        g=g.sort_values([g.columns[-1],'fipsCode'],ascending=[False,True]).head(k)
        assert g.fipsCode.tolist()==selected[selected.horizon==h].sort_values('rank').fipsCode.tolist()
    metric_checks=0
    for name,cols in [('overall',['cv_seed','horizon']),('county',['cv_seed','horizon','fipsCode']),('day',['cv_seed','horizon','target_day']),('county_day',['cv_seed','horizon','fipsCode','target_day'])]:
        saved=csv(OUT/f'summary/{name}_metrics.csv',dtype={'fipsCode':str} if 'fipsCode' in cols else None).set_index(cols)
        sums=d.groupby(cols)[['T_sse','S_sse','common_score']].sum(min_count=1)
        counts=d.groupby(cols).size();assert len(saved)==len(sums)
        for c in sums:close(saved.loc[sums.index,c],sums[c])
        close(saved.loc[sums.index,'n'],counts);close(saved.loc[sums.index,'T_rmse'],np.sqrt(sums.T_sse/counts))
        metric_checks+=len(saved)
    decomp=csv(OUT/'summary/component_error_decomposition.csv',dtype={'fipsCode':str})
    terms=[c for c in decomp if c.endswith('_squared') or '_x_' in c]
    close(decomp[terms].sum(axis=1),decomp.T_sse)
    group=d.groupby(['cv_seed','horizon','fipsCode','target_day'])
    terms_in=['e_P','e_N','e_D','e_R','e_remainder'];dec=decomp.set_index(['cv_seed','horizon','fipsCode','target_day'])
    for a in terms_in:
        sums=(d.assign(value=d[a]**2).groupby(['cv_seed','horizon','fipsCode','target_day']).value.sum())
        close(dec.loc[sums.index,a+'_squared'],sums)
    for a,b in itertools.combinations(terms_in,2):
        sums=d.assign(value=2*d[a]*d[b]).groupby(['cv_seed','horizon','fipsCode','target_day']).value.sum()
        close(dec.loc[sums.index,a+'_x_'+b],sums)
    for (s,h),g in six.groupby(['cv_seed','horizon']):
        saved=codes(pd.read_parquet(ROOT/f'versions/stella_v112_strict/runs/stella_v112_nested_v1_split{s}_model42/candidate_predictions.parquet')).set_index(['fipsCode','hour_idx'])
        saved=saved.reindex(pd.MultiIndex.from_frame(g[['fipsCode','hour_idx']]))
        sn=['lgbm_direct','lgbm_component','xgboost_direct','catboost_direct','xgboost_component','catboost_component']
        for name in sn:np.testing.assert_array_equal(g[name],saved[f'processed_{name}_osi_target_t{h:02d}h'])
        np.testing.assert_array_equal(g.sources_under,(g[sn].to_numpy()<g.truth.to_numpy()[:,None]).sum(axis=1))
        np.testing.assert_array_equal(g.gate,saved[f'anchor_gt_theta_osi_target_t{h:02d}h'])
    print('Rows, sources, component reconstruction and metrics PASS',flush=True)
    inputs=pd.read_parquet(OUT/'matching/inputs.parquet');blocks=read(OUT/'matching/cache_blocks.json')
    ranks=pd.read_parquet(OUT/'matching/rankings.parquet');support=csv(OUT/'matching/support.csv',dtype={'fipsCode':str})
    freeze=read(OUT/'matching/RANKINGS_FROZEN.json')
    for n,h in freeze['files'].items():assert sha(OUT/'matching'/n)==h
    assert (ranks.matched_fold!=ranks.excluded_fold).all() and (ranks.fipsCode!=ranks.matched_fips).all()
    ix=['cv_seed','horizon','target_day','excluded_fold','variant','fipsCode']
    indexed=ranks.set_index(ix).sort_index();sup=support.set_index(ix).sort_index();match_checks=0
    for s in cfg['seeds']:
        cv=codes(csv(ROOT/f'cv/cv_assignments_balanced_v1_seed{s}.csv')).set_index('fipsCode').fold
        for (h,day),full in inputs.groupby(['horizon','target_day']):
            full=full.sort_values('fipsCode');fs=full.fipsCode.map(cv)
            for fold in range(5):
                c=full.loc[fs!=fold].copy();q=full.loc[fs==fold].copy()
                for variant in ['context','source_cache_rank']:
                    groups=[cfg['history_columns'],cfg['weather_columns'],cfg['static_columns']] if variant=='context' else list(blocks[str(h)].values())
                    cr=[];qr=[]
                    for names in groups:
                        cc=c[names].to_numpy(float,copy=True);qq=q[names].to_numpy(float,copy=True)
                        for j,name in enumerate(names):
                            if variant=='source_cache_rank':
                                if np.isnan(cc[:,j]).all() and np.isnan(qq[:,j]).all():cc[:,j]=qq[:,j]=0;continue
                                ref=np.sort(cc[:,j].copy())
                                for arr in [cc,qq]:arr[:,j]=[float((ref<v).sum()+.5*(ref==v).sum())/len(ref) for v in arr[:,j]]
                            elif name in cfg['history_columns']:
                                cc[:,j]=np.arcsinh(cc[:,j]/.01);qq[:,j]=np.arcsinh(qq[:,j]/.01)
                            elif name in ['customers_at_cutoff','pop_density','n_utilities','tp_sum','prior24_tp_sum']:
                                cc[:,j]=np.log1p(cc[:,j]);qq[:,j]=np.log1p(qq[:,j])
                        if variant=='context':
                            mu=cc.mean(axis=0);sd=cc.std(axis=0);sd[sd<=1e-12]=1
                            cc=(cc-mu)/sd;qq=(qq-mu)/sd
                        cr.append(cc/np.sqrt(len(names)*len(groups)));qr.append(qq/np.sqrt(len(names)*len(groups)))
                    cc=np.column_stack(cr);qq=np.column_stack(qr)
                    dist=np.sqrt(((qq[:,None]-cc)**2).sum(axis=2))
                    own=np.sqrt(((cc[:,None]-cc)**2).sum(axis=2));np.fill_diagonal(own,np.inf);loo=own.min(axis=1)
                    for i,f in enumerate(q.fipsCode):
                        selected_rows=indexed.loc[(s,h,day,fold,variant,f)].sort_values('rank')
                        order=np.lexsort((c.fipsCode.to_numpy(),dist[i]))[:5]
                        assert selected_rows.matched_fips.tolist()==c.fipsCode.iloc[order].tolist(),(s,h,day,variant,f)
                        close(selected_rows.distance,dist[i,order])
                        sr=sup.loc[(s,h,day,fold,variant,f)]
                        close(sr.nearest_percentile,100*(loo<=dist[i,order[0]]).mean())
                        assert sr.pool_n==len(c)
                        match_checks+=1
        print('Independent matching PASS',s,flush=True)
    joined=pd.read_parquet(OUT/'matching/ranked_controls_with_outcomes.parquet')
    pd.testing.assert_frame_equal(joined[ranks.columns],ranks,check_exact=True)
    for n,h in freeze['files'].items():assert sha(OUT/'matching'/n)==h
    outcomes=csv(OUT/'matching/window_outcomes.csv',dtype={'fipsCode':str})
    expected=d.groupby(['cv_seed','horizon','target_day','fipsCode']).agg(osi_peak=('truth','max'),osi_mean=('truth','mean'),P_peak=('P_true','max'),P_mean=('P_true','mean'),N_peak=('N_true','max'),R_mean=('R_true','mean'),D_mean=('D_true','mean'))
    oo=outcomes.set_index(['cv_seed','horizon','target_day','fipsCode'])
    for col in expected:close(oo.loc[expected.index,col],expected[col])
    for p,h in manifest.items():assert sha(p)==h
    result=dict(status='PASS',new_fits=0,models_reloaded=0,rows_checked=len(d),stella_rows_checked=len(six),metric_groups_checked=metric_checks,
                matching_queries_checked=match_checks,ranked_pairs_checked=len(ranks),whole_outer_fold_exclusion=True,
                common_score_checked=True,component_sse_closure=True,source_files_unchanged=len(manifest),
                outcome_blind_ranking_hashes_unchanged=True,absolute_tolerance=1e-12,verifier_sha256=sha(__file__))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':verify()
