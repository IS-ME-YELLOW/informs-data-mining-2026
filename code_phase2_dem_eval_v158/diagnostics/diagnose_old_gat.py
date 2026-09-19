"""Read-only analysis of saved old GAT evidence. All writes stay in diagnostics."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import importlib.util
import itertools
import json
import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;PKG=OUT.parent;REPO=PKG.parent
RUN=PKG/'outputs/runs/dem_v158_p2_component_seed42_cuda_log'
CACHE=REPO/'versions/xyy/v1.5.6'
HOURS=(1,6,24,48);HORIZONS=tuple(f'osi_target_t{h:02d}h' for h in HOURS)
COMPONENTS=('P_t','N_t','D_t','R_t');WEIGHTS=(.4,.35,.25,-.1)
NEWTON='18111';H1=HORIZONS[0]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def target(path):
    p=(OUT/path).resolve()
    if p==OUT or not p.is_relative_to(OUT):raise ValueError('Output escapes diagnostics')
    p.parent.mkdir(parents=True,exist_ok=True);return p
def save_json(path,value):target(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def save_csv(path,frame):frame.to_csv(target(path),index=False)
def csv(path):return pd.read_csv(path,dtype={'fipsCode':str},float_precision='round_trip')
def close(a,b,tol=1e-12):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
def post(values):
    result=np.clip(np.asarray(values,dtype=float),0,.65)
    return np.where(result<.001,0.,result)
def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float);assert len(y)>0 and np.isfinite(y).all() and np.isfinite(p).all()
    e=p-y;ss=((y-y.mean())**2).sum()
    return {'n':len(y),'mse':float(np.mean(e**2)),'rmse':float(np.sqrt(np.mean(e**2))),
        'mae':float(np.mean(np.abs(e))),'medae':float(np.median(np.abs(e))),'bias':float(e.mean()),
        'error_std':float(e.std()),'r2':float(1-np.sum(e**2)/ss) if ss else np.nan,'max_abs_error':float(np.abs(e).max()),'sse':float(np.sum(e**2))}
def comparison(frame,pred=None):
    y=frame.y_true.to_numpy();b=metrics(y,post(frame.base_prediction));g=metrics(y,frame.prediction if pred is None else pred)
    return {'n':b['n'],**{'base_'+k:v for k,v in b.items() if k!='n'},**{'gat_'+k:v for k,v in g.items() if k!='n'},
        'rmse_change_pct':100*(g['rmse']/b['rmse']-1) if b['rmse'] else np.nan,'delta_sse':g['sse']-b['sse']}
def scope(text):return tuple(int(x) for x in str(text).split(':S')[-1].split(','))


def source_audit():
    m=read(RUN/'run_manifest.json');cv=read(RUN/'cv_manifest.json');definition=m['identity']['definition']
    digest=hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    assert digest==m['identity']['digest']
    checks=[]
    for n,h in {'run_manifest.json':cv['run_manifest_sha256'],**cv['artifact_hashes']}.items():
        path=RUN/n;actual=sha(path);match='exact'
        if actual!=h:
            assert path.suffix in ('.json','.csv'),n
            raw=path.read_bytes();original_windows=raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
            assert hashlib.sha256(original_windows).hexdigest()==h,n
            match='same_text_after_LF_to_CRLF_conversion_without_writing'
        checks.append({'category':'CV_artifact','path':str(path),'expected_sha256':h,'actual_sha256':actual,'match':match})
    cv_hash=hashlib.sha256(json.dumps(cv['artifact_hashes'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert cv_hash==cv['cv_hash']==(RUN/'CV_COMPLETE').read_text()
    for n,h in definition['sources'].items():
        got=sha(PKG/n);assert got==h,n
        checks.append({'category':'source_code','path':str(PKG/n),'expected_sha256':h,'actual_sha256':got,'match':'exact'})
    for n,obj in m['inputs']['package_manifest']['files'].items():
        h=obj['sha256'].lower();got=sha(CACHE/n);assert got==h,n
        checks.append({'category':'feature_package','path':str(CACHE/n),'expected_sha256':h,'actual_sha256':got,'match':'exact'})
    mapped={'cv_sha256':REPO/'cv/cv_assignments_balanced_v1_seed42.csv','terrain_sha256':REPO/'data/geo/county_terrain.csv',
        'geo_dbf_sha256':REPO/'data/geo/c_16ap26.dbf','geo_shp_sha256':REPO/'data/geo/c_16ap26.shp',
        'component_targets_sha256':REPO/'versions/xyy/v1.5.8/component_targets_v1.8.parquet',
        'submission_template_sha256':REPO/'data/sample_submission.csv'}
    for key,path in mapped.items():
        expected=m['inputs'][key];actual=sha(path);match='exact'
        if actual!=expected:
            raw=path.read_bytes();windows=raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
            assert hashlib.sha256(windows).hexdigest()==expected,(key,path)
            match='same_text_after_LF_to_CRLF_conversion_without_writing'
        checks.append({'category':key,'path':str(path),'expected_sha256':expected,'actual_sha256':actual,'match':match})
    graph=np.load(RUN/'graph.npz',allow_pickle=False)
    nodes=graph['all_fips'].tolist();edges=graph['edge_index'];attrs=graph['edge_attr'];coords=graph['coords']
    gh=hashlib.sha256();gh.update(json.dumps(nodes,separators=(',',':')).encode());gh.update(edges.astype(np.int64).tobytes());gh.update(attrs.astype(np.float32).tobytes())
    assert gh.hexdigest()==m['inputs']['graph_hash'] and nodes==m['inputs']['all_fips']
    assert coords.shape==(302,2) and edges.shape==(2,3028) and attrs.shape==(3028,4)
    final=read(RUN/'verification.json')
    missing=[]
    base=cv_models=pd.read_parquet(RUN/'base_fit_manifest.parquet')
    full=pd.read_parquet(RUN/'base_fit_manifest_final.parquet')
    for row in full.itertuples(index=False):
        path=RUN/row.model_path.replace('\\','/')
        if not path.is_file():missing.append({'type':'base_model','expected_path':str(path),'model_id':row.model_id})
        receipt=Path(str(path)+'.complete.json')
        if not receipt.is_file():missing.append({'type':'base_receipt','expected_path':str(receipt),'model_id':row.model_id})
    for n in (3,4,5):
        for folds in itertools.combinations(range(5),n):
            for h in HORIZONS:
                suffix=h.replace('osi_target_','');path=RUN/'models/gat'/f'gat_component_v158_{suffix}_S{"-".join(map(str,folds))}.pt'
                mid=f'stack:component_v158:{h}:S{",".join(map(str,folds))}'
                for kind,p in [('gat_checkpoint',path),('gat_receipt',Path(str(path)+'.complete.json'))]:
                    if not p.is_file():missing.append({'type':kind,'expected_path':str(p),'model_id':mid})
    save_csv('tables/source_hash_audit.csv',pd.DataFrame(checks));save_csv('tables/missing_model_artifacts.csv',pd.DataFrame(missing))
    audit={'status':'saved_numeric_artifacts_and_source_code_hashes_PASS','run_identity':digest,'cv_hash':cv_hash,
        'CV_COMPLETE_present':True,'COMPLETE_present':(RUN/'COMPLETE').exists(),'FINAL_READY_text':(RUN/'FINAL_READY').read_text(),
        'saved_verification':final,'local_torch_available':importlib.util.find_spec('torch') is not None,
        'local_gat_checkpoints':len(list(RUN.rglob('*.pt'))),'local_model_receipts':len(list(RUN.rglob('*.complete.json'))),
        'expected_base_models':len(full),'expected_gat_checkpoints':64,
        'model_reload_performed':False,'full_remote_acceptance':'not established by this copied package',
        'line_ending_difference':'run JSON, CV CSVs and terrain match recorded hashes after in-memory LF to CRLF conversion; no source file modified'}
    save_json('source_audit.json',audit)
    return m,nodes,edges,attrs,coords


def check_rows(m):
    outer=pd.read_parquet(RUN/'outer_oof.parquet');inner=pd.read_parquet(RUN/'inner_oof.parquet')
    alpha=pd.read_parquet(RUN/'alpha_selection.parquet');foldrows=csv(RUN/'folds.csv')
    fm=foldrows.groupby('fipsCode').fold.first();assert foldrows.groupby('fipsCode').fold.nunique().eq(1).all()
    raw=csv(REPO/'data/DM_Train.csv');raw['hour_idx']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta(hours=1)).astype(int)
    truth=raw.set_index(['fipsCode','hour_idx']).osi;names=raw[['fipsCode','countyName']].drop_duplicates().set_index('fipsCode').countyName
    assert len(outer)==137664 and not outer.duplicated(['horizon','fipsCode','hour_idx']).any()
    assert len(inner)==475132 and not inner.duplicated(['outer_fold','horizon','fipsCode','hour_idx']).any()
    for h,hours in zip(HORIZONS,HOURS):
        d=outer.loc[outer.horizon==h];valid=(d.hour_idx+hours<=215).to_numpy();np.testing.assert_array_equal(d.is_scoreable,valid)
        assert len(d)==34416 and d.fipsCode.nunique()==239
        close(d.loc[valid,'y_true'],truth.reindex(pd.MultiIndex.from_arrays([d.loc[valid,'fipsCode'],d.loc[valid,'hour_idx']+hours])))
        assert d.loc[~valid,['y_true','prediction','prediction_before_postprocess']].isna().all().all()
        close(d.loc[valid,'prediction_before_postprocess'],d.loc[valid,'base_prediction']+d.loc[valid,'alpha']*d.loc[valid,'correction_osi'])
        close(d.loc[valid,'prediction'],post(d.loc[valid,'prediction_before_postprocess']))
        composed=np.zeros(len(d))
        for c,w in zip(COMPONENTS,WEIGHTS):composed+=w*d['base_'+c].to_numpy()
        close(d.base_prediction,post(composed).astype(np.float32).astype(float))
        for q,g in d.groupby('outer_fold'):
            S=tuple(k for k in range(5) if k!=q);assert fm.loc[g.fipsCode].eq(q).all()
            assert g.stack_id.eq(f'stack:component_v158:{h}:S{",".join(map(str,S))}').all()
            for c in COMPONENTS:assert g['base_source_'+c].eq(f'base:{c}:{h}:S{",".join(map(str,S))}').all()
    for (q,h,r),g in inner.groupby(['outer_fold','horizon','inner_fold']):
        S=tuple(f for f in range(5) if f not in (q,r));assert q!=r
        assert g.allowed_folds.map(lambda x:tuple(json.loads(x))==S).all()
        assert g.stack_id.eq(f'stack:component_v158:{h}:S{",".join(map(str,S))}').all()
        assert g.base_source_id.eq(f'base:P_t:{h}:S{",".join(map(str,S))}').all()
        assert fm.loc[g.fipsCode].eq(r).all() and g.is_scoreable.all()
        hour=int(h[-3:-1]);close(g.y_true,truth.reindex(pd.MultiIndex.from_arrays([g.fipsCode,g.hour_idx+hour])))
    full=pd.read_parquet(RUN/'base_fit_manifest_final.parquet')
    for row in full.itertuples(index=False):
        S=tuple(json.loads(row.allowed_folds));best=json.loads(row.best_iterations);probes=json.loads(row.probe_records)
        assert len(best)==len(S)==len(probes) and row.final_rounds==max(1,int(np.mean(best))) and 1<=row.actual_rounds<=row.final_rounds
        assert row.model_id==f'base:{row.component}:{row.horizon}:S{",".join(map(str,S))}'
        assert row.run_identity==m['identity']['digest']
        for q,b,pr in zip(S,best,probes):
            assert pr['probe_fold']==q and pr['train_folds']==[k for k in S if k!=q] and pr['valid_folds']==[q] and pr['best_iteration']==b
    valid=outer.loc[outer.is_scoreable].copy();valid['base_post']=post(valid.base_prediction)
    valid['base_error']=valid.base_post-valid.y_true;valid['gat_error']=valid.prediction-valid.y_true
    valid['delta_sse_row']=valid.gat_error**2-valid.base_error**2;valid['applied_correction']=valid.alpha*valid.correction_osi
    valid['countyName']=valid.fipsCode.map(names)
    return valid,outer,inner,alpha,raw,fm


def metric_tables(o,inner,alpha):
    totals=[];folds=[];counties=[];windows=[];bootstrap=[]
    for h,g in o.groupby('horizon',sort=False):
        totals.append({'horizon':h,**comparison(g)})
        for q,f in g.groupby('outer_fold'):folds.append({'horizon':h,'outer_fold':q,'alpha':float(f.alpha.iloc[0]),**comparison(f)})
        for code,f in g.groupby('fipsCode'):
            counties.append({'horizon':h,'fipsCode':code,'countyName':f.countyName.iloc[0],'outer_fold':int(f.outer_fold.iloc[0]),
                'alpha':float(f.alpha.iloc[0]),'actual_max':float(f.y_true.max()),'prediction_max':float(f.prediction.max()),
                'correction_min':float(f.correction_osi.min()),'correction_mean':float(f.correction_osi.mean()),'correction_max':float(f.correction_osi.max()),**comparison(f)})
        for label,mask in [('all_counties',np.ones(len(g),bool)),('exclude_Newton_hindsight',g.fipsCode.ne(NEWTON).to_numpy())]:
            v=g.loc[mask];group=pd.DataFrame({'county':v.fipsCode,'base':v.base_error**2,'gat':v.gat_error**2}).groupby('county').agg(base=('base','sum'),gat=('gat','sum'),n=('base','size')).to_numpy()
            rng=np.random.default_rng(20260910);samples=rng.integers(0,len(group),size=(2000,len(group)));s=group[samples].sum(axis=1)
            delta=np.sqrt(s[:,1]/s[:,2])-np.sqrt(s[:,0]/s[:,2]);lo,hi=np.quantile(delta,[.025,.975])
            bootstrap.append({'horizon':h,'population':label,'n_counties':len(group),'replicates':2000,'seed':20260910,
                'ci_low':lo,'ci_high':hi,'fraction_delta_negative':float(np.mean(delta<0)),**comparison(v)})
    totals=pd.DataFrame(totals);folds=pd.DataFrame(folds);counties=pd.DataFrame(counties)
    counties['fraction_of_net_loss_increase']=counties.delta_sse/counties.groupby('horizon').delta_sse.transform('sum')
    original=csv(RUN/'cv_summary.csv')
    for row in original.itertuples(index=False):
        d=o.loc[o.horizon==row.horizon];score=metrics(d.y_true,d.base_post if row.model=='base' else d.prediction)
        for key in ('n','mse','rmse','mae','medae','bias','error_std','r2','max_abs_error'):close(getattr(row,key),score[key])
    oldfold=csv(RUN/'fold_metrics.csv')
    for row in oldfold.itertuples(index=False):
        d=o.loc[(o.horizon==row.horizon)&(o.outer_fold==row.outer_fold)]
        score=metrics(d.y_true,d.base_post if row.model=='base' else d.prediction)
        for key in ('n','mse','rmse','mae','medae','bias','error_std','r2','max_abs_error'):close(getattr(row,key),score[key])
    for population,subset in [('all',o),('Newton',o.loc[o.fipsCode==NEWTON]),('others',o.loc[o.fipsCode!=NEWTON])]:
        for h,g in subset.groupby('horizon'):
            target=g.hour_idx+int(h[-3:-1])
            for lo,hi in [(73,76),(77,95),(96,143),(144,215)]:
                d=g.loc[target.between(lo,hi)]
                if len(d):windows.append({'population':population,'horizon':h,'target_window':f'{lo}-{hi}',**comparison(d)})
    save_csv('tables/overall_metrics.csv',totals);save_csv('tables/fold_effects.csv',folds)
    save_csv('tables/county_effects.csv',counties.sort_values(['horizon','delta_sse'],ascending=[True,False]))
    save_csv('tables/window_effects.csv',pd.DataFrame(windows));save_csv('tables/county_bootstrap_recomputed.csv',pd.DataFrame(bootstrap))
    return totals,folds,counties


def alpha_analysis(o,inner,alpha):
    grid=sorted(alpha.alpha.unique());inside=[];outside=[];transfer=[]
    for (q,h),g in inner.groupby(['outer_fold','horizon'],sort=True):
        saved=alpha.loc[(alpha.outer_fold==q)&(alpha.horizon==h)].sort_values('alpha')
        assert saved.alpha.tolist()==grid and int(saved.selected.sum())==1
        rows=[]
        for a in grid:
            score=metrics(g.y_true,post(g.base_prediction+a*g.correction_osi));s=saved.loc[saved.alpha==a].iloc[0]
            for key in ('n','sse','rmse','mae'):close(score[key],s[key])
            rows.append({'horizon':h,'outer_fold':q,'alpha':a,'selected':bool(s.selected),**score})
        best=min(rows,key=lambda z:(z['rmse'],z['alpha']));assert best['selected']
        inside.extend(rows);outer=o.loc[(o.horizon==h)&(o.outer_fold==q)]
        assert outer.alpha.eq(best['alpha']).all()
        ref=rows[0];train_scale=max(float(np.std(g.y_true.to_numpy()-g.base_prediction.to_numpy(),ddof=0)),.003)
        for a in grid:
            result=comparison(outer,post(outer.base_prediction+a*outer.correction_osi))
            outside.append({'horizon':h,'outer_fold':q,'alpha':a,'selected_by_inner':a==best['alpha'],'diagnostic_only':True,**result})
        result=comparison(outer)
        transfer.append({'horizon':h,'outer_fold':q,'selected_alpha':best['alpha'],'inner_base_rmse':ref['rmse'],
            'inner_selected_rmse':best['rmse'],'inner_change_pct':100*(best['rmse']/ref['rmse']-1),
            'inner_Newton_rows':int(g.fipsCode.eq(NEWTON).sum()),'estimated_outer_residual_scale':train_scale,
            'inner_correction_abs_p99':float(g.correction_osi.abs().quantile(.99)),
            'inner_correction_abs_max':float(g.correction_osi.abs().max()),
            'outer_correction_abs_p99':float(outer.correction_osi.abs().quantile(.99)),
            'outer_correction_abs_max':float(outer.correction_osi.abs().max()),**result})
    final= pd.read_parquet(RUN/'alpha_selection_final.parquet')
    final=final.loc[final.selection_type=='full_train_cv_for_final'];final_records=[]
    test=pd.read_parquet(RUN/'test_predictions.parquet')
    for h,g in o.groupby('horizon'):
        table=final.loc[final.horizon==h].sort_values('alpha');assert table.alpha.tolist()==grid
        scores=[]
        for a in grid:
            value=metrics(g.y_true,post(g.base_prediction+a*g.correction_osi));saved=table.loc[table.alpha==a].iloc[0]
            for key in ('n','sse','rmse','mae'):close(value[key],saved[key])
            scores.append((value['rmse'],a))
        selected=min(scores)[1];assert table.loc[table.selected.astype(bool),'alpha'].tolist()==[selected]
        t=test.loc[(test.horizon==h)&test.is_scoreable];assert t.alpha.eq(selected).all()
        close(t.prediction_before_postprocess,t.base_prediction+selected*t.correction_osi);close(t.prediction,post(t.prediction_before_postprocess))
        final_records.append({'horizon':h,'final_alpha':selected,'test_valid_rows':len(t),'test_changed_vs_base':int(np.sum(t.prediction.to_numpy()!=post(t.base_prediction))),
            'test_replay':'saved rows only, not model inference'})
    table=pd.DataFrame(outside)
    table['hindsight_best']=False
    for _,g in table.groupby(['horizon','outer_fold']):table.loc[g.sort_values(['gat_rmse','alpha']).index[0],'hindsight_best']=True
    save_csv('tables/inner_alpha_recomputed.csv',pd.DataFrame(inside));save_csv('tables/outer_alpha_hindsight_diagnostic.csv',table)
    save_csv('tables/inner_outer_transfer.csv',pd.DataFrame(transfer));save_csv('tables/final_alpha_and_test_replay.csv',pd.DataFrame(final_records))
    return pd.DataFrame(transfer),pd.DataFrame(final_records)


def newton_analysis(o,inner,raw,transfer):
    history=raw.loc[raw.fipsCode==NEWTON].copy();save_csv('tables/Newton_official_history_and_future.csv',history)
    n=o.loc[o.fipsCode==NEWTON].copy();save_csv('tables/Newton_outer_rows.csv',n)
    ni=inner.loc[inner.fipsCode==NEWTON].copy();save_csv('tables/Newton_inner_rows.csv',ni)
    summaries=[]
    for source,frame in [('outer',n),('inner',ni)]:
        for (h,q,stack),g in frame.groupby(['horizon','outer_fold','stack_id']):
            summaries.append({'source':source,'horizon':h,'outer_fold':q,'stack_id':stack,'n':len(g),
                'correction_min':g.correction_osi.min(),'correction_mean':g.correction_osi.mean(),'correction_max':g.correction_osi.max(),
                'base_mean':g.base_prediction.mean(),'truth_mean':g.y_true.mean(),'truth_max':g.y_true.max()})
    save_csv('tables/Newton_across_stack_contexts.csv',pd.DataFrame(summaries))
    h1=o.loc[o.horizon==H1];nw=h1.fipsCode.eq(NEWTON).to_numpy();sens=[]
    for label,mask in [('all',np.ones(len(h1),bool)),('exclude_Newton_hindsight',~nw),('exclude_fold2_hindsight',h1.outer_fold.ne(2).to_numpy()),('Newton_only',nw)]:
        sens.append({'scenario':label,'diagnostic_only':True,**comparison(h1.loc[mask])})
    for label,mask in [('Newton_alpha0_hindsight',nw),('fold2_alpha0_hindsight',h1.outer_fold.eq(2).to_numpy())]:
        pred=h1.prediction.to_numpy(copy=True);pred[mask]=post(h1.base_prediction.to_numpy()[mask])
        sens.append({'scenario':label,'diagnostic_only':True,**comparison(h1,pred)})
    save_csv('tables/diagnostic_exclusions_and_rollbacks.csv',pd.DataFrame(sens))
    decomposition=[]
    for label,mask in [('Newton',nw),('others',~nw),('all',np.ones(len(h1),bool))]:
        g=h1.loc[mask];r=g.y_true.to_numpy()-g.base_prediction.to_numpy();delta=g.alpha.to_numpy()*g.correction_osi.to_numpy()
        quadratic=float(np.sum(delta**2));cross=float(-2*np.sum(r*delta))
        raw_added=float(np.sum((g.base_prediction.to_numpy()+delta-g.y_true.to_numpy())**2-(g.base_prediction.to_numpy()-g.y_true.to_numpy())**2))
        close(raw_added,quadratic+cross)
        decomposition.append({'population':label,'n':len(g),'correction_square_SSE':quadratic,'residual_cross_SSE':cross,
            'raw_delta_SSE':raw_added,'postprocessed_delta_SSE':g.delta_sse_row.sum(),
            'postprocessing_delta_SSE_difference':g.delta_sse_row.sum()-raw_added})
    save_csv('tables/correction_error_decomposition.csv',pd.DataFrame(decomposition))
    history_rows=[]
    for lo,hi in [(0,47),(48,71),(66,71),(73,215)]:
        g=history.loc[history.hour_idx.between(lo,hi)]
        history_rows.append({'start_hour':lo,'end_hour':hi,'n':len(g),'mean_P':g.P_t.mean(),'max_P':g.P_t.max(),
            'mean_OSI':g.osi.mean(),'max_OSI':g.osi.max(),'customers_min':g.customersTracked.min(),'customers_max':g.customersTracked.max(),
            'outage_count_above_customers_rows':int((g.outageCount>g.customersTracked).sum())})
    save_csv('tables/Newton_history_windows.csv',pd.DataFrame(history_rows))


def feature_diagnostics(m,nodes,edges,coords,o,inner,fm,transfer):
    names=read(CACHE/'feature_names_v1.5.6.json');schema=read(RUN/'feature_schema.json')
    assert len(names)==163 and schema['graph_input_dim']==205
    xt=pd.read_parquet(CACHE/'features_train_v1.5.6.parquet');xs=pd.read_parquet(CACHE/'features_test_v1.5.6.parquet')
    mt=pd.read_parquet(CACHE/'meta_train_v1.5.6.parquet');ms=pd.read_parquet(CACHE/'meta_test_v1.5.6.parquet')
    meta=pd.concat([mt,ms],ignore_index=True);meta['fipsCode']=meta.fipsCode.astype(str).str.zfill(5)
    x=pd.concat([xt,xs],ignore_index=True);assert list(x)==names
    node_index={f:i for i,f in enumerate(nodes)};ti=meta.hour_idx.to_numpy()-72;ni=meta.fipsCode.map(node_index).to_numpy()
    finite=x.fillna(0).to_numpy(dtype=np.float32)
    graph=np.zeros((144,302,173),dtype=np.float32);graph[ti,ni,1:164]=finite
    graph[:,:,164:166]=coords
    terrain=csv(REPO/'data/geo/county_terrain.csv').set_index('fipsCode');terrain_cols=list(terrain)
    assert len(terrain_cols)==7;graph[:,:,166:173]=terrain.loc[nodes].to_numpy(np.float32)
    # Source helper is pure and hash-matched. No training/loading of checkpoints.
    sys.path.insert(0,str(PKG))
    from data import add_neighbor_feature_aggregates
    raw,extra=add_neighbor_feature_aggregates(graph,names,edges,H1)
    graph_names=['base',*names,'latitude','longitude',*terrain_cols,*extra]
    assert graph_names==schema['graph_schemas'][H1]
    train_nodes=np.array([f in fm.index for f in nodes]);nodefold=np.array([int(fm[f]) if f in fm.index else -1 for f in nodes])
    valid_time=np.arange(144)<143;newton_node=node_index[NEWTON]
    contexts=[];stats=[];allnodes=[]
    for S in [tuple(f for f in range(5) if f!=q) for q in range(5)]+[S for S in itertools.combinations([0,1,3,4],3)]:
        scope_id=','.join(map(str,S));mask=train_nodes&np.isin(nodefold,S)
        use=raw[:143,mask,1:];flat=use.reshape(-1,204)
        mean=flat.mean(axis=0,dtype=np.float64);rawstd=flat.std(axis=0,dtype=np.float64);std=np.where(rawstd<1e-6,1.,rawstd)
        low=flat.min(axis=0);high=flat.max(axis=0)
        newton=raw[:143,newton_node,1:].astype(np.float64)
        z=(newton-mean)/std
        for j,name in enumerate(graph_names[1:]):
            stats.append({'scope':scope_id,'scope_folds':len(S),'Newton_in_supervision':2 in S,'feature':name,
                'training_mean':mean[j],'training_std':rawstd[j],'applied_std':std[j],'training_min':float(low[j]),'training_max':float(high[j]),
                'Newton_min':float(newton[:,j].min()),'Newton_max':float(newton[:,j].max()),
                'Newton_max_abs_z':float(np.max(np.abs(z[:,j]))),'Newton_outside_training_range_rows':int(np.sum((newton[:,j]<low[j])|(newton[:,j]>high[j])))})
        if S==(0,1,3,4):
            for node,f in enumerate(nodes):
                znode=(raw[:143,node,1:].astype(np.float64)-mean)/std
                index=np.unravel_index(np.argmax(np.abs(znode)),znode.shape)
                allnodes.append({'fipsCode':f,'is_training_county':f in fm.index,'fold':int(nodefold[node]),
                    'max_abs_z':float(np.abs(znode[index])),'feature':graph_names[index[1]+1],'origin_hour':index[0]+72})
    stats=pd.DataFrame(stats);save_csv('tables/Newton_feature_support_by_scope.csv',stats)
    save_csv('tables/outer2_all_nodes_feature_support.csv',pd.DataFrame(allnodes).sort_values('max_abs_z',ascending=False))
    # Full training residual scale and the base input channel are reconstructible
    # from outer-q rows plus q's inner validation rows. Test base is not available;
    # no artificial test base is ever treated as an inference input.
    scale=[]
    for q in range(5):
        inn=inner.loc[(inner.outer_fold==q)&(inner.horizon==H1)]
        out=o.loc[(o.outer_fold==q)&(o.horizon==H1)]
        combined=pd.concat([inn[['fipsCode','hour_idx','base_prediction']],out[['fipsCode','hour_idx','base_prediction']]])
        assert len(combined)==34177 and not combined.duplicated(['fipsCode','hour_idx']).any()
        r=inn.y_true.to_numpy()-inn.base_prediction.to_numpy();s=max(float(np.std(r,ddof=0)),.003)
        if q==2:
            n=out.loc[out.fipsCode==NEWTON]
            scale.append({'outer_fold':q,'train_residual_std_reconstructed':float(np.std(r,ddof=0)),
                'residual_scale_reconstructed':s,'Newton_normalized_correction_min':n.correction_osi.min()/s,
                'Newton_normalized_correction_mean':n.correction_osi.mean()/s,'Newton_normalized_correction_max':n.correction_osi.max()/s,
                'train_true_residual_min':r.min(),'train_true_residual_max':r.max(),'checkpoint_stats_verified':False})
    save_csv('tables/outer2_residual_scale_reconstruction.csv',pd.DataFrame(scale))
    pd.DataFrame(raw[:143,newton_node,1:],columns=graph_names[1:]).assign(origin_hour=np.arange(72,215)).to_csv(target('tables/Newton_nonbase_graph_features.csv'),index=False)
    save_json('feature_reconstruction_scope.json',{'status':'source-backed_nonbase_input_reconstruction','source_code_hash_matches_run':True,
        'nonbase_dimensions':204,'base_channel_excluded_from_zscore_table':True,'test_base_predictions_for_outer_stack_available':False,
        'checkpoint_scalers_available':False,'statistics':'recomputed from float32 raw graph features using source train-node/time masks',
        'model_inference_or_feature_ablation_performed':False,'attribution_limit':'large z scores support an extrapolation hypothesis; do not prove specific learned weights or skip/message-path causality'})


def main():
    m,nodes,edges,attrs,coords=source_audit()
    o,outer,inner,alpha,raw,fm=check_rows(m)
    totals,folds,counties=metric_tables(o,inner,alpha)
    transfer,final=alpha_analysis(o,inner,alpha)
    newton_analysis(o,inner,raw,transfer)
    feature_diagnostics(m,nodes,edges,coords,o,inner,fm,transfer)
    o.to_parquet(target('tables/outer_row_effects.parquet'),index=False)
    save_json('diagnostic_status.json',{'status':'NUMERIC_DIAGNOSIS_COMPLETE','source_run':str(RUN),
        'source_run_identity':m['identity']['digest'],'all_outer_predictions_and_labels_replayed':True,
        'all_160_inner_alpha_candidates_recomputed':True,'all_32_final_alpha_candidates_recomputed':True,
        'declared_scope_and_round_lineage_checked':True,'model_training_performed':False,'checkpoint_reload_performed':False,
        'all_writes_restricted_to':str(OUT),'created_at_utc':datetime.now(timezone.utc).isoformat()})
    print(totals[['horizon','base_rmse','gat_rmse','rmse_change_pct','delta_sse']].to_string(index=False))
    print(counties.query('horizon==@H1').sort_values('delta_sse',ascending=False).head(3)[['fipsCode','countyName','delta_sse','fraction_of_net_loss_increase']].to_string(index=False))


if __name__=='__main__':main()
