"""Frozen sources, county scopes, and product metric; writes confined to this experiment."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys,platform,importlib.metadata
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=OUT.parents[3]
sys.path.insert(0,str(BASE))
from protocol import load_data,build_controls,expected_mask,component_target_name
from config import HORIZONS,HORIZON_HOURS,COMPONENTS,make_lgbm_params
RUN=OUT/'runs/v18_nr2stage_v1_split42_model42'
TREE=BASE/'p_neighbor_horizon_increment/runs/v18_pneighbor_v1_split42_model42'
OLD=BASE/'runs/v18_tree_nested_v2_split42_model42'
F2=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42'
INDEX=BASE/'p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json'
TREE_ID='bd4468e5c37fba7e93832153bc9847f807b8ae0054e45586f3c3853e96835bbd'
H1='osi_target_t01h';KEYS=['fipsCode','timestamp_et','hour_idx','fold'];CASES=['B0','N_only','R_only','NR_both']
PAIRS=[('N_only','B0'),('R_only','B0'),('NR_both','B0'),('NR_both','N_only'),('NR_both','R_only')]
FOCUS=['42053','39117','39115','18013','54015','54013','54007','39119','54087','54109','54041']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrsha(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,Path):return str(v)
    return v
def safe(p):
    p=Path(p).resolve();assert p.is_relative_to(OUT) and p!=OUT;p.parent.mkdir(parents=True,exist_ok=True);return p
def atomic(p,writer):
    p=safe(p);q=p.with_name(p.name+'.partial');writer(q);q.replace(p)
def save(p,v):atomic(p,lambda q:q.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n'))
def frame(p,d):
    d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)
    atomic(p,lambda q:d.to_parquet(q,index=False) if Path(p).suffix=='.parquet' else d.to_csv(q,index=False))
def norm(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5);d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    assert not d.duplicated(['fipsCode','hour_idx']).any();return d.sort_values(['fipsCode','hour_idx']).reset_index(drop=True)
def params(head):
    p=make_lgbm_params(42);p.update(objective='binary' if head=='q' else 'regression',metric='binary_logloss' if head=='q' else 'None');return p
def product(q,m):
    q=np.asarray(q);m=np.asarray(m);assert q.shape==m.shape
    assert ((q[np.isfinite(q)]>=0)&(q[np.isfinite(q)]<=1)).all()
    return q*np.clip(m,0,1)
def product_metric(q,official):
    q=np.asarray(q,float).copy();official=np.asarray(official,float).copy()
    assert q.shape==official.shape and np.isfinite(q).all() and np.isfinite(official).all()
    def feval(pred,_dataset):return 'product_rmse',float(np.sqrt(np.mean((product(q,pred)-official)**2))),False
    return feval
def load_context():
    data=load_data();meta=data.meta_train.copy();meta['fold']=data.row_folds
    item=next(x for x in read(INDEX)['splits'] if x['split_seed']==42)
    assert item['case']=='NB_P24' and item['run_identity_hash']==TREE_ID and Path(item['run_directory'])==TREE
    for rec in item['files'].values():assert sha(rec['path'])==rec['sha256'],rec['path']
    tm=read(TREE/'run_manifest.json');assert digest(tm['identity'])==tm['identity_hash']==TREE_ID
    part=norm(pd.read_parquet(TREE/'component_predictions.parquet'));saved=norm(pd.read_parquet(TREE/'control_predictions.parquet'))
    original=norm(pd.read_parquet(F2/'control_predictions.parquet'))
    for d in [part,saved,original]:pd.testing.assert_frame_equal(d[KEYS],meta[KEYS],check_dtype=False)
    parts={h:{c:part['pred_NB_P24_'+component_target_name(c,h)].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS}
    direct={h:original['raw_direct_'+h].to_numpy(copy=True) for h in HORIZONS};controls,aux=build_controls(meta,direct,parts)
    for h in HORIZONS:
        for mode in controls:np.testing.assert_array_equal(controls[mode][h],saved[f'pred_NB_P24_{mode}_{h}'])
    names=pd.read_csv(ROOT/'data/DM_Train.csv',usecols=['fipsCode','countyName','stateAbbr'],dtype={'fipsCode':str}).drop_duplicates().set_index('fipsCode')
    ys={c:data.component_targets[component_target_name(c,H1)].to_numpy(float,copy=True) for c in ['N_t','R_t']}
    return SimpleNamespace(data=data,meta=meta,X=data.X_train,parts=parts,direct=direct,controls=controls,aux=aux,names=names,
        reference_parts=part,reference_saved=saved,valid=expected_mask(meta,H1),ys=ys)
def initialize(ctx):
    source={}
    def add(p,h=None):
        p=Path(p).resolve();v=sha(p)
        if h is not None:assert v==h,str(p)
        source[str(p)]=v
    for n,p in ctx.data.input_paths.items():add(p,ctx.data.loaded_hashes[n])
    for marker in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']:
        add(TREE/marker)
        for p,h in read(TREE/marker)['files'].items():add(TREE/p,h)
    for p in [TREE/'run_manifest.json',TREE/'VERIFIED_COMPLETE',TREE/'logs/independent_verification.json',INDEX,F2/'control_predictions.parquet',
        F2/'INFO_CV_COMPLETE',F2/'run_manifest.json',BASE/'config.py',BASE/'protocol.py',BASE/'run_artifacts.py',
        BASE/'train_test_feature_support/data/nearest_five.parquet',BASE/'train_test_feature_support/completion.json',OLD/'run_manifest.json']:
        add(p)
    for outer in range(5):
        for c in ['N_t','R_t']:
            p=OLD/f'models/outer{outer}/{component_target_name(c,H1)}.json';r=read(p);add(p);add(OLD/r['model_path'],r['model_sha256'])
    assert sha(F2/'control_predictions.parquet')==read(F2/'INFO_CV_COMPLETE')['files']['control_predictions.parquet']
    sp=OUT/'reference/source_hashes.json'
    if sp.exists():assert read(sp)==source
    else:save(sp,source)
    votes=pd.read_parquet(BASE/'train_test_feature_support/data/nearest_five.parquet');votes=votes[(votes.scope=='test_to_train')&(votes.view=='context')]
    counts=votes.groupby(['horizon','target_day','matched_fips']).fipsCode.nunique();rows=[]
    for h in HORIZONS:
        hh=HORIZON_HOURS[h]
        for day in range((72+hh)//24,9):
            for f in ctx.meta.fipsCode.unique():rows.append(dict(horizon=hh,target_day=day,fipsCode=f,test_like=int(counts.get((hh,day,f),0))>0,test_votes=int(counts.get((hh,day,f),0))))
    member=pd.DataFrame(rows);mp=OUT/'reference/test_like_membership.parquet'
    if mp.exists():pd.testing.assert_frame_equal(pd.read_parquet(mp),member)
    else:frame(mp,member)
    cfg=dict(protocol='v18_nr2stage_v1',cv_seed=42,model_seed=42,cases=CASES,pairs=PAIRS,changed_sources=['N1','R1'],reference_tree_id=TREE_ID,
        feature_count=163,occurrence='official_Z>0',magnitude_training='official_Z>0',magnitude_validation='all_inner_rows_q_times_clip_m_RMSE_float64',
        params={head:params(head) for head in ['q','m']},max_rounds=2000,early_stopping=100,refit_rule='floor_mean_four_inner_best',
        fits=100,outer_models=20,saved_inner_models=80,bootstrap_replicates=2000,bootstrap_seed=20260910,focus_counties=FOCUS,
        membership_sha256=sha(mp),protocol_sha256=sha(OUT/'Experiment_Protocol_2026-09-21.md'),plots=False,test_inference=False)
    cfg=clean(cfg);cp=OUT/'experiment_config.json'
    if cp.exists():assert read(cp)==cfg
    else:save(cp,cfg)
    env=dict(python=platform.python_version(),libraries={p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow','lightgbm','scipy','scikit-learn']})
    prior=read(TREE/'run_manifest.json')['identity']['environment'];assert env['python']==prior['python'] and env['libraries']==prior['libraries']
    ctx.identity=dict(protocol=cfg['protocol'],config_sha256=sha(cp),source_hashes=source,feature_columns=list(ctx.X),
        training_code_sha256={p:sha(OUT/p) for p in ['nr_core.py','preflight.py','train_nr.py']},environment=env)
    ctx.identity_hash=digest(ctx.identity);ctx.sources=source;ctx.membership=member;return ctx
def subset(ctx,c,folds,positive=False):
    folds=tuple(sorted(set(folds)));assert len(folds) in (1,3,4) and set(folds)<=set(range(5))
    hit=ctx.valid&np.isin(ctx.meta.fold.to_numpy(),folds)
    if positive:hit&=ctx.ys[c]>0
    rows=np.flatnonzero(hit);assert len(rows)>0;y=ctx.ys[c][rows].copy();assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
    return SimpleNamespace(rows=rows,X=ctx.X.iloc[rows].reset_index(drop=True),y=y,folds=folds,positive=positive,meta=ctx.meta.iloc[rows].reset_index(drop=True))
def describe(s):
    return dict(folds=list(s.folds),positive_only=s.positive,counties=sorted(s.meta.fipsCode.unique()),rows=len(s.rows),positive_rows=int((s.y>0).sum()),
        row_sha256=arrsha(s.rows),feature_sha256=arrsha(s.X.to_numpy()),label_sha256=arrsha(s.y))
