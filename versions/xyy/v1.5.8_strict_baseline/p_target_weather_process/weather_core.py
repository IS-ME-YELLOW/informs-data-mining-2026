"""Frozen sources and training scopes for P24 exposure-process information."""
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
RUN=OUT/'runs/v18_pweather_v1_split42_model42'
TREE=BASE/'p_neighbor_horizon_increment/runs/v18_pneighbor_v1_split42_model42'
OLD_FEATURES=BASE/'p_neighbor_horizon_increment/features/v1'
F2=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42'
INDEX=BASE/'p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json'
TREE_ID='bd4468e5c37fba7e93832153bc9847f807b8ae0054e45586f3c3853e96835bbd'
H24='osi_target_t24h';PT='P_t_target_t24h';KEYS=['fipsCode','timestamp_et','hour_idx','fold']
NEW_COLUMNS=['exposure_decay_6h','exposure_decay_24h','hours_since_last_high_gust','has_high_gust_48h',
    'high_gust_episode_count_48h','max_lull_between_episodes_48h','exposure_weighted_age_48h','recent6_exposure_fraction']
TRAIN_CODE=['weather_core.py','process_features.py','preflight.py','verify_features.py','train_weather.py']

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
    p=safe(p);tmp=p.with_name(p.name+'.partial');writer(tmp);tmp.replace(p)
def save(p,v):atomic(p,lambda q:q.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n'))
def frame(p,d):
    d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)
    atomic(p,lambda q:d.to_parquet(q,index=False) if Path(p).suffix=='.parquet' else d.to_csv(q,index=False))
def norm(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5);d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    assert not d.duplicated(['fipsCode','hour_idx']).any();return d.sort_values(['fipsCode','hour_idx']).reset_index(drop=True)
def params():return make_lgbm_params(42)

def load_context():
    data=load_data();meta=data.meta_train.copy();meta['fold']=data.row_folds
    item=next(x for x in read(INDEX)['splits'] if x['split_seed']==42)
    assert item['case']=='NB_P24' and item['run_identity_hash']==TREE_ID and Path(item['run_directory'])==TREE
    for rec in item['files'].values():assert sha(rec['path'])==rec['sha256'],rec['path']
    tm=read(TREE/'run_manifest.json');assert digest(tm['identity'])==tm['identity_hash']==TREE_ID
    fm=read(OLD_FEATURES/'feature_manifest.json');assert digest(fm['identity'])==fm['identity_hash']
    for name,h in fm['files'].items():assert sha(OLD_FEATURES/name)==h,name
    X0=pd.read_parquet(OLD_FEATURES/'h24_train.parquet');Xt0=pd.read_parquet(OLD_FEATURES/'h24_test.parquet')
    assert X0.shape==(34416,183) and Xt0.shape==(9072,183)
    pd.testing.assert_frame_equal(X0[list(data.X_train)],data.X_train);pd.testing.assert_frame_equal(Xt0[list(data.X_test)],data.X_test)
    part=norm(pd.read_parquet(TREE/'component_predictions.parquet'));saved=norm(pd.read_parquet(TREE/'control_predictions.parquet'))
    original=norm(pd.read_parquet(F2/'control_predictions.parquet'))
    for d in [part,saved,original]:pd.testing.assert_frame_equal(d[KEYS],meta[KEYS],check_dtype=False)
    parts={h:{c:part['pred_NB_P24_'+component_target_name(c,h)].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS}
    direct={h:original['raw_direct_'+h].to_numpy(copy=True) for h in HORIZONS};controls,aux=build_controls(meta,direct,parts)
    for h in HORIZONS:
        for mode in controls:np.testing.assert_array_equal(controls[mode][h],saved[f'pred_NB_P24_{mode}_{h}'])
    raw=pd.read_csv(ROOT/'data/DM_Train.csv',usecols=['fipsCode','countyName','stateAbbr'],dtype={'fipsCode':str}).drop_duplicates().set_index('fipsCode')
    return SimpleNamespace(data=data,meta=meta,X0=X0,Xt0=Xt0,parts=parts,direct=direct,controls=controls,aux=aux,names=raw,
        reference_parts=part,reference_saved=saved,valid=expected_mask(meta,H24),y=data.component_targets[PT].to_numpy(float,copy=True),old_feature_identity=fm['identity_hash'])

def snapshot(ctx):
    source={}
    def add(p,h=None):
        p=Path(p).resolve();v=sha(p)
        if h is not None:assert v==h,str(p)
        source[str(p)]=v
    for n,p in ctx.data.input_paths.items():add(p,ctx.data.loaded_hashes[n])
    for marker in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']:
        add(TREE/marker)
        for p,h in read(TREE/marker)['files'].items():add(TREE/p,h)
    for p in [TREE/'run_manifest.json',TREE/'VERIFIED_COMPLETE',TREE/'logs/independent_verification.json',INDEX,
        F2/'control_predictions.parquet',F2/'INFO_CV_COMPLETE',F2/'run_manifest.json',OLD_FEATURES/'feature_manifest.json',
        BASE/'config.py',BASE/'protocol.py',BASE/'run_artifacts.py',
        BASE/'train_test_feature_support/data/nearest_five.parquet',BASE/'train_test_feature_support/completion.json',
        BASE/'train_test_feature_support/Growth_Exposure_Experiment_Design_2026-09-20.md']:
        add(p)
    f2marker=read(F2/'INFO_CV_COMPLETE');assert sha(F2/'control_predictions.parquet')==f2marker['files']['control_predictions.parquet']
    for p,h in read(OLD_FEATURES/'feature_manifest.json')['files'].items():add(OLD_FEATURES/p,h)
    path=OUT/'reference/source_hashes.json'
    if path.exists():assert read(path)==source
    else:save(path,source)
    design=BASE/'train_test_feature_support/Growth_Exposure_Experiment_Design_2026-09-20.md';dest=OUT/'reference/approved_design.md'
    if dest.exists():assert dest.read_bytes()==design.read_bytes()
    else:atomic(dest,lambda q:q.write_bytes(design.read_bytes()))
    ctx.sources=source;return ctx

def initialize(ctx):
    fm=read(OUT/'features/feature_manifest.json')
    for p,h in fm['files'].items():assert sha(OUT/'features'/p)==h,p
    ctx.X=pd.read_parquet(OUT/'features/train_191.parquet');ctx.Xt=pd.read_parquet(OUT/'features/test_191.parquet')
    assert list(ctx.X)==list(ctx.X0)+NEW_COLUMNS and ctx.X.shape==(34416,191)
    pd.testing.assert_frame_equal(ctx.X[list(ctx.X0)],ctx.X0);pd.testing.assert_frame_equal(ctx.Xt[list(ctx.Xt0)],ctx.Xt0)
    votes=pd.read_parquet(BASE/'train_test_feature_support/data/nearest_five.parquet');votes=votes[(votes.scope=='test_to_train')&(votes.view=='context')]
    count=votes.groupby(['horizon','target_day','matched_fips']).fipsCode.nunique();rows=[]
    for h in HORIZONS:
        hh=HORIZON_HOURS[h]
        for day in range((72+hh)//24,9):
            for f in ctx.meta.fipsCode.unique():rows.append(dict(horizon=hh,target_day=day,fipsCode=f,test_like=int(count.get((hh,day,f),0))>0,test_votes=int(count.get((hh,day,f),0))))
    member=pd.DataFrame(rows);mp=OUT/'reference/test_like_membership.parquet'
    if mp.exists():pd.testing.assert_frame_equal(pd.read_parquet(mp),member)
    else:frame(mp,member)
    cfg=dict(protocol='v18_pweather_v1',cv_seed=42,model_seed=42,cases=['W0','W1'],changed_source='P24_direct',reference_tree_id=TREE_ID,
        new_columns=NEW_COLUMNS,window_points=48,wind_threshold_mph=30,strict_threshold=True,decay_constants=[6,24],missing_age_code=49,
        sum_method='chronological_math_fsum_float64',params=params(),max_rounds=2000,early_stopping=100,refit_rule='floor_mean_four_inner_best',
        fits=25,outer_models=5,saved_inner_models=20,bootstrap_replicates=2000,bootstrap_seed=20260910,
        membership_sha256=sha(mp),protocol_sha256=sha(OUT/'Experiment_Protocol_2026-09-21.md'),plots=False,test_inference=False)
    cp=OUT/'experiment_config.json'
    if cp.exists():assert read(cp)==clean(cfg)
    else:save(cp,cfg)
    env=dict(python=platform.python_version(),libraries={p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow','lightgbm','scipy','scikit-learn']})
    prior=read(TREE/'run_manifest.json')['identity']['environment'];assert env['python']==prior['python'] and env['libraries']==prior['libraries']
    ctx.identity=dict(protocol=cfg['protocol'],config_sha256=sha(cp),source_hashes=ctx.sources,feature_manifest_sha256=sha(OUT/'features/feature_manifest.json'),
        feature_columns=list(ctx.X),training_code_sha256={p:sha(OUT/p) for p in TRAIN_CODE},environment=env)
    ctx.identity_hash=digest(ctx.identity);ctx.membership=member;return ctx

def subset(ctx,folds):
    folds=tuple(sorted(set(folds)));assert len(folds) in (1,3,4) and set(folds)<=set(range(5))
    rows=np.flatnonzero(ctx.valid&np.isin(ctx.meta.fold.to_numpy(),folds));assert len(rows)>0
    y=ctx.y[rows].copy();assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
    return SimpleNamespace(rows=rows,X=ctx.X.iloc[rows].reset_index(drop=True),y=y,folds=folds,meta=ctx.meta.iloc[rows].reset_index(drop=True))
def describe(s):
    return dict(folds=list(s.folds),counties=sorted(s.meta.fipsCode.unique()),rows=len(s.rows),row_sha256=arrsha(s.rows),
        feature_sha256=arrsha(s.X.to_numpy()),label_sha256=arrsha(s.y),metric_label_sha256=arrsha(s.y.astype(np.float32).astype(float)))
