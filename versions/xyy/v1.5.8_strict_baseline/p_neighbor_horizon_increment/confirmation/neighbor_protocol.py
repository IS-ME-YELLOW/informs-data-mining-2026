"""Current P6-enhanced baseline and feature-only neighbor experiments."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

from confirmation_runtime import CONFIRMATION,PARENT,OUT,BASE,ROOT,SEED,CV_FILE,REFERENCE_ID
sys.path.insert(0,str(BASE))
from protocol import load_data as _baseline_load_data,expected_mask,build_controls,clip_component,component_target_name
from config import HORIZONS,HORIZON_HOURS,COMPONENTS,make_lgbm_params
from run_artifacts import atomic_write,digest_object,sha256_file,environment

RUN=OUT/f'runs/v18_pneighbor_v1_split{SEED}_model42'
PTRANSFER=BASE/'p_structure_horizon_transfer'
CURRENT=PTRANSFER/f'confirmation/split{SEED}/runs/v18_ptransfer_ab_v1_split{SEED}_model42'
F2=BASE/f'p_information_increment/runs/v18_pinfo_v1_split{SEED}_model42'
OLD_FEATURES=BASE/'p_information_increment/features/v1'
FEATURES=PARENT/'features/v1'
SOURCE_MANIFEST=PTRANSFER/'confirmation/summary/candidate_manifest.json'
CHANGED=('osi_target_t24h',)
CASES={'B0':None,'NB_P24':CHANGED[0]}
TASKS=((CHANGED[0],'direct'),)
KEYS=['fipsCode','timestamp_et','hour_idx','fold']
VIEW_KEYS=['fipsCode','timestamp_et','hour_idx']
HISTORY=['last_P_t','last_N_t','last_D_t','last_R_t','osi_trend_last6h']
TRAIN_CODE=('confirmation_runtime.py','neighbor_protocol.py','neighbor_features.py','independent_features.py','train_neighbors.py')

def load_data():
    return _baseline_load_data(cv_file=CV_FILE)


def safe(path):
    p=Path(path).resolve()
    if p==OUT or not p.is_relative_to(OUT):raise ValueError(f'Output escapes experiment: {p}')
    return p


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,v):atomic_write(safe(p),lambda q:q.write_text(json.dumps(clean(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n'))
def write_frame(p,d):
    p=safe(p);d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)
    atomic_write(p,lambda q:d.to_parquet(q,index=False) if p.suffix=='.parquet' else d.to_csv(q,index=False))
def array_sha(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()


def normalize(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    assert not d.duplicated(['fipsCode','hour_idx']).any()
    return d.sort_values(['fipsCode','hour_idx']).reset_index(drop=True)


def whitelist(h):return HISTORY+['gust_t',f'gust_at_t{h}h','gust_max_next_6h','gust_mean_next_6h']
def added_columns(h):return [prefix+q for q in whitelist(h) for prefix in ('nbr8_mean_','nbr8_max_')]+['self_minus_nbr8_mean_last_P_t',f'self_minus_nbr8_mean_gust_at_t{h}h']


def params(kind):
    p=make_lgbm_params(42)
    if kind!='direct':p.update(objective='regression',metric='None')
    return p


def support(p,kind):return np.ones(len(p),bool) if kind=='direct' else (p>0 if kind=='a' else p<1)


def transform(y,p,kind):
    assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all() and support(p,kind).all()
    if kind=='direct':return y.copy(),np.ones(len(y))
    if kind=='a':return np.minimum(y,p)/p,p**2
    return np.maximum(y-p,0)/(1-p),(1-p)**2


def reconstruct(p,a,b,valid):
    applied=[]
    for kind,raw in [('a',a),('b',b)]:
        hit=valid&support(p,kind);assert np.isfinite(raw[hit]).all() and np.isnan(raw[~hit]).all()
        v=np.where(valid,0.,np.nan);v[hit]=np.clip(raw[hit],0,1);applied.append(v)
    ca=p*applied[0];cb=(1-p)*applied[1]
    return np.clip(ca+cb,0,1),applied[0],applied[1],ca,cb


def load_context():
    data=load_data();meta=data.meta_train.copy();meta['fold']=data.row_folds
    item=next(v for v in read_json(SOURCE_MANIFEST)['splits'] if v['split_seed']==SEED)
    assert item['case']=='AB_P06' and item['run_identity_hash']==REFERENCE_ID and Path(item['run_directory'])==CURRENT
    for rec in item['files'].values():assert sha256_file(rec['path'])==rec['sha256'],rec['path']
    part=normalize(pd.read_parquet(CURRENT/'component_predictions.parquet'))
    saved=normalize(pd.read_parquet(CURRENT/'control_predictions.parquet'))
    original=normalize(pd.read_parquet(F2/'control_predictions.parquet'))
    for d in (part,saved,original):pd.testing.assert_frame_equal(d[KEYS],meta[KEYS],check_dtype=False)
    assert part.candidate_identity_hash.eq(REFERENCE_ID).all() and saved.candidate_identity_hash.eq(REFERENCE_ID).all()
    parts={h:{c:part[f'pred_AB_P06_{component_target_name(c,h)}'].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS}
    direct={h:original[f'raw_direct_{h}'].to_numpy(copy=True) for h in HORIZONS}
    controls,aux=build_controls(meta,direct,parts)
    for h in HORIZONS:
        for mode in controls:np.testing.assert_allclose(controls[mode][h],saved[f'pred_AB_P06_{mode}_{h}'],atol=1e-12,rtol=0)
    assert read_json(CURRENT/'logs/independent_verification.json')['status']=='PASS'
    raw=pd.read_csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str},float_precision='round_trip')
    hist=raw.loc[pd.to_datetime(raw.timestamp_et)==pd.Timestamp('2026-03-13 23:00')].set_index('fipsCode')
    p=meta.fipsCode.map(hist.P_t).to_numpy(float);np.testing.assert_array_equal(p,data.X_train.last_P_t)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    prior_env=read_json(CURRENT/'run_manifest.json')['identity']['environment'];now=environment()
    assert prior_env['libraries']==now['libraries'] and prior_env['python']==now['python']
    return SimpleNamespace(data=data,meta=meta,p=p,parts=parts,direct=direct,controls=controls,aux=aux,
        reference_parts=part,reference_controls=saved,names=hist[['countyName','stateAbbr']])


def snapshot_sources(ctx):
    expected={}
    def add(p,h=None):
        p=str(Path(p).resolve())
        if h is not None and expected.get(p) is not None:assert expected[p]==h,p
        expected[p]=h if h is not None else expected.get(p)
    m=read_json(CURRENT/'run_manifest.json');assert digest_object(m['identity'])==m['identity_hash']==REFERENCE_ID
    for p,h in m['identity']['source_sha256'].items():add(p,h)
    for name in ('MODEL_CV_COMPLETE','EVALUATION_COMPLETE'):
        marker=read_json(CURRENT/name);assert marker['identity_hash']==REFERENCE_ID
        add(CURRENT/name)
        for p,h in marker['files'].items():add(CURRENT/p,h)
    for name in ('VERIFIED_COMPLETE','logs/independent_verification.json','run_manifest.json'):add(CURRENT/name)
    add(SOURCE_MANIFEST)
    add(CONFIRMATION/'reference/adaptation_verification.json')
    add(CONFIRMATION/'Confirmation_Protocol_2026-09-20.md')
    add(FEATURES/'feature_manifest.json')
    for path,digest in read_json(FEATURES/'feature_manifest.json')['files'].items():add(FEATURES/path,digest)
    fm=read_json(OLD_FEATURES/'feature_manifest.json');assert digest_object(fm['identity'])==fm['identity_hash']
    add(OLD_FEATURES/'feature_manifest.json')
    for p,h in fm['files'].items():add(OLD_FEATURES/p,h)
    for key,p in ctx.data.input_paths.items():add(p,ctx.data.loaded_hashes[key])
    for name in ('transfer_protocol.py','verify_transfer.py','train_transfer.py','evaluate_transfer.py'):add(PTRANSFER/name)
    for name in ('build_information_features.py','independent_features.py','verify_information_exact.py'):add(BASE/'p_information_increment'/name)
    add(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv')
    result={}
    for p,h in sorted(expected.items()):
        current=sha256_file(p)
        if h is not None:assert current==h,p
        result[p]=current
    old=OUT/'reference/initial_input_hashes.json'
    if old.exists():assert read_json(old)==result
    else:write_json(old,result)
    return result


def initialize(ctx):
    cfg=dict(protocol='v18_pneighbor_v1',split_seed=SEED,model_seed=42,reference_id=REFERENCE_ID,reference_case='AB_P06',
        cases=CASES,tasks=TASKS,feature_count=183,new_features={h:added_columns(HORIZON_HOURS[h]) for h in CHANGED},
        neighbors=8,history_end=71,exposure_window='[t,min(t+6,215)] including origin; unchanged',
        max_rounds=2000,early_stopping_rounds=100,bootstrap_replicates=2000,bootstrap_seed=20260910,
        params={k:params(k) for k in ('a','b','direct')},formal_fits=25,new_outer_models=5,
        root=str(ROOT),output=str(OUT),run=str(RUN),protocol_sha256=sha256_file(CONFIRMATION/'Confirmation_Protocol_2026-09-20.md'))
    path=OUT/'experiment_config.json'
    if path.exists():assert read_json(path)==clean(cfg)
    else:write_json(path,cfg)
    ctx.sources=snapshot_sources(ctx)
    return ctx


def attach_features(ctx):
    m=read_json(FEATURES/'feature_manifest.json');assert digest_object(m['identity'])==m['identity_hash']
    for p,h in m['files'].items():assert sha256_file(FEATURES/p)==h,p
    ctx.X={h:pd.read_parquet(FEATURES/f'h{HORIZON_HOURS[h]:02d}_train.parquet') for h in CHANGED}
    for h,x in ctx.X.items():
        pd.testing.assert_frame_equal(x[list(ctx.data.X_train)],ctx.data.X_train)
        assert list(x)==list(ctx.data.X_train)+added_columns(HORIZON_HOURS[h]) and len(x.columns)==183
    ctx.feature_identity=m['identity_hash']
    ctx.identity=dict(protocol='v18_pneighbor_v1',config_sha256=sha256_file(OUT/'experiment_config.json'),
        source_sha256=ctx.sources,feature_identity=m['identity_hash'],feature_manifest_sha256=sha256_file(FEATURES/'feature_manifest.json'),
        training_code_sha256={p:sha256_file(CONFIRMATION/p) for p in TRAIN_CODE},environment=environment(),reference_identity=REFERENCE_ID,
        p71_sha256=array_sha(ctx.p),feature_names={h:list(x) for h,x in ctx.X.items()})
    ctx.identity_hash=digest_object(ctx.identity)
    return ctx


def subset(ctx,folds,h,kind):
    folds=tuple(sorted(folds));assert len(folds) in (1,3,4) and (h,kind) in TASKS
    eligible=np.isin(ctx.data.row_folds,folds)&expected_mask(ctx.meta,h)
    rows=np.flatnonzero(eligible&support(ctx.p,kind))
    y=ctx.data.component_targets[component_target_name('P_t',h)].iloc[rows].to_numpy(float,copy=True)
    p=ctx.p[rows].copy();labels,weights=transform(y,p,kind)
    assert len(labels)>0 and np.isfinite(weights).all() and (weights>0).all()
    return SimpleNamespace(X=ctx.X[h].iloc[rows].reset_index(drop=True),y=labels,w=weights,p=p,rows=rows,
        meta=ctx.meta.iloc[rows].reset_index(drop=True),folds=folds,total_valid=int(eligible.sum()))


def describe(s):
    return dict(folds=list(s.folds),counties=sorted(s.meta.fipsCode.unique()),rows=len(s.y),total_valid=s.total_valid,
        row_sha=array_sha(s.rows),feature_sha=array_sha(s.X.to_numpy()),label_sha=array_sha(s.y),p_sha=array_sha(s.p),
        weight_sha=array_sha(s.w),weight_mean=float(s.w.mean()),normalized_weight_sha=array_sha(s.w/s.w.mean()))


def specification(ctx,outer,h,kind,identity):
    allowed=[f for f in range(5) if f!=outer]
    return dict(identity_hash=identity,outer_fold=outer,horizon=h,kind=kind,allowed_folds=allowed,
        feature_names=list(ctx.X[h]),params=params(kind),
        probes=[dict(inner_fold=q,train=describe(subset(ctx,[f for f in allowed if f!=q],h,kind)),
                     validation=describe(subset(ctx,[q],h,kind))) for q in allowed],
        refit=describe(subset(ctx,allowed,h,kind)))


def contribution_metric(y,w,n):
    y,w=np.asarray(y).copy(),np.asarray(w).copy()
    def evaluate(pred,dataset):return 'raw_P_contribution_rmse',float(np.sqrt(np.sum(w*(pred-y)**2)/n)),False
    return evaluate


def assert_inputs(ctx):
    assert all(sha256_file(p)==h for p,h in ctx.sources.items())
    assert all(sha256_file(CONFIRMATION/p)==h for p,h in ctx.identity['training_code_sha256'].items())
    assert sha256_file(FEATURES/'feature_manifest.json')==ctx.identity['feature_manifest_sha256']


def build_cases(ctx,values):
    parts={};controls={};aux={}
    for case,changed in CASES.items():
        p={h:{c:v.copy() for c,v in comps.items()} for h,comps in ctx.parts.items()}
        if changed:p[changed]['P_t']=values[changed]
        parts[case]=p;controls[case],aux[case]=build_controls(ctx.meta,ctx.direct,p)
    return parts,controls,aux
