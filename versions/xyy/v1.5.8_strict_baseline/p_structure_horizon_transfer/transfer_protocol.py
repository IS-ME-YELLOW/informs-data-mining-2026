"""Frozen F2 reference and exact P a/b transfer to three independent sources."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(BASE))
from protocol import load_data,expected_mask,build_controls,clip_component,component_target_name
from config import HORIZONS,HORIZON_HOURS,COMPONENTS,make_lgbm_params
from run_artifacts import atomic_write,digest_object,sha256_file,environment

RUN=OUT/'runs/v18_ptransfer_ab_v1_split42_model42'
REF=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42'
F2_ID='e39b76d75f191891abda1939dc47933815cc4fa15796227b0c200eb1b96bfcac'
TRANSFER=tuple(h for h in HORIZONS if HORIZON_HOURS[h]>1)
KINDS=('a','b')
CASES={'B0':None,'AB_P06':TRANSFER[0],'AB_P24':TRANSFER[1],'AB_P48':TRANSFER[2]}
KEYS=['fipsCode','timestamp_et','hour_idx','fold']
TRAIN_CODE=('transfer_protocol.py','train_transfer.py')


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


def write_json(path,value):atomic_write(safe(path),lambda p:p.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n'))


def write_frame(path,value):
    p=safe(path);d=value if isinstance(value,pd.DataFrame) else pd.DataFrame(value)
    atomic_write(p,lambda q:d.to_parquet(q,index=False) if p.suffix=='.parquet' else d.to_csv(q,index=False))


def array_sha(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()


def normalize(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    assert not d.duplicated(['fipsCode','hour_idx']).any()
    return d.sort_values(['fipsCode','hour_idx']).reset_index(drop=True)


def params():
    p=make_lgbm_params(42);p.update(objective='regression',metric='None');return p


def model_target(h,kind):return ('P_anchor_a_' if kind=='a' else 'P_excess_b_')+h.rsplit('_',1)[-1]


def support(p,kind):
    assert kind in KINDS
    return p>0 if kind=='a' else p<1


def transform(y,p,kind):
    assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and support(p,kind).all()
    if kind=='a':return np.minimum(y,p)/p,p**2
    return np.maximum(y-p,0)/(1-p),(1-p)**2


def reconstruct(p,raw_a,raw_b,valid):
    applied=[]
    for kind,raw in [('a',raw_a),('b',raw_b)]:
        active=valid&support(p,kind)
        assert np.isfinite(raw[active]).all() and np.isnan(raw[~active]).all()
        values=np.where(valid,0.,np.nan);values[active]=np.clip(raw[active],0,1);applied.append(values)
    ca=p*applied[0];cb=(1-p)*applied[1];out=ca+cb
    assert np.all(out[valid]>=-1e-12) and np.all(out[valid]<=1+1e-12)
    return np.clip(out,0,1),applied[0],applied[1],ca,cb


def load_context():
    data=load_data();meta=data.meta_train.copy();meta['fold']=data.row_folds
    part=normalize(pd.read_parquet(REF/'component_predictions.parquet'));saved=normalize(pd.read_parquet(REF/'control_predictions.parquet'))
    for d in (part,saved):
        pd.testing.assert_frame_equal(d[KEYS],meta[KEYS],check_dtype=False)
        assert d.candidate_identity_hash.eq(F2_ID).all()
    raw=pd.read_csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str},float_precision='round_trip')
    raw['timestamp_et']=pd.to_datetime(raw.timestamp_et)
    hist=raw.loc[raw.timestamp_et==pd.Timestamp('2026-03-13 23:00')].set_index('fipsCode')
    assert not hist.index.duplicated().any() and set(hist.index)==set(meta.fipsCode)
    p=meta.fipsCode.map(hist.P_t).to_numpy(float)
    np.testing.assert_array_equal(p,data.X_train.last_P_t)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    parts={h:{c:part[f'pred_F2_{component_target_name(c,h)}'].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS}
    direct={h:saved[f'raw_direct_{h}'].to_numpy(copy=True) for h in HORIZONS}
    controls,aux=build_controls(meta,direct,parts)
    for h in HORIZONS:
        for mode in controls:np.testing.assert_allclose(controls[mode][h],saved[f'pred_F2_{mode}_{h}'],atol=1e-12,rtol=0)
    assert read_json(REF/'logs/independent_verification.json')['status']=='PASS'
    old=read_json(REF/'run_manifest.json')['identity']['environment'];now=environment()
    assert old['libraries']==now['libraries'] and old['python']==now['python']
    return SimpleNamespace(data=data,meta=meta,p=p,parts=parts,direct=direct,controls=controls,aux=aux,
        reference_parts=part,reference_controls=saved,names=hist[['countyName','stateAbbr']])


def sources(ctx):
    expected={}
    def add(path,digest=None):
        path=str(Path(path).resolve())
        if digest is not None and expected.get(path) is not None:assert expected[path]==digest,path
        expected[path]=digest if digest is not None else expected.get(path)
    m=read_json(REF/'run_manifest.json');marker=read_json(REF/'INFO_CV_COMPLETE')
    assert digest_object(m['identity'])==m['identity_hash']==marker['identity_hash']==F2_ID
    for name in ('run_manifest.json','INFO_CV_COMPLETE','source_manifest.json'):add(REF/name)
    for name,digest in marker['files'].items():
        path=(REF/name).resolve();assert path.is_relative_to(REF);add(path,digest)
    for path,digest in read_json(REF/'source_manifest.json')['source_sha256'].items():add(path,digest)
    for key,path in ctx.data.input_paths.items():add(path,ctx.data.loaded_hashes[key])
    for name in ('p_structure_protocol.py','weighted_scoped_training.py','Experiment_Plan_2026-09-18.md'):
        add(BASE/'p_initial_state_structure'/name)
    add(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv')
    result={}
    for path,digest in sorted(expected.items()):
        actual=sha256_file(path)
        if digest is not None:assert actual==digest,path
        result[path]=actual
    p=OUT/'reference/initial_input_hashes.json'
    if p.exists():assert read_json(p)==result,'Sources changed'
    else:write_json(p,result)
    return result


def initialize(ctx):
    config=dict(protocol='v18_ptransfer_ab_v1',split_seed=42,model_seed=42,reference_F2_identity=F2_ID,
        transfer_horizons=TRANSFER,cases=CASES,feature_count=163,params=params(),max_rounds=2000,
        early_stopping_rounds=100,bootstrap_replicates=2000,bootstrap_seed=20260910,
        formal_fits=150,new_outer_models=30,root=str(ROOT),output=str(OUT),run=str(RUN),
        weighting='raw p^2 or (1-p)^2; normalize by fit-training support mean only',
        early_stop_metric='sqrt(sum(raw_validation_weight * branch_error^2)/all_valid_validation_rows)',
        authorized_scope='seed42; three single-source AB migrations only; no joint or feature increment',
        protocol_sha256=sha256_file(OUT/'Experiment_Protocol_2026-09-19.md'))
    cfg=OUT/'experiment_config.json'
    if cfg.exists():assert read_json(cfg)==clean(config)
    else:write_json(cfg,config)
    ctx.sources=sources(ctx)
    ctx.identity=dict(protocol=config['protocol'],config_sha256=sha256_file(cfg),source_sha256=ctx.sources,
        training_code_sha256={p:sha256_file(OUT/p) for p in TRAIN_CODE},environment=environment(),
        p71_sha256=array_sha(ctx.p),feature_names=list(ctx.data.X_train),reference_F2_identity=F2_ID)
    ctx.identity_hash=digest_object(ctx.identity)
    return ctx


def subset(ctx,folds,h,kind):
    folds=tuple(sorted(folds));assert len(folds) in (1,3,4) and set(folds).issubset(range(5))
    eligible=np.isin(ctx.data.row_folds,folds)&expected_mask(ctx.meta,h)
    rows=np.flatnonzero(eligible&support(ctx.p,kind))
    official=ctx.data.component_targets[component_target_name('P_t',h)].iloc[rows].to_numpy(float,copy=True)
    p=ctx.p[rows].copy();y,w=transform(official,p,kind)
    assert len(y)>0 and np.isfinite(w).all() and (w>0).all()
    return SimpleNamespace(X=ctx.data.X_train.iloc[rows].reset_index(drop=True),y=y,w=w,p=p,rows=rows,
        meta=ctx.meta.iloc[rows].reset_index(drop=True),folds=folds,total_valid=int(eligible.sum()),
        excluded_rows=np.flatnonzero(eligible&~support(ctx.p,kind)))


def describe(s):
    return dict(folds=list(s.folds),counties=sorted(s.meta.fipsCode.unique()),rows=len(s.y),total_valid_rows=s.total_valid,
        omitted_rows=len(s.excluded_rows),row_indices_sha256=array_sha(s.rows),
        row_keys_sha256=digest_object((s.meta.fipsCode+'|'+s.meta.timestamp_et.astype(str)).tolist()),
        omitted_indices_sha256=array_sha(s.excluded_rows),label_sha256=array_sha(s.y),
        feature_sha256=array_sha(s.X.to_numpy()),p_sha256=array_sha(s.p),weight_sha256=array_sha(s.w),
        weight_mean=float(s.w.mean()),weight_min=float(s.w.min()),weight_max=float(s.w.max()),
        normalized_weight_sha256=array_sha(s.w/s.w.mean()),row_weight_ess=float(s.w.sum()**2/np.sum(s.w**2)))


def specification(ctx,outer,h,kind,identity):
    allowed=[f for f in range(5) if f!=outer]
    return dict(identity_hash=identity,outer_fold=outer,horizon=h,kind=kind,target=model_target(h,kind),
        allowed_folds=allowed,feature_names=list(ctx.data.X_train),params=params(),
        probes=[dict(inner_fold=q,train=describe(subset(ctx,[f for f in allowed if f!=q],h,kind)),
                     validation=describe(subset(ctx,[q],h,kind))) for q in allowed],
        refit=describe(subset(ctx,allowed,h,kind)))


def contribution_metric(y,w,n_all):
    y,w=np.asarray(y).copy(),np.asarray(w).copy()
    assert n_all>=len(y)>0
    def evaluate(pred,dataset):return 'raw_P_contribution_rmse',float(np.sqrt(np.sum(w*(pred-y)**2)/n_all)),False
    return evaluate


def assert_inputs(ctx):
    assert all(sha256_file(p)==h for p,h in ctx.sources.items())
    assert all(sha256_file(OUT/p)==h for p,h in ctx.identity['training_code_sha256'].items())


def build_cases(ctx,structured):
    parts={};controls={};aux={}
    for case,changed in CASES.items():
        p={h:{c:v.copy() for c,v in comps.items()} for h,comps in ctx.parts.items()}
        if changed:p[changed]['P_t']=structured[changed].copy()
        parts[case]=p;controls[case],aux[case]=build_controls(ctx.meta,ctx.direct,p)
    return parts,controls,aux
