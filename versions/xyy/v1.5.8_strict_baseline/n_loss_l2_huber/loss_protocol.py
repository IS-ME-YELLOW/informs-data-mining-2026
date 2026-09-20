"""Frozen N-loss experiment: source identity, scoped labels and pure reconstruction."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.metadata
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=BASE.parents[2]
sys.path.insert(0,str(BASE))
from config import make_lgbm_params,HORIZONS,HORIZON_HOURS,COMPONENTS
from protocol import load_data,expected_mask,build_controls,clip_component,component_target_name
from run_artifacts import digest_object,sha256_file,atomic_write,environment

RUN=OUT/'runs/v18_nloss_l2_v1_split42_model42'
REF=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42'
HUBER=BASE/'runs/v18_tree_nested_v2_split42_model42'
F2_ID='e39b76d75f191891abda1939dc47933815cc4fa15796227b0c200eb1b96bfcac'
CASES=('H0_Huber','L2_N01','L2_N06','L2_N24','L2_N48','L2_all')
TRAIN_CODE=('loss_protocol.py','train_n_loss.py')
KEYS=['fipsCode','timestamp_et','hour_idx','fold']


def safe(path):
    p=Path(path).resolve()
    if p==OUT or not p.is_relative_to(OUT):raise ValueError(f'Output outside experiment: {p}')
    return p


def read_json(path):return json.loads(Path(path).read_text())


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def write_json(path,value):
    atomic_write(safe(path),lambda p:p.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n'))


def write_frame(path,value):
    p=safe(path);d=value if isinstance(value,pd.DataFrame) else pd.DataFrame(value)
    atomic_write(p,lambda q:d.to_parquet(q,index=False) if p.suffix=='.parquet' else d.to_csv(q,index=False))


def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def l2_params():
    p=make_lgbm_params(42);p['objective']='regression';return p


def snapshot_sources(data):
    expected={}
    def add(p,digest=None):
        p=str(Path(p).resolve())
        if digest is not None and expected.get(p) is not None:assert expected[p]==digest,p
        expected[p]=digest if digest is not None else expected.get(p)
    for root,marker in [(REF,'INFO_CV_COMPLETE'),(HUBER,'CV_COMPLETE')]:
        m=read_json(root/'run_manifest.json');done=read_json(root/marker)
        assert digest_object(m['identity'])==m['identity_hash']==done['identity_hash']
        if root==REF:assert m['identity_hash']==F2_ID
        add(root/'run_manifest.json');add(root/marker)
        for path,digest in done['files'].items():
            resolved=(root/path).resolve();assert resolved.is_relative_to(root);add(resolved,digest)
    sm=read_json(REF/'source_manifest.json');add(REF/'source_manifest.json')
    for p,h in sm['source_sha256'].items():add(p,h)
    add(HUBER/'verification_cv.json')
    for key,path in data.input_paths.items():add(path,data.loaded_hashes[key])
    add(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv')
    add(BASE/'n_four_horizon_diagnosis/Results_2026-09-19.md')
    actual={}
    for p,digest in sorted(expected.items()):
        current=sha256_file(p)
        if digest is not None and digest!=current:raise ValueError(f'Changed frozen source: {p}')
        actual[p]=current
    old=OUT/'reference/initial_input_hashes.json'
    if old.exists():assert read_json(old)==actual,'Sources changed since preflight'
    else:write_json(old,actual)
    return actual


def normalize(frame):
    d=frame.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    assert not d.duplicated(['fipsCode','hour_idx']).any()
    return d.sort_values(['fipsCode','hour_idx']).reset_index(drop=True)


def load_context():
    data=load_data();meta=data.meta_train.copy();meta['fold']=data.row_folds
    part=normalize(pd.read_parquet(REF/'component_predictions.parquet'))
    saved=normalize(pd.read_parquet(REF/'control_predictions.parquet'))
    raw=normalize(pd.read_parquet(HUBER/'base_predictions_cv.parquet'))
    old=normalize(pd.read_parquet(HUBER/'oof_component_predictions.parquet'))
    for d in (part,saved,raw,old):pd.testing.assert_frame_equal(d[KEYS],meta[KEYS],check_dtype=False)
    assert part.candidate_identity_hash.eq(F2_ID).all() and saved.candidate_identity_hash.eq(F2_ID).all()
    parts={h:{c:part[f'pred_F2_{component_target_name(c,h)}'].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS}
    direct={h:saved[f'raw_direct_{h}'].to_numpy(copy=True) for h in HORIZONS}
    nraw={h:raw[f'raw_{component_target_name("N_t",h)}'].to_numpy(copy=True) for h in HORIZONS}
    ctl,aux=build_controls(meta,direct,parts)
    for h in HORIZONS:
        for mode in ctl:
            np.testing.assert_allclose(ctl[mode][h],saved[f'pred_F2_{mode}_{h}'],atol=1e-12,rtol=0)
        tag=component_target_name('N_t',h)
        np.testing.assert_array_equal(parts[h]['N_t'],old['pred_'+tag])
        np.testing.assert_array_equal(clip_component(nraw[h]),parts[h]['N_t'])
        np.testing.assert_array_equal(part['source_F2_'+tag],old['source_'+tag])
    assert read_json(REF/'logs/independent_verification.json')['status']=='PASS'
    assert read_json(HUBER/'verification_cv.json')['status']=='passed'
    oldenv=read_json(HUBER/'run_manifest.json')['identity']['environment']
    now=environment();assert oldenv['libraries']==now['libraries'] and oldenv['python']==now['python']
    return SimpleNamespace(data=data,meta=meta,parts=parts,direct=direct,nraw=nraw,ctl=ctl,aux=aux,
                           reference_parts=part,reference_controls=saved)


def scope(data,folds,target):
    folds=tuple(sorted(folds));assert len(folds) in (1,3,4) and set(folds).issubset(range(5))
    h='osi_target_'+target.rsplit('_',1)[-1]
    rows=np.flatnonzero(np.isin(data.row_folds,folds)&expected_mask(data.meta_train,h))
    # Restrict county/row scope BEFORE materializing any supervision array.
    y=data.component_targets[target].iloc[rows].to_numpy(dtype=float,copy=True)
    assert len(y)>0 and np.isfinite(y).all()
    return SimpleNamespace(X=data.X_train.iloc[rows].reset_index(drop=True),y=y,rows=rows,
         meta=data.meta_train.iloc[rows].reset_index(drop=True),folds=folds,target=target)


def describe(s):
    return dict(folds=list(s.folds),counties=sorted(s.meta.fipsCode.unique()),rows=len(s.y),
        row_indices_sha256=array_sha(s.rows),row_keys_sha256=digest_object((s.meta.fipsCode+'|'+s.meta.timestamp_et.astype(str)).tolist()),
        feature_sha256=array_sha(s.X.to_numpy()),label_sha256=array_sha(s.y))


def specification(ctx,outer,h,identity):
    target=component_target_name('N_t',h);allowed=tuple(f for f in range(5) if f!=outer)
    probes=[]
    for q in allowed:
        probes.append(dict(inner_fold=q,train=describe(scope(ctx.data,[f for f in allowed if f!=q],target)),
                           validation=describe(scope(ctx.data,[q],target))))
    return dict(identity_hash=identity,outer_fold=outer,target=target,allowed_folds=list(allowed),
        feature_names=list(ctx.data.X_train),params=l2_params(),probes=probes,
        refit=describe(scope(ctx.data,allowed,target)))


def initialize(ctx):
    config=dict(protocol='v18_nloss_l2_v1',split_seed=42,model_seed=42,horizons=list(HORIZONS),
        source_F2_identity=F2_ID,feature_count=163,cases=CASES,primary_candidate='L2_all',
        max_rounds=2000,early_stopping_rounds=100,bootstrap_replicates=2000,bootstrap_seed=20260910,
        huber_params=make_lgbm_params(42),l2_params=l2_params(),training_count=100,new_outer_models=20,
        root=str(ROOT),output=str(OUT),run=str(RUN),feature_path=str(ctx.data.input_paths['features_train']),
        scope='seed42 first round authorized; no final training or automatic CV confirmation',
        protocol_sha256=sha256_file(OUT/'Experiment_Protocol_2026-09-19.md'))
    p=OUT/'experiment_config.json'
    if p.exists():assert read_json(p)==clean(config)
    else:write_json(p,config)
    sources=snapshot_sources(ctx.data)
    identity=dict(protocol=config['protocol'],config_sha256=sha256_file(p),source_sha256=sources,
        training_code_sha256={name:sha256_file(OUT/name) for name in TRAIN_CODE},environment=environment(),
        reference_F2_identity=F2_ID,feature_names=list(ctx.data.X_train))
    ctx.identity=identity;ctx.identity_hash=digest_object(identity);ctx.sources=sources
    return ctx


def assert_inputs(ctx):
    assert all(sha256_file(p)==h for p,h in ctx.sources.items()),'Existing input file changed'
    assert all(sha256_file(OUT/p)==h for p,h in ctx.identity['training_code_sha256'].items()),'Training code changed'


def construct_cases(ctx,new_raw):
    pp={};cc={};aa={}
    for case in CASES:
        parts={h:{c:v.copy() for c,v in comp.items()} for h,comp in ctx.parts.items()}
        for h in HORIZONS:
            if case=='L2_all' or case=='L2_N'+h.rsplit('_',1)[-1][1:3]:parts[h]['N_t']=clip_component(new_raw[h])
        pp[case]=parts;cc[case],aa[case]=build_controls(ctx.meta,ctx.direct,parts)
    return pp,cc,aa
