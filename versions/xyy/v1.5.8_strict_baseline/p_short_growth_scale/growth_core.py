"""Immutable seed42 sources, label-scope isolation and frozen growth protocol."""
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
RUN=OUT/'runs/v18_pgrowth_v1_split42_model42'
F2=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42'
FEATURES=BASE/'p_information_increment/features/v1'
TREE=BASE/'p_neighbor_horizon_increment/runs/v18_pneighbor_v1_split42_model42'
INDEX=BASE/'p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json'
TREE_ID='bd4468e5c37fba7e93832153bc9847f807b8ae0054e45586f3c3853e96835bbd'
F2_ID='e39b76d75f191891abda1939dc47933815cc4fa15796227b0c200eb1b96bfcac'
H1='osi_target_t01h';PT='P_t_target_t01h';CASES=('G0','G1','G2');NEW=('G1','G2')
KEYS=['fipsCode','timestamp_et','hour_idx','fold']
TRAIN_CODE=('growth_core.py','preflight.py','train_growth.py')

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
def normalize(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    d['timestamp_et']=pd.to_datetime(d.timestamp_et);assert not d.duplicated(['fipsCode','hour_idx']).any()
    return d.sort_values(['fipsCode','hour_idx']).reset_index(drop=True)
def params():
    p=make_lgbm_params(42);p.update(objective='regression',metric='None');return p

def load_context():
    data=load_data();meta=data.meta_train.copy();meta['fold']=data.row_folds
    item=next(x for x in read(INDEX)['splits'] if x['split_seed']==42)
    assert item['case']=='NB_P24' and item['run_identity_hash']==TREE_ID and Path(item['run_directory'])==TREE
    for rec in item['files'].values():assert sha(rec['path'])==rec['sha256'],rec['path']
    tm=read(TREE/'run_manifest.json');assert tm['identity_hash']==digest(tm['identity'])==TREE_ID
    fm=read(F2/'run_manifest.json');assert fm['identity_hash']==digest(fm['identity'])==F2_ID
    assert read(F2/'verification.json')['status']=='PASS'
    feature=read(FEATURES/'feature_manifest.json');assert feature['identity_hash']==digest(feature['identity'])
    for n,h in feature['files'].items():assert sha(FEATURES/n)==h,n
    X=pd.read_parquet(FEATURES/'F2_train.parquet');assert X.shape==(34416,183)
    base_names=list(data.X_train);pd.testing.assert_frame_equal(X[base_names],data.X_train)
    part=normalize(pd.read_parquet(TREE/'component_predictions.parquet'));saved=normalize(pd.read_parquet(TREE/'control_predictions.parquet'))
    rawbranch=pd.read_parquet(F2/'branch_oof_predictions.parquet');branch=normalize(rawbranch[rawbranch['case']=='F2'])
    original=normalize(pd.read_parquet(F2/'control_predictions.parquet'))
    for d in [part,saved,branch,original]:pd.testing.assert_frame_equal(d[KEYS],meta[KEYS],check_dtype=False)
    parts={h:{c:part['pred_NB_P24_'+component_target_name(c,h)].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS}
    direct={h:original['raw_direct_'+h].to_numpy(copy=True) for h in HORIZONS}
    controls,aux=build_controls(meta,direct,parts)
    for h in HORIZONS:
        for mode in controls:np.testing.assert_array_equal(controls[mode][h],saved[f'pred_NB_P24_{mode}_{h}'])
    raw=pd.read_csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str},float_precision='round_trip')
    cutoff=raw[pd.to_datetime(raw.timestamp_et)==pd.Timestamp('2026-03-13 23:00')].set_index('fipsCode')
    p=meta.fipsCode.map(cutoff.P_t).to_numpy(float);g=meta.fipsCode.map(cutoff.N_t).to_numpy(float)
    np.testing.assert_array_equal(p,X.last_P_t);np.testing.assert_array_equal(g,X.last_N_t)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.isfinite(g).all() and (g>=0).all()
    y=data.component_targets[PT].to_numpy(float,copy=True)
    np.testing.assert_array_equal(branch.pred_P,parts[H1]['P_t'])
    aa=branch.a_applied.to_numpy(copy=True);np.testing.assert_array_equal(np.clip(p*aa+(1-p)*branch.b_applied.to_numpy(),0,1),parts[H1]['P_t'])
    early=expected_mask(meta,H1)&(meta.hour_idx.to_numpy()+1<=95)
    assert early.sum()==239*23
    return SimpleNamespace(data=data,meta=meta,X=X,p=p,g=g,y=y,early=early,parts=parts,controls=controls,aux=aux,direct=direct,
        branch=branch,a=aa,names=cutoff[['countyName','stateAbbr']],reference_parts=part,reference_saved=saved,feature_identity=feature['identity_hash'])

def initialize(ctx):
    expected={}
    def add(p,h=None):
        p=Path(p).resolve();actual=sha(p)
        if h is not None:assert actual==h,str(p)
        expected[str(p)]=actual
    for n,p in ctx.data.input_paths.items():add(p,ctx.data.loaded_hashes[n])
    for directory,marker in [(TREE,'EVALUATION_COMPLETE'),(F2,'INFO_CV_COMPLETE')]:
        add(directory/marker);m=read(directory/marker)
        for n,h in m['files'].items():add(directory/n,h)
        add(directory/'run_manifest.json')
    for n,h in read(FEATURES/'feature_manifest.json')['files'].items():add(FEATURES/n,h)
    add(FEATURES/'feature_manifest.json');add(INDEX)
    for p in [BASE/'protocol.py',BASE/'config.py',BASE/'run_artifacts.py',BASE/'p_information_increment/information_training.py',
              BASE/'train_test_feature_support/Growth_Exposure_Experiment_Design_2026-09-20.md',
              BASE/'train_test_feature_support/data/nearest_five.parquet',BASE/'train_test_feature_support/completion.json']:
        add(p)
    old=OUT/'reference/input_hashes.json'
    if old.exists():assert read(old)==expected
    else:save(old,expected)
    source_design=BASE/'train_test_feature_support/Growth_Exposure_Experiment_Design_2026-09-20.md'
    snapshot=OUT/'reference/approved_design.md'
    if snapshot.exists():assert snapshot.read_bytes()==source_design.read_bytes()
    else:atomic(snapshot,lambda q:q.write_bytes(source_design.read_bytes()))
    # Freeze the label-free auxiliary membership before any new fits.
    nearest=pd.read_parquet(BASE/'train_test_feature_support/data/nearest_five.parquet')
    nearest=nearest[(nearest.scope=='test_to_train')&(nearest.view=='context')]
    votes=nearest.groupby(['horizon','target_day','matched_fips']).fipsCode.nunique()
    rows=[]
    for h in HORIZONS:
        hh=HORIZON_HOURS[h]
        for day in range((72+hh)//24,9):
            for f in ctx.meta.fipsCode.unique():rows.append(dict(horizon=hh,target_day=day,fipsCode=f,test_like=int(votes.get((hh,day,f),0))>0,test_votes=int(votes.get((hh,day,f),0))))
    member=pd.DataFrame(rows);path=OUT/'reference/test_like_membership.parquet'
    if path.exists():pd.testing.assert_frame_equal(pd.read_parquet(path),member)
    else:frame(path,member)
    cfg=dict(protocol='v18_pgrowth_v1',split_seed=42,model_seed=42,cases=CASES,changed_source='P1_b',feature_count=183,
        reference_tree_id=TREE_ID,reference_F2_id=F2_ID,early_target_hours=[73,95],history_end=71,growth_scale_unit=.01,
        objective='regression',params=params(),max_rounds=2000,early_stopping=100,refit_round_rule='floor_mean_of_four_inner_best',
        formal_fits=50,new_outer_models=10,saved_inner_models=40,bootstrap_replicates=2000,bootstrap_seed=20260910,
        protocol_sha256=sha(OUT/'Experiment_Protocol_2026-09-20.md'),aux_membership_sha256=sha(path),no_plots=True,no_test_inference=True)
    cfgpath=OUT/'experiment_config.json'
    if cfgpath.exists():assert read(cfgpath)==clean(cfg)
    else:save(cfgpath,cfg)
    env=dict(python=platform.python_version(),libraries={p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow','lightgbm','scipy','scikit-learn']})
    oldenv=read(F2/'run_manifest.json')['identity']['environment'];assert env['python']==oldenv['python'] and env['libraries']==oldenv['libraries']
    identity=dict(protocol=cfg['protocol'],config_sha256=sha(cfgpath),source_hashes=expected,training_code_sha256={n:sha(OUT/n) for n in TRAIN_CODE},
        feature_identity=ctx.feature_identity,feature_columns=list(ctx.X),environment=env,p71_sha256=arrsha(ctx.p),N71_sha256=arrsha(ctx.g))
    ctx.identity=identity;ctx.identity_hash=digest(identity);ctx.sources=expected;ctx.membership=member
    return ctx

def scales(p,g,case):
    if case not in NEW:raise ValueError('Unknown new case')
    return (1-p)*(1+g/.01) if case=='G2' else 1-p

def subset(ctx,folds,case):
    folds=tuple(sorted(set(folds)));assert len(folds) in (1,3,4) and set(folds)<=set(range(5))
    allowed=ctx.early&np.isin(ctx.meta.fold.to_numpy(),folds)
    rows=np.flatnonzero(allowed&(ctx.p<1));assert len(rows)>0
    # No full-table transformed target: select allowed labels before transforming.
    y=ctx.y[rows].copy();p=ctx.p[rows].copy();g=ctx.g[rows].copy();scale=scales(p,g,case)
    assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all() and (scale>0).all()
    u=np.maximum(y-p,0)/scale;w=scale**2
    assert np.isfinite(u).all() and ((u>=0)&(u<=1+1e-12)).all()
    return SimpleNamespace(rows=rows,X=ctx.X.iloc[rows].reset_index(drop=True),u=u,w=w,scale=scale,p=p,g=g,
        meta=ctx.meta.iloc[rows].reset_index(drop=True),folds=folds,total_valid=int(allowed.sum()))

def describe(s):
    return dict(folds=list(s.folds),counties=sorted(s.meta.fipsCode.unique()),rows=len(s.rows),total_valid=s.total_valid,
        row_sha256=arrsha(s.rows),feature_sha256=arrsha(s.X.to_numpy()),label_sha256=arrsha(s.u),scale_sha256=arrsha(s.scale),
        weight_sha256=arrsha(s.w),weight_mean=float(s.w.mean()),normalized_weight_sha256=arrsha(s.w/s.w.mean()),
        minimum_weight=float(s.w.min()),maximum_weight=float(s.w.max()))

def reconstruct(ctx,raw,case):
    result=ctx.parts[H1]['P_t'].copy();hit=ctx.early&(ctx.p<1)
    assert np.isfinite(raw[hit]).all() and np.isnan(raw[~hit]).all()
    increment=np.zeros(len(ctx.p));c=scales(ctx.p[hit],ctx.g[hit],case)
    increment[hit]=np.clip(c*np.clip(raw[hit],0,1),0,1-ctx.p[hit])
    result[ctx.early]=np.clip(ctx.p[ctx.early]*ctx.a[ctx.early]+increment[ctx.early],0,1)
    np.testing.assert_array_equal(result[~ctx.early],ctx.parts[H1]['P_t'][~ctx.early])
    return result
