"""Immutable source contracts and label-free final-OSI route composition. No model imports."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]
SEEDS=(42,20260917,20260918);HS=(1,6,24,48);CASES=('B0','I1','I2','I3')
PAIRS=(('I3','I1'),('I3','I2'),('I2','I1'),('I1','B0'),('I3','B0'))
TREE_INDEX=BASE/'p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json'
T_IDS={42:'bd4468e5c37fba7e93832153bc9847f807b8ae0054e45586f3c3853e96835bbd',
       20260917:'847bc47ba2970bcebb9711cd554874f4e423d6b3d7a0df9683c93afc89dd6fcb',
       20260918:'e8cf35263d42073aaae928f05ba58ab0b1f66df89b5d9cfe7a3944345c44dfe1'}
S_IDS={42:'0a019474a437758214bb2eb3de8e13fb0a112320f6c69df0fe76b41166788be4',
       20260917:'a859aa7b9cbd9edc5121c6755f0a8eb0c32850bda08f31457187988fb7564b1b',
       20260918:'3e67a67e6fa8ed7d435ee58805a8af4584dfee8974a298d710da60ac2abe6699'}
FOCUS=('42053','39117','18013','54015','54013','39115','42131','39157','39149','54101','54105')
CODE_FILES=('integration_protocol.py','integrate_routes.py')


def tag(h):return f'osi_target_t{h:02d}h'
def read_json(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(v,ascii=True):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=ascii,allow_nan=False).encode()).hexdigest()


def safe(p):
    p=Path(p).resolve()
    if p==OUT or not p.is_relative_to(OUT):raise ValueError(f'Write outside experiment: {p}')
    p.parent.mkdir(parents=True,exist_ok=True);return p


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def write_json(p,v):safe(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_frame(p,d):
    p=safe(p);d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)
    if p.suffix=='.parquet':d.to_parquet(p,index=False)
    else:d.to_csv(p,index=False)


def normalized(d):
    d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    assert d.fipsCode.str.fullmatch(r'\d{5}').all()
    d['timestamp_et']=pd.to_datetime(d.timestamp_et)
    assert not d.duplicated(['fipsCode','timestamp_et']).any()
    return d


def ordered(d):return normalized(d).sort_values(['fipsCode','hour_idx']).reset_index(drop=True)


def metric(y,p):
    y,p=np.asarray(y),np.asarray(p);assert y.shape==p.shape
    assert np.isfinite(y).all() and np.isfinite(p).all()
    if not y.size:return dict(n=0,rmse=np.nan,mae=np.nan,sse=0.,sae=0.,bias=np.nan)
    e=p-y
    return dict(n=y.size,rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),sse=float(np.sum(e**2)),
                sae=float(np.sum(abs(e))),bias=float(np.mean(e)))


def compare(y,b,p):
    before,after=metric(y,b),metric(y,p)
    return dict(**after,baseline_rmse=before['rmse'],rmse_delta=after['rmse']-before['rmse'],
        rmse_change_pct=100*(after['rmse']/before['rmse']-1) if before['rmse'] else np.nan,
        sse_reduction=before['sse']-after['sse'])


def require_source(a,mask):
    a=np.asarray(a,dtype=float)
    if a.shape!=mask.shape or not np.isfinite(a[mask]).all() or not np.isnan(a[~mask]).all():raise ValueError('Invalid prediction shape/mask/finite coverage')
    if ((a[mask]<0)|(a[mask]>.65)|((a[mask]>0)&(a[mask]<.001))).any():raise ValueError('Source violates frozen output convention')


def compose(T,S,valid):
    assert set(T)==set(HS) and set(S)=={24,48}
    for h in HS:require_source(T[h],valid[h])
    for h in S:require_source(S[h],valid[h])
    result={case:{h:a.copy() for h,a in T.items()} for case in CASES}
    for case in ('I1','I2','I3'):result[case][48]=S[48].copy()
    result['I2'][24]=S[24].copy()
    average=(T[24]+S[24])/2.0
    clipped=np.clip(average,0.,.65);result['I3'][24]=np.where(clipped<.001,0.,clipped)
    for case in CASES:
        for h in HS:require_source(result[case][h],valid[h])
    return result,average


def paired_bootstrap(meta,y,b,p):
    if np.array_equal(b,p):return dict(ci_low=0.,ci_high=0.,replicates=0,unit='exact_invariant')
    count=np.bincount(meta.county_index,minlength=239)
    s0=np.bincount(meta.county_index,weights=(b-y)**2,minlength=239)
    s1=np.bincount(meta.county_index,weights=(p-y)**2,minlength=239)
    sample=np.random.default_rng(20260910).integers(0,239,(2000,239))
    delta=np.sqrt(s1[sample].sum(axis=1)/count[sample].sum(axis=1))-np.sqrt(s0[sample].sum(axis=1)/count[sample].sum(axis=1))
    low,high=np.quantile(delta,[.025,.975])
    return dict(ci_low=float(low),ci_high=float(high),replicates=2000,bootstrap_seed=20260910,unit='county_full_trajectory')


def initialize():
    plan=OUT/'Experiment_Plan_2026-09-20.md';approved=OUT/'reference/approved_plan.md'
    if approved.exists():assert approved.read_bytes()==plan.read_bytes()
    else:safe(approved).write_bytes(plan.read_bytes())
    config=dict(protocol='tree_stella_final_osi_v1',split_seeds=SEEDS,model_seed=42,cases=CASES,pairs=PAIRS,
        tree_identities=T_IDS,stella_identities=S_IDS,tree_case='NB_P24',stella_case='L2',
        primary_case='I3',weights_24h=[.5,.5],postprocess=dict(min=0.,max=.65,zero_below=.001),
        bootstrap_replicates=2000,bootstrap_seed=20260910,windows=[[96,119],[120,143],[144,215]],
        severity_edges=[0.,.01,.05],fixed_focus_counties=FOCUS,approved_plan_sha256=sha(approved),
        root=str(ROOT),output=str(OUT),new_fits=0,new_models=0,allow_weight_search=False)
    path=OUT/'integration_config.json'
    if path.exists():assert read_json(path)==clean(config)
    else:write_json(path,config)


def source_inventory():
    """Check the exact saved predictions and their upstream completion records, not model replay."""
    files={};checks=[]
    def add(path,expected=None,provider='shared'):
        p=Path(path).resolve();assert not p.is_relative_to(OUT)
        raw=sha(p);mode='raw'
        if expected is not None and raw!=expected:
            # Stella's documented portability rule applies only to text artifacts.
            if provider!='Stella' or p.suffix not in ('.json','.csv','.md','.py','.txt'):
                raise ValueError(f'Source bytes do not match: {p}')
            if hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
                raise ValueError(f'Source canonical digest mismatch: {p}')
            mode='LF-canonical'
        if str(p) in files:assert files[str(p)]==raw
        files[str(p)]=raw;checks.append(dict(path=str(p),provider=provider,actual_sha256=raw,expected_sha256=expected,comparison=mode))
    add(TREE_INDEX);items=read_json(TREE_INDEX)['splits']
    for seed in SEEDS:
        item=next(x for x in items if x['split_seed']==seed);tr=Path(item['run_directory']).resolve()
        expected_tr=(BASE/'p_neighbor_horizon_increment' if seed==42 else BASE/f'p_neighbor_horizon_increment/confirmation/split{seed}')/f'runs/v18_pneighbor_v1_split{seed}_model42'
        assert tr==expected_tr and item['case']=='NB_P24' and item['run_identity_hash']==T_IDS[seed]
        for rec in item['files'].values():
            assert Path(rec['path']).resolve().is_relative_to(tr);add(rec['path'],rec['sha256'],'tree')
        tm=read_json(tr/'run_manifest.json');assert digest(tm['identity'])==tm['identity_hash']==T_IDS[seed]
        tv=read_json(tr/'VERIFIED_COMPLETE');assert tv['identity_hash']==T_IDS[seed]
        add(tr/'logs/independent_verification.json',tv['verification_sha256'],'tree')
        assert read_json(tr/'logs/independent_verification.json')['status']=='PASS'
        te=read_json(tr/'EVALUATION_COMPLETE');assert te['identity_hash']==T_IDS[seed]
        add(tr/'metrics/primary_scores.csv',te['files']['metrics/primary_scores.csv'],'tree')
        sr=ROOT/f'versions/stella_v112_strict/runs/stella_v112_nested_v1_split{seed}_model42'
        sm=read_json(sr/'run_manifest.json');marker=read_json(sr/'CV_COMPLETE')
        assert digest(sm['identity'],ascii=False)==sm['identity_hash']==marker['identity_hash']==S_IDS[seed]
        assert marker['status']=='CV_COMPLETE';add(sr/'CV_COMPLETE',provider='Stella')
        for name in ('run_manifest.json','candidate_predictions.parquet','summary_metrics.csv','paired_county_bootstrap.csv','thresholds.parquet','cv_assignments.csv'):
            add(sr/name,marker['files'].get(name),'Stella')
        add(sr/'verification.json',marker['verification_sha256'],'Stella')
        sv=read_json(sr/'verification.json');assert sv['status']=='PASS' and sv['identity_hash']==S_IDS[seed]
        add(ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv')
    for name in ('meta_train_v1.5.6.parquet','targets_train_v1.5.6.parquet'):add(ROOT/'versions/xyy/v1.5.6'/name)
    add(ROOT/'data/DM_Train.csv')
    add(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv')
    previous=OUT/'reference/initial_input_hashes.json'
    if previous.exists():assert read_json(previous)==files,'Frozen input set or bytes changed'
    else:write_json(previous,files)
    schema=[]
    for path in files:
        if path.endswith('.parquet'):
            pf=pq.ParquetFile(path);schema.append(dict(path=path,rows=pf.metadata.num_rows,schema=str(pf.schema_arrow)))
    write_json(OUT/'reference/source_manifest.json',dict(files=files,checks=checks,parquet_schemas=schema,
        verification_boundary='saved prediction provenance and upstream PASS records; no source-model reload',source_models_reloaded=0))
    return files


def load_context(seed):
    assert seed in SEEDS
    item=next(x for x in read_json(TREE_INDEX)['splits'] if x['split_seed']==seed)
    tr=Path(item['run_directory']);sr=ROOT/f'versions/stella_v112_strict/runs/stella_v112_nested_v1_split{seed}_model42'
    meta=normalized(pd.read_parquet(ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet'))
    labels=pd.read_parquet(ROOT/'versions/xyy/v1.5.6/targets_train_v1.5.6.parquet')
    assert len(meta)==len(labels)==34416 and labels.index.equals(meta.index)
    official=meta.copy()
    for h in HS:official['truth_'+tag(h)]=labels[tag(h)]
    official=ordered(official);assert official.fipsCode.nunique()==239
    assert np.array_equal(official.hour_idx,np.tile(np.arange(72,216),239))
    assert np.array_equal(official.timestamp_et,pd.Timestamp('2026-03-11')+pd.to_timedelta(official.hour_idx,unit='h'))
    t=ordered(pd.read_parquet(tr/'control_predictions.parquet'));s=ordered(pd.read_parquet(sr/'candidate_predictions.parquet'))
    cv=pd.read_csv(ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv',dtype={'fipsCode':str},float_precision='round_trip')
    assert not cv.fipsCode.duplicated().any() and set(cv.fipsCode)==set(official.fipsCode)
    official['fold']=official.fipsCode.map(cv.set_index('fipsCode').fold).to_numpy(int)
    for d in (t,s):
        pd.testing.assert_frame_equal(d[['fipsCode','timestamp_et','hour_idx','fold']],official[['fipsCode','timestamp_et','hour_idx','fold']],check_dtype=False)
    assert t.candidate_identity_hash.eq(T_IDS[seed]).all() and s.split_id.eq(f'seed{seed}').all()
    source=official[['fipsCode','timestamp_et','hour_idx','stateAbbr','fold']].copy()
    source.insert(0,'cv_seed',seed);source['T_identity']=T_IDS[seed];source['S_identity']=S_IDS[seed]
    codes={f:i for i,f in enumerate(sorted(official.fipsCode.unique()))}
    source['county_index']=source.fipsCode.map(codes).to_numpy(int)
    county_names=pd.read_csv(ROOT/'data/DM_Train.csv',usecols=['fipsCode','countyName','stateAbbr'],dtype={'fipsCode':str}).drop_duplicates()
    assert not county_names.fipsCode.duplicated().any()
    names=county_names.set_index('fipsCode');assert set(FOCUS)<=set(names.index)
    source['countyName']=source.fipsCode.map(names.countyName)
    tscores=pd.read_csv(tr/'metrics/primary_scores.csv',float_precision='round_trip')
    sscores=pd.read_csv(sr/'summary_metrics.csv',float_precision='round_trip')
    thresholds=pd.read_parquet(sr/'thresholds.parquet');assert len(thresholds)==10
    T={};S={};y={};valid={};metrics_rows=[]
    for h in HS:
        name=tag(h);mask=official.hour_idx.to_numpy()+h<=215
        truth=official['truth_'+name].to_numpy(float)
        assert np.array_equal(np.isfinite(truth),mask) and np.isnan(truth[~mask]).all()
        np.testing.assert_array_equal(truth,t['true_'+name]);np.testing.assert_array_equal(truth,s['truth_'+name])
        np.testing.assert_array_equal(mask,t['scoreable_'+name]);np.testing.assert_array_equal(mask,s['scoreable_'+name])
        T[h]=t['pred_NB_P24_v18_rule_'+name].to_numpy(copy=True);require_source(T[h],mask)
        row=metric(truth[mask],T[h][mask]);saved=tscores[(tscores['case']=='NB_P24')&(tscores.horizon==name)].iloc[0]
        for col in ('n','rmse','mae','sse','bias'):np.testing.assert_allclose(row[col],saved[col],atol=1e-12,rtol=0)
        metrics_rows.append(dict(cv_seed=seed,source='T',horizon=h,**row))
        for level in ('L0','L1','L2'):
            p=s[f'pred_{level}_{name}'].to_numpy();require_source(p,mask)
            row=metric(truth[mask],p[mask]);saved=sscores[(sscores.candidate==level)&(sscores.horizon==name)].iloc[0]
            for col in ('n','rmse','mae'):np.testing.assert_allclose(row[col],saved[col],atol=1e-12,rtol=0)
            metrics_rows.append(dict(cv_seed=seed,source='Stella_'+level,horizon=h,**row))
        source['truth_'+name]=truth;source['scoreable_'+name]=mask;source['target_hour_'+name]=official.hour_idx+h;source['T_'+name]=T[h]
        if h>=24:
            S[h]=s['pred_L2_'+name].to_numpy(copy=True)
            theta=s['theta_'+name].to_numpy();gate=s['anchor_gt_theta_'+name].to_numpy(bool)
            a=s['pred_L0_'+name].to_numpy();b=s['pred_L1_'+name].to_numpy()
            assert np.isfinite(theta[mask]).all() and np.isnan(theta[~mask]).all()
            np.testing.assert_array_equal(gate,mask&(a>theta))
            np.testing.assert_array_equal(S[h],np.where(gate,a,b))
            for f in range(5):
                th=thresholds[(thresholds.horizon==name)&(thresholds.outer_fold==f)]
                assert len(th)==1 and th.quantile_probability.iloc[0]==.95 and th.method.iloc[0]=='linear'
                assert np.all(theta[mask&(official.fold.to_numpy()==f)]==th.theta.iloc[0])
            for field,values in [('S',S[h]),('S_L0',a),('S_L1',b),('S_theta',theta),('S_gate',gate)]:source[field+'_'+name]=values
        valid[h]=mask;y[h]=truth
    return SimpleNamespace(seed=seed,meta=source[['cv_seed','fipsCode','timestamp_et','hour_idx','stateAbbr','countyName','fold','county_index']],
        source=source,T=T,S=S,y=y,valid=valid,metrics=pd.DataFrame(metrics_rows),tr=tr,sr=sr,run=OUT/f'runs/split{seed}')
