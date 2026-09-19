"""Prediction-only K-positive routing on immutable, same-split OOF sources."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.metadata
import json
import sys

sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'experiment_config.json').read_text())
ROOT=Path(CONFIG['project_root']).resolve();BASE=Path(CONFIG['baseline_root']).resolve()
assert HERE==Path(CONFIG['experiment_root']).resolve()==BASE/'d_unknown_contribution/k_positive_switch'
sys.path.insert(0,str(BASE))
from protocol import HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,expected_mask,build_controls,sha256_file
from run_artifacts import digest_object,write_json,write_frame,check_stage
from known_history import observed_history,known_bounds

H1='osi_target_t01h'
KEYS=['fipsCode','timestamp_et','hour_idx','stateAbbr','fold','split_id']
CODE_FILES=('switch_protocol.py','known_history.py','metric_helpers.py','independent_numerics.py',
            'gated_evaluation.py','evaluate_switch.py','verify_switch.py')


def read_json(path):return json.loads(Path(path).read_text())


def output_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE) or path==HERE:raise ValueError('Output escapes experiment directory')
    return path


def switch_d(old_d,raw_u,k):
    """Only predictions and observed-history K enter the branch decision."""
    old_d,raw_u,k=[np.asarray(v,dtype=float) for v in (old_d,raw_u,k)]
    valid=np.isfinite(k)
    if old_d.shape!=k.shape or raw_u.shape!=k.shape or np.isinf(k).any():raise ValueError('Invalid shapes or K')
    for values in (old_d,raw_u):
        if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():raise ValueError('Prediction mask mismatch')
    if not ((k[valid]>=0)&(k[valid]<=1)&(old_d[valid]>=0)&(old_d[valid]<=1)).all():raise ValueError('Use clipped original D and valid K')
    gate=valid&(k>0)
    rebuilt=np.clip(k+np.maximum(raw_u,0),0,1)
    result=old_d.copy();result[gate]=rebuilt[gate]
    return result,gate


def load_sources(seed):
    item=next((row for row in CONFIG['splits'] if row['split_seed']==seed),None)
    if item is None:raise ValueError('Unexpected split')
    run=Path(item['u_run']);marker=read_json(run/'U_CV_COMPLETE');m=read_json(run/'run_manifest.json')
    if sha256_file(run/'U_CV_COMPLETE')!=item['u_marker_sha256']:raise ValueError('U stage marker changed')
    assert marker['identity_hash']==m['identity_hash']==item['u_identity_hash']
    assert m['identity']['split_seed']==seed
    assert digest_object(m['identity'])==m['identity_hash']
    check_stage(run,'U_CV_COMPLETE',item['u_identity_hash'])
    independent=read_json(run/'logs/independent_verification.json')
    assert independent['status']=='PASS' and independent['identity_hash']==m['identity_hash']
    sources={**m['identity']['source_sha256'],**{str(run/name):digest for name,digest in marker['files'].items()},
             str(run/'U_CV_COMPLETE'):item['u_marker_sha256'],
             str(run/'logs/independent_verification.json'):sha256_file(run/'logs/independent_verification.json')}
    for name,digest in m['identity']['code_sha256'].items():sources[str(run.parent.parent/name)]=digest
    for name in ('protocol.py','config.py'):
        assert str(BASE/name) in sources and sha256_file(BASE/name)==sources[str(BASE/name)]
    for path,digest in sources.items():
        if sha256_file(path)!=digest:raise ValueError(f'Changed source: {path}')
    baseline=Path(item['baseline_run']);bm=read_json(baseline/'run_manifest.json')
    assert bm['identity_hash']==item['baseline_identity_hash']==m['identity']['baseline_identity_hash']
    check_stage(baseline,'CV_COMPLETE',item['baseline_identity_hash'])
    part=pd.read_parquet(run/'component_predictions.parquet')
    control=pd.read_parquet(run/'control_predictions.parquet')
    u=pd.read_parquet(run/'u_oof_predictions.parquet')
    truth=pd.read_parquet(baseline/'oof_component_predictions.parquet')
    meta=control[KEYS].copy()
    for frame in (part,u,truth):pd.testing.assert_frame_equal(frame[KEYS],meta)
    assert len(meta)==34416 and meta.fipsCode.nunique()==239 and not meta.duplicated(['fipsCode','timestamp_et']).any()
    assert meta.split_id.eq(bm['identity']['inputs']['cv']).all()
    for frame in (part,control,u):
        assert frame.candidate_identity_hash.eq(item['u_identity_hash']).all()
        assert frame.baseline_identity_hash.eq(item['baseline_identity_hash']).all()
    raw_history=pd.read_csv(ROOT/'data/DM_Train.csv',usecols=['fipsCode','timestamp_et','P_t'],dtype={'fipsCode':str})
    history=observed_history(raw_history);bounds=known_bounds(meta,history)
    np.testing.assert_array_equal(bounds[H1],u.K)
    direct={h:control[f'raw_direct_{h}'].to_numpy(copy=True) for h in HORIZONS}
    parts={case:{h:{c:part[f'pred_{case}_{c}_target_{h.rsplit("_",1)[-1]}'].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS} for case in ('A','B','C')}
    controls={};aux={}
    for case in ('A','B','C'):
        controls[case],aux[case]=build_controls(meta,direct,parts[case])
        for h in HORIZONS:
            for mode in MODES:np.testing.assert_array_equal(controls[case][mode][h],control[f'pred_{case}_{mode}_{h}'])
    raw_u=u.pred_U_raw.to_numpy(copy=True)
    np.testing.assert_array_equal(np.clip(bounds[H1]+np.maximum(raw_u,0),0,1),parts['C'][H1]['D_t'])
    parts['G']={h:{c:v.copy() for c,v in values.items()} for h,values in parts['A'].items()}
    parts['G'][H1]['D_t'],gate=switch_d(parts['A'][H1]['D_t'],raw_u,bounds[H1])
    np.testing.assert_array_equal(parts['G'][H1]['D_t'][gate],parts['C'][H1]['D_t'][gate])
    np.testing.assert_array_equal(parts['G'][H1]['D_t'][~gate],parts['B'][H1]['D_t'][~gate])
    controls['G'],aux['G']=build_controls(meta,direct,parts['G'])
    for h in HORIZONS:
        valid=expected_mask(meta,h)
        for c in COMPONENTS:
            if (h,c)!=(H1,'D_t'):np.testing.assert_array_equal(parts['G'][h][c],parts['A'][h][c])
        for mode in MODES:
            if h!=H1 or mode=='C0_direct_osi':np.testing.assert_array_equal(controls['G'][mode][h],controls['B'][mode][h])
            if h==H1:np.testing.assert_array_equal(controls['G'][mode][h][~gate],controls['B'][mode][h][~gate])
    data=SimpleNamespace(input_paths={'raw_train':ROOT/'data/DM_Train.csv'},
        y_train=pd.DataFrame({h:control[f'actual_{h}'].to_numpy() for h in HORIZONS}),
        component_targets=pd.DataFrame({f'{c}_target_{h.rsplit("_",1)[-1]}':truth[f'actual_{c}_target_{h.rsplit("_",1)[-1]}'].to_numpy() for h in HORIZONS for c in COMPONENTS}))
    return SimpleNamespace(seed=seed,item=item,source_run=run,sources=sources,meta=meta,data=data,
        raw_history=raw_history,history=history,bounds=bounds,source_part=part,source_control=control,source_u=u,
        raw_u=raw_u,gate=gate,parts=parts,controls=controls,aux=aux,direct=direct)


def identity_for(ctx):
    return {'protocol':CONFIG['protocol'],'split_seed':ctx.seed,'source_u_identity_hash':ctx.item['u_identity_hash'],
        'config_sha256':sha256_file(HERE/'experiment_config.json'),'code_sha256':{name:sha256_file(HERE/name) for name in CODE_FILES},
        'source_sha256':ctx.sources,'python':sys.version,'libraries':{name:importlib.metadata.version(name) for name in ('numpy','pandas','pyarrow')},
        'gate':'K>0 exactly','primary':CONFIG['primary'],'training_performed':False}


def check_prediction_boundaries(ctx):
    from independent_numerics import brute_bounds,exact,close
    for h,values in brute_bounds(ctx.raw_history,ctx.meta).items():close(values,ctx.bounds[f'osi_target_t{h:02d}h'])
    old=np.array([.2,.2,.2,np.nan]);u=np.array([.1,-.3,.1,np.nan]);k=np.array([0.,1e-12,.3,np.nan])
    result,gate=switch_d(old,u,k);exact(gate,[False,True,True,False]);close(result,[.2,1e-12,.4,np.nan])
    future=pd.to_datetime(ctx.raw_history.timestamp_et)>=pd.Timestamp('2026-03-14')
    for replacement in (np.nan,999.,np.random.default_rng(17).normal(size=int(future.sum()))):
        modified=ctx.raw_history.copy();modified.loc[future,'P_t']=replacement
        pd.testing.assert_frame_equal(observed_history(modified),ctx.history)
    pd.testing.assert_frame_equal(observed_history(ctx.raw_history.loc[~future]),ctx.history)
    changed=ctx.raw_u.copy();changed[(~ctx.gate)&np.isfinite(changed)]=999.
    exact(switch_d(ctx.parts['A'][H1]['D_t'],changed,ctx.bounds[H1])[0],ctx.parts['G'][H1]['D_t'])
    order=ctx.meta.sample(frac=1,random_state=17)
    reordered=known_bounds(order,ctx.history.sample(frac=1,random_state=19))[H1]
    r,_=switch_d(ctx.parts['A'][H1]['D_t'][order.index],ctx.raw_u[order.index],reordered)
    exact(r,ctx.parts['G'][H1]['D_t'][order.index])
    def reject(call):
        try:call()
        except ValueError:return
        raise AssertionError('Invalid input was accepted')
    reject(lambda:known_bounds(pd.concat([ctx.meta.iloc[:1]]*2),ctx.history))
    reject(lambda:observed_history(ctx.raw_history.drop(ctx.raw_history.index[0])))
    reject(lambda:output_path(HERE.parent/'forbidden'))
    reject(lambda:switch_d(np.array([.2]),np.array([np.nan]),np.array([0.])))
    valid=expected_mask(ctx.meta,H1)
    assert int(ctx.gate.sum())==784 and int(ctx.meta.loc[ctx.gate,'fipsCode'].nunique())==207
    assert ((ctx.meta.hour_idx.to_numpy()[ctx.gate]+1>=73)&(ctx.meta.hour_idx.to_numpy()[ctx.gate]+1<=76)).all()
    assert (ctx.parts['G'][H1]['D_t'][valid]>=ctx.bounds[H1][valid]).all()
    return {'status':'PASS','gate_rows':784,'gate_counties':207,'strict_positive_including_tiny_values':True,
        'future_P_and_unused_U_perturbation_invariance':True,'row_order_and_invalid_inputs':'PASS',
        'all_6_24_48h_controls_exactly_equal_B':True,'training_performed':False}
