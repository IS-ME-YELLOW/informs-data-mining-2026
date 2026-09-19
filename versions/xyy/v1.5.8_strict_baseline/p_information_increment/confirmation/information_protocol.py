"""Same mathematical F2 experiment with explicit same-split E2 and immutable features."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
from confirmation_runtime import HERE,CODE_ROOT,CONFIG_PATH,META_ROOT,RUN_ID,CONFIG,SEED
ROOT=Path(CONFIG['project_root']);BASE=Path(CONFIG['baseline_root']);PBASE=Path(CONFIG['p_reference_root'])
sys.path.append(str(PBASE))
import p_structure_protocol as reference
from p_structure_protocol import (PT,H1,HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,KEYS,support,transform,
    reconstruct,params,array_sha,digest_object,read_json,sha256_file,write_json,write_frame,safe_artifact,
    expected_mask,build_controls)
from run_artifacts import check_stage,environment,data_identity

KINDS=('a','b')
TARGETS={k:reference.TARGETS[k] for k in KINDS}
CASES=('F0','F2')
NEW_CASES=('F2',)
ADAPTED=('information_protocol.py','information_preflight.py','evaluate_information.py','train_information.py',
         'verify_information.py','verify_information_exact.py','confirmation_runtime.py')
SHARED=('build_information_features.py','independent_features.py','information_training.py')


def output_path(path):
    path=Path(path).resolve()
    if path==HERE or not path.is_relative_to(HERE):raise ValueError('Output escapes information experiment')
    return path


def load_reference(seed=SEED):
    if seed!=SEED or seed not in CONFIG['authorized_split_seeds']:raise ValueError('Mismatched confirmation split')
    data=reference.load_data(ROOT/'versions/xyy/v1.5.6',ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv')
    if digest_object(data_identity(data,CONFIG['settings']))!=CONFIG['baseline_identity_hash']:
        raise ValueError('Underlying same-split strict baseline differs')
    run=Path(CONFIG['reference_run']);m=read_json(run/'run_manifest.json')
    if m['identity_hash']!=CONFIG['reference_identity_hash'] or digest_object(m['identity'])!=m['identity_hash']:
        raise ValueError('E2 reference identity mismatch')
    assert m['identity']['split_seed']==seed and m['identity']['baseline_identity_hash']==CONFIG['baseline_identity_hash']
    marker=check_stage(run,'P_CV_COMPLETE',m['identity_hash'])
    independent=read_json(run/'logs/independent_verification.json')
    assert independent['status']=='PASS' and independent['identity_hash']==m['identity_hash']
    sources=dict(m['identity']['source_sha256'])
    sources.update({str(run/n):h for n,h in marker['files'].items()})
    for n,h in m['identity']['code_sha256'].items():
        path=Path(n) if Path(n).is_absolute() else PBASE/n
        sources[str(path)]=h
    ref_config=PBASE/'confirmation/experiment_config.json'
    assert sha256_file(ref_config)==m['identity']['config_sha256']
    sources[str(ref_config)]=sha256_file(ref_config);sources[str(run/'P_CV_COMPLETE')]=sha256_file(run/'P_CV_COMPLETE')
    # Pin the previously accepted feature package and both original and corrected audit code.
    s42=Path(CONFIG['source_seed42_run']);sm=read_json(s42/'run_manifest.json')
    assert sm['identity_hash']==CONFIG['source_seed42_identity_hash']
    assert sha256_file(s42/'INFO_CV_COMPLETE')==CONFIG['source_seed42_marker_sha256']
    check_stage(s42,'INFO_CV_COMPLETE',sm['identity_hash'])
    sv=read_json(s42/'logs/independent_verification.json');assert sv['status']=='PASS'
    for n,h in sm['identity']['code_sha256'].items():sources[str(HERE/n)]=h
    sources[str(HERE/'verify_information_exact.py')]=sv['audit_identity']['adapter_sha256']
    sources[str(s42/'INFO_CV_COMPLETE')]=CONFIG['source_seed42_marker_sha256']
    sources[str(s42/'logs/independent_verification.json')]=sha256_file(s42/'logs/independent_verification.json')
    for item in CONFIG['geo_sources'].values():
        for field in ('source','snapshot'):sources[item[field]]=item['sha256']
    for path,h in sources.items():
        if not Path(path).is_file() or sha256_file(path)!=h:raise ValueError(f'Changed source: {path}')
    part=pd.read_parquet(run/'component_predictions.parquet');saved=pd.read_parquet(run/'control_predictions.parquet')
    branch=pd.read_parquet(run/'p_branch_oof_predictions.parquet');meta=saved[KEYS].copy()
    pd.testing.assert_frame_equal(meta[KEYS[:4]],data.meta_train[KEYS[:4]])
    np.testing.assert_array_equal(meta.fold,data.row_folds);assert meta.split_id.eq(data.loaded_hashes['cv']).all()
    for f in (part,saved,branch):
        pd.testing.assert_frame_equal(f[KEYS],meta);assert f.candidate_identity_hash.eq(m['identity_hash']).all()
    raw=pd.read_csv(ROOT/'data/DM_Train.csv',usecols=['fipsCode','timestamp_et','P_t','countyName'],dtype={'fipsCode':str},float_precision='round_trip')
    p=reference.p_from_history(raw,meta);np.testing.assert_array_equal(p,data.X_train.last_P_t)
    direct={h:saved[f'raw_direct_{h}'].to_numpy(copy=True) for h in HORIZONS}
    parts={h:{c:part[f'pred_E2_{c}_target_{h.rsplit("_",1)[-1]}'].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS}
    controls,aux=build_controls(meta,direct,parts)
    for h in HORIZONS:
        np.testing.assert_array_equal(data.y_train[h],saved[f'actual_{h}'])
        for mode in MODES:np.testing.assert_array_equal(controls[mode][h],saved[f'pred_E2_{mode}_{h}'])
    return SimpleNamespace(data=data,meta=meta,p=p,raw=raw,valid=expected_mask(meta,H1),direct=direct,
        parts=parts,controls=controls,aux=aux,f0_component=part,f0_saved=saved,f0_branch=branch,sources=sources,
        names=raw[['fipsCode','countyName']].drop_duplicates().set_index('fipsCode').countyName)


def attach_features(ctx):
    # No feature generation: CV is deliberately absent from the frozen package identity.
    folder=HERE/'features/v1';m=read_json(folder/'feature_manifest.json')
    if sha256_file(folder/'feature_manifest.json')!=CONFIG['feature_manifest_sha256'] or m['identity_hash']!=CONFIG['feature_package_identity']:
        raise ValueError('Frozen feature package changed')
    assert digest_object(m['identity'])==m['identity_hash']
    for name,h in m['identity']['inputs'].items():assert ctx.data.loaded_hashes[name]==h
    assert sha256_file(HERE/'experiment_config.json')==m['identity']['config_sha256']
    assert sha256_file(HERE/'build_information_features.py')==m['identity']['builder_sha256']
    for name,h in m['files'].items():
        path=(folder/name).resolve();assert path.is_relative_to(folder.resolve()) and sha256_file(path)==h
    features={}
    # Retain both matrices for source-package QA only; only F2 gets a fit context.
    for case in ('F1','F2'):
        features[case]={};names=read_json(folder/f'{case}_feature_names.json')
        for side in ('train','test'):
            frame=pd.read_parquet(folder/f'{case}_{side}.parquet');base=getattr(ctx.data,f'X_{side}')
            assert list(frame)==names and frame.shape==(len(base),CONFIG['feature_counts'][case])
            pd.testing.assert_frame_equal(frame[list(base)],base);features[case][side]=frame
    ctx.feature_bundle={'manifest':m,'features':features};ctx.cases={}
    for case in NEW_CASES:
        child=SimpleNamespace(**vars(ctx));child.data=SimpleNamespace(**vars(ctx.data))
        child.data.X_train=features[case]['train'];child.data.X_test=features[case]['test']
        child.case=case;child.feature_identity=m['identity_hash'];ctx.cases[case]=child
    return ctx


def experiment_identity(ctx):
    return {'protocol':CONFIG['protocol'],'split_seed':SEED,'trained_cases':['F2'],'config_sha256':sha256_file(CONFIG_PATH),
        'code_sha256':{str(p):sha256_file(p) for p in [*(CODE_ROOT/n for n in ADAPTED),*(HERE/n for n in SHARED)]},
        'source_sha256':ctx.sources,'reference_E2_identity_hash':CONFIG['reference_identity_hash'],
        'feature_package_identity':ctx.feature_bundle['manifest']['identity_hash'],
        'feature_manifest_sha256':sha256_file(HERE/'features/v1/feature_manifest.json'),
        'environment':environment(),'settings':CONFIG['settings'],'params':{k:params(k) for k in KINDS},
        'feature_names':{'F2':list(ctx.cases['F2'].data.X_train)},'p_sha256':array_sha(ctx.p),
        'P_label_sha256':array_sha(ctx.data.component_targets[PT])}


def build_candidates(ctx,p_values):
    assert set(p_values)==set(NEW_CASES)
    parts={};controls={};aux={}
    for case in CASES:
        part={h:{c:v.copy() for c,v in values.items()} for h,values in ctx.parts.items()}
        if case!='F0':part[H1]['P_t']=p_values[case]
        parts[case]=part;controls[case],aux[case]=build_controls(ctx.meta,ctx.direct,part)
        for h in HORIZONS:
            for c in COMPONENTS:
                if (h,c)!=(H1,'P_t'):np.testing.assert_array_equal(part[h][c],ctx.parts[h][c])
            for mode in MODES:
                if case=='F0' or mode=='C0_direct_osi' or (h!=H1 and mode in ('C1_component_osi','C2_equal_blend')) or (HORIZON_HOURS[h]>=24 and mode=='v18_rule'):
                    np.testing.assert_array_equal(controls[case][mode][h],ctx.controls[mode][h])
    return parts,controls,aux
