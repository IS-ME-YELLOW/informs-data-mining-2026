"""Frozen E2 reference, case-isolated feature views and P-only reconstruction."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'experiment_config.json').read_text())
ROOT=Path(CONFIG['project_root']);BASE=Path(CONFIG['baseline_root']);PBASE=Path(CONFIG['p_reference_root'])
assert HERE==Path(CONFIG['experiment_root']).resolve()==BASE/'p_information_increment'
sys.path.append(str(PBASE))
import p_structure_protocol as reference
from p_structure_protocol import (PT,H1,HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,KEYS,support,transform,
    reconstruct,params,array_sha,digest_object,read_json,sha256_file,write_json,write_frame,safe_artifact,
    expected_mask,build_controls)
from run_artifacts import check_stage,environment

KINDS=('a','b')
TARGETS={k:reference.TARGETS[k] for k in KINDS}
CASES=('F0','F1','F2')
NEW_CASES=('F1','F2')
CODE_FILES=('information_protocol.py','build_information_features.py','independent_features.py',
    'information_training.py','information_preflight.py','evaluate_information.py',
    'train_information.py','verify_information.py')


def output_path(path):
    path=Path(path).resolve()
    if path==HERE or not path.is_relative_to(HERE):raise ValueError('Output escapes information experiment')
    return path


def load_reference(seed=42):
    if seed!=42 or CONFIG['authorized_split_seeds']!=[42]:raise ValueError('Only seed42 authorized')
    ctx=reference.load_context(42)
    run=Path(CONFIG['reference_run']);m=read_json(run/'run_manifest.json')
    if m['identity_hash']!=CONFIG['reference_identity_hash'] or digest_object(m['identity'])!=m['identity_hash']:
        raise ValueError('E2 reference identity mismatch')
    marker=check_stage(run,'P_CV_COMPLETE',m['identity_hash'])
    verified=read_json(run/'logs/independent_verification.json')
    assert verified['status']=='PASS' and verified['identity_hash']==m['identity_hash']
    for name,digest in m['identity']['code_sha256'].items():
        if sha256_file(PBASE/name)!=digest:raise ValueError('Frozen E2 code changed')
    assert reference.experiment_identity(ctx)==m['identity']
    sources=dict(ctx.sources)
    sources.update({str(run/n):h for n,h in marker['files'].items()})
    sources.update({str(PBASE/n):h for n,h in m['identity']['code_sha256'].items()})
    for path in (PBASE/'experiment_config.json',run/'P_CV_COMPLETE'):
        sources[str(path)]=sha256_file(path)
    for item in CONFIG['geo_sources'].values():
        for field in ('source','snapshot'):
            if sha256_file(item[field])!=item['sha256']:raise ValueError('Changed geographic source')
            sources[item[field]]=item['sha256']
    part=pd.read_parquet(run/'component_predictions.parquet')
    saved=pd.read_parquet(run/'control_predictions.parquet')
    branch=pd.read_parquet(run/'p_branch_oof_predictions.parquet')
    for f in (part,saved,branch):
        pd.testing.assert_frame_equal(f[KEYS],ctx.meta)
        assert f.candidate_identity_hash.eq(m['identity_hash']).all()
    parts={h:{c:part[f'pred_E2_{c}_target_{h.rsplit("_",1)[-1]}'].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS}
    controls,aux=build_controls(ctx.meta,ctx.direct,parts)
    for h in HORIZONS:
        for mode in MODES:np.testing.assert_array_equal(controls[mode][h],saved[f'pred_E2_{mode}_{h}'])
    ctx.parts=parts;ctx.controls=controls;ctx.aux=aux
    ctx.f0_component=part;ctx.f0_saved=saved;ctx.f0_branch=branch;ctx.sources=sources
    return ctx


def attach_features(ctx):
    from build_information_features import load_features,feature_inputs
    bundle=load_features(feature_inputs(ctx.data))
    ctx.feature_bundle=bundle
    ctx.cases={}
    for case in NEW_CASES:
        child=SimpleNamespace(**vars(ctx));child.data=SimpleNamespace(**vars(ctx.data))
        child.data.X_train=bundle['features'][case]['train'];child.data.X_test=bundle['features'][case]['test']
        child.case=case;child.feature_identity=bundle['manifest']['identity_hash']
        ctx.cases[case]=child
    return ctx


def experiment_identity(ctx):
    return {'protocol':CONFIG['protocol'],'split_seed':42,'config_sha256':sha256_file(HERE/'experiment_config.json'),
        'code_sha256':{name:sha256_file(HERE/name) for name in CODE_FILES},
        'source_sha256':ctx.sources,'reference_E2_identity_hash':CONFIG['reference_identity_hash'],
        'feature_package_identity':ctx.feature_bundle['manifest']['identity_hash'],
        'feature_manifest_sha256':sha256_file(HERE/'features/v1/feature_manifest.json'),
        'environment':environment(),'settings':CONFIG['settings'],'params':{k:params(k) for k in KINDS},
        'feature_names':{c:list(ctx.cases[c].data.X_train) for c in NEW_CASES},
        'p_sha256':array_sha(ctx.p),'P_label_sha256':array_sha(ctx.data.component_targets[PT])}


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
