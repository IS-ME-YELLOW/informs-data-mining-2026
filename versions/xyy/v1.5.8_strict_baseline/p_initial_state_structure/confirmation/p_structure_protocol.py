"""Frozen repeated-split P experiment: source verification and label-free reconstruction."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

from _runtime import HERE, CODE_ROOT, CONFIG_PATH, META_ROOT, RUN_ID, CONFIG, SEED
ROOT, BASE = (Path(CONFIG[k]).resolve() for k in ('project_root', 'baseline_root'))
assert HERE == Path(CONFIG['experiment_root']).resolve() == BASE / 'p_initial_state_structure'
sys.path.insert(0, str(BASE))
from protocol import load_data, expected_mask, build_controls, HORIZONS, HORIZON_HOURS, COMPONENTS, MODES, sha256_file
from config import make_lgbm_params
from run_artifacts import digest_object, data_identity, environment, write_json, write_frame, check_stage, safe_artifact

H1 = 'osi_target_t01h'
PT = 'P_t_target_t01h'
KEYS = ['fipsCode', 'timestamp_et', 'hour_idx', 'stateAbbr', 'fold', 'split_id']
KINDS = ('direct', 'a', 'b')
TARGETS = {'direct': 'P_direct_l2_t01h', 'a': 'P_anchor_a_t01h', 'b': 'P_excess_b_t01h'}
CODE_FILES = ('p_structure_protocol.py', 'weighted_scoped_training.py', 'train_p_structure.py',
              'evaluate_p_structure.py', 'verify_p_structure.py', 'preflight_checks.py',
              'metric_helpers.py', 'independent_numerics.py')


def read_json(path):
    return json.loads(Path(path).read_text())


def output_path(path):
    path = Path(path).resolve()
    if path == HERE or not path.is_relative_to(HERE):
        raise ValueError('Output escapes P experiment')
    return path


def array_sha(a):
    return hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest()


def require_unit(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Official P/history outside [0,1] or non-finite')


def p_from_history(raw, meta):
    frame = raw[['fipsCode', 'timestamp_et', 'P_t']].copy()
    frame['fipsCode'] = frame.fipsCode.astype(str).str.zfill(5)
    frame['timestamp_et'] = pd.to_datetime(frame.timestamp_et)
    if meta.duplicated(['fipsCode', 'timestamp_et']).any():
        raise ValueError('Duplicate forecast keys')
    selected = frame.loc[frame.timestamp_et == pd.Timestamp('2026-03-13 23:00')]
    if selected.fipsCode.duplicated().any() or set(selected.fipsCode) != set(meta.fipsCode):
        raise ValueError('Missing, duplicate, or mismatched hour71 history')
    p = meta.fipsCode.map(selected.set_index('fipsCode').P_t).to_numpy(dtype=float, copy=True)
    require_unit(p)
    return p


def support(p, kind):
    if kind not in KINDS:
        raise ValueError('Unknown P target')
    return np.ones(len(p), dtype=bool) if kind == 'direct' else (p > 0 if kind == 'a' else p < 1)


def transform(y, p, kind):
    require_unit(y); require_unit(p)
    if np.asarray(y).shape != np.asarray(p).shape or not support(p, kind).all():
        raise ValueError('Unsupported branch rows passed to label transform')
    if kind == 'direct':
        return np.asarray(y, dtype=float).copy(), np.ones(len(y))
    if kind == 'a':
        return np.minimum(y, p) / p, p ** 2
    return np.maximum(y - p, 0) / (1 - p), (1 - p) ** 2


def reconstruct(p, raw_a, raw_b, valid):
    p, raw_a, raw_b = [np.asarray(a, dtype=float) for a in (p, raw_a, raw_b)]
    valid = np.asarray(valid, dtype=bool)
    require_unit(p)
    if not (p.shape == raw_a.shape == raw_b.shape == valid.shape):
        raise ValueError('Reconstruction shape mismatch')
    applied = []
    for kind, raw in (('a', raw_a), ('b', raw_b)):
        active = valid & support(p, kind)
        if not np.isfinite(raw[active]).all() or not np.isnan(raw[~active]).all():
            raise ValueError('Branch prediction/support mask mismatch')
        out = np.where(valid, 0., np.nan)
        out[active] = np.clip(raw[active], 0, 1)
        applied.append(out)
    ca, cb = p * applied[0], (1 - p) * applied[1]
    result = ca + cb
    if (result[valid] < -1e-12).any() or (result[valid] > 1 + 1e-12).any():
        raise ValueError('Unexpected reconstructed bound violation')
    return np.clip(result, 0, 1), applied[0], applied[1], ca, cb


def verify_sources():
    base_run, grun = Path(CONFIG['baseline_run']), Path(CONFIG['g_run'])
    bm, gm = read_json(base_run/'run_manifest.json'), read_json(grun/'manifest.json')
    for m, expected in ((bm, CONFIG['baseline_identity_hash']), (gm, CONFIG['g_identity_hash'])):
        if m['identity_hash'] != expected or digest_object(m['identity']) != expected:
            raise ValueError('Source identity mismatch')
    check_stage(base_run, 'CV_COMPLETE', bm['identity_hash'])
    gmarker = check_stage(grun, 'G_COMPLETE', gm['identity_hash'])
    independent = read_json(grun/'logs/independent_verification.json')
    if independent['status'] != 'PASS' or independent['identity_hash'] != gm['identity_hash']:
        raise ValueError('D_G lacks matching independent verification')
    sources = dict(gm['identity']['source_sha256'])
    sources.update({str(grun/name): h for name, h in gmarker['files'].items()})
    for name, h in gm['identity']['code_sha256'].items():
        sources[str(grun.parent.parent/name)] = h
    for name in ('G_COMPLETE', 'logs/independent_verification.json'):
        sources[str(grun/name)] = sha256_file(grun/name)
    for path, h in sources.items():
        if not Path(path).is_file() or sha256_file(path) != h:
            raise ValueError(f'Changed source: {path}')
    return sources, bm, gm


def load_context(seed=SEED):
    if seed != SEED or seed not in CONFIG['authorized_split_seeds']:
        raise ValueError('Confirmation split does not match the selected source package')
    sources, bm, gm = verify_sources()
    data = load_data(ROOT/'versions/xyy/v1.5.6', ROOT/f'cv/cv_assignments_balanced_v1_seed{SEED}.csv')
    if digest_object(data_identity(data, CONFIG['settings'])) != CONFIG['baseline_identity_hash']:
        raise ValueError('Frozen baseline data/code/environment changed')
    grun = Path(CONFIG['g_run'])
    component = pd.read_parquet(grun/'component_predictions.parquet')
    saved = pd.read_parquet(grun/'control_predictions.parquet')
    meta = saved[KEYS].copy()
    pd.testing.assert_frame_equal(meta[KEYS[:4]], data.meta_train[KEYS[:4]])
    pd.testing.assert_frame_equal(component[KEYS], meta)
    np.testing.assert_array_equal(meta.fold, data.row_folds)
    assert meta.split_id.eq(data.loaded_hashes['cv']).all()
    for table in (component, saved):
        assert table.gated_identity_hash.eq(CONFIG['g_identity_hash']).all()
        assert table.baseline_identity_hash.eq(CONFIG['baseline_identity_hash']).all()
    raw = pd.read_csv(ROOT/'data/DM_Train.csv', usecols=['fipsCode','timestamp_et','P_t','countyName'],
                      dtype={'fipsCode':str}, float_precision='round_trip')
    p = p_from_history(raw, meta)
    np.testing.assert_array_equal(p, data.X_train.last_P_t)
    direct = {h: saved[f'raw_direct_{h}'].to_numpy(copy=True) for h in HORIZONS}
    parts = {h:{c:component[f'pred_G_{c}_target_{h.rsplit("_",1)[-1]}'].to_numpy(copy=True)
                for c in COMPONENTS} for h in HORIZONS}
    controls, aux = build_controls(meta, direct, parts)
    for h in HORIZONS:
        np.testing.assert_array_equal(data.y_train[h], saved[f'actual_{h}'])
        for mode in MODES:
            np.testing.assert_array_equal(controls[mode][h], saved[f'pred_G_{mode}_{h}'])
    # Original D_B comes from G's verified, same-split U dependency package.
    u_cfg = read_json(grun.parent.parent/'experiment_config.json')
    item = next(x for x in u_cfg['splits'] if x['split_seed'] == seed)
    u_source = pd.read_parquet(Path(item['u_run'])/'component_predictions.parquet')
    pd.testing.assert_frame_equal(u_source[KEYS], meta)
    b_d = {h:u_source[f'pred_B_D_t_target_{h.rsplit("_",1)[-1]}'].to_numpy(copy=True) for h in HORIZONS}
    return SimpleNamespace(data=data, meta=meta, p=p, raw=raw, valid=expected_mask(meta,H1),
        component=component, saved=saved, direct=direct, parts=parts, controls=controls, aux=aux,
        b_d=b_d, sources=sources, source_manifests={'baseline':bm,'g':gm}, names=raw[['fipsCode','countyName']].drop_duplicates().set_index('fipsCode').countyName)


def params(kind):
    result = make_lgbm_params(42)
    result.update(objective='regression', metric='rmse' if kind == 'direct' else 'None')
    return result


def experiment_identity(ctx):
    return {'protocol': CONFIG['protocol'], 'split_seed':SEED, 'config_sha256':sha256_file(CONFIG_PATH),
            'code_sha256':{str(p):sha256_file(p) for p in [*(CODE_ROOT/n if n in ('p_structure_protocol.py','train_p_structure.py','verify_p_structure.py') else HERE/n for n in CODE_FILES), CODE_ROOT/'_runtime.py']}, 'source_sha256':ctx.sources,
            'environment':environment(), 'settings':CONFIG['settings'],
            'params':{k:params(k) for k in KINDS}, 'features':list(ctx.data.X_train.columns),
            'p_sha256':array_sha(ctx.p), 'P_label_sha256':array_sha(ctx.data.component_targets[PT]),
            'baseline_identity_hash':CONFIG['baseline_identity_hash'], 'g_identity_hash':CONFIG['g_identity_hash']}


def build_candidates(ctx, direct_p, structured_p):
    candidates, controls, aux = {}, {}, {}
    for case, p in (('E0', ctx.parts[H1]['P_t']), ('E1',direct_p), ('E2',structured_p)):
        part = {h:{c:v.copy() for c,v in comps.items()} for h,comps in ctx.parts.items()}
        part[H1]['P_t'] = p.copy()
        candidates[case] = part
        controls[case],aux[case] = build_controls(ctx.meta,ctx.direct,part)
        for h in HORIZONS:
            for c in COMPONENTS:
                if (h,c) != (H1,'P_t'):
                    np.testing.assert_array_equal(part[h][c],ctx.parts[h][c])
            np.testing.assert_array_equal(controls[case]['C0_direct_osi'][h],ctx.controls['C0_direct_osi'][h])
            if h != H1:
                for mode in ('C1_component_osi','C2_equal_blend'):
                    np.testing.assert_array_equal(controls[case][mode][h],ctx.controls[mode][h])
            if HORIZON_HOURS[h] >= 24:
                np.testing.assert_array_equal(controls[case]['v18_rule'][h],ctx.controls['v18_rule'][h])
    return candidates, controls, aux
