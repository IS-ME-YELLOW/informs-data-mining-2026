"""Read-only source contracts and diagnostic numerics. No model fitting imports."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[2]
SEEDS = (42, 20260917, 20260918)
HS = (1, 6, 24, 48)
CS = ('P', 'N', 'D', 'R')
W = np.array([.40, .35, .25, -.10])
IDS = {42: 'e39b76d75f191891abda1939dc47933815cc4fa15796227b0c200eb1b96bfcac',
       20260917: '64f8263f5356378124c5e981fcb7c1117ad808c97b46c383b64b7cdbe109484d',
       20260918: 'a3f16d3463442c01b1e1315df498317eed53298a1f89ef087b016ab8bd0c9532'}
MATCH = ['last_P_t', 'last_N_t', 'last_R_t', 'N_t_mean_72h', 'N_t_max_72h',
         'gust_at_t{h}h', 'log_customers', 'pct_forest']
WINDOWS = ((73, 76), (77, 95), (96, 143), (144, 215))
FOCUS = {'42053': ('Forest County', 'PA'), '39117': ('Morrow County', 'OH'),
         '18013': ('Brown County', 'IN'), '54013': ('Calhoun County', 'WV'),
         '54015': ('Clay County', 'WV'), '39115': ('Morgan County', 'OH')}


def safe(path):
    p = Path(path).resolve()
    if not p.is_relative_to(OUT) or p == OUT:
        raise ValueError(f'Output escapes diagnostic directory: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, Path): return str(v)
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def write_json(path, obj):
    safe(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path): return json.loads(Path(path).read_text())


def frame(path, data):
    d = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    p = safe(path)
    if p.suffix == '.parquet': d.to_parquet(p, index=False)
    else: d.to_csv(p, index=False)


def csv(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def fips(x):
    s = pd.Series(x).astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
    assert s.str.fullmatch(r'\d{5}').all()
    return s.to_numpy()


def ordered(d):
    d = d.copy()
    d['fipsCode'] = fips(d.fipsCode)
    if 'timestamp_et' in d: d['timestamp_et'] = pd.to_datetime(d.timestamp_et)
    assert not d.duplicated(['fipsCode', 'hour_idx']).any()
    return d.sort_values(['fipsCode', 'hour_idx']).reset_index(drop=True)


def post(z):
    v = np.clip(z, 0, .65)
    return np.where(v < .001, 0, v)


def linear(p):
    return .40*p[..., 0] + .35*p[..., 1] + .25*p[..., 2] - .10*p[..., 3]


def align(parts):
    # Each source contains all four components for exactly its permitted target hours.
    available = np.isfinite(parts[..., 1])
    count = available.sum(axis=0)
    total = np.where(available[..., None], np.clip(parts, 0, 1), 0).sum(axis=0)
    return np.divide(total, count[..., None], out=np.full_like(total, np.nan),
                     where=count[..., None] > 0), count


def controls(parts):
    a, count = align(parts)
    c1 = post(linear(np.clip(parts, 0, 1)))
    c3 = np.broadcast_to(post(linear(a)), c1.shape).copy()
    main = c1.copy()
    main[:2] = c3[:2]
    for i, h in enumerate(HS):
        c1[i, :, :72+h] = np.nan
        c3[i, :, :72+h] = np.nan
        main[i, :, :72+h] = np.nan
    return dict(C1=c1, C3=c3, main=main, aligned=a, count=count)


def metric(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    assert y.shape == p.shape
    if not len(y): return dict(n=0, rmse=np.nan, mae=np.nan, sse=0., bias=np.nan)
    assert np.isfinite(y).all() and np.isfinite(p).all()
    e = p-y
    return dict(n=len(e), rmse=float(np.sqrt(np.mean(e*e))), mae=float(np.mean(abs(e))),
                sse=float(e@e), bias=float(e.mean()))


def correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compare(y, base, pred):
    b, p = metric(y, base), metric(y, pred)
    return dict(**p, baseline_rmse=b['rmse'], rmse_delta=p['rmse']-b['rmse'],
                rmse_change_pct=100*(p['rmse']/b['rmse']-1) if b['rmse'] else np.nan,
                sse_reduction=b['sse']-p['sse'])


def bootstrap(y, base, pred):
    assert y.shape == base.shape == pred.shape and y.ndim == 2
    idx = np.random.default_rng(20260910).integers(0, len(y), (2000, len(y)))
    n = y.shape[1] * len(y)
    bs = ((base-y)**2).sum(axis=1)
    ps = ((pred-y)**2).sum(axis=1)
    diff = np.sqrt(ps[idx].sum(axis=1)/n) - np.sqrt(bs[idx].sum(axis=1)/n)
    lo, hi = np.quantile(diff, [.025, .975])
    return dict(ci_low=float(lo), ci_high=float(hi), replicates=2000,
                bootstrap_seed=20260910, unit='county_full_trajectory')


def initialize():
    plan = OUT/'Diagnosis_Plan_2026-09-19.md'
    approved = safe(OUT/'reference/approved_plan.md')
    if not approved.exists(): approved.write_bytes(plan.read_bytes())
    assert approved.read_bytes() == plan.read_bytes()
    config = dict(protocol='n_four_horizon_diagnosis_v1', authorized='user approved full diagnosis; no fitting',
                  root=str(ROOT), base=str(BASE), output=str(OUT), seeds=SEEDS, horizons=HS,
                  reference_F2_identities=IDS, model_seed=42, known_history_end=71,
                  weights=W, common_target_hours=[120, 215], windows=WINDOWS,
                  N_bins=[0, .0001, .001, .01], event_threshold=.001, lags=list(range(-3,4)),
                  partial_alphas=[.25, .5, 1.], bootstrap_replicates=2000, bootstrap_seed=20260910,
                  matching_columns=MATCH, matching_top_k=5, matching_excludes_full_outer_fold=True,
                  historical_focus=FOCUS, approved_plan_sha256=sha(approved))
    path = OUT/'diagnostic_config.json'
    if path.exists(): assert read_json(path) == clean(config)
    else: write_json(path, config)


def source_inventory():
    """Validate recorded artifact and dependency hashes, without invoking old verifiers."""
    expected = {}
    def add(p, h=None):
        p = Path(p).resolve()
        assert not p.is_relative_to(OUT), p
        if h is not None and expected.get(str(p)) is not None: assert expected[str(p)] == h, p
        expected[str(p)] = h if h is not None else expected.get(str(p))
    for seed in SEEDS:
        for r, marker in [(BASE/f'runs/v18_tree_nested_v2_split{seed}_model42', 'CV_COMPLETE'),
                          (BASE/f'p_information_increment/runs/v18_pinfo_v1_split{seed}_model42', 'INFO_CV_COMPLETE')]:
            m = read_json(r/'run_manifest.json'); v = read_json(r/marker)
            assert digest(m['identity']) == m['identity_hash'] == v['identity_hash']
            if marker.startswith('INFO'): assert m['identity_hash'] == IDS[seed]
            add(r/'run_manifest.json'); add(r/marker)
            for rel, h in v['files'].items():
                p = (r/rel).resolve(); assert p.is_relative_to(r)
                add(p, h)
            if marker.startswith('INFO'):
                sm = read_json(r/'source_manifest.json'); add(r/'source_manifest.json')
                for path, h in sm['source_sha256'].items(): add(path, h)
            else:
                im = read_json(r/'input_manifest.json'); add(r/'input_manifest.json')
                for v in im.values(): add(ROOT/v['path'], v['sha256'])
                add(r/'verification_cv.json')
        add(ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv')
    fd = BASE/'p_information_increment/features/v1'
    fm = read_json(fd/'feature_manifest.json'); add(fd/'feature_manifest.json')
    assert digest(fm['identity']) == fm['identity_hash']
    for rel, h in fm['files'].items(): add(fd/rel, h)
    add(ROOT/'review/2026-09-16/similar_counties/focus_match_summary.csv')
    actual = {}
    for p, h in sorted(expected.items()):
        a = sha(p)
        if h is not None and a != h: raise ValueError(f'Source hash changed: {p}')
        actual[p] = a
    frozen = OUT/'reference/initial_input_hashes.json'
    if frozen.exists(): assert read_json(frozen) == actual, 'Input set or bytes changed'
    else: write_json(frozen, actual)
    write_json(OUT/'reference/source_manifest.json', dict(files=actual, count=len(actual),
               source_hashes_verified=True, diagnostic_config_sha256=sha(OUT/'diagnostic_config.json'),
               environment=dict(python=platform.python_version(), libraries={k:importlib.metadata.version(k)
                               for k in ('numpy','pandas','pyarrow','scipy')})))
    return actual


def load_context(seed):
    assert seed in SEEDS
    r = BASE/f'p_information_increment/runs/v18_pinfo_v1_split{seed}_model42'
    old = BASE/f'runs/v18_tree_nested_v2_split{seed}_model42'
    meta = ordered(pd.read_parquet(ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet'))
    counties = meta.fipsCode.unique(); nc = len(counties)
    assert nc == 239 and len(meta) == 34416
    assert np.array_equal(meta.hour_idx, np.tile(np.arange(72,216), nc))
    assert np.array_equal(meta.timestamp_et, pd.Timestamp('2026-03-11')+pd.to_timedelta(meta.hour_idx,unit='h'))
    cv = csv(ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv', dtype={'fipsCode':str}).set_index('fipsCode')
    assert not cv.index.duplicated().any() and set(cv.index) == set(counties)
    folds = cv.loc[counties, 'fold'].to_numpy(int)
    raw = csv(ROOT/'data/DM_Train.csv', dtype={'fipsCode':str})
    raw['timestamp_et'] = pd.to_datetime(raw.timestamp_et)
    raw['hour_idx'] = ((raw.timestamp_et-pd.Timestamp('2026-03-11'))/pd.Timedelta(hours=1)).astype(int)
    raw = ordered(raw)
    assert np.array_equal(raw.hour_idx, np.tile(np.arange(216),nc))
    assert np.array_equal(raw.fipsCode.unique(), counties)
    truth = raw[[c+'_t' for c in CS]].to_numpy().reshape(nc,216,4)
    y = raw.osi.to_numpy().reshape(nc,216)
    names = raw.drop_duplicates('fipsCode').set_index('fipsCode')[['countyName','stateAbbr']]
    features = pd.read_parquet(ROOT/'versions/xyy/v1.5.6/features_train_v1.5.6.parquet')
    f2features = pd.read_parquet(BASE/'p_information_increment/features/v1/F2_train.parquet')
    assert len(features) == len(f2features) == len(meta)
    pd.testing.assert_frame_equal(features, f2features[list(features)])
    tables = {k: ordered(pd.read_parquet(p)) for k,p in dict(
        parts=r/'component_predictions.parquet', saved=r/'control_predictions.parquet',
        oldparts=old/'oof_component_predictions.parquet', raw=old/'base_predictions_cv.parquet',
        targets=BASE/'component_targets_v1.8.parquet', oldsaved=old/'oof_predictions.parquet').items()}
    for key,d in tables.items():
        pd.testing.assert_frame_equal(d[['fipsCode','hour_idx','timestamp_et']],meta[['fipsCode','hour_idx','timestamp_et']],check_dtype=False)
        if 'fold' in d: np.testing.assert_array_equal(d.fold,np.repeat(folds,144))
    for key in ['parts','saved']:
        assert tables[key].candidate_identity_hash.eq(IDS[seed]).all()
    parts = np.full((4,nc,216,4),np.nan); oldparts = parts.copy()
    rawN = np.full((4,nc,216),np.nan)
    sources = np.full((4,nc,216,4),'',dtype=object)
    for i,h in enumerate(HS):
        start = 72+h; sl = slice(start,216)
        for j,c in enumerate(CS):
            tag=f'{c}_t_target_t{h:02d}h'
            pp=tables['parts'][f'pred_F2_{tag}'].to_numpy().reshape(nc,144)
            op=tables['oldparts'][f'pred_{tag}'].to_numpy().reshape(nc,144)
            yy=tables['targets'][tag].to_numpy().reshape(nc,144)
            assert np.isfinite(pp[:,:144-h]).all() and np.isnan(pp[:,144-h:]).all()
            assert np.isnan(yy[:,144-h:]).all()
            np.testing.assert_array_equal(yy[:,:144-h],truth[:,sl,j])
            parts[i,:,sl,j]=pp[:,:144-h]; oldparts[i,:,sl,j]=op[:,:144-h]
            sources[i,:,sl,j]=tables['parts'][f'source_F2_{tag}'].to_numpy().reshape(nc,144)[:,:144-h]
            if c == 'N':
                np.testing.assert_array_equal(pp,op)
                np.testing.assert_array_equal(tables['parts'][f'source_F2_{tag}'],tables['oldparts'][f'source_{tag}'])
        rawN[i,:,sl]=tables['raw'][f'raw_N_t_target_t{h:02d}h'].to_numpy().reshape(nc,144)[:,:144-h]
        np.testing.assert_array_equal(np.clip(rawN[i,:,sl],0,1),parts[i,:,sl,1])
        for f in range(5):
            rec=read_json(old/f'models/outer{f}/N_t_target_t{h:02d}h.json')
            spec=rec['spec']; fit=rec['fit']; allowed=sorted(set(range(5))-{f})
            assert digest(spec)==rec['model_id'] and spec['scope']==fit['allowed_folds']==allowed
            train_counties=sorted(counties[folds!=f]); held=set(counties[folds==f])
            assert fit['refit']['counties']==train_counties
            assert {k:fit[k] for k in ('allowed_folds','refit','probes')}==spec['provenance']
            assert len(fit['probes'])==4
            for probe in fit['probes']:
                q=probe['inner_fold']
                assert q in allowed and probe['train_folds']==[a for a in allowed if a!=q]
                assert probe['early_stop_folds']==[q]
                assert set(probe['train']['counties'])==set(counties[(folds!=f)&(folds!=q)])
                assert set(probe['early_stop']['counties'])==set(counties[folds==q])
                assert not held.intersection(probe['train']['counties']+probe['early_stop']['counties'])
            assert np.all(sources[i,folds==f,sl,1]==rec['model_id'])
    ctl=controls(parts); oldctl=controls(oldparts)
    saved=tables['saved']; maxdiff=0.
    for i,h in enumerate(HS):
        sl=slice(72+h,216)
        for mode,col in [('C1','C1_component_osi'),('C3','C3_aligned_component'),('main','v18_rule')]:
            want=saved[f'pred_F2_{col}_osi_target_t{h:02d}h'].to_numpy().reshape(nc,144)
            got=ctl[mode][i,:,sl]
            np.testing.assert_allclose(got,want[:,:144-h],atol=1e-12,rtol=0)
            assert np.isnan(want[:,144-h:]).all()
            maxdiff=max(maxdiff,float(np.max(abs(got-want[:,:144-h]))))
        official=saved[f'actual_osi_target_t{h:02d}h'].to_numpy().reshape(nc,144)
        np.testing.assert_array_equal(official[:,:144-h],y[:,sl])
        oldwant=tables['oldsaved'][f'pred_v18_rule_osi_target_t{h:02d}h'].to_numpy().reshape(nc,144)[:,:144-h]
        np.testing.assert_allclose(oldctl['main'][i,:,sl],oldwant,atol=1e-12,rtol=0)
    au=pd.read_parquet(r/'aligned_unique_components.parquet')
    au=au.loc[au['case'].eq('F2')].copy(); au['fipsCode']=fips(au.fipsCode)
    au['target_timestamp']=pd.to_datetime(au.target_timestamp)
    assert not au.duplicated(['component','fipsCode','target_timestamp']).any()
    for j,c in enumerate(CS):
        a=au.loc[au.component.eq(c+'_t')].sort_values(['fipsCode','target_timestamp'])
        np.testing.assert_allclose(a.prediction.to_numpy().reshape(nc,143),ctl['aligned'][:,73:,j],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(a.candidate_count.to_numpy().reshape(nc,143),ctl['count'][:,73:])
    assert read_json(r/'logs/independent_verification.json')['status']=='PASS'
    assert read_json(old/'verification_cv.json')['status']=='passed'
    return SimpleNamespace(seed=seed,run=r,oldrun=old,meta=meta,counties=counties,nc=nc,folds=folds,
        raw=raw,truth=truth,y=y,names=names,features=features,f2features=f2features,parts=parts,
        oldparts=oldparts,rawN=rawN,sources=sources,ctl=ctl,oldctl=oldctl,maxdiff=maxdiff,
        output=OUT/f'runs/split{seed}')


def scenario_parts(ctx, name):
    p=ctx.parts.copy()
    if name.startswith('O_N_source_'):
        i=HS.index(int(name.rsplit('_',1)[-1])); h=HS[i]
        p[i,:,72+h:,1]=ctx.truth[:,72+h:,1]
    elif name.startswith('O_N_partial_'):
        alpha=float(name.rsplit('_',1)[-1])
        for i,h in enumerate(HS): p[i,:,72+h:,1]=(1-alpha)*p[i,:,72+h:,1]+alpha*ctx.truth[:,72+h:,1]
    elif name in ('A_N_zero','A_N71'):
        for i,h in enumerate(HS): p[i,:,72+h:,1]=0 if name=='A_N_zero' else ctx.truth[:,71,1,None]
    else:
        comps=CS if name=='O_all_components' else (name.split('_')[1],)
        for c in comps:
            j=CS.index(c)
            for i,h in enumerate(HS): p[i,:,72+h:,j]=ctx.truth[:,72+h:,j]
    return p


SCENARIOS = ('O_N_all', *(f'O_N_source_{h}' for h in HS), 'O_N_partial_0.25','O_N_partial_0.5',
             'O_P_all','O_D_all','O_R_all','O_all_components','A_N_zero','A_N71')
