"""Frozen, validated 141-column v1.5.5 dataset; no model dependency."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
for path in (ROOT / '.python_packages', ROOT / 'code_phase1', ROOT / 'data/external/scripts'):
    sys.path.insert(0, str(path))
import numpy as np
import pandas as pd
from config import FORBIDDEN_AS_FEATURES, HORIZONS, HORIZON_HOURS
from data_loader import load_train, load_test, preprocess
from features import ALL_FEATURE_NAMES, OBSERVED_FEATURES, compute_observed_features, build_feature_matrix
from build_county_features import build_county_features, FINAL_COLUMNS, COUNTY_DBF, EIA_FILE

VERSION = 'v1.5.5'
PARENT = 'v1.5.2'
CACHE = ROOT / 'cache'
RELEASE = ROOT / 'versions' / VERSION
MANIFEST = CACHE / f'manifest_{VERSION}.json'
TABLE_NAMES = ['features_train', 'targets_train', 'meta_train', 'features_test', 'meta_test']

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def table_path(name, version=VERSION):
    return CACHE / f'{name}_{version}.parquet'

def frozen_schema():
    return list(ALL_FEATURE_NAMES) + list(FINAL_COLUMNS) + ['forest_x_customers']

@dataclass
class FeatureDataset:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    meta_train: pd.DataFrame
    X_test: pd.DataFrame
    meta_test: pd.DataFrame

def _validate_split(X, y, meta, tag):
    county_count = 239 if tag == 'train' else 63
    if X.shape != (county_count * 144, 141) or list(X) != frozen_schema():
        raise ValueError(f'{tag}: incorrect shape or column order: {X.shape}')
    if set(X) & set(FORBIDDEN_AS_FEATURES):
        raise ValueError('Forbidden input columns.')
    if len(meta) != len(X) or meta.duplicated(['fipsCode', 'hour_idx']).any():
        raise ValueError(f'{tag}: invalid metadata keys.')
    if not meta.equals(meta.sort_values(['fipsCode', 'hour_idx']).reset_index(drop=True)):
        raise ValueError(f'{tag}: rows must be sorted by county and hour.')
    if meta.fipsCode.nunique() != county_count:
        raise ValueError(f'{tag}: county count differs.')
    for _, group in meta.groupby('fipsCode'):
        if not np.array_equal(group.hour_idx.to_numpy(), np.arange(72, 216)):
            raise ValueError(f'{tag}: missing or misaligned hours.')
    dates = pd.to_datetime(meta.timestamp_et, format='%m/%d/%Y %H:%M')
    expected_dates = pd.Timestamp('2026-03-11') + pd.to_timedelta(meta.hour_idx, unit='h')
    if not dates.equals(expected_dates.rename('timestamp_et')):
        raise ValueError(f'{tag}: timestamps differ from hour indexes.')
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(f'{tag}: infinite feature values.')
    allowed_nan = {f'{name}_at_t{h}h': meta.hour_idx + h >= 216
                   for h in (1, 6, 24, 48) for name in ('gust', 'wind_speed', 't2m', 'tp')}
    # No positive observed OSI means there is no defined positive peak.
    allowed_nan['hours_since_peak'] = X.osi_max_72h.eq(0)
    for col in X:
        expected = allowed_nan.get(col, np.zeros(len(X), dtype=bool))
        if not np.array_equal(X[col].isna(), expected):
            raise ValueError(f'{tag}: unexpected missing-value pattern in {col}.')
    if y is not None:
        if list(y) != HORIZONS or len(y) != len(X):
            raise ValueError('Target schema mismatch.')
        for col, h in HORIZON_HOURS.items():
            if not np.array_equal(y[col].isna(), meta.hour_idx + h >= 216):
                raise ValueError(f'Incorrect target mask: {col}')
    return {'rows': len(X), 'columns': len(X.columns), 'counties': county_count,
            'nan_cells': int(X.isna().sum().sum()),
            'target_valid_counts': None if y is None else y.notna().sum().to_dict()}

def _validate_raw(raw, tag):
    expected_count = 239 if tag == 'train' else 63
    if raw.fipsCode.nunique() != expected_count or raw.duplicated(['fipsCode', '_dt']).any():
        raise ValueError(f'{tag}: invalid raw county/time keys.')
    for fips, group in raw.groupby('fipsCode'):
        expected = pd.date_range('2026-03-11', periods=216, freq='h')
        if len(group) != 216 or not np.array_equal(group['_dt'].to_numpy(), expected.to_numpy()):
            raise ValueError(f'{tag}: county {fips} must have 216 contiguous hours.')

def attach_external(X, meta, county):
    keys = meta.fipsCode.astype(str).str.zfill(5)
    external = county.reindex(keys).reset_index(drop=True)
    if external.isna().any().any():
        raise ValueError('Unmatched external feature rows.')
    X = pd.concat([X.reset_index(drop=True), external], axis=1)
    X['forest_x_customers'] = X.pct_forest * X.log_customers
    return X[frozen_schema()]

def build_dataset(from_raw=False, overwrite=False):
    """Default: reuse immutable parent; --from-raw: rebuild all 141 columns."""
    outputs = [table_path(name) for name in TABLE_NAMES] + [
        CACHE / f'feature_names_{VERSION}.json', ROOT / f'data/county_features_{VERSION}.csv',
        RELEASE / 'eia_county_matches.csv', RELEASE / 'feature_changes.csv', RELEASE / 'validation.json']
    if not overwrite and any(p.exists() for p in outputs + [MANIFEST]):
        raise FileExistsError('v1.5.5 output exists. Use --verify, or explicitly --overwrite to rebuild.')
    if not from_raw:
        fingerprints = json.loads((RELEASE / 'parent_inputs_sha256.json').read_text(encoding='utf-8'))
        for name, digest in fingerprints.items():
            if not (ROOT / name).exists() or sha256(ROOT / name) != digest:
                raise ValueError(f'Frozen parent input missing or changed: {name}. Use --from-raw; do not patch stale caches.')
    county, matches = build_county_features()
    old_county = pd.read_csv(ROOT / 'data/county_features.csv', dtype={'fipsCode': str}).set_index('fipsCode')
    matches['old_n_utilities'] = old_county.n_utilities.reindex(matches.index)
    matches['changed'] = matches.n_utilities.ne(matches.old_n_utilities)
    frames, changes, validation = {}, [], {}
    raw_paths = [ROOT / 'data/DM_Train.csv', ROOT / 'data/DM_Test.csv']
    for tag, loader in [('train', load_train), ('test', load_test)]:
        print(f'Building {tag} ({"raw" if from_raw else "parent cache + corrected fields"})...', flush=True)
        raw = loader()
        _validate_raw(raw, tag)
        prepared = preprocess(raw)
        oldX = pd.read_parquet(table_path(f'features_{tag}', PARENT)) if table_path(f'features_{tag}', PARENT).exists() else None
        oldmeta = pd.read_parquet(table_path(f'meta_{tag}', PARENT)) if table_path(f'meta_{tag}', PARENT).exists() else None
        oldy = pd.read_parquet(table_path('targets_train', PARENT)) if tag == 'train' and table_path('targets_train', PARENT).exists() else None
        if from_raw:
            X, y, meta = build_feature_matrix(prepared, is_train=tag == 'train')
            X = attach_external(X, meta, county)
        else:
            if oldX is None or oldmeta is None or (tag == 'train' and oldy is None):
                raise FileNotFoundError('Parent cache incomplete; use --from-raw.')
            X, y, meta = oldX.copy(), oldy, oldmeta.copy()
            if list(X) != frozen_schema():
                raise ValueError('Parent schema is not the full 141-column v1.5.2 dataset.')
            obs = pd.DataFrame({fips: compute_observed_features(g)
                                for fips, g in prepared.groupby('fipsCode')}).T
            for col in OBSERVED_FEATURES:
                values = meta.fipsCode.map(obs[col]).to_numpy()
                # Preserve unchanged parent bytes (e.g. polyfit library roundoff).
                if not np.allclose(values, oldX[col], rtol=1e-12, atol=1e-12, equal_nan=True):
                    X[col] = values
            keys = meta.fipsCode.astype(str).str.zfill(5)
            for col in FINAL_COLUMNS:
                values = keys.map(county[col]).to_numpy()
                if col == 'n_utilities':
                    # Keep the parent column's numeric dtype for downstream compatibility.
                    X[col] = values.astype(oldX[col].dtype)
                elif not np.allclose(values, oldX[col], rtol=1e-12, atol=1e-12):
                    raise ValueError(f'Unexpected external change: {col}; investigate before release.')
        # Verify row identity and targets against raw input, not merely the parent cache.
        pred = raw.loc[raw._hour_idx.between(72, 215)].reset_index(drop=True)
        pd.testing.assert_frame_equal(meta[['fipsCode', 'timestamp_et']], pred[['fipsCode', 'timestamp_et']])
        if y is not None:
            pd.testing.assert_frame_equal(y, pred[HORIZONS], check_exact=True)
        if oldmeta is not None:
            pd.testing.assert_frame_equal(meta, oldmeta, check_exact=True)
        if oldy is not None:
            pd.testing.assert_frame_equal(y, oldy, check_exact=True)
        validation[tag] = _validate_split(X, y, meta, tag)
        if oldX is not None:
            allowed = {c for c in OBSERVED_FEATURES if 'osi' in c} | {'frac_zero_72h', 'hours_since_peak', 'n_utilities'}
            for col in X:
                a, b = oldX[col].to_numpy(), X[col].to_numpy()
                changed = ~((a == b) | (pd.isna(a) & pd.isna(b)))
                material = ~np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
                if col not in allowed and material.any():
                    raise ValueError(f'Unexpected feature change outside correction scope: {tag}/{col}')
                if changed.any():
                    delta = np.abs(a-b)
                    changes.append({'split': tag, 'feature': col, 'changed_rows_exact': int(changed.sum()),
                        'changed_rows_tolerance_1e-12': int(material.sum()),
                        'changed_counties': int(meta.loc[changed, 'fipsCode'].nunique()),
                        'nan_status_changed_rows': int((pd.isna(a) != pd.isna(b)).sum()),
                        'max_abs_change': float(np.nanmax(delta)) if np.isfinite(delta).any() else None})
        frames[f'features_{tag}'] = X
        frames[f'meta_{tag}'] = meta
        if y is not None:
            frames['targets_train'] = y
        obs_mask = raw._hour_idx.lt(72)
        validation[tag]['observed_reconstruction_differs_from_stored_osi'] = (
            int(raw.loc[obs_mask, 'osi'].ne(prepared.loc[obs_mask, 'osi']).sum()) if 'osi' in raw else None)
    validation['utilities'] = {'matched_counties': len(matches), 'changed_counties': int(matches.changed.sum()),
                               'unmatched_counties': 0}
    validation['official_targets_and_meta_unchanged'] = True
    validation['build_mode'] = 'from_raw' if from_raw else 'parent_cache_patch'
    inputs = raw_paths + [COUNTY_DBF, EIA_FILE, ROOT / 'data/rural_urban_codes.csv',
        ROOT / 'data/county_landcover.csv', ROOT / 'data/county_tree_canopy_2025.csv', ROOT / 'data/county_features.csv',
        ROOT / 'cv/cv_assignments_balanced_v1_seed42.csv', RELEASE / 'parent_inputs_sha256.json']
    inputs += [table_path(n, PARENT) for n in TABLE_NAMES if table_path(n, PARENT).exists()]
    sources = [Path(__file__), ROOT / 'code_phase1/data_loader.py', ROOT / 'code_phase1/features.py',
               ROOT / 'code_phase1/config.py', ROOT / 'data/external/scripts/build_county_features.py',
               RELEASE / 'build_features.py']
    manifest = {'version': VERSION, 'parent': PARENT, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'build_mode': validation['build_mode'], 'feature_count': 141,
        'inputs_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in inputs},
        'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in sources},
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__},
        'semantics': {'observed_osi': 'stored P/N/D/R weighted sum, clip lower=0, round(4), hours 0..71 only',
            'weather_windows': '[t,t+h], truncated at hour 215', 'peak4h': 'highest min(4, available) values; not necessarily consecutive'}}
    RELEASE.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # Stage all artifacts before replacing anything. Manifest is the commit marker and is written last.
    with tempfile.TemporaryDirectory(prefix='v155_stage_', dir=CACHE) as stage_dir:
        staged = {}
        for name, frame in frames.items():
            dest = table_path(name)
            temp = Path(stage_dir) / dest.name
            frame.to_parquet(temp, index=False)
            staged[dest] = temp
        payloads = {CACHE / f'feature_names_{VERSION}.json': frozen_schema(), RELEASE / 'validation.json': validation}
        for dest, payload in payloads.items():
            temp = Path(stage_dir) / dest.name
            temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
            staged[dest] = temp
        for dest, frame, index in [(ROOT / f'data/county_features_{VERSION}.csv', county, True),
                (RELEASE / 'eia_county_matches.csv', matches, True),
                (RELEASE / 'feature_changes.csv', pd.DataFrame(changes), False)]:
            temp = Path(stage_dir) / dest.name
            frame.to_csv(temp, index=index)
            staged[dest] = temp
        manifest['outputs_sha256'] = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(t) for p, t in staged.items()}
        marker = Path(stage_dir) / MANIFEST.name
        marker.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        for dest, temp in staged.items():
            os.replace(temp, dest)
        os.replace(marker, MANIFEST)
    load_feature_dataset()
    print(json.dumps(validation, indent=2), flush=True)
    return validation

def load_feature_dataset(verify=True):
    """Load the complete frozen version; validate checksums and semantic shape by default."""
    if not MANIFEST.exists():
        raise FileNotFoundError('Build v1.5.5 first: python versions/v1.5.5/build_features.py')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['version'] != VERSION:
        raise ValueError('Manifest version mismatch.')
    if verify:
        for name, digest in manifest['outputs_sha256'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Dataset artifact changed or incomplete: {name}')
    frames = {name: pd.read_parquet(table_path(name)) for name in TABLE_NAMES}
    for tag in ('train', 'test'):
        _validate_split(frames[f'features_{tag}'], frames.get('targets_train') if tag == 'train' else None,
                        frames[f'meta_{tag}'], tag)
    names = json.loads((CACHE / f'feature_names_{VERSION}.json').read_text(encoding='utf-8'))
    if names != list(frames['features_train']):
        raise ValueError('Feature-name manifest differs from data.')
    return FeatureDataset(frames['features_train'], frames['targets_train'], frames['meta_train'],
                          frames['features_test'], frames['meta_test'])
