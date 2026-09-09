"""Frozen v1.5.6 dataset: v1.5.5 plus land-cover and window-quality features."""
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
for path in (ROOT / '.python_packages', ROOT / 'code_phase1'):
    sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

import feature_dataset as v155
from config import FORBIDDEN_AS_FEATURES, HORIZONS, HORIZON_HOURS

VERSION = 'v1.5.6'
PARENT = 'v1.5.5'
CACHE = ROOT / 'cache'
RELEASE = ROOT / 'versions' / VERSION
MANIFEST = CACHE / f'manifest_{VERSION}.json'
TABLE_NAMES = ['features_train', 'targets_train', 'meta_train', 'features_test', 'meta_test']
EXTRA_COLUMNS = ['pct_unmapped', 'pct_forest_classified']
WINDOW_HOURS = [1, 6, 24, 48]
WINDOW_QUALITY_COLUMNS = [
    name
    for h in WINDOW_HOURS
    for name in (
        f'window_len_next_{h}h',
        f'is_full_next_{h}h',
        f'tp_rate_next_{h}h',
        f'gust_gt30_frac_next_{h}h',
        f'has_weather_at_t{h}h',
    )
]


@dataclass
class FeatureDataset:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    meta_train: pd.DataFrame
    X_test: pd.DataFrame
    meta_test: pd.DataFrame


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def table_path(name, version=VERSION):
    return CACHE / f'{name}_{version}.parquet'


def landcover_schema():
    names = []
    for name in v155.frozen_schema():
        names.append(name)
        if name == 'pct_wetland':
            names.extend(EXTRA_COLUMNS)
    return names


def frozen_schema():
    return landcover_schema() + WINDOW_QUALITY_COLUMNS


def _load_landcover_features(expected_fips):
    landcover = pd.read_csv(ROOT / 'data/county_landcover.csv', dtype={'fipsCode': str})
    landcover['fipsCode'] = landcover['fipsCode'].str.zfill(5)
    if landcover['fipsCode'].duplicated().any():
        raise ValueError('Duplicate FIPS in county_landcover.csv.')
    landcover = landcover.set_index('fipsCode').reindex(expected_fips)
    values = landcover[['pct_forest', 'pct_unmapped']].apply(pd.to_numeric, errors='raise')
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError('Missing or non-finite land-cover sensitivity values.')
    if ((values < 0) | (values > 100)).any().any():
        raise ValueError('Land-cover percentages must be in [0,100].')
    classified = 100.0 - values['pct_unmapped']
    if not classified.gt(0).all():
        raise ValueError('pct_unmapped leaves no classified pixels for at least one county.')
    result = pd.DataFrame(index=expected_fips)
    result['pct_unmapped'] = values['pct_unmapped']
    result['pct_forest_classified'] = 100.0 * values['pct_forest'] / classified
    if ((result['pct_forest_classified'] < 0) | (result['pct_forest_classified'] > 100)).any():
        raise ValueError('Classified forest percentage is outside [0,100].')
    return result


def _add_landcover_features(X, meta, landcover):
    keys = meta.fipsCode.astype(str).str.zfill(5)
    additions = landcover.reindex(keys).reset_index(drop=True)
    if additions.isna().any().any():
        raise ValueError('Unmatched land-cover sensitivity rows.')
    X = X.copy()
    for col in EXTRA_COLUMNS:
        X[col] = additions[col].to_numpy()
    return X[landcover_schema()]


def _add_window_quality_features(X, meta):
    X = X.copy()
    hour_idx = meta.hour_idx.to_numpy(dtype=int)
    remaining = 216 - hour_idx
    for h in WINDOW_HOURS:
        window_len = np.minimum(h + 1, remaining).astype(float)
        is_full = (hour_idx + h < 216).astype(int)
        X[f'window_len_next_{h}h'] = window_len
        X[f'is_full_next_{h}h'] = is_full
        X[f'tp_rate_next_{h}h'] = X[f'total_tp_next_{h}h'].to_numpy(dtype=float) / window_len
        X[f'gust_gt30_frac_next_{h}h'] = X[f'gust_gt30_next_{h}h'].to_numpy(dtype=float) / window_len
        X[f'has_weather_at_t{h}h'] = is_full
    return X[frozen_schema()]


def _validate_split(X, y, meta, tag):
    county_count = 239 if tag == 'train' else 63
    if X.shape != (county_count * 144, len(frozen_schema())) or list(X) != frozen_schema():
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
    values = X.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f'{tag}: infinite feature values.')
    allowed_nan = {f'{name}_at_t{h}h': meta.hour_idx + h >= 216
                   for h in (1, 6, 24, 48) for name in ('gust', 'wind_speed', 't2m', 'tp')}
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


def build_dataset(overwrite=False):
    outputs = [table_path(name) for name in TABLE_NAMES] + [
        CACHE / f'feature_names_{VERSION}.json', ROOT / f'data/county_features_{VERSION}.csv',
        RELEASE / 'feature_changes.csv', RELEASE / 'validation.json']
    if not overwrite and any(p.exists() for p in outputs + [MANIFEST]):
        raise FileExistsError('v1.5.6 output exists. Use --verify, or explicitly --overwrite to rebuild.')

    parent = v155.load_feature_dataset()
    expected_fips = pd.Index(
        sorted(set(parent.meta_train.fipsCode.astype(str).str.zfill(5))
               | set(parent.meta_test.fipsCode.astype(str).str.zfill(5)))
    )
    landcover = _load_landcover_features(expected_fips)
    county_parent = pd.read_csv(ROOT / f'data/county_features_{PARENT}.csv', dtype={'fipsCode': str})
    county_parent['fipsCode'] = county_parent['fipsCode'].str.zfill(5)
    county_parent = county_parent.set_index('fipsCode').reindex(expected_fips)
    county = county_parent.join(landcover, validate='one_to_one')
    if county.isna().any().any():
        raise ValueError('Incomplete v1.5.6 county feature table.')

    frames = {
        'features_train': _add_window_quality_features(
            _add_landcover_features(parent.X_train, parent.meta_train, landcover),
            parent.meta_train,
        ),
        'targets_train': parent.y_train.copy(),
        'meta_train': parent.meta_train.copy(),
        'features_test': _add_window_quality_features(
            _add_landcover_features(parent.X_test, parent.meta_test, landcover),
            parent.meta_test,
        ),
        'meta_test': parent.meta_test.copy(),
    }
    validation = {
        'train': _validate_split(frames['features_train'], frames['targets_train'], frames['meta_train'], 'train'),
        'test': _validate_split(frames['features_test'], None, frames['meta_test'], 'test'),
        'official_targets_and_meta_unchanged': True,
        'parent': PARENT,
        'new_features': {
            'pct_unmapped': 'unclassified share in county_landcover.csv; percent of valid pixels',
            'pct_forest_classified': '100 * pct_forest / (100 - pct_unmapped)',
            'window_len_next_{h}h': 'number of available points in the existing [t,t+h] weather window',
            'is_full_next_{h}h': '1 if the existing [t,t+h] weather window reaches t+h, else 0',
            'tp_rate_next_{h}h': 'total_tp_next_{h}h divided by window_len_next_{h}h',
            'gust_gt30_frac_next_{h}h': 'gust_gt30_next_{h}h divided by window_len_next_{h}h',
            'has_weather_at_t{h}h': '1 if exact weather at t+h is available, else 0',
        },
    }
    pd.testing.assert_frame_equal(frames['targets_train'], parent.y_train, check_exact=True)
    pd.testing.assert_frame_equal(frames['meta_train'], parent.meta_train, check_exact=True)
    pd.testing.assert_frame_equal(frames['meta_test'], parent.meta_test, check_exact=True)
    pd.testing.assert_frame_equal(frames['features_train'][parent.X_train.columns], parent.X_train, check_exact=True)
    pd.testing.assert_frame_equal(frames['features_test'][parent.X_test.columns], parent.X_test, check_exact=True)

    changes = []
    for tag in ('train', 'test'):
        meta = frames[f'meta_{tag}']
        for col in EXTRA_COLUMNS + WINDOW_QUALITY_COLUMNS:
            changes.append({
                'split': tag,
                'feature': col,
                'changed_rows_exact': len(meta),
                'changed_rows_tolerance_1e-12': len(meta),
                'changed_counties': int(meta.fipsCode.nunique()),
                'nan_status_changed_rows': 0,
                'min': float(frames[f'features_{tag}'][col].min()),
                'max': float(frames[f'features_{tag}'][col].max()),
                'mean': float(frames[f'features_{tag}'][col].mean()),
            })

    inputs = [ROOT / 'data/county_landcover.csv', ROOT / f'data/county_features_{PARENT}.csv',
              CACHE / f'manifest_{PARENT}.json']
    inputs += [CACHE / f'{name}_{PARENT}.parquet' for name in TABLE_NAMES]
    inputs += [CACHE / f'feature_names_{PARENT}.json']
    sources = [Path(__file__), RELEASE / 'build_features.py', ROOT / 'code_phase1/config.py',
               ROOT / 'code_phase1/cache.py']
    manifest = {
        'version': VERSION,
        'parent': PARENT,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'build_mode': 'parent_cache_plus_landcover_sensitivity',
        'feature_count': len(frozen_schema()),
        'inputs_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in inputs},
        'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in sources if p.exists()},
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__},
        'semantics': {
            'pct_unmapped': 'from county_landcover.csv; percent of valid pixels not in modeled major groups',
            'pct_forest_classified': '100 * pct_forest / (100 - pct_unmapped); sensitivity view only',
            'window_quality': 'availability and normalized-rate descriptors for existing [t,t+h] weather windows',
            'parent_semantics': PARENT,
        },
    }

    RELEASE.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v156_stage_', dir=CACHE) as stage_dir:
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
    print(json.dumps(validation, indent=2, ensure_ascii=False), flush=True)
    return validation


def load_feature_dataset(verify=True):
    if not MANIFEST.exists():
        raise FileNotFoundError('Build v1.5.6 first: python versions/v1.5.6/build_features.py')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['version'] != VERSION:
        raise ValueError('Manifest version mismatch.')
    if verify:
        for name, digest in manifest['outputs_sha256'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Dataset artifact changed or incomplete: {name}')
    frames = {name: pd.read_parquet(table_path(name)) for name in TABLE_NAMES}
    _validate_split(frames['features_train'], frames['targets_train'], frames['meta_train'], 'train')
    _validate_split(frames['features_test'], None, frames['meta_test'], 'test')
    names = json.loads((CACHE / f'feature_names_{VERSION}.json').read_text(encoding='utf-8'))
    if names != list(frames['features_train']):
        raise ValueError('Feature-name manifest differs from data.')
    return FeatureDataset(frames['features_train'], frames['targets_train'], frames['meta_train'],
                          frames['features_test'], frames['meta_test'])
