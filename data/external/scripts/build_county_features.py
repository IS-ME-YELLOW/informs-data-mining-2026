"""Build corrected county covariates from existing official tables (no GIS downloads)."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if (PROJECT_ROOT / '.python_packages').is_dir():
    sys.path.insert(0, str(PROJECT_ROOT / '.python_packages'))
import numpy as np
import pandas as pd

DATA_DIR = PROJECT_ROOT / 'data'
COUNTY_DBF = DATA_DIR / 'external/tl_2025_us_county/tl_2025_us_county.dbf'
EIA_FILE = DATA_DIR / 'external/eia861_2024/Service_Territory_2024.xlsx'
STATE_ABBR = {'18': 'IN', '39': 'OH', '42': 'PA', '54': 'WV'}
FINAL_COLUMNS = ['rural_urban_code', 'is_metro', 'pop_density', 'n_utilities',
    'pct_forest', 'pct_developed', 'pct_agriculture', 'pct_water', 'pct_wetland',
    'tree_canopy_pct', 'tree_canopy_std']

def normalize_county_name(value):
    """Exact case/whitespace/punctuation normalization; no fuzzy matching."""
    if pd.isna(value) or not str(value).strip():
        raise ValueError('Missing county name.')
    return re.sub(r'[\W_]+', '', str(value).casefold())

def load_county_reference():
    # Only attributes are needed, not the .shp/.shx geometries.
    from dbfread import DBF
    frame = pd.DataFrame(iter(DBF(str(COUNTY_DBF), encoding='utf-8')))
    frame['STATEFP'] = frame['STATEFP'].astype(str).str.zfill(2)
    frame = frame.loc[frame['STATEFP'].isin(STATE_ABBR)].copy()
    frame['fipsCode'] = frame['GEOID'].astype(str).str.zfill(5)
    frame['stateAbbr'] = frame['STATEFP'].map(STATE_ABBR)
    frame['countyName'] = frame['NAME']
    frame['county_area_sqmi'] = pd.to_numeric(frame['ALAND']) / 2589988.11
    if len(frame) != 302 or frame['fipsCode'].duplicated().any():
        raise ValueError('County reference must contain 302 unique four-state FIPS.')
    if not (frame['county_area_sqmi'] > 0).all():
        raise ValueError('Non-positive county land area.')
    return frame.set_index('fipsCode').sort_index()

def aggregate_eia_utilities(reference, service_territory):
    """Match uniquely within each state; never default unmatched counts."""
    ref = reference.reset_index()[['fipsCode', 'stateAbbr', 'countyName']].copy()
    ref['county_key'] = ref['countyName'].map(normalize_county_name)
    if ref.duplicated(['stateAbbr', 'county_key']).any():
        raise ValueError('County normalization is ambiguous within a state.')
    src = service_territory.copy()
    src['State'] = src['State'].astype(str).str.strip().str.upper()
    src = src.loc[src['State'].isin(STATE_ABBR.values())].copy()
    if src[['County', 'Utility Number']].isna().any().any():
        raise ValueError('Missing EIA county or utility number.')
    src['county_key'] = src['County'].map(normalize_county_name)
    grouped = src.groupby(['State', 'county_key'], as_index=False).agg(
        n_utilities=('Utility Number', 'nunique'),
        eia_county_names=('County', lambda x: ' | '.join(sorted(set(x)))))
    matched = ref.merge(grouped, left_on=['stateAbbr', 'county_key'],
        right_on=['State', 'county_key'], how='left', validate='one_to_one')
    missing = matched.loc[matched['n_utilities'].isna(), ['fipsCode', 'stateAbbr', 'countyName']]
    if len(missing):
        raise ValueError(f'Unmatched EIA counties: {missing.to_dict("records")}')
    matched['n_utilities'] = matched['n_utilities'].astype(int)
    if (matched['n_utilities'] < 1).any():
        raise ValueError('Invalid EIA utility count.')
    return matched.set_index('fipsCode').sort_index()

def _read_county_table(path, columns, expected):
    frame = pd.read_csv(path, dtype={'fipsCode': str})
    frame['fipsCode'] = frame['fipsCode'].str.zfill(5)
    if frame['fipsCode'].duplicated().any():
        raise ValueError(f'Duplicate FIPS in {path.name}.')
    frame = frame.set_index('fipsCode').reindex(expected)[columns]
    frame = frame.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f'Missing or non-finite county data in {path.name}.')
    return frame

def build_county_features():
    """Return 11 static features and an auditable EIA match table."""
    ref = load_county_reference()
    usda = pd.read_csv(DATA_DIR / 'rural_urban_codes.csv', encoding='latin-1', dtype={'FIPS': str})
    usda['FIPS'] = usda['FIPS'].str.zfill(5)
    usda = usda.loc[usda['FIPS'].isin(ref.index)]
    # pivot rejects duplicate attributes, unlike pivot_table(aggfunc='first').
    usda = usda.pivot(index='FIPS', columns='Attribute', values='Value').reindex(ref.index)
    rucc = pd.to_numeric(usda['RUCC_2023'], errors='raise')
    population = pd.to_numeric(usda['Population_2020'], errors='raise')
    if not (rucc.between(1, 9) & rucc.eq(rucc.round())).all() or not population.ge(0).all():
        raise ValueError('Invalid or unmatched USDA values.')
    result = pd.DataFrame(index=ref.index)
    result['rural_urban_code'] = rucc.astype(int)
    result['is_metro'] = rucc.le(3).astype(int)
    result['pop_density'] = population / ref['county_area_sqmi']
    matches = aggregate_eia_utilities(ref, pd.read_excel(EIA_FILE))
    result['n_utilities'] = matches['n_utilities']
    groups = [(DATA_DIR / 'county_landcover.csv',
        ['pct_forest', 'pct_developed', 'pct_agriculture', 'pct_water', 'pct_wetland']),
        (DATA_DIR / 'county_tree_canopy_2025.csv', ['tree_canopy_pct', 'tree_canopy_std'])]
    for path, columns in groups:
        values = _read_county_table(path, columns, ref.index)
        if ((values < 0) | (values > 100)).any().any():
            raise ValueError(f'County percentages outside [0,100]: {path.name}')
        result = result.join(values, validate='one_to_one')
    result = result[FINAL_COLUMNS]
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Incomplete county features; cannot fill missing data with zero.')
    return result, matches

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA_DIR / 'county_features_v1.5.5.csv')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output.resolve() == (DATA_DIR / 'county_features.csv').resolve():
        raise ValueError('Preserve historical county_features.csv; choose a versioned output.')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'{args.output} exists; use --overwrite explicitly.')
    features, matches = build_county_features()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output)
    matches.to_csv(args.output.with_name(args.output.stem + '_eia_matches.csv'))
    print(f'Saved {features.shape} county features to {args.output}')

if __name__ == '__main__':
    main()
