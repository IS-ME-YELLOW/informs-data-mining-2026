"""Persisted, balanced, county-grouped cross-validation folds."""

import os

import numpy as np
import pandas as pd

from config import CV_ASSIGNMENT_FILE, N_FOLDS, SEED


REQUIRED_META_COLUMNS = ('fipsCode', 'stateAbbr', 'severity_tier')


def _county_table(meta_df):
    """Return exactly one validated metadata row per county."""
    missing = [column for column in REQUIRED_META_COLUMNS if column not in meta_df]
    if missing:
        raise ValueError(f'CV metadata is missing required columns: {missing}')

    counties = meta_df.loc[:, REQUIRED_META_COLUMNS].copy()
    counties['fipsCode'] = counties['fipsCode'].astype(str)
    counties['stateAbbr'] = counties['stateAbbr'].astype(str)
    counties['severity_tier'] = pd.to_numeric(
        counties['severity_tier'], errors='raise'
    ).astype(int)

    unique_counties = counties.drop_duplicates()
    if unique_counties['fipsCode'].duplicated().any():
        bad_fips = sorted(
            unique_counties.loc[unique_counties['fipsCode'].duplicated(False), 'fipsCode']
            .unique()
        )
        raise ValueError(f'A FIPS code belongs to multiple CV strata: {bad_fips[:10]}')

    return unique_counties.sort_values(
        ['stateAbbr', 'severity_tier', 'fipsCode']
    ).reset_index(drop=True)


def _validate_assignment(assignment, counties, n_splits):
    """Reject stale, incomplete, or imbalanced persisted assignments."""
    required_columns = {'fipsCode', 'stateAbbr', 'severity_tier', 'fold'}
    if set(assignment.columns) != required_columns:
        raise ValueError(
            f'CV assignment columns must be {sorted(required_columns)}, '
            f'got {assignment.columns.tolist()}'
        )

    assignment = assignment.copy()
    assignment['fipsCode'] = assignment['fipsCode'].astype(str)
    assignment['stateAbbr'] = assignment['stateAbbr'].astype(str)
    assignment['severity_tier'] = pd.to_numeric(
        assignment['severity_tier'], errors='raise'
    ).astype(int)
    assignment['fold'] = pd.to_numeric(assignment['fold'], errors='raise').astype(int)

    if assignment['fipsCode'].duplicated().any():
        raise ValueError('CV assignment contains duplicate fipsCode values.')
    if set(assignment['fold']) != set(range(n_splits)):
        raise ValueError(f'CV assignment must contain folds 0..{n_splits - 1}.')

    expected = counties.set_index('fipsCode').sort_index()
    actual = assignment.set_index('fipsCode')[
        ['stateAbbr', 'severity_tier', 'fold']
    ].sort_index()
    if not expected.index.equals(actual.index):
        raise ValueError('CV assignment FIPS set does not match the current metadata.')
    if not expected[['stateAbbr', 'severity_tier']].equals(
            actual[['stateAbbr', 'severity_tier']]):
        raise ValueError('CV assignment strata do not match the current metadata.')

    fold_counts = assignment['fold'].value_counts().sort_index()
    if fold_counts.max() - fold_counts.min() > 1:
        raise ValueError(f'CV assignment is globally imbalanced: {fold_counts.to_dict()}')

    stratum_counts = assignment.groupby(
        ['stateAbbr', 'severity_tier', 'fold']
    ).size().unstack(fill_value=0).reindex(columns=range(n_splits), fill_value=0)
    if ((stratum_counts.max(axis=1) - stratum_counts.min(axis=1)) > 1).any():
        raise ValueError('CV assignment is not balanced within every stratum.')


def _build_assignment(counties, n_splits, seed):
    """Balance every stratum and keep total county counts across folds equal."""
    rng = np.random.RandomState(seed)
    fold_sizes = np.zeros(n_splits, dtype=int)
    records = []

    for (state, tier), stratum in counties.groupby(
            ['stateAbbr', 'severity_tier'], sort=True):
        fips_codes = stratum['fipsCode'].to_numpy(copy=True)
        rng.shuffle(fips_codes)
        base, remainder = divmod(len(fips_codes), n_splits)

        # Every fold receives the stratum's base allocation. Residual counties
        # go to globally smallest folds; random ties prevent early-fold bias.
        tie_order = rng.permutation(n_splits)
        tie_rank = np.empty(n_splits, dtype=int)
        tie_rank[tie_order] = np.arange(n_splits)
        ranked_folds = sorted(
            range(n_splits), key=lambda fold: (fold_sizes[fold], tie_rank[fold])
        )
        extra_folds = ranked_folds[:remainder]
        slots = [fold for fold in range(n_splits) for _ in range(base)] + extra_folds
        rng.shuffle(slots)

        for fips, fold in zip(fips_codes, slots):
            records.append({
                'fipsCode': fips,
                'stateAbbr': state,
                'severity_tier': int(tier),
                'fold': int(fold),
            })
        fold_sizes += base
        fold_sizes[extra_folds] += 1

    assignment = pd.DataFrame(records).sort_values('fipsCode').reset_index(drop=True)
    _validate_assignment(assignment, counties, n_splits)
    return assignment


def _load_or_create_assignment(counties, n_splits, seed, assignment_file):
    if os.path.exists(assignment_file):
        assignment = pd.read_csv(
            assignment_file, dtype={'fipsCode': str, 'stateAbbr': str}
        )
        _validate_assignment(assignment, counties, n_splits)
        return assignment

    assignment = _build_assignment(counties, n_splits, seed)
    os.makedirs(os.path.dirname(assignment_file), exist_ok=True)
    assignment.to_csv(assignment_file, index=False)
    return assignment


def get_cv_folds(meta_df, n_splits=N_FOLDS, seed=SEED):
    """Return county-grouped folds from a validated persisted assignment."""
    counties = _county_table(meta_df)
    assignment_file = CV_ASSIGNMENT_FILE
    if n_splits != N_FOLDS or seed != SEED:
        stem, extension = os.path.splitext(CV_ASSIGNMENT_FILE)
        assignment_file = f'{stem}_folds{n_splits}_seed{seed}{extension}'

    assignment = _load_or_create_assignment(
        counties, n_splits, seed, assignment_file
    )
    fold_by_fips = assignment.set_index('fipsCode')['fold']
    row_folds = meta_df['fipsCode'].astype(str).map(fold_by_fips)
    if row_folds.isna().any():
        raise ValueError('At least one metadata row has no CV fold assignment.')

    row_folds = row_folds.to_numpy(dtype=int)
    return [
        (np.flatnonzero(row_folds != fold), np.flatnonzero(row_folds == fold))
        for fold in range(n_splits)
    ]
