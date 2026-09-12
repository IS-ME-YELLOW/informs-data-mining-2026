"""Load Phase-1 feature caches and align rows to a county x time tensor."""

import numpy as np
import pandas as pd

from config import (GAT_FEATURES, HORIZON_HOURS, PRED_END, PRED_START,
                    TEST_FEATURES, TEST_META, TRAIN_FEATURES, TRAIN_META,
                    TRAIN_TARGETS)


def load_cached_data():
    X_train = pd.read_parquet(TRAIN_FEATURES)
    X_test = pd.read_parquet(TEST_FEATURES)
    meta_train = pd.read_parquet(TRAIN_META)
    meta_test = pd.read_parquet(TEST_META)
    y_train = pd.read_parquet(TRAIN_TARGETS)
    for meta in (meta_train, meta_test):
        meta["fips_str"] = meta["fipsCode"].astype(str).str.zfill(5)
    return X_train, X_test, meta_train, meta_test, y_train


def make_county_time_view(X_train, X_test, meta_train, meta_test, base_train, base_test,
                          horizon, coords, state_map):
    """Construct features/base predictions in [144 timestamps, 302 counties]."""
    h = str(HORIZON_HOURS[horizon])
    names = []
    for template in GAT_FEATURES:
        names.append(template.format(h=h))
    missing = [n for n in names if n not in X_train.columns or n not in X_test.columns]
    if missing:
        raise ValueError(f"Missing GAT features: {missing}")
    train_fips = sorted(meta_train["fips_str"].unique())
    test_fips = sorted(meta_test["fips_str"].unique())
    all_fips = sorted(set(train_fips + test_fips))
    node_of = {f: i for i, f in enumerate(all_fips)}
    n_time = PRED_END - PRED_START
    n_nodes = len(all_fips)
    n_feat = 1 + len(names) + 2 + 4
    features = np.zeros((n_time, n_nodes, n_feat), dtype=np.float32)
    base = np.zeros((n_time, n_nodes), dtype=np.float32)
    train_rows = np.full((n_time, n_nodes), -1, dtype=np.int64)
    test_rows = np.full((n_time, n_nodes), -1, dtype=np.int64)
    state_codes = {s: i for i, s in enumerate(sorted(set(state_map.values())))}

    def fill(X, meta, base_pred, row_lookup, is_train):
        for row_idx, row in meta.iterrows():
            t = int(row["hour_idx"]) - PRED_START
            if not 0 <= t < n_time:
                continue
            node = node_of[row["fips_str"]]
            row_lookup[t, node] = row_idx
            values = X.iloc[row_idx][names].to_numpy(dtype=float)
            state = state_codes[state_map[row["fips_str"]]]
            features[t, node, 0] = float(base_pred[row_idx])
            features[t, node, 1:1 + len(names)] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            features[t, node, 1 + len(names):1 + len(names) + 2] = coords[node]
            one_hot = np.zeros(4, dtype=np.float32)
            if state < 4:
                one_hot[state] = 1.0
            features[t, node, 1 + len(names) + 2:] = one_hot

    fill(X_train, meta_train, base_train, train_rows, True)
    fill(X_test, meta_test, base_test, test_rows, False)
    # Metadata arrays are easier to use for exact OOF indexing later.
    return features, base, train_rows, test_rows, all_fips, names


def make_state_map(meta_train, meta_test):
    out = {}
    for meta in (meta_train, meta_test):
        for row in meta.itertuples():
            out[row.fips_str] = row.stateAbbr
    return out
