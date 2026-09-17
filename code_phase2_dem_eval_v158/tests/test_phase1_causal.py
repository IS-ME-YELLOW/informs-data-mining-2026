import os
from pathlib import Path
import subprocess
import sys


def test_real_phase1_v156_future_outage_invariance():
    """Exercise the actual Phase-1/v1.5.6 builder in an isolated interpreter."""

    root = Path(__file__).resolve().parents[2]
    script = r'''
import numpy as np
import pandas as pd
import data_loader
import features
import feature_dataset as v155
import feature_dataset_v156 as v156

raw = pd.read_csv("data/DM_Train.csv")
raw = raw[raw.fipsCode.astype(str).str.zfill(5) == "18009"].copy()
raw["_dt"] = pd.to_datetime(raw.timestamp_et, format="%m/%d/%Y %H:%M")
raw = raw.sort_values(["fipsCode", "_dt"]).reset_index(drop=True)
raw["_hour_idx"] = raw.groupby("fipsCode").cumcount()
county = pd.read_csv("data/county_features_v1.5.6.csv", dtype={"fipsCode": str}).set_index("fipsCode")
land = county[["pct_unmapped", "pct_forest_classified"]]

def build(frame):
    prepared = data_loader.preprocess(frame.copy())
    x, y, meta = features.build_feature_matrix(prepared, is_train=True)
    x = v155.attach_external(x, meta, county[v155.FINAL_COLUMNS])
    x = v156._add_landcover_features(x, meta, land)
    x = v156._add_window_quality_features(x, meta)
    return x, y, meta

base_x, base_y, base_meta = build(raw)
changed = raw.copy()
future = changed["_hour_idx"] >= 72
outage_fields = [
    "outageCount", "outage_pct", "P_t", "N_t", "D_t", "R_t", "osi",
    "outage_pct_lag1h", "osi_lag1h", "outage_pct_lag3h", "osi_lag3h",
    "outage_pct_lag6h", "osi_lag6h", "outage_pct_lag24h", "osi_lag24h",
    "outage_pct_lag48h", "osi_lag48h", "osi_target_t01h", "osi_target_t03h",
    "osi_target_t06h", "osi_target_t24h", "osi_target_t48h",
    "osi_delta_t01h", "osi_delta_t03h", "osi_delta_t06h", "osi_delta_t24h",
    "osi_delta_t48h",
]
changed.loc[future, outage_fields] = 999.0
new_x, new_y, new_meta = build(changed)
assert base_x.shape == (144, 163)
assert list(base_x) == v156.frozen_schema()
assert base_meta.reset_index(drop=True).equals(new_meta.reset_index(drop=True))
assert np.array_equal(base_x.to_numpy(dtype=float), new_x.to_numpy(dtype=float), equal_nan=True)
assert not base_y.equals(new_y)
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "code_phase1")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
