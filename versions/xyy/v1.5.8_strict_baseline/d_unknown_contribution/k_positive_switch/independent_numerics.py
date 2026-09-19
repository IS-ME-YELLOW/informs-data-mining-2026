"""Independent scalar bounds and bucket-based controls; no training API."""
import numpy as np
import pandas as pd
START=pd.Timestamp("2026-03-11")
HOURS=(1,6,24,48)
COMPONENTS=("P_t","N_t","D_t","R_t")
MODES=("C0_direct_osi","C1_component_osi","C2_equal_blend","C3_aligned_component","v18_rule")

def exact(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)

def brute_bounds(raw, meta):
    """Separate scalar implementation: iterate every window, never use future P."""
    lookup = {}
    for row in raw.itertuples(index=False):
        hour = int((pd.Timestamp(row.timestamp_et) - START) / pd.Timedelta(hours=1))
        if 0 <= hour <= 71:
            key = (str(row.fipsCode).zfill(5), hour)
            assert key not in lookup
            lookup[key] = float(row.P_t)
    answer = {}
    for h in HOURS:
        values = []
        for county, origin in zip(meta.fipsCode, meta.hour_idx):
            target = int(origin) + h
            if target > 215:
                values.append(np.nan)
            else:
                values.append(sum(lookup[(county, t)] for t in range(target-5, target+1) if 0 <= t <= 71)/6)
        answer[h] = np.array(values)
    return answer

def finish_osi(values):
    result = np.clip(values, 0, .65)
    result[result < .001] = 0
    return result

def compose(parts):
    return np.maximum(.40*parts["P_t"] + .35*parts["N_t"] + .25*parts["D_t"] - .10*parts["R_t"], 0)

def independent_controls(meta, raw_direct, parts):
    """Use scalar county/target-hour buckets, independent of protocol.py."""
    buckets = {}
    for h in HOURS:
        for i, (county, origin) in enumerate(zip(meta.fipsCode, meta.hour_idx)):
            target = int(origin)+h
            if target <= 215:
                key = (county, target)
                buckets.setdefault(key, []).append((h, [parts[h][c][i] for c in COMPONENTS]))
    means = {key: np.mean([v for _, v in values], axis=0) for key, values in buckets.items()}
    controls = {}
    for h in HOURS:
        aligned = np.array([means.get((f, int(t)+h), [np.nan]*4) for f, t in zip(meta.fipsCode, meta.hour_idx)])
        c1_raw = compose(parts[h])
        direct = raw_direct[f"raw_osi_target_t{h:02d}h"].to_numpy()
        c3_raw = compose(dict(zip(COMPONENTS, aligned.T)))
        values = [finish_osi(direct.copy()), finish_osi(c1_raw), finish_osi((direct+c1_raw)/2), finish_osi(c3_raw)]
        values.append(values[3 if h <= 6 else 1].copy())
        controls[h] = dict(zip(MODES, values))
    return controls, buckets, means

def compare_stats(y, b, a):
    be, ae = b-y, a-y
    br, ar = np.sqrt(np.mean(be**2)), np.sqrt(np.mean(ae**2))
    return {"n": len(y), "baseline_rmse": br, "candidate_rmse": ar,
            "baseline_mae": np.abs(be).mean(), "candidate_mae": np.abs(ae).mean(),
            "baseline_sse": np.dot(be, be), "candidate_sse": np.dot(ae, ae),
            "baseline_bias": be.mean(), "candidate_bias": ae.mean(),
            "rmse_delta": ar-br, "rmse_change_pct": 100*(ar/br-1) if br else np.nan,
            "sse_reduction": np.dot(be, be)-np.dot(ae, ae), "changed_predictions": int(np.sum(a != b))}

def assert_stats(row, y, b, a):
    for key, value in compare_stats(y, b, a).items():
        close(row[key], value)
