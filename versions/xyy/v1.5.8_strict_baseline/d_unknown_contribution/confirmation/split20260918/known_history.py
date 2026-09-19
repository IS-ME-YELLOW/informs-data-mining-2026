"""Prediction-only D lower-bound projection. No labels or model fitting API."""
import numpy as np
import pandas as pd

START = pd.Timestamp("2026-03-11 00:00:00")
HORIZON_HOURS = {f"osi_target_t{h:02d}h": h for h in (1,6,24,48)}


def observed_history(raw):
    """Discard future rows BEFORE reading P; official hour71 is retained."""
    required = {"fipsCode", "timestamp_et", "P_t"}
    if not required.issubset(raw.columns):
        raise ValueError("Missing observed-history columns")
    times = pd.to_datetime(raw["timestamp_et"])
    hours = ((times - START) / pd.Timedelta(hours=1)).to_numpy()
    if not np.isfinite(hours).all() or not (hours == np.floor(hours)).all():
        raise ValueError("Invalid hourly timestamps")
    observed = (hours >= 0) & (hours <= 71)
    frame = raw.loc[observed, ["fipsCode", "P_t"]].copy()
    frame["hour_idx"] = hours[observed].astype(int)
    frame["fipsCode"] = frame.fipsCode.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    if not frame.fipsCode.str.fullmatch(r"\d{5}").all() or frame.duplicated(["fipsCode", "hour_idx"]).any():
        raise ValueError("Invalid/duplicate observed county-hour keys")
    if not np.isfinite(frame.P_t).all() or not frame.P_t.between(0,1).all():
        raise ValueError("Observed P must be finite and in [0,1]")
    if not frame.groupby("fipsCode").hour_idx.apply(lambda v: set(v) == set(range(72))).all():
        raise ValueError("Incomplete observed county history")
    return frame.sort_values(["fipsCode", "hour_idx"]).reset_index(drop=True)


def known_bounds(meta, history):
    """Return D bounds by horizon, aligned to meta row order; invalid tails NaN."""
    if set(history.columns) != {"fipsCode", "P_t", "hour_idx"}:
        raise ValueError("Pass only the observed_history result")
    if not history.hour_idx.between(0,71).all():
        raise ValueError("Future P supplied to bound builder")
    if history.duplicated(["fipsCode", "hour_idx"]).any():
        raise ValueError("Duplicate history keys")
    matrix = history.pivot(index="fipsCode", columns="hour_idx", values="P_t").reindex(columns=range(72))
    if not np.isfinite(matrix.to_numpy()).all() or not ((matrix>=0)&(matrix<=1)).all().all():
        raise ValueError("Incomplete/invalid observed history")
    fips = meta.fipsCode.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    time = pd.to_datetime(meta.timestamp_et)
    hour = np.asarray(meta.hour_idx, dtype=int)
    if not np.array_equal(START + pd.to_timedelta(hour, unit="h"), time):
        raise ValueError("Forecast hour/timestamp mismatch")
    if not ((hour>=72)&(hour<=215)).all() or pd.DataFrame({"fips":fips,"hour":hour}).duplicated().any():
        raise ValueError("Invalid forecast keys")
    if not fips.isin(matrix.index).all():
        raise ValueError("County missing from observed history")
    output = {}
    for name, ahead in HORIZON_HOURS.items():
        target = hour + ahead
        bound = np.zeros(len(meta), dtype=float)
        # Only target hours73..76 share any observed P with a six-hour window.
        for s in range(73,77):
            take = target == s
            if take.any():
                county_sum = matrix.loc[:, list(range(s-5,72))].sum(axis=1) / 6.0
                bound[take] = fips[take].map(county_sum).to_numpy()
        bound[target>215] = np.nan
        output[name] = bound
    return output


def project_components(components, bounds):
    """Copy P/N/D/R arrays and project D only. Does not inspect true D or OSI."""
    if set(components) != set(HORIZON_HOURS) or set(bounds) != set(HORIZON_HOURS):
        raise ValueError("Incorrect horizons")
    projected = {}
    for horizon in HORIZON_HOURS:
        if set(components[horizon]) != {"P_t", "N_t", "D_t", "R_t"}:
            raise ValueError("Incorrect components")
        bound = np.asarray(bounds[horizon], dtype=float)
        valid = np.isfinite(bound)
        if np.isinf(bound).any() or not ((bound[valid]>=0)&(bound[valid]<=1)).all():
            raise ValueError("Invalid lower bound")
        projected[horizon] = {}
        for component, values in components[horizon].items():
            values = np.asarray(values, dtype=float)
            if values.shape != bound.shape or not np.array_equal(np.isfinite(values),valid) or not np.isnan(values[~valid]).all():
                raise ValueError("Component prediction mask/shape mismatch")
            if not ((values[valid]>=0)&(values[valid]<=1)).all():
                raise ValueError("Use the baseline's clipped component predictions")
            result = values.copy()
            if component == "D_t":
                result[valid] = np.maximum(values[valid], bound[valid])
            projected[horizon][component] = result
    return projected
