"""Frozen metric definitions copied from the D lower-bound experiment."""
import numpy as np

def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or not len(y):
        raise ValueError("Invalid score inputs")
    e = p-y
    return {"n":len(y), "rmse":float(np.sqrt(np.mean(e**2))), "mae":float(np.mean(np.abs(e))),
            "sse":float(np.sum(e**2)), "bias":float(e.mean())}

def comparison(y, before, after):
    b, a = metrics(y,before), metrics(y,after)
    return {"n":b["n"], **{f"baseline_{k}":v for k,v in b.items() if k!="n"},
            **{f"candidate_{k}":v for k,v in a.items() if k!="n"},
            "rmse_delta":a["rmse"]-b["rmse"], "rmse_change_pct":100*(a["rmse"]/b["rmse"]-1) if b["rmse"] else np.nan,
            "sse_reduction":b["sse"]-a["sse"], "changed_predictions":int(np.count_nonzero(np.asarray(before)!=np.asarray(after)))}

def bootstrap(y, before, after, codes, seed, replicates):
    counties, inverse = np.unique(np.asarray(codes), return_inverse=True)
    n = np.bincount(inverse)
    b = np.bincount(inverse, weights=(np.asarray(before)-np.asarray(y))**2)
    a = np.bincount(inverse, weights=(np.asarray(after)-np.asarray(y))**2)
    take = np.random.default_rng(seed).integers(0,len(counties),(replicates,len(counties)))
    delta = np.sqrt(a[take].sum(axis=1)/n[take].sum(axis=1))-np.sqrt(b[take].sum(axis=1)/n[take].sum(axis=1))
    low, high = np.quantile(delta,[.025,.975])
    return {"counties":len(counties),"replicates":replicates,"seed":seed,"ci_low":float(low),"ci_high":float(high),
            "bootstrap_fraction_delta_lt0":float(np.mean(delta<0)),"bootstrap_fraction_delta_eq0":float(np.mean(delta==0))}
