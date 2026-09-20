"""Approved diagnostic entry point. Reads frozen artifacts; never fits models."""
import argparse
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from diagnostic_core import *


def preflight(ctx):
    nr=[]
    for f,g in ctx.raw.groupby('fipsCode',sort=True):
        delta=g.outageCount.diff()
        counts=g.customersTracked.to_numpy(float)
        assert np.isfinite(counts).all() and (counts>0).all()
        q=delta/g.customersTracked
        n=q.clip(lower=0).rolling(3,center=True,min_periods=3).mean()
        r=(-q).clip(lower=0).rolling(3,center=True,min_periods=3).mean()
        valid=n.notna().to_numpy()
        nd=float(np.max(abs(n.to_numpy()[valid]-g.N_t.to_numpy()[valid])))
        rd=float(np.max(abs(r.to_numpy()[valid]-g.R_t.to_numpy()[valid])))
        net=q.rolling(3,center=True,min_periods=3).mean().to_numpy()[valid]
        netdiff=float(np.max(abs(net-(g.N_t-g.R_t).to_numpy()[valid])))
        assert nd<1e-6 and rd<1e-6 and netdiff<2e-6
        nr.append(dict(fipsCode=f,complete_windows=int(valid.sum()),N_max_abs_difference=nd,
            R_max_abs_difference=rd,net_max_abs_difference=netdiff,customers_min=counts.min(),
            customers_max=counts.max(),denominator_change_hours=int(np.count_nonzero(np.diff(counts))),
            N_min=g.N_t.min(),N_max=g.N_t.max(),R_min=g.R_t.min(),R_max=g.R_t.max(),
            N_and_R_positive_hours=int(((g.N_t>0)&(g.R_t>0)).sum()),
            formula_unavailable_hours=','.join(g.hour_idx[~valid].astype(str))))
    composed=np.maximum(linear(ctx.truth),0)
    gap=float(np.max(abs(composed-ctx.y)))
    assert gap<=5.1e-5
    np.testing.assert_allclose(controls(scenario_parts(ctx,'O_N_partial_0.0'))['main'],ctx.ctl['main'],atol=0,rtol=0)
    foldfile=pd.read_csv(ctx.run/'cv_assignments.csv',dtype={'fipsCode':str}).sort_values('fipsCode').reset_index(drop=True)
    rootfile=pd.read_csv(ROOT/f'cv/cv_assignments_balanced_v1_seed{ctx.seed}.csv',dtype={'fipsCode':str}).sort_values('fipsCode').reset_index(drop=True)
    pd.testing.assert_frame_equal(foldfile,rootfile)
    frame(ctx.output/'tables/label_formula_audit.csv',nr)
    result=dict(status='PASS',split_seed=ctx.seed,reference_identity=IDS[ctx.seed],counties=ctx.nc,
        origin_rows=len(ctx.meta),valid_counts={h:ctx.nc*(144-h) for h in HS},
        N_models_scope_checked=20,N_matches_original=True,official_targets_match=True,
        max_current_prediction_difference=ctx.maxdiff,max_official_formula_difference=gap,
        N_formula_max=max(x['N_max_abs_difference'] for x in nr),
        R_formula_max=max(x['R_max_abs_difference'] for x in nr),
        oracle_alpha_zero_exact=True,training_performed=False)
    write_json(ctx.output/'tables/integrity_checks.json',result)
    print(f'split{ctx.seed} preflight PASS; max prediction difference={ctx.maxdiff:.3g}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','analyze'],required=True)
    ap.add_argument('--split-seeds',nargs='+',type=int,default=list(SEEDS));args=ap.parse_args()
    assert tuple(args.split_seeds)==SEEDS,'This protocol executes all three fixed CVs in order'
    if (OUT/'completion.json').exists(): raise RuntimeError('Completed diagnostic is immutable; use verifier for read-only checks')
    initialize();sources=source_inventory()
    print(f'Validated {len(sources)} frozen input files',flush=True)
    for seed in args.split_seeds:
        ctx=load_context(seed);preflight(ctx)
        if args.stage=='analyze':
            from diagnostic_analysis import analyze_split
            analyze_split(ctx)
    if args.stage=='analyze':
        from diagnostic_analysis import summarize
        summarize()
    assert all(sha(p)==h for p,h in sources.items())
    print(f'{args.stage} completed; all frozen input hashes unchanged',flush=True)


if __name__=='__main__': main()
