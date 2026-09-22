"""Per-split P24 confirmation tables and component/OSI accounting."""
from neighbor_protocol import *


def main():
    assert read_json(RUN/'logs/independent_verification.json')['status']=='PASS'
    ctx=load_context()
    for name in ('primary_scores','P_component_metrics','P71_bins','county_metrics','fold_metrics','window_metrics','bootstrap_intervals','branch_diagnostics'):
        d=pd.read_csv(RUN/f'metrics/{name}.csv',float_precision='round_trip',dtype={'fipsCode':str} if name=='county_metrics' else {})
        d.insert(0,'split_seed',SEED);write_frame(OUT/f'summary/{name}.csv',d)
    pp=pd.read_parquet(RUN/'P_component_diagnostics.parquet');saved=normalize(pd.read_parquet(RUN/'control_predictions.parquet'))
    rows=[];summary=[];weights=np.array([.4,.35,.25,-.1])
    for h in HORIZONS:
        hh=HORIZON_HOURS[h];valid=expected_mask(ctx.meta,h)
        d=normalize(pp[(pp['case']=='NB_P24')&(pp.horizon==h)])
        truth=np.column_stack([ctx.data.component_targets[component_target_name(c,h)].to_numpy() for c in COMPONENTS])
        old=ctx.parts[h] if hh>=24 else ctx.aux['aligned_components'][h]
        old=np.column_stack([old[c] for c in COMPONENTS]);new=old.copy();new[:,0]=d['component_clipped' if hh>=24 else 'aligned'].to_numpy()
        y=ctx.data.y_train[h].to_numpy();p0=saved[f'pred_B0_v18_rule_{h}'].to_numpy();p1=saved[f'pred_NB_P24_v18_rule_{h}'].to_numpy()
        e0=.4*(old[:,0]-truth[:,0]);e1=.4*(new[:,0]-truth[:,0]);other=(old[:,1:]-truth[:,1:])@weights[1:]
        b0=p0-old@weights+truth@weights-y;b1=p1-new@weights+truth@weights-y
        sg=e0**2-e1**2;cg=2*(e0-e1)*other;bg=b0**2-b1**2+2*((e0+other)*b0-(e1+other)*b1)
        gain=(p0-y)**2-(p1-y)**2
        np.testing.assert_allclose(sg+cg+bg,gain,atol=1e-12,rtol=0)
        out=ctx.meta.loc[valid].copy();out['split_seed']=SEED;out['case']='NB_P24';out['horizon']=h
        for k,v in [('P_self_sse_gain',sg),('P_other_cross_sse_gain',cg),('postprocessing_sse_gain',bg),('OSI_sse_gain',gain)]:out[k]=v[valid]
        rows.append(out);summary.append(dict(split_seed=SEED,case='NB_P24',horizon=h,
            P_self_sse_gain=float(np.nansum(sg)),P_other_cross_sse_gain=float(np.nansum(cg)),
            postprocessing_sse_gain=float(np.nansum(bg)),OSI_sse_gain=float(np.nansum(gain))))
    write_frame(OUT/'summary/OSI_change_decomposition.csv',summary)
    write_frame(OUT/'summary/OSI_change_decomposition.parquet',pd.concat(rows,ignore_index=True))
    primary=pd.read_csv(OUT/'summary/primary_scores.csv',float_precision='round_trip')
    for row in summary:
        val=primary[(primary['case']=='NB_P24')&(primary.horizon==row['horizon'])].sse_reduction.iloc[0]
        np.testing.assert_allclose(row['OSI_sse_gain'],val,atol=1e-12,rtol=0)
    # Fixed, outcome-selected diagnostic county; no changes to training or scoring population.
    clay=saved[saved.fipsCode=='54015'].copy()
    cols=KEYS+[c for c in clay if c.startswith(('true_','pred_B0_v18_rule','pred_NB_P24_v18_rule','error_B0_','error_NB_P24_'))]
    write_frame(OUT/'summary/Clay_54015_predictions.parquet',clay[cols])
    source=read_json(OUT/'reference/initial_input_hashes.json')
    assert all(sha256_file(p)==h for p,h in source.items())
    write_json(OUT/'analysis_verification.json',dict(status='PASS',split_seed=SEED,decomposition_closes=True,
        source_inputs_unchanged=len(source),code_sha256=sha256_file(Path(__file__)),training_performed=False))
    print(f'split{SEED}: confirmation summaries and Clay trace saved',flush=True)


if __name__=='__main__':main()
