"""Per-CV summaries and exact P-versus-OSI error-change accounting."""
from transfer_protocol import *


def main():
    assert read_json(RUN/'logs/independent_verification.json')['status']=='PASS'
    ctx=load_context()
    for name in ('primary_scores','P_component_metrics','P71_bins','county_metrics','fold_metrics','window_metrics',
                 'bootstrap_intervals','branch_diagnostics','branch_loss_decomposition'):
        df=pd.read_csv(RUN/f'metrics/{name}.csv',float_precision='round_trip',dtype={'fipsCode':str} if name=='county_metrics' else {})
        df.insert(0,'split_seed',SEED);write_frame(OUT/f'summary/{name}.csv',df)
    pp=pd.read_parquet(RUN/'P_component_diagnostics.parquet')
    pred=normalize(pd.read_parquet(RUN/'control_predictions.parquet'));rows=[];summary=[]
    w=np.array([.4,.35,.25,-.1])
    for h in HORIZONS:
        hh=HORIZON_HOURS[h];valid=expected_mask(ctx.meta,h)
        candidate=normalize(pp[(pp['case']=='AB_P06')&(pp.horizon==h)])
        actual=np.column_stack([ctx.data.component_targets[component_target_name(c,h)].to_numpy() for c in COMPONENTS])
        old=ctx.parts[h] if hh>=24 else ctx.aux['aligned_components'][h]
        old=np.column_stack([old[c] for c in COMPONENTS]);new=old.copy()
        new[:,0]=candidate['component_clipped' if hh>=24 else 'aligned'].to_numpy()
        y=ctx.data.y_train[h].to_numpy();p0=pred[f'pred_B0_v18_rule_{h}'].to_numpy();p1=pred[f'pred_AB_P06_v18_rule_{h}'].to_numpy()
        e0=.4*(old[:,0]-actual[:,0]);e1=.4*(new[:,0]-actual[:,0]);other=(old[:,1:]-actual[:,1:])@w[1:]
        b0=p0-old@w+actual@w-y;b1=p1-new@w+actual@w-y
        self_gain=e0**2-e1**2;cross=2*(e0-e1)*other
        remainder=b0**2-b1**2+2*((e0+other)*b0-(e1+other)*b1)
        gain=(p0-y)**2-(p1-y)**2
        np.testing.assert_allclose(e0+other+b0,p0-y,atol=1e-12,rtol=0)
        np.testing.assert_allclose(e1+other+b1,p1-y,atol=1e-12,rtol=0)
        np.testing.assert_allclose(self_gain+cross+remainder,gain,atol=1e-12,rtol=0)
        d=ctx.meta.loc[valid].copy();d['split_seed']=SEED;d['case']='AB_P06';d['horizon']=h
        for name,a in [('P_self_sse_gain',self_gain),('P_other_cross_sse_gain',cross),('postprocessing_and_precision_sse_gain',remainder),('actual_OSI_sse_gain',gain)]:d[name]=a[valid]
        rows.append(d)
        summary.append(dict(split_seed=SEED,case='AB_P06',horizon=h,n=int(valid.sum()),
            P_self_sse_gain=float(np.nansum(self_gain)),P_other_cross_sse_gain=float(np.nansum(cross)),
            postprocessing_and_precision_sse_gain=float(np.nansum(remainder)),actual_OSI_sse_gain=float(np.nansum(gain))))
    write_frame(OUT/'summary/OSI_change_decomposition.csv',summary)
    write_frame(OUT/'summary/OSI_change_decomposition.parquet',pd.concat(rows,ignore_index=True))
    primary=pd.read_csv(OUT/'summary/primary_scores.csv',float_precision='round_trip')
    for row in summary:
        expected=primary[(primary['case']=='AB_P06')&(primary.horizon==row['horizon'])].sse_reduction.iloc[0]
        np.testing.assert_allclose(row['actual_OSI_sse_gain'],expected,atol=1e-12,rtol=0)
    sources=read_json(OUT/'reference/initial_input_hashes.json')
    for path,digest in sources.items():assert sha256_file(path)==digest,path
    write_json(OUT/'analysis_verification.json',dict(status='PASS',split_seed=SEED,decomposition_closes=True,
        source_inputs_unchanged=len(sources),training_performed=False,code_sha256=sha256_file(Path(__file__))))
    print(f'split{SEED}: summaries and component/OSI accounting saved',flush=True)


if __name__=='__main__':main()
