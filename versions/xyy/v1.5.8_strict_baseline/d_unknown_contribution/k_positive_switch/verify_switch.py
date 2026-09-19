"""Independent gate, composition, metric and bootstrap reconstruction; no fitting."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from switch_protocol import (HERE,HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,H1,KEYS,load_sources,
    identity_for,output_path,read_json,digest_object,sha256_file,expected_mask,check_prediction_boundaries)
from run_artifacts import check_stage
from independent_numerics import exact,close,brute_bounds,independent_controls,assert_stats


def csv(path):return pd.read_csv(path,dtype={'fipsCode':str},float_precision='round_trip')


def verify_run(run,require_marker=True):
    run=output_path(run);manifest=read_json(run/'manifest.json');seed=manifest['identity']['split_seed']
    ctx=load_sources(seed);assert manifest['identity']==identity_for(ctx)
    identity=manifest['identity_hash'];assert identity==digest_object(manifest['identity'])
    marker=check_stage(run,'G_COMPLETE',identity) if require_marker else None
    gate=pd.read_parquet(run/'gate_decisions.parquet');components=pd.read_parquet(run/'component_predictions.parquet')
    oof=pd.read_parquet(run/'control_predictions.parquet')
    for frame in (gate,components,oof):
        pd.testing.assert_frame_equal(frame[KEYS],ctx.meta)
        assert frame.gated_identity_hash.eq(identity).all()
        assert frame.source_u_identity_hash.eq(ctx.item['u_identity_hash']).all()
        assert frame.baseline_identity_hash.eq(ctx.item['baseline_identity_hash']).all()
    k=brute_bounds(ctx.raw_history,ctx.meta)[1];valid=np.isfinite(k);eligible=valid&(k>0)
    close(gate.K,k);exact(gate.use_U,eligible);exact(gate.scoreable,valid)
    exact(gate.target_timestamp,ctx.meta.timestamp_et+pd.Timedelta(hours=1));exact(gate.pred_U_raw,ctx.raw_u)
    dg=ctx.parts['A'][H1]['D_t'].copy()
    for i in np.flatnonzero(eligible):dg[i]=min(1.,k[i]+max(0.,ctx.raw_u[i]))
    close(gate.G_D1,dg)
    for case in ('A','B','C'):exact(gate[f'{case}_D1'],ctx.parts[case][H1]['D_t'])
    exact(gate.G_D1.to_numpy()[~eligible],ctx.parts['A'][H1]['D_t'][~eligible])
    exact(gate.G_D1.to_numpy()[eligible],ctx.parts['C'][H1]['D_t'][eligible])
    original=ctx.source_part.source_A_D_t_target_t01h.to_numpy();uid=ctx.source_u.source_U_model_id.to_numpy()
    selected=np.where(eligible,uid,original)
    exact(gate.original_D_model_id,original);exact(gate.U_model_id,uid);exact(gate.selected_model_id,selected)
    exact(gate.selected_rule,np.where(valid,np.where(eligible,'U_reconstruction','original_D'),'unscoreable'))
    for case in ('A','B','C'):exact(gate[f'D_changed_vs_{case}'],valid&(gate.G_D1.to_numpy()!=gate[f'{case}_D1'].to_numpy()))
    independent_parts={h:{} for h in (1,6,24,48)}
    for h in HORIZONS:
        for c in COMPONENTS:
            target=f'{c}_target_{h.rsplit("_",1)[-1]}'
            reference=dg if (h,c)==(H1,'D_t') else ctx.parts['A'][h][c]
            close(components[f'pred_G_{target}'],reference)
            independent_parts[HORIZON_HOURS[h]][c]=reference
            source=selected if (h,c)==(H1,'D_t') else ctx.source_part[f'source_A_{target}'].to_numpy()
            exact(components[f'source_G_{target}'],source)
            if (h,c)!=(H1,'D_t'):exact(components[f'pred_G_{target}'],reference)
    direct_frame=pd.DataFrame({f'raw_{h}':ctx.direct[h] for h in HORIZONS})
    controls,buckets,means=independent_controls(ctx.meta,direct_frame,independent_parts)
    pred={};actual={};masks={};county_indices={f:np.flatnonzero(ctx.meta.fipsCode.to_numpy()==f) for f in ctx.meta.fipsCode.unique()}
    folds=ctx.meta.fold.to_numpy();codes=ctx.meta.fipsCode.to_numpy();hours=ctx.meta.hour_idx.to_numpy()
    for h in HORIZONS:
        mask=expected_mask(ctx.meta,h);masks[h]=mask;actual[h]=ctx.data.y_train[h].to_numpy()
        exact(oof[f'actual_{h}'],actual[h]);exact(oof[f'scoreable_{h}'],mask)
        exact(oof[f'raw_direct_{h}'],ctx.direct[h]);exact(oof[f'source_direct_{h}'],ctx.source_control[f'source_direct_{h}'])
        for case in ('A','B','C','G'):
            for mode in MODES:
                value=oof[f'pred_{case}_{mode}_{h}'].to_numpy();pred[case,mode,h]=value
                exact(np.isfinite(value),mask)
                if case!='G':exact(value,ctx.controls[case][mode][h])
                else:
                    close(value,controls[HORIZON_HOURS[h]][mode])
                    if h!=H1 or mode=='C0_direct_osi':exact(value,ctx.controls['B'][mode][h])
                    if h==H1:exact(value[~eligible],ctx.controls['B'][mode][h][~eligible])
    pairs={'G_minus_B':'B','G_minus_A':'A','G_minus_C':'C'}
    for pair,before in pairs.items():
        for mode in MODES:
            for h in HORIZONS:exact(oof[f'changed_{pair}_{mode}_{h}'],masks[h]&(pred['G',mode,h]!=pred[before,mode,h]))
    unique=pd.read_parquet(run/'aligned_unique_components.parquet')
    assert len(unique)==4*len(means) and not unique.duplicated(['component','fipsCode','target_timestamp']).any()
    for row in unique.itertuples(index=False):
        key=(row.fipsCode,int((row.target_timestamp-pd.Timestamp('2026-03-11'))/pd.Timedelta(hours=1)))
        close(row.prediction,means[key][COMPONENTS.index(row.component)])
        assert row.candidate_count==len(buckets[key])
        assert row.source_horizons==','.join(f'osi_target_t{h:02d}h' for h,_ in sorted(buckets[key]))
    tables={name:csv(run/f'metrics/{name}.csv') for name in ('pooled','fold','county','window','bootstrap','D_component','primary_county_influence','D_spaces','D_bootstrap','gate_windows')}
    for name,n in (('pooled',60),('fold',300),('county',14340),('bootstrap',66),('D_component',12),('D_spaces',9),('D_bootstrap',18),('gate_windows',9)):
        assert len(tables[name])==n,(name,len(tables[name]))
    for name in ('pooled','fold','county','window','primary_county_influence'):
        for _,row in tables[name].iterrows():
            h,mode=row.horizon,row.model;b=pred[pairs[row.comparison],mode,h];a=pred['G',mode,h];y=actual[h]
            if name in ('county','primary_county_influence'):
                idx=county_indices[row.fipsCode];idx=idx[masks[h][idx]]
            else:
                take=masks[h].copy()
                if name=='fold':take &= folds==row.fold
                if name=='window':take &= {'first_four_target_hours':hours<=75,'later_target_hours':hours>75,'changed_predictions':a!=b}[row.window]
                idx=np.flatnonzero(take)
            assert_stats(row,y[idx],b[idx],a[idx])
    for _,row in tables['gate_windows'].iterrows():
        before=pairs[row.comparison];b=pred[before,'v18_rule',H1];a=pred['G','v18_rule',H1]
        take={'K_positive':eligible,'early_K_zero':valid&(hours<=75)&~eligible,'later_K_zero':valid&(hours>75)}[row.window]
        assert_stats(row,actual[H1][take],b[take],a[take])
    aligned_g={}
    saved_d_lookup=unique.loc[unique.component=='D_t'].set_index(['fipsCode','target_timestamp']).prediction
    for h in HORIZONS:
        values=np.array([means.get((f,int(t)+HORIZON_HOURS[h]),[np.nan]*4)[2] for f,t in zip(codes,hours)])
        values[~masks[h]]=np.nan
        keys=pd.MultiIndex.from_arrays([ctx.meta.fipsCode,ctx.meta.timestamp_et+pd.Timedelta(hours=HORIZON_HOURS[h])])
        saved=saved_d_lookup.reindex(keys).to_numpy(copy=True);saved[~masks[h]]=np.nan
        close(saved,values)
        if h!=H1:exact(saved,ctx.aux['B']['aligned_components'][h]['D_t'])
        aligned_g[h]=saved
    def d_arrays(row):
        h=row.horizon;before=pairs[row.comparison]
        if 'space' not in row or row.space=='raw_D1':
            b=ctx.parts[before][h]['D_t'];a=components[f'pred_G_D_t_target_{h.rsplit("_",1)[-1]}'].to_numpy()
        else:
            b=ctx.aux[before]['aligned_components'][h]['D_t'];a=aligned_g[h]
        y=ctx.data.component_targets[f'D_t_target_{h.rsplit("_",1)[-1]}'].to_numpy()
        return y,b,a
    for name in ('D_component','D_spaces'):
        for _,row in tables[name].iterrows():
            y,b,a=d_arrays(row);take=np.isfinite(y);assert_stats(row,y[take],b[take],a[take])
    # Independent group-wise bootstrap, sequential draws; no evaluator bootstrap helper.
    for name in ('bootstrap','D_bootstrap'):
        for _,row in tables[name].iterrows():
            if name=='D_bootstrap':y,b,a=d_arrays(row)
            else:y,b,a=actual[row.horizon],pred[pairs[row.comparison],row.model,row.horizon],pred['G',row.model,row.horizon]
            take=np.isfinite(y)
            if row.population=='exclude_Morrow_diagnostic_only':take &= codes!='39117'
            assert_stats(row,y[take],b[take],a[take])
            frame=pd.DataFrame({'county':codes[take],'b':(b[take]-y[take])**2,'a':(a[take]-y[take])**2})
            grouped=frame.groupby('county',sort=True).agg(b=('b','sum'),a=('a','sum'),n=('b','size'))
            values=grouped.to_numpy();assert len(values)==row.counties
            rng=np.random.default_rng(int(row.seed));delta=[]
            for _ in range(int(row.replicates)):
                bs,ass,ns=values[rng.integers(0,len(values),len(values))].sum(axis=0)
                delta.append(np.sqrt(ass/ns)-np.sqrt(bs/ns))
            close([row.ci_low,row.ci_high],np.quantile(delta,[.025,.975]))
            close(row.bootstrap_fraction_delta_lt0,np.mean(np.asarray(delta)<0));close(row.bootstrap_fraction_delta_eq0,np.mean(np.asarray(delta)==0))
    county_groups=tables['county'].groupby(['comparison','horizon','model'])
    for _,row in tables['pooled'].iterrows():
        c=county_groups.get_group((row.comparison,row.horizon,row.model));assert len(c)==239 and c.n.sum()==row.n
        close(c.baseline_sse.sum(),row.baseline_sse);close(c.candidate_sse.sum(),row.candidate_sse)
    influence=tables['primary_county_influence']
    close(influence.fraction_of_net_sse_reduction,influence.sse_reduction/influence.groupby('horizon').sse_reduction.transform('sum'))
    eligible_saved=csv(run/'eligible_rows.csv')
    expected=gate.loc[eligible].reset_index(drop=True)
    actual_gate=eligible_saved[expected.columns].copy()
    for name in ('timestamp_et','target_timestamp'):actual_gate[name]=pd.to_datetime(actual_gate[name])
    pd.testing.assert_frame_equal(actual_gate,expected,check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
    close(eligible_saved.actual_D,ctx.data.component_targets.D_t_target_t01h.to_numpy()[eligible])
    close(eligible_saved.actual_osi,actual[H1][eligible])
    for case in ('A','B','C','G'):exact(eligible_saved[f'{case}_osi'],pred[case,'v18_rule',H1][eligible])
    close(eligible_saved.sse_reduction_G_vs_B,(eligible_saved.B_osi-eligible_saved.actual_osi)**2-(eligible_saved.G_osi-eligible_saved.actual_osi)**2)
    stats=read_json(run/'routing_stats.json');changed=valid&(pred['G','v18_rule',H1]!=pred['B','v18_rule',H1])
    assert stats['eligible_rows']==int(eligible.sum())==784
    assert stats['eligible_counties']==ctx.meta.loc[eligible,'fipsCode'].nunique()==207
    assert stats['D_changed_vs_B']==int(gate.D_changed_vs_B.sum())
    assert stats['OSI_changed_vs_B']==int(changed.sum()) and stats['OSI_changed_counties']==len(set(codes[changed]))
    assert check_prediction_boundaries(ctx)['status']=='PASS'
    for path,digest in manifest['identity']['source_sha256'].items():assert sha256_file(path)==digest,path
    if require_marker:
        check_stage(run,'G_COMPLETE',identity)
        files={str(f.relative_to(run)) for f in run.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.parquet') and 'logs' not in f.relative_to(run).parts}
        assert set(marker['files'])==files
    return {'status':'PASS','split_seed':seed,'identity_hash':identity,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'verifier_sha256':sha256_file(__file__),'absolute_tolerance':1e-12,'independent_K_and_gate_reconstruction':True,
        'independent_G_controls_and_D_alignment':True,'all_metrics_and_bootstrap_recomputed':True,
        'all_6_24_48h_controls_exactly_equal_A_B':True,'off_gate_G_exactly_equal_A_B':True,
        'eligible_gate_uses_no_labels_or_future_P':True,'source_lineage_and_hashes':'PASS','training_performed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-seed',type=int,required=True,choices=(42,20260917,20260918))
    args=parser.parse_args();print(json.dumps(verify_run(HERE/'runs'/f'split{args.split_seed}'),indent=2))


if __name__=='__main__':main()
