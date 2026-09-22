"""No fitting: compose the four frozen final-OSI routes and evaluate all CVs."""
import argparse
import sys
sys.dont_write_bytecode=True
from integration_protocol import *


def boundary_checks():
    v=np.array([True,True,True,True,False]);a=np.array([0.,.001,.002,.65,np.nan]);b=np.array([0.,.001,0.,.65,np.nan])
    T={h:a.copy() for h in HS};S={24:b.copy(),48:b.copy()};valid={h:v for h in HS}
    out,_=compose(T,S,valid)
    np.testing.assert_array_equal(out['I3'][24],np.array([0.,.001,.001,.65,np.nan]))
    same,_=compose(T,{24:a.copy(),48:a.copy()},valid);np.testing.assert_array_equal(same['I3'][24],a)
    b2=b.copy();b2[1]=0.;out,_=compose(T,{24:b2,48:b},valid);assert out['I3'][24][1]==0
    for bad in (np.nan,np.inf):
        broken=b.copy();broken[2]=bad
        try:compose(T,{24:broken,48:b},valid)
        except ValueError:pass
        else:raise AssertionError('Missing or infinite valid source accepted')
    broken=b.copy();broken[-1]=0.
    try:compose(T,{24:broken,48:b},valid)
    except ValueError:pass
    else:raise AssertionError('Invalid tail was not rejected')
    return dict(status='PASS',equal_source=True,zero_and_threshold_boundary=True,missing_valid_source_rejected=True,invalid_tail_rejected=True)


def evaluate(ctx):
    if (ctx.run/'EVALUATION_COMPLETE').exists():raise RuntimeError('Completed run cannot be overwritten')
    cases,average=compose(ctx.T,ctx.S,ctx.valid)
    pred=ctx.source.copy();lineage=[];primary=[];comparison_rows=[];boot=[];county=[];fold=[];window=[];day=[];severity=[];gate_rows=[];changes=[]
    for case in CASES:
        for h in HS:
            p=cases[case][h];mask=ctx.valid[h];y=ctx.y[h];pred[f'pred_{case}_{tag(h)}']=p
            if h<=6 or case=='B0':np.testing.assert_array_equal(p,ctx.T[h])
            if case=='I1' and h==24:np.testing.assert_array_equal(p,ctx.T[24])
            if case=='I2' and h==24:np.testing.assert_array_equal(p,ctx.S[24])
            if case!='B0' and h==48:np.testing.assert_array_equal(p,ctx.S[48])
            source='T' if h<=6 or case=='B0' or (case=='I1' and h==24) else ('T_and_S' if case=='I3' and h==24 else 'S')
            lineage.append(dict(candidate=case,horizon=h,source=source,T_weight=.5 if source=='T_and_S' else float(source=='T'),
                S_weight=.5 if source=='T_and_S' else float(source=='S'),apply_postprocess=source=='T_and_S',
                T_identity=T_IDS[ctx.seed],S_identity=S_IDS[ctx.seed]))
            primary.append(dict(cv_seed=ctx.seed,candidate=case,horizon=h,**metric(y[mask],p[mask])))
    for candidate,reference in PAIRS:
        pair=f'{candidate}_minus_{reference}'
        for h in HS:
            valid=ctx.valid[h];yy=ctx.y[h];b=cases[reference][h];p=cases[candidate][h]
            base=dict(cv_seed=ctx.seed,pair=pair,candidate=candidate,reference=reference,horizon=h)
            stats=compare(yy[valid],b[valid],p[valid]);comparison_rows.append(dict(**base,**stats))
            boot.append(dict(**base,**stats,**paired_bootstrap(ctx.meta.loc[valid],yy[valid],b[valid],p[valid])))
            d=ctx.meta.loc[valid].copy();d['pair']=pair;d['candidate']=candidate;d['reference']=reference;d['horizon']=h
            d['target_hour']=d.hour_idx+h;d['truth']=yy[valid];d['baseline_prediction']=b[valid];d['candidate_prediction']=p[valid]
            d['sse_reduction']=(b[valid]-yy[valid])**2-(p[valid]-yy[valid])**2
            d['original_stella_gate']=ctx.source['S_gate_'+tag(h)].to_numpy()[valid] if h>=24 else False
            changes.append(d)
            for f,g in ctx.meta.groupby('fipsCode',sort=True):
                mask=valid&(ctx.meta.fipsCode.to_numpy()==f)
                county.append(dict(**base,fipsCode=f,countyName=g.countyName.iloc[0],stateAbbr=g.stateAbbr.iloc[0],fold=int(g.fold.iloc[0]),**compare(yy[mask],b[mask],p[mask])))
            for f in range(5):
                mask=valid&(ctx.meta.fold.to_numpy()==f);fold.append(dict(**base,fold=f,**compare(yy[mask],b[mask],p[mask])))
            target=ctx.meta.hour_idx.to_numpy()+h
            for lo,hi,common in [(96,119,False),(120,143,False),(144,215,False),(120,215,True)]:
                mask=valid&(target>=lo)&(target<=hi);window.append(dict(**base,start=lo,end=hi,common_target=common,accounting_only=False,**compare(yy[mask],b[mask],p[mask])))
            # Short-horizon rows before 96 are immutable; account for them so the
            # nonoverlapping window totals still reproduce each full score.
            mask=valid&(target<96)
            window.append(dict(**base,start=72+h,end=95,common_target=False,accounting_only=True,**compare(yy[mask],b[mask],p[mask])))
            dates=(ctx.meta.timestamp_et+pd.to_timedelta(h,unit='h')).dt.strftime('%Y-%m-%d').to_numpy()
            for date in sorted(set(dates[valid])):
                mask=valid&(dates==date);day.append(dict(**base,target_day=date,**compare(yy[mask],b[mask],p[mask])))
            for group,take in [('zero',yy==0),('0_to_.01',(yy>0)&(yy<=.01)),('.01_to_.05',(yy>.01)&(yy<=.05)),('above_.05',yy>.05)]:
                mask=valid&take;severity.append(dict(**base,severity=group,**compare(yy[mask],b[mask],p[mask])))
            if h==24:
                gate=ctx.source['S_gate_'+tag(24)].to_numpy(bool)
                for level in (False,True):
                    mask=valid&(gate==level);e0=b[mask]-yy[mask];e1=p[mask]-yy[mask]
                    gate_rows.append(dict(**base,original_gate=level,**compare(yy[mask],b[mask],p[mask]),
                        baseline_underprediction_sse=float(np.sum(e0[e0<0]**2)),candidate_underprediction_sse=float(np.sum(e1[e1<0]**2)),
                        baseline_underprediction_rows=int((e0<0).sum()),candidate_underprediction_rows=int((e1<0).sum())))
    # Exact error decomposition for the new 24h fixed mixture, before and after H.
    mask=ctx.valid[24];y=ctx.y[24];a=ctx.T[24];b=ctx.S[24];final=cases['I3'][24]
    audit=ctx.meta.copy();audit['horizon']=24;audit['target_hour']=ctx.meta.hour_idx+24;audit['scoreable']=mask
    audit['truth']=y;audit['T24']=a;audit['S24']=b;audit['mean_before_post']=average;audit['I3_after_post']=final
    audit['original_gate']=ctx.source['S_gate_'+tag(24)];audit['post_changed']=mask&(average!=final)
    audit['mean_squared_error']=(average-y)**2
    audit['mean_mse_identity_rhs']=.5*(a-y)**2+.5*(b-y)**2-.25*(a-b)**2
    audit['postprocessing_sse_increase']=(final-y)**2-(average-y)**2
    np.testing.assert_allclose(audit.loc[mask,'mean_squared_error'],audit.loc[mask,'mean_mse_identity_rhs'],atol=1e-12,rtol=0)
    complement=[]
    for label,take in [('all',mask),('gate',mask&ctx.source['S_gate_'+tag(24)].to_numpy(bool)),('not_gate',mask&~ctx.source['S_gate_'+tag(24)].to_numpy(bool))]:
        eT=a[take]-y[take];eS=b[take]-y[take];l0=ctx.source['S_L0_'+tag(24)].to_numpy()[take]
        def corr(x,z):return float(np.corrcoef(x,z)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(z)>0 else np.nan
        before=metric(y[take],average[take]);after=metric(y[take],final[take])
        diversity=.25*float(np.sum((a[take]-b[take])**2))
        np.testing.assert_allclose(before['sse'],.5*np.sum(eT**2)+.5*np.sum(eS**2)-diversity,atol=1e-12,rtol=0)
        complement.append(dict(cv_seed=ctx.seed,group=label,n=int(take.sum()),error_correlation=corr(eT,eS),
            correction_correlation=corr(a[take]-l0,b[take]-l0),T_sse=float(np.sum(eT**2)),S_sse=float(np.sum(eS**2)),
            diversity_sse_reduction=diversity,mean_before_post_sse=before['sse'],I3_sse=after['sse'],
            postprocessing_sse_increase=after['sse']-before['sse'],post_changed_rows=int(np.sum(take&(average!=final)))))
    write_frame(ctx.run/'source_predictions.parquet',ctx.source)
    write_frame(ctx.run/'integrated_predictions.parquet',pred)
    write_json(ctx.run/'source_lineage.json',dict(cv_seed=ctx.seed,rows=lineage,source_models_modified=False,new_training=0))
    write_frame(ctx.run/'row_changes.parquet',pd.concat(changes,ignore_index=True))
    write_frame(ctx.run/'blend_postprocessing_audit.parquet',audit)
    for name,d in [('primary_scores',primary),('comparisons',comparison_rows),('paired_county_bootstrap',boot),('county_metrics',county),
        ('fold_metrics',fold),('target_window_metrics',window),('target_day_metrics',day),('severity_metrics',severity),
        ('original_gate_metrics',gate_rows),('complementarity',complement)]:write_frame(ctx.run/f'metrics/{name}.csv',d)
    manifest=dict(protocol='tree_stella_final_osi_v1',cv_seed=ctx.seed,T_identity=T_IDS[ctx.seed],S_identity=S_IDS[ctx.seed],
        config_sha256=sha(OUT/'integration_config.json'),input_manifest_sha256=sha(OUT/'reference/initial_input_hashes.json'),
        evaluation_code_sha256={p:sha(OUT/p) for p in CODE_FILES},environment=dict(python=platform.python_version(),
            packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','pyarrow')}),new_fits=0,source_models_reloaded=0)
    write_json(ctx.run/'run_manifest.json',dict(identity=manifest,identity_hash=digest(manifest)))
    write_json(ctx.run/'EVALUATION_COMPLETE',dict(identity_hash=digest(manifest),files={str(p.relative_to(ctx.run)):sha(p)
        for p in sorted(ctx.run.rglob('*')) if p.is_file() and p.name not in ('EVALUATION_COMPLETE','verification.json')}))
    print(pd.DataFrame(comparison_rows).query('horizon == 24')[['cv_seed','pair','rmse_delta','rmse_change_pct']].to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','evaluate'],required=True)
    ap.add_argument('--split-seeds',type=int,nargs='+',default=list(SEEDS));args=ap.parse_args()
    assert tuple(args.split_seeds)==SEEDS,'The frozen plan evaluates all three CVs without adaptation'
    if (OUT/'completion.json').exists():raise RuntimeError('Completed experiment is immutable')
    initialize();checks=boundary_checks();write_json(OUT/'reference/boundary_checks.json',checks)
    inputs=source_inventory();print(f'{len(inputs)} source files verified',flush=True)
    for seed in SEEDS:
        ctx=load_context(seed)
        write_frame(OUT/f'reference/source_metrics_split{seed}.csv',ctx.metrics)
        write_json(OUT/f'reference/preflight_split{seed}.json',dict(status='PASS',cv_seed=seed,counties=239,rows=34416,
            scoreable={h:int(ctx.valid[h].sum()) for h in HS},source_metrics_reproduced=True,
            saved_gate_matches_thresholds=True,source_models_reloaded=0,training_performed=False))
        print(f'split{seed}: source identities, folds, labels, masks, saved gate and scores PASS',flush=True)
        if args.stage=='evaluate':evaluate(ctx)
    assert all(sha(p)==h for p,h in inputs.items())
    print(f'{args.stage} complete; old inputs unchanged; no fitting',flush=True)


if __name__=='__main__':main()
