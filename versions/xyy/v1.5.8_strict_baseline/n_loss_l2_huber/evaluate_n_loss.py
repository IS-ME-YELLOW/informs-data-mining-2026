"""Frozen single-source and joint comparisons on full county OOF trajectories."""
from loss_protocol import *


def metrics(y,p):
    y,p=np.asarray(y),np.asarray(p);assert y.shape==p.shape
    valid=np.isfinite(y);assert np.isfinite(p[valid]).all()
    y,p=y[valid],p[valid]
    if not len(y):return dict(n=0,rmse=np.nan,mae=np.nan,sse=0.,bias=np.nan)
    e=p-y
    return dict(n=len(y),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),sse=float(np.sum(e**2)),bias=float(e.mean()))


def comparison(y,b,p):
    before,after=metrics(y,b),metrics(y,p)
    return dict(**after,baseline_rmse=before['rmse'],rmse_delta=after['rmse']-before['rmse'],
        rmse_change_pct=100*(after['rmse']/before['rmse']-1) if before['rmse'] else np.nan,
        sse_reduction=before['sse']-after['sse'])


def bootstrap(meta,y,b,p):
    county=meta.fipsCode.unique();codes=meta.fipsCode.map({f:i for i,f in enumerate(county)}).to_numpy(int)
    valid=np.isfinite(y);assert np.isfinite(b[valid]).all() and np.isfinite(p[valid]).all()
    n=np.bincount(codes[valid],minlength=len(county))
    bs=np.bincount(codes[valid],weights=(b[valid]-y[valid])**2,minlength=len(county))
    ps=np.bincount(codes[valid],weights=(p[valid]-y[valid])**2,minlength=len(county))
    draws=np.random.default_rng(20260910).integers(0,len(county),(2000,len(county)))
    diff=np.sqrt(ps[draws].sum(axis=1)/n[draws].sum(axis=1))-np.sqrt(bs[draws].sum(axis=1)/n[draws].sum(axis=1))
    lo,hi=np.quantile(diff,[.025,.975])
    return dict(ci_low=float(lo),ci_high=float(hi),bootstrap_replicates=2000,bootstrap_seed=20260910,unit='county_full_trajectory')


def evaluate():
    if (RUN/'EVALUATION_COMPLETE').exists():raise RuntimeError('Completed evaluation is immutable')
    ctx=initialize(load_context());run=read_json(RUN/'run_manifest.json')
    assert run['identity_hash']==ctx.identity_hash
    marker=read_json(RUN/'MODEL_CV_COMPLETE');assert marker['identity_hash']==ctx.identity_hash
    for path,digest in marker['files'].items():assert sha256_file(RUN/path)==digest,path
    fresh=normalize(pd.read_parquet(RUN/'n_l2_oof_predictions.parquet'))
    pd.testing.assert_frame_equal(fresh[KEYS],ctx.meta[KEYS],check_dtype=False)
    new={h:fresh['raw_'+component_target_name('N_t',h)].to_numpy() for h in HORIZONS}
    pp,cc,aa=construct_cases(ctx,new)
    part={k:ctx.meta[k].to_numpy() for k in ctx.meta};pred=dict(part);aligned=[];nrows=[]
    part['candidate_identity_hash']=ctx.identity_hash;pred['candidate_identity_hash']=ctx.identity_hash
    allmetrics=[];primary=[];componentmetrics=[];bins=[];county=[];fold=[];window=[];boots=[];invariants=[]
    for case in CASES:
        a=aa[case]['unique_components'].copy();a.insert(0,'case',case);a['candidate_identity_hash']=ctx.identity_hash;aligned.append(a)
        for h in HORIZONS:
            hh=HORIZON_HOURS[h];nt=component_target_name('N_t',h);yn=ctx.data.component_targets[nt].to_numpy()
            y=ctx.data.y_train[h].to_numpy();valid=expected_mask(ctx.meta,h);s=ctx.meta.hour_idx.to_numpy()+hh
            changes=case=='L2_all' or case==f'L2_N{hh:02d}'
            for c in COMPONENTS:
                tag=component_target_name(c,h);v=pp[case][h][c]
                part[f'pred_{case}_{tag}']=v
                part[f'source_{case}_{tag}']=fresh['source_'+tag].to_numpy() if c=='N_t' and changes else ctx.reference_parts['source_F2_'+tag].to_numpy()
                if c!='N_t' or not changes:
                    np.testing.assert_array_equal(v,ctx.parts[h][c]);invariants.append(dict(case=case,horizon=h,component=c,unchanged=True))
            for mode in cc[case]:
                p=cc[case][mode][h];base=cc['H0_Huber'][mode][h]
                pred[f'pred_{case}_{mode}_{h}']=p
                allmetrics.append(dict(case=case,horizon=h,mode=mode,**comparison(y,base,p)))
                if mode=='C0_direct_osi' or (case=='H0_Huber') or (mode in ('C1_component_osi','C2_equal_blend') and not changes) or (mode=='v18_rule' and hh>=24 and not changes):
                    np.testing.assert_array_equal(p,base)
                if mode!='v18_rule':continue
                pred[f'error_{case}_{h}']=p-y;pred[f'true_{h}']=y;pred[f'scoreable_{h}']=valid
                row=dict(case=case,horizon=h,**comparison(y,base,p));primary.append(row)
                if case!='H0_Huber':boots.append(dict(domain='OSI',space='v18_rule',**row,**bootstrap(ctx.meta,y,base,p)))
                for f in ctx.meta.fipsCode.unique():
                    hit=ctx.meta.fipsCode.to_numpy()==f
                    county.append(dict(case=case,horizon=h,fipsCode=f,fold=int(ctx.meta.loc[hit,'fold'].iloc[0]),**comparison(y[hit],base[hit],p[hit])))
                for f in range(5):
                    hit=ctx.meta.fold.to_numpy()==f;fold.append(dict(case=case,horizon=h,fold=f,**comparison(y[hit],base[hit],p[hit])))
                for lo,hi in [(73,76),(77,95),(96,143),(144,215),(120,215)]:
                    hit=(s>=lo)&(s<=hi);window.append(dict(case=case,horizon=h,start=lo,end=hi,common_target=lo==120,
                        **comparison(y[hit],base[hit],p[hit])))
            base_n={'model_raw':ctx.nraw[h],'component_clipped':ctx.parts[h]['N_t'],'aligned':ctx.aux['aligned_components'][h]['N_t']}
            candidate_n={'model_raw':new[h] if changes else ctx.nraw[h],
                'component_clipped':pp[case][h]['N_t'],'aligned':aa[case]['aligned_components'][h]['N_t']}
            nr=ctx.meta.copy();nr['case']=case;nr['horizon']=h;nr['target_hour']=s;nr['scoreable']=valid
            nr['true_N']=yn;nr['source_N_model_id']=part[f'source_{case}_{nt}'];nr['candidate_identity_hash']=ctx.identity_hash
            for space,p in candidate_n.items():
                b=base_n[space];nr['N_'+space]=p
                for lo,hi,group in [(72+hh,215,'full'),(120,215,'common_120_215')]:
                    hit=(s>=lo)&(s<=hi)
                    componentmetrics.append(dict(case=case,horizon=h,space=space,group=group,**comparison(yn[hit],b[hit],p[hit]),
                        negative_fraction=float(np.mean(p[hit]<0)),above_one_fraction=float(np.mean(p[hit]>1))))
                if case=='L2_all':boots.append(dict(domain='N',space=space,case=case,horizon=h,
                    **comparison(yn,b,p),**bootstrap(ctx.meta,yn,b,p)))
                groups=[('N=0',yn==0),('0<N<=1e-4',(yn>0)&(yn<=1e-4)),('1e-4<N<=1e-3',(yn>1e-4)&(yn<=1e-3)),
                    ('1e-3<N<=1e-2',(yn>1e-3)&(yn<=1e-2)),('N>1e-2',yn>1e-2)]
                for group,hit in groups:
                    hit=hit&valid
                    bins.append(dict(case=case,horizon=h,space=space,group=group,**comparison(yn[hit],b[hit],p[hit]),
                        true_mean=float(np.mean(yn[hit])) if hit.any() else np.nan,
                        baseline_mean=float(np.mean(b[hit])) if hit.any() else np.nan,
                        prediction_mean=float(np.mean(p[hit])) if hit.any() else np.nan))
            nrows.append(nr)
    write_frame(RUN/'component_predictions.parquet',part)
    write_frame(RUN/'control_predictions.parquet',pred)
    write_frame(RUN/'n_component_diagnostics.parquet',pd.concat(nrows,ignore_index=True))
    write_frame(RUN/'aligned_unique_components.parquet',pd.concat(aligned,ignore_index=True))
    for name,values in [('all_control_metrics',allmetrics),('primary_scores',primary),('N_component_metrics',componentmetrics),
        ('N_bins',bins),('county_metrics',county),('fold_metrics',fold),('window_metrics',window),('bootstrap_intervals',boots)]:
        write_frame(RUN/f'metrics/{name}.csv',values)
    write_json(RUN/'invariants.json',dict(status='PASS',checks=invariants,P_D_R_unchanged=True,direct_OSI_unchanged=True))
    assert_inputs(ctx)
    before=set(marker['files'])|{'MODEL_CV_COMPLETE','EVALUATION_COMPLETE','VERIFIED_COMPLETE'}
    files={str(p.relative_to(RUN)):sha256_file(p) for p in sorted(RUN.rglob('*')) if p.is_file() and str(p.relative_to(RUN)) not in before}
    write_json(RUN/'EVALUATION_COMPLETE',dict(identity_hash=ctx.identity_hash,files=files,
        evaluation_code_sha256=sha256_file(Path(__file__)),primary_case='L2_all',case_count=6))
    print(pd.DataFrame(primary).query('case == "L2_all"')[['horizon','baseline_rmse','rmse','rmse_change_pct']].to_string(index=False),flush=True)


if __name__=='__main__':evaluate()
