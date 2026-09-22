"""Three predeclared single-source P transfers; no joint or adaptive blending."""
from neighbor_protocol import *
TRANSFER=CHANGED


def metric(y,p):
    y,p=np.asarray(y),np.asarray(p);mask=np.isfinite(y)
    assert np.isfinite(p[mask]).all()
    e=p[mask]-y[mask]
    if not len(e):return dict(n=0,rmse=np.nan,mae=np.nan,sse=0.,bias=np.nan)
    return dict(n=len(e),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),sse=float(np.sum(e**2)),bias=float(np.mean(e)))


def comparison(y,b,p):
    bm,pm=metric(y,b),metric(y,p)
    return dict(**pm,baseline_rmse=bm['rmse'],rmse_delta=pm['rmse']-bm['rmse'],
        rmse_change_pct=100*(pm['rmse']/bm['rmse']-1) if bm['rmse'] else np.nan,sse_reduction=bm['sse']-pm['sse'])


def bootstrap(meta,y,b,p):
    counties=meta.fipsCode.unique();codes=meta.fipsCode.map({f:i for i,f in enumerate(counties)}).to_numpy(int)
    good=np.isfinite(y);assert np.isfinite(b[good]).all() and np.isfinite(p[good]).all()
    n=np.bincount(codes[good],minlength=len(counties))
    sb=np.bincount(codes[good],weights=(b[good]-y[good])**2,minlength=len(counties))
    sp=np.bincount(codes[good],weights=(p[good]-y[good])**2,minlength=len(counties))
    draws=np.random.default_rng(20260910).integers(0,len(counties),(2000,len(counties)))
    diff=np.sqrt(sp[draws].sum(axis=1)/n[draws].sum(axis=1))-np.sqrt(sb[draws].sum(axis=1)/n[draws].sum(axis=1))
    low,high=np.quantile(diff,[.025,.975])
    return dict(ci_low=float(low),ci_high=float(high),replicates=2000,bootstrap_seed=20260910,unit='county_full_trajectory')


def evaluate():
    if (RUN/'EVALUATION_COMPLETE').exists():raise RuntimeError('Completed evaluation is immutable')
    ctx=attach_features(initialize(load_context()));m=read_json(RUN/'MODEL_CV_COMPLETE');assert m['identity_hash']==ctx.identity_hash
    for f,h in m['files'].items():assert sha256_file(RUN/f)==h,f
    branch=pd.read_parquet(RUN/'new_P_oof_predictions.parquet')
    byh={h:normalize(branch.loc[branch.horizon==h]) for h in TRANSFER}
    structured={h:d.pred_P.to_numpy() for h,d in byh.items()}
    for h,d in byh.items():
        pd.testing.assert_frame_equal(d[KEYS],ctx.meta[KEYS],check_dtype=False)
        close=(reconstruct(ctx.p,d.raw_a.to_numpy(),d.raw_b.to_numpy(),expected_mask(ctx.meta,h))[0] if HORIZON_HOURS[h]==6 else clip_component(d.raw_direct.to_numpy()))
        np.testing.assert_array_equal(close,structured[h])
    pp,cc,aa=build_cases(ctx,structured)
    comp={k:ctx.meta[k].to_numpy() for k in ctx.meta};pred=dict(comp)
    comp['candidate_identity_hash']=ctx.identity_hash;pred['candidate_identity_hash']=ctx.identity_hash
    comp['reference_current_identity_hash']=REFERENCE_ID;pred['reference_current_identity_hash']=REFERENCE_ID
    aligned=[];pframes=[];primary=[];allm=[];pm=[];boots=[];county=[];folds=[];windows=[];pbins=[];invariants=[]
    for case,changed in CASES.items():
        a=aa[case]['unique_components'].copy();a.insert(0,'case',case);a['candidate_identity_hash']=ctx.identity_hash;aligned.append(a)
        for h in HORIZONS:
            hh=HORIZON_HOURS[h];valid=expected_mask(ctx.meta,h);s=ctx.meta.hour_idx.to_numpy()+hh
            for c in COMPONENTS:
                tag=component_target_name(c,h);value=pp[case][h][c];comp[f'pred_{case}_{tag}']=value
                if h==changed and c=='P_t':comp[f'source_{case}_{tag}']=byh[h].P_source_id.to_numpy()
                else:
                    comp[f'source_{case}_{tag}']=ctx.reference_parts['source_AB_P06_'+tag].to_numpy()
                    np.testing.assert_array_equal(value,ctx.parts[h][c]);invariants.append(dict(case=case,horizon=h,component=c,unchanged=True))
            y=ctx.data.y_train[h].to_numpy();pred['true_'+h]=y;pred['scoreable_'+h]=valid
            for mode in cc[case]:
                p=cc[case][mode][h];b=cc['B0'][mode][h]
                pred[f'pred_{case}_{mode}_{h}']=p;allm.append(dict(case=case,horizon=h,mode=mode,**comparison(y,b,p)))
                if case=='B0' or mode=='C0_direct_osi' or (h!=changed and mode in ('C1_component_osi','C2_equal_blend')) or (mode=='v18_rule' and hh>=24 and h!=changed):
                    np.testing.assert_array_equal(p,b)
                if changed:
                    unaffected=valid&(s<72+HORIZON_HOURS[changed]);np.testing.assert_array_equal(p[unaffected],b[unaffected])
                if mode!='v18_rule':continue
                pred[f'error_{case}_{h}']=p-y;pred[f'sse_reduction_{case}_{h}']=(b-y)**2-(p-y)**2
                row=dict(case=case,horizon=h,**comparison(y,b,p));primary.append(row)
                if case!='B0':boots.append(dict(domain='OSI',space='v18_rule',**row,**bootstrap(ctx.meta,y,b,p)))
                for f in ctx.meta.fipsCode.unique():
                    hit=ctx.meta.fipsCode.to_numpy()==f
                    county.append(dict(case=case,horizon=h,fipsCode=f,countyName=ctx.names.loc[f,'countyName'],
                        stateAbbr=ctx.names.loc[f,'stateAbbr'],fold=int(ctx.meta.loc[hit,'fold'].iloc[0]),**comparison(y[hit],b[hit],p[hit])))
                for f in range(5):
                    hit=ctx.meta.fold.to_numpy()==f;folds.append(dict(case=case,horizon=h,fold=f,**comparison(y[hit],b[hit],p[hit])))
                for lo,hi in [(73,76),(77,95),(96,143),(144,215),(120,215)]:
                    hit=(s>=lo)&(s<=hi);windows.append(dict(case=case,horizon=h,start=lo,end=hi,common_target=lo==120,
                        **comparison(y[hit],b[hit],p[hit])))
            yp=ctx.data.component_targets[component_target_name('P_t',h)].to_numpy()
            f=ctx.meta.copy();f['case']=case;f['horizon']=h;f['target_hour']=s;f['P71']=ctx.p;f['true_P']=yp;f['scoreable']=valid
            f['candidate_identity_hash']=ctx.identity_hash
            for space,values,baseline in [('component_clipped',pp[case][h]['P_t'],ctx.parts[h]['P_t']),
                ('aligned',aa[case]['aligned_components'][h]['P_t'],ctx.aux['aligned_components'][h]['P_t'])]:
                f[space]=values
                for lo,hi,group in [(72+hh,215,'full'),(120,215,'common_120_215')]:
                    hit=(s>=lo)&(s<=hi);pm.append(dict(case=case,horizon=h,space=space,group=group,**comparison(yp[hit],baseline[hit],values[hit])))
                if changed==h:boots.append(dict(domain='P',space=space,case=case,horizon=h,
                    **comparison(yp,baseline,values),**bootstrap(ctx.meta,yp,baseline,values)))
                for name,hit in [('p=0',ctx.p==0),('0<p<.01',(ctx.p>0)&(ctx.p<.01)),('.01<=p<.1',(ctx.p>=.01)&(ctx.p<.1)),
                    ('.1<=p<.5',(ctx.p>=.1)&(ctx.p<.5)),('.5<=p<=1',ctx.p>=.5)]:
                    pbins.append(dict(case=case,horizon=h,space=space,P71_group=name,**comparison(yp[hit],baseline[hit],values[hit])))
            pframes.append(f)
    branch_stats=[];decompositions=[];audits=[]
    for h,d in byh.items():
        y=d.true_P.to_numpy();valid=d.scoreable.to_numpy(bool);p=ctx.p
        audit=ctx.meta.copy();audit['horizon']=h;audit['P71']=p;audit['true_P']=y;audit['scoreable']=valid
        for kind in (('a','b') if HORIZON_HOURS[h]==6 else ('direct',)):
            active=valid&support(p,kind);labels=np.full(len(p),np.nan);weights=np.where(valid,0.,np.nan)
            labels[active],weights[active]=transform(y[active],p[active],kind)
            raw=d['raw_'+kind].to_numpy();applied=np.clip(raw,0,1)
            audit['label_'+kind]=labels;audit['weight_'+kind]=weights;audit['support_'+kind]=active
            branch_stats.append(dict(horizon=h,kind=kind,supported_rows=int(active.sum()),all_valid_rows=int(valid.sum()),
                raw_label_rmse=metric(labels,raw)['rmse'],clipped_label_rmse=metric(labels,applied)['rmse'],
                raw_P_contribution_rmse=float(np.sqrt(np.sum(weights[active]*(raw[active]-labels[active])**2)/valid.sum())),
                clipped_P_contribution_rmse=float(np.sqrt(np.sum(weights[active]*(applied[active]-labels[active])**2)/valid.sum())),
                clip_low_rows=int((raw[active]<0).sum()),clip_high_rows=int((raw[active]>1).sum())))
        if HORIZON_HOURS[h]==6:
            ea=d.contribution_a.to_numpy()[valid]-np.minimum(y[valid],p[valid]);eb=d.contribution_b.to_numpy()[valid]-np.maximum(y[valid]-p[valid],0)
            mse=np.mean((d.pred_P.to_numpy()[valid]-y[valid])**2)
            np.testing.assert_allclose(np.mean(ea**2)+np.mean(eb**2)+2*np.mean(ea*eb),mse,atol=1e-12,rtol=0)
            decompositions.append(dict(horizon=h,a_mse=np.mean(ea**2),b_mse=np.mean(eb**2),twice_cross_mean=2*np.mean(ea*eb),P_mse=mse))
        audits.append(audit)
    write_frame(RUN/'component_predictions.parquet',comp);write_frame(RUN/'control_predictions.parquet',pred)
    write_frame(RUN/'P_component_diagnostics.parquet',pd.concat(pframes,ignore_index=True))
    write_frame(RUN/'aligned_unique_components.parquet',pd.concat(aligned,ignore_index=True))
    write_frame(RUN/'target_audit.parquet',pd.concat(audits,ignore_index=True))
    for name,values in [('primary_scores',primary),('all_control_metrics',allm),('P_component_metrics',pm),('bootstrap_intervals',boots),
        ('county_metrics',county),('fold_metrics',folds),('window_metrics',windows),('P71_bins',pbins),
        ('branch_diagnostics',branch_stats),('branch_loss_decomposition',pd.DataFrame(decompositions,columns=['horizon','a_mse','b_mse','twice_cross_mean','P_mse']))]:write_frame(RUN/f'metrics/{name}.csv',values)
    write_json(RUN/'invariants.json',dict(status='PASS',unchanged_component_checks=invariants,P1_N_D_R_fixed=True,
        direct_OSI_fixed=True,single_source_propagation_checked=True,no_joint_candidate=True))
    assert_inputs(ctx)
    prior=set(m['files'])|{'MODEL_CV_COMPLETE','EVALUATION_COMPLETE','VERIFIED_COMPLETE'}
    files={str(p.relative_to(RUN)):sha256_file(p) for p in sorted(RUN.rglob('*')) if p.is_file() and str(p.relative_to(RUN)) not in prior}
    write_json(RUN/'EVALUATION_COMPLETE',dict(identity_hash=ctx.identity_hash,files=files,evaluation_code_sha256=sha256_file(Path(__file__)),cases=list(CASES)))
    print(pd.DataFrame(primary).query('case != "B0"')[['case','horizon','baseline_rmse','rmse','rmse_change_pct']].to_string(index=False),flush=True)


if __name__=='__main__':evaluate()
