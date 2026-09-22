"""Four candidates from the same fits; complete OSI and component diagnostics."""
from nr_core import *
def metric(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);hit=np.isfinite(y);e=p[hit]-y[hit];assert np.isfinite(e).all();n=len(e)
    return dict(n=n,rmse=float(np.sqrt(np.mean(e**2))) if n else np.nan,sse=float(np.sum(e**2)),
        mae=float(np.mean(abs(e))) if n else np.nan,bias=float(np.mean(e)) if n else np.nan)
def compare(y,b,p):
    a,c=metric(y,b),metric(y,p)
    return dict(**c,baseline_rmse=a['rmse'],rmse_delta=c['rmse']-a['rmse'],
        rmse_change_pct=100*(c['rmse']/a['rmse']-1) if a['rmse']>0 else np.nan,sse_reduction=a['sse']-c['sse'])
def bootstrap(fips,y,b,p):
    hit=np.isfinite(y);codes=pd.Categorical(fips,categories=sorted(set(fips))).codes
    n=np.bincount(codes[hit],minlength=239);sb=np.bincount(codes[hit],weights=(b[hit]-y[hit])**2,minlength=239);sp=np.bincount(codes[hit],weights=(p[hit]-y[hit])**2,minlength=239)
    if np.array_equal(p[hit],b[hit]):return dict(ci_low=0.,ci_high=0.,bootstrap_replicates=0,bootstrap_unit='exact_invariant')
    draw=np.random.default_rng(20260910).integers(0,239,(2000,239));nn=n[draw].sum(axis=1);assert (nn>0).all()
    delta=np.sqrt(sp[draw].sum(axis=1)/nn)-np.sqrt(sb[draw].sum(axis=1)/nn)
    lo,hi=np.quantile(delta,[.025,.975]);return dict(ci_low=float(lo),ci_high=float(hi),bootstrap_replicates=2000,bootstrap_unit='county_full_trajectory')
def component_groups(y):return dict(all=np.ones(len(y),bool),zero=y==0,small_positive=(y>0)&(y<=.01),tail_gt_001=y>.01,positive=y>0)
def classification(y,q):
    q=np.asarray(q);a=np.clip(q,1e-15,1-1e-15);binary=(np.asarray(y)>0).astype(float)
    return dict(n=len(y),observed_positive_fraction=float(binary.mean()),mean_q=float(q.mean()),
        logloss=float(-np.mean(binary*np.log(a)+(1-binary)*np.log1p(-a))),brier=float(np.mean((q-binary)**2)))
def evaluate():
    if (RUN/'EVALUATION_COMPLETE').exists():raise RuntimeError('Completed evaluation will not be overwritten')
    ctx=initialize(load_context());done=read(RUN/'MODEL_CV_COMPLETE');assert done['identity_hash']==ctx.identity_hash
    for p,h in done['files'].items():assert sha(RUN/p)==h,p
    new=norm(pd.read_parquet(RUN/'NR1_oof_predictions.parquet'));pd.testing.assert_frame_equal(new[KEYS],ctx.meta[KEYS],check_dtype=False)
    comp={};controls={};aux={};wide={k:ctx.meta[k] for k in KEYS};wide['identity_hash']=ctx.identity_hash;partout=dict(wide);aligned=[]
    for case in CASES:
        pp={h:{c:v.copy() for c,v in p.items()} for h,p in ctx.parts.items()}
        for c in ['N_t','R_t']:
            if case in ([c[0]+'_only','NR_both']):pp[H1][c]=new[c+'_product'].to_numpy(copy=True)
        cc,aa=build_controls(ctx.meta,ctx.direct,pp);comp[case]=pp;controls[case]=cc;aux[case]=aa
        for h in HORIZONS:
            hh=HORIZON_HOURS[h];wide['true_'+h]=ctx.data.y_train[h];wide['scoreable_'+h]=expected_mask(ctx.meta,h)
            for mode in cc:wide[f'pred_{case}_{mode}_{h}']=cc[mode][h]
            for c in COMPONENTS:
                tag=component_target_name(c,h);changed=h==H1 and c in ['N_t','R_t'] and case in [c[0]+'_only','NR_both']
                partout[f'pred_{case}_{tag}']=pp[h][c];partout[f'source_{case}_{tag}']=new[c+'_product_id'] if changed else ctx.reference_parts['source_NB_P24_'+tag]
                if not changed:np.testing.assert_array_equal(pp[h][c],ctx.parts[h][c])
            if hh>=24:np.testing.assert_array_equal(cc['v18_rule'][h],ctx.controls['v18_rule'][h])
            rr=ctx.meta[KEYS].copy();rr['case']=case;rr['horizon']=hh;rr['target_hour']=ctx.meta.hour_idx+hh
            for c in COMPONENTS:rr[c]=aa['aligned_components'][h][c]
            aligned.append(rr)
    frame(RUN/'control_predictions.parquet',pd.DataFrame(wide));frame(RUN/'component_predictions.parquet',pd.DataFrame(partout));frame(RUN/'aligned_components.parquet',pd.concat(aligned,ignore_index=True))
    definitions=dict(all='All 239 counties, complete scoreable rows',early_73_95='Target hour <=95',late_96_215='Target hour >=96',
        test_like='Frozen context test_to_train top5 matched county union by horizon and target day',not_test_like='Complement of test_like',
        all_except_Forest='All valid rows except FIPS42053, diagnostic only',test_like_except_Forest='Frozen test_like except FIPS42053, diagnostic only')
    save(RUN/'group_definitions.json',definitions)
    primary=[];allm=[];comparisons=[];counties=[];folds=[];days=[];groups=[];boot=[];components=[];changes=[];decomp=[]
    fips=ctx.meta.fipsCode.to_numpy();hours=ctx.meta.hour_idx.to_numpy();fold=ctx.meta.fold.to_numpy();membership=ctx.membership.set_index(['horizon','target_day','fipsCode']).test_like
    for h in HORIZONS:
        hh=HORIZON_HOURS[h];y=ctx.data.y_train[h].to_numpy();valid=expected_mask(ctx.meta,h);s=hours+hh
        idx=pd.MultiIndex.from_arrays([np.full(len(fips),hh),s//24,fips]);test_like=membership.reindex(idx,fill_value=False).to_numpy(bool)
        masks=dict(all=np.ones(len(y),bool),early_73_95=s<=95,late_96_215=s>=96,test_like=test_like,not_test_like=~test_like,
            all_except_Forest=fips!='42053',test_like_except_Forest=test_like&(fips!='42053'))
        for case in CASES:
            primary.append(dict(case=case,horizon=hh,**metric(y,controls[case]['v18_rule'][h])))
            for mode in controls[case]:allm.append(dict(case=case,horizon=hh,mode=mode,**metric(y,controls[case][mode][h])))
        for case,ref in PAIRS:
            b=controls[ref]['v18_rule'][h];pred=controls[case]['v18_rule'][h];base=dict(case=case,reference=ref,pair=case+'_minus_'+ref,horizon=hh)
            comparisons.append(dict(**base,**compare(y,b,pred)))
            for name,mask in masks.items():
                yy=np.where(mask,y,np.nan);row=dict(**base,group=name,**compare(yy,b,pred));groups.append(row)
                if name in ['all','test_like']:boot.append(dict(**row,**bootstrap(fips,yy,b,pred)))
            for f in sorted(set(fips)):
                hit=fips==f;counties.append(dict(**base,fipsCode=f,countyName=ctx.names.loc[f,'countyName'],stateAbbr=ctx.names.loc[f,'stateAbbr'],fold=int(fold[hit][0]),**compare(y[hit],b[hit],pred[hit])))
            for ff in range(5):
                hit=fold==ff;folds.append(dict(**base,fold=ff,**compare(y[hit],b[hit],pred[hit])))
            for day in sorted(set(s[valid]//24)):
                hit=valid&(s//24==day);days.append(dict(**base,target_day=day,target_date=str((pd.Timestamp('2026-03-11')+pd.Timedelta(days=int(day))).date()),**compare(y[hit],b[hit],pred[hit])))
            rr=ctx.meta.loc[valid,KEYS].copy();rr['pair']=base['pair'];rr['horizon']=hh;rr['target_hour']=s[valid];rr['truth']=y[valid];rr['baseline']=b[valid];rr['candidate']=pred[valid]
            rr['sse_reduction']=(b[valid]-y[valid])**2-(pred[valid]-y[valid])**2;rr['test_like']=test_like[valid];changes.append(rr)
            for c in ['N_t','R_t']:
                yc=ctx.data.component_targets[component_target_name(c,h)].to_numpy()
                for space,bb,pp in [('source',comp[ref][h][c],comp[case][h][c]),('aligned',aux[ref]['aligned_components'][h][c],aux[case]['aligned_components'][h][c])]:
                    baseline_sse=metric(yc,bb)['sse']
                    for name,mask in component_groups(yc).items():
                        yy=np.where(mask,yc,np.nan);row=compare(yy,bb,pp);components.append(dict(**base,component=c,space=space,group=name,**row,
                            baseline_group_sse=metric(yy,bb)['sse'],baseline_group_sse_share=metric(yy,bb)['sse']/baseline_sse if baseline_sse else np.nan))
            if hh<=6 and ref=='B0':
                ba=aux[ref]['aligned_components'][h];ca=aux[case]['aligned_components'][h];oldsum=.4*ba['P_t']+.35*ba['N_t']+.25*ba['D_t']-.1*ba['R_t']
                en=(.35*(ca['N_t']-ba['N_t']))[valid];er=(-.1*(ca['R_t']-ba['R_t']))[valid];old_error=(oldsum-y)[valid];delta=en+er
                cross=-2*np.sum(old_error*delta);sq=-np.sum(delta**2);actual=np.sum((b[valid]-y[valid])**2-(pred[valid]-y[valid])**2)
                decomp.append(dict(**base,n=int(valid.sum()),old_error_cross_reduction=cross,change_squared_reduction=sq,
                    N_R_change_interaction=-2*np.sum(en*er),latent_sse_reduction=cross+sq,postprocess_remainder=actual-cross-sq,OSI_sse_reduction=actual))
    for name,records in [('primary_scores',primary),('comparisons',comparisons),('all_controls',allm),('county_metrics',counties),('fold_metrics',folds),('day_metrics',days),('group_metrics',groups),('bootstrap',boot),('component_metrics',components),('OSI_change_decomposition',decomp)]:frame(RUN/f'metrics/{name}.csv',records)
    frame(RUN/'row_changes.parquet',pd.concat(changes,ignore_index=True))
    cls=[];cal=[];amps=[]
    for c in ['N_t','R_t']:
        y=ctx.ys[c];q=new[c+'_q'].to_numpy();m=new[c+'_m'].to_numpy();prior=new[c+'_prior_q'].to_numpy();mc=np.clip(m,0,1)
        for ff in ['all',0,1,2,3,4]:
            hit=ctx.valid if ff=='all' else ctx.valid&(fold==ff)
            for variant,a in [('q_model',q),('training_prior',prior)]:cls.append(dict(component=c,fold=str(ff),variant=variant,**classification(y[hit],a[hit])))
        for lo in np.arange(10)/10:
            hit=ctx.valid&(q>=lo)&((q<lo+.1) if lo<.9 else (q<=1));n=int(hit.sum())
            cal.append(dict(component=c,bin_low=lo,bin_high=lo+.1,n=n,mean_q=float(q[hit].mean()) if n else np.nan,observed_positive_fraction=float((y[hit]>0).mean()) if n else np.nan))
        for name,mask in component_groups(y).items():
            hit=mask&ctx.valid
            amps.append(dict(component=c,group=name,n=int(hit.sum()),true_mean=float(y[hit].mean()),q_mean=float(q[hit].mean()),raw_m_mean=float(m[hit].mean()),
                clipped_m_mean=float(mc[hit].mean()),product_mean=float(new.loc[hit,c+'_product'].mean()),conditional_m_rmse=float(np.sqrt(np.mean((mc[hit]-y[hit])**2))),
                raw_m_below_zero=int((m[hit]<0).sum()),raw_m_above_one=int((m[hit]>1).sum())))
    frame(RUN/'metrics/classification.csv',cls);frame(RUN/'metrics/calibration.csv',cal);frame(RUN/'metrics/amplitude_diagnostics.csv',amps)
    for p,h in ctx.sources.items():assert sha(p)==h,p
    files={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and 'models' not in p.parts and p.name not in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']}
    save(RUN/'EVALUATION_COMPLETE',dict(identity_hash=ctx.identity_hash,evaluation_code_sha256=sha(__file__),files=files))
    print(pd.DataFrame(comparisons)[['pair','horizon','rmse','rmse_change_pct']].to_string(index=False),flush=True)
if __name__=='__main__':evaluate()
