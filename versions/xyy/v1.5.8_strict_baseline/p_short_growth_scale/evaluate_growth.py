"""Frozen full-county and target-related evaluations; no new model fitting."""
from growth_core import *
PAIRS=(('G1','G0'),('G2','G0'),('G2','G1'))

def metric(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);valid=np.isfinite(y)
    assert np.isfinite(p[valid]).all() and not np.isinf(y).any()
    e=p[valid]-y[valid];n=len(e)
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
    draw=np.random.default_rng(20260910).integers(0,239,(2000,239));nn=n[draw].sum(axis=1)
    good=nn>0;assert good.all()
    delta=np.sqrt(sp[draw].sum(axis=1)/nn)-np.sqrt(sb[draw].sum(axis=1)/nn)
    lo,hi=np.quantile(delta,[.025,.975]);return dict(ci_low=float(lo),ci_high=float(hi),bootstrap_replicates=2000,bootstrap_unit='county_full_trajectory')

def evaluate():
    if (RUN/'EVALUATION_COMPLETE').exists():raise RuntimeError('Completed evaluation will not be overwritten')
    ctx=initialize(load_context());done=read(RUN/'MODEL_CV_COMPLETE');assert done['identity_hash']==ctx.identity_hash
    for n,h in done['files'].items():assert sha(RUN/n)==h,n
    branch=normalize(pd.read_parquet(RUN/'branch_predictions.parquet'));pd.testing.assert_frame_equal(branch[KEYS],ctx.meta[KEYS],check_dtype=False)
    comp={};controls={};aux={};wide=ctx.meta[KEYS].copy();wide['identity_hash']=ctx.identity_hash
    parts_frame=wide.copy();source_rows=[]
    for case in CASES:
        pp={h:{c:v.copy() for c,v in part.items()} for h,part in ctx.parts.items()}
        if case!='G0':pp[H1]['P_t']=reconstruct(ctx,branch[case+'_raw_b'].to_numpy(),case)
        if case!='G0':np.testing.assert_array_equal(pp[H1]['P_t'],branch[case+'_P1'])
        cc,aa=build_controls(ctx.meta,ctx.direct,pp);comp[case]=pp;controls[case]=cc;aux[case]=aa
        for h in HORIZONS:
            hh=HORIZON_HOURS[h];wide['true_'+h]=ctx.data.y_train[h];wide['scoreable_'+h]=expected_mask(ctx.meta,h)
            for mode,v in cc.items():wide[f'pred_{case}_{mode}_{h}']=v[h]
            for c in COMPONENTS:
                tag=component_target_name(c,h);parts_frame[f'pred_{case}_{tag}']=pp[h][c]
                if case=='G0' or (h,c)!=(H1,'P_t'):np.testing.assert_array_equal(pp[h][c],ctx.parts[h][c])
            unchanged=expected_mask(ctx.meta,h)&((ctx.meta.hour_idx.to_numpy()+hh>=96)|(hh>=24))
            np.testing.assert_array_equal(cc['v18_rule'][h][unchanged],ctx.controls['v18_rule'][h][unchanged])
            s=ctx.meta.hour_idx.to_numpy()+hh
            rr=ctx.meta[KEYS].copy();rr['case']=case;rr['horizon']=hh;rr['target_hour']=s;rr['P_source']=pp[h]['P_t'];rr['P_aligned']=aa['aligned_components'][h]['P_t']
            rr['true_P']=ctx.data.component_targets[component_target_name('P_t',h)];source_rows.append(rr)
    frame(RUN/'control_predictions.parquet',wide);frame(RUN/'component_predictions.parquet',parts_frame)
    frame(RUN/'P_component_predictions.parquet',pd.concat(source_rows,ignore_index=True))
    primary=[];comparisons=[];counties=[];folds=[];groups=[];boot=[];pmetrics=[];changes=[];allm=[]
    fips=ctx.meta.fipsCode.to_numpy();p=ctx.p;g=ctx.g;membership=ctx.membership.set_index(['horizon','target_day','fipsCode']).test_like
    early_truth=pd.DataFrame(dict(fipsCode=fips[ctx.early],truth=ctx.data.y_train[H1].to_numpy()[ctx.early])).groupby('fipsCode').truth.max()
    low=ctx.meta.fipsCode.map(early_truth).to_numpy()<=.01
    group_definitions={
        'all':'All 239 counties, complete valid rows for this horizon',
        'early_73_95':'Valid target hours <=95',
        'late_96_215':'Valid target hours >=96; explicit source reuse',
        'test_like':'Frozen test-to-train context top5 union for the corresponding horizon and target day',
        'not_test_like':'Complement of frozen test_like on valid rows',
        'all_except_Forest':'All valid rows excluding FIPS42053; diagnostic only',
        'test_like_except_Forest':'Frozen test_like excluding FIPS42053; diagnostic only',
        'g_zero':'Known official N71 == 0',
        'g_positive':'Known official N71 > 0',
        'g_positive_low_early_damage':'N71 > 0 and official early (73-95) peak OSI <=.01; outcome-defined diagnosis only'}
    save(RUN/'group_definitions.json',group_definitions)
    for h in HORIZONS:
        hh=HORIZON_HOURS[h];y=ctx.data.y_train[h].to_numpy();valid=expected_mask(ctx.meta,h);s=ctx.meta.hour_idx.to_numpy()+hh
        idx=pd.MultiIndex.from_arrays([np.full(len(fips),hh),s//24,fips]);test_like=membership.reindex(idx,fill_value=False).to_numpy(bool)
        masks={'all':np.ones(len(y),bool),'early_73_95':s<=95,'late_96_215':s>=96,'test_like':test_like,'not_test_like':~test_like,
               'all_except_Forest':fips!='42053','test_like_except_Forest':test_like&(fips!='42053'),'g_zero':g==0,'g_positive':g>0,'g_positive_low_early_damage':(g>0)&low}
        for case in CASES:
            pred=controls[case]['v18_rule'][h];primary.append(dict(case=case,horizon=hh,**metric(y,pred)))
            for mode in controls[case]:allm.append(dict(case=case,horizon=hh,mode=mode,**metric(y,controls[case][mode][h])))
        for case,ref in PAIRS:
            b=controls[ref]['v18_rule'][h];pred=controls[case]['v18_rule'][h];base=dict(case=case,reference=ref,pair=case+'_minus_'+ref,horizon=hh)
            comparisons.append(dict(**base,**compare(y,b,pred)))
            for name,mask in masks.items():
                yy=np.where(mask,y,np.nan);row=dict(**base,group=name,**compare(yy,b,pred));groups.append(row)
                if name in ('all','test_like'):
                    boot.append(dict(**row,**bootstrap(fips,yy,b,pred)))
            for f in sorted(set(fips)):
                mask=fips==f;counties.append(dict(**base,fipsCode=f,countyName=ctx.names.loc[f,'countyName'],stateAbbr=ctx.names.loc[f,'stateAbbr'],fold=int(ctx.meta.loc[mask,'fold'].iloc[0]),**compare(y[mask],b[mask],pred[mask])))
            for fold in range(5):
                mask=ctx.meta.fold.to_numpy()==fold;folds.append(dict(**base,fold=fold,**compare(y[mask],b[mask],pred[mask])))
            rr=ctx.meta.loc[valid,KEYS].copy();rr['pair']=base['pair'];rr['horizon']=hh;rr['target_hour']=s[valid];rr['truth']=y[valid];rr['baseline']=b[valid];rr['candidate']=pred[valid]
            rr['sse_reduction']=(b[valid]-y[valid])**2-(pred[valid]-y[valid])**2;rr['test_like']=test_like[valid];changes.append(rr)
            yp=ctx.data.component_targets[component_target_name('P_t',h)].to_numpy()
            for space,bb,pp in [('source',comp[ref][h]['P_t'],comp[case][h]['P_t']),('aligned',aux[ref]['aligned_components'][h]['P_t'],aux[case]['aligned_components'][h]['P_t'])]:
                for group,mask in [('all',np.ones(len(y),bool)),('early_73_95',s<=95)]:
                    pmetrics.append(dict(**base,space=space,group=group,**compare(np.where(mask,yp,np.nan),bb,pp)))
    for name,data in [('primary_scores',primary),('comparisons',comparisons),('all_controls',allm),('county_metrics',counties),('fold_metrics',folds),('group_metrics',groups),('bootstrap',boot),('P_metrics',pmetrics)]:frame(RUN/f'metrics/{name}.csv',data)
    frame(RUN/'row_changes.parquet',pd.concat(changes,ignore_index=True))
    ba=ctx.meta[KEYS].copy();ba['p71']=p;ba['N71']=g;ba['early_eligible']=ctx.early;ba['true_P1']=ctx.y
    ba['a_contribution']=p*ctx.a
    bstats=[]
    for case in CASES:
        if case=='G0':inc=(1-p)*ctx.branch.b_applied.to_numpy();raw=ctx.branch.pred_b_raw.to_numpy();scale=1-p
        else:
            raw=branch[case+'_raw_b'].to_numpy();scale=scales(p,g,case)
            inc=np.full(len(p),np.nan);hit=ctx.early&(p<1);inc[hit]=np.clip(scale[hit]*np.clip(raw[hit],0,1),0,1-p[hit]);inc[ctx.early&(p==1)]=0
        ba[case+'_scale']=scale;ba[case+'_increment']=inc
        mask=ctx.early;truth_inc=np.maximum(ctx.y[mask]-p[mask],0);ea=p[mask]*ctx.a[mask]-np.minimum(ctx.y[mask],p[mask]);eb=inc[mask]-truth_inc
        p_error=comp[case][H1]['P_t'][mask]-ctx.y[mask]
        close=np.mean(ea**2)+np.mean(eb**2)+2*np.mean(ea*eb)
        np.testing.assert_allclose(close,np.mean(p_error**2),atol=1e-12,rtol=0)
        bstats.append(dict(case=case,n=int(mask.sum()),b_increment_rmse=float(np.sqrt(np.mean(eb**2))),P1_early_rmse=float(np.sqrt(np.mean(p_error**2))),
            a_squared_term=float(np.mean(ea**2)),b_squared_term=float(np.mean(eb**2)),two_ab_cross_term=float(2*np.mean(ea*eb)),
            raw_b_below_zero=int((raw[mask]<0).sum()),raw_b_above_one=int((raw[mask]>1).sum()),
            increment_capped_at_remaining=int(((scale[mask]*np.clip(raw[mask],0,1))>1-p[mask]).sum())))
    frame(RUN/'branch_target_audit.parquet',ba);frame(RUN/'metrics/branch_contribution.csv',bstats)
    boundaries=[]
    for case in CASES:
        for domain in ['P1_source','OSI1','OSI6']:
            h=H1 if domain!='OSI6' else 'osi_target_t06h';hh=HORIZON_HOURS[h]
            values=comp[case][H1]['P_t'] if domain=='P1_source' else controls[case]['v18_rule'][h]
            ref=ctx.parts[H1]['P_t'] if domain=='P1_source' else ctx.controls['v18_rule'][h]
            for f in sorted(set(fips)):
                before=np.flatnonzero((fips==f)&(ctx.meta.hour_idx.to_numpy()+hh==95))[0];after=np.flatnonzero((fips==f)&(ctx.meta.hour_idx.to_numpy()+hh==96))[0]
                boundaries.append(dict(case=case,domain=domain,fipsCode=f,value95=values[before],value96=values[after],jump=values[after]-values[before],
                    baseline_jump=ref[after]-ref[before],additional_jump=(values[after]-values[before])-(ref[after]-ref[before])))
    frame(RUN/'metrics/boundary_95_96.csv',boundaries)
    for p,h in ctx.sources.items():assert sha(p)==h,p
    files={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and 'models' not in p.parts and p.name not in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']}
    save(RUN/'EVALUATION_COMPLETE',dict(identity_hash=ctx.identity_hash,evaluation_code_sha256=sha(__file__),files=files))
    print(pd.DataFrame(comparisons)[['pair','horizon','rmse','rmse_change_pct']].to_string(index=False),flush=True)

if __name__=='__main__':evaluate()
