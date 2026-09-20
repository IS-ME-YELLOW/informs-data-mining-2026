"""Predeclared N diagnostics and descriptive comparisons, with row-level artifacts."""
import pyarrow as pa
import pyarrow.parquet as pq
from diagnostic_core import *


def grid(ctx,h,full=False):
    ss=np.arange(72+h,216+h if full else 216); n=len(ss)
    return pd.DataFrame(dict(split_seed=ctx.seed,model_seed=42,fipsCode=np.repeat(ctx.counties,n),
        fold=np.repeat(ctx.folds,n),horizon=h,hour_idx=np.tile(ss-h,ctx.nc),
        target_hour=np.tile(ss,ctx.nc),scoreable=np.tile(ss<=215,ctx.nc),
        target_timestamp=np.tile(pd.Timestamp('2026-03-11')+pd.to_timedelta(ss,unit='h'),ctx.nc),
        reference_identity=IDS[ctx.seed]))


def spaces(ctx,i):
    return dict(model_raw=ctx.rawN[i],component_clipped=ctx.parts[i,:,:,1],aligned=ctx.ctl['aligned'][:,:,1])


def n_stats(y,p):
    result=metric(y,p)
    if not len(np.asarray(y).ravel()): return result
    yy=np.asarray(y).ravel(); pp=np.asarray(p).ravel(); e=pp-yy
    for name,a in [('truth',yy),('prediction',pp)]:
        result.update({f'{name}_{k}':v for k,v in zip(['q00','q50','q90','q95','q99','q100'],np.quantile(a,[0,.5,.9,.95,.99,1]))})
        result[f'{name}_mean']=a.mean();result[f'{name}_variance']=a.var();result[f'{name}_zero_fraction']=(a==0).mean()
    result.update(negative_fraction=(pp<0).mean(),above_one_fraction=(pp>1).mean(),
                  underestimated_sse=float((e[e<0]**2).sum()),overestimated_sse=float((e[e>0]**2).sum()))
    yp=yy>.001; pp_event=pp>.001
    tp=int((yp&pp_event).sum());fp=int((~yp&pp_event).sum());fn=int((yp&~pp_event).sum())
    result.update(tp=tp,fp=fp,fn=fn,tn=int((~yp&~pp_event).sum()),
                  precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan)
    return result


def groups(ctx,h,y):
    ss=np.arange(72+h,216); shape=y.shape
    yield 'all','all',np.ones(shape,bool)
    yield 'common_target','120_215',np.broadcast_to(ss>=120,shape)
    for lo,hi in WINDOWS: yield 'target_window',f'{lo}_{hi}',np.broadcast_to((ss>=lo)&(ss<=hi),shape)
    for count in (1,2,3,4): yield 'candidate_count',str(count),ctx.ctl['count'][:,72+h:]==count
    for name,hit in [('N=0',y==0),('0<N<=1e-4',(y>0)&(y<=1e-4)),('1e-4<N<=1e-3',(y>1e-4)&(y<=1e-3)),
                     ('1e-3<N<=1e-2',(y>1e-3)&(y<=1e-2)),('N>1e-2',y>1e-2)]: yield 'truth_N',name,hit
    p=ctx.truth[:,71,0,None];n=ctx.truth[:,71,1,None]
    for name,hit in [('p=0',p==0),('0<p<.01',(p>0)&(p<.01)),('.01<=p<.1',(p>=.01)&(p<.1)),
                     ('.1<=p<.5',(p>=.1)&(p<.5)),('.5<=p<=1',p>=.5)]: yield 'P71',name,np.broadcast_to(hit,shape)
    for name,hit in [('n=0',n==0),('0<n<=.001',(n>0)&(n<=.001)),('n>.001',n>.001)]: yield 'N71',name,np.broadcast_to(hit,shape)
    for f in range(5): yield 'fold',str(f),np.broadcast_to(ctx.folds[:,None]==f,shape)


def component_diagnostics(ctx):
    metrics=[];bins=[];peaklag=[];counties=[];alignment=[];boots=[];rowframes=[];sourceframes=[];cross=[]
    for i,h in enumerate(HS):
        sl=slice(72+h,216);ss=np.arange(72+h,216);y=ctx.truth[:,sl,1]
        g=grid(ctx,h,full=True)
        def pad(a,value=np.nan):
            a=np.asarray(a);return np.concatenate([a,np.full((ctx.nc,h),value,dtype=a.dtype)],axis=1).ravel()
        g['true_N']=pad(y)
        g['baseline_OSI']=pad(ctx.ctl['main'][i,:,sl]);g['true_OSI']=pad(ctx.y[:,sl])
        g['candidate_count']=pad(ctx.ctl['count'][:,sl],0)
        for j,c in enumerate(CS):
            g[f'source_{c}_model_id']=pad(ctx.sources[i,:,sl,j],'')
            g[f'true_{c}']=pad(ctx.truth[:,sl,j])
            g[f'pred_{c}_clipped']=pad(ctx.parts[i,:,sl,j])
        for space,pred in spaces(ctx,i).items():
            p=pred[:,sl];g[f'N_{space}']=pad(p)
            total=metric(y,p)['sse']
            for kind,name,mask in groups(ctx,h,y):
                row=dict(split_seed=ctx.seed,horizon=h,space=space,group_type=kind,group=name,
                         **n_stats(y[mask],p[mask]))
                row['sse_fraction_of_full']=row['sse']/total if total else np.nan
                bins.append(row)
                if kind in ('all','common_target','fold','target_window'):metrics.append(row.copy())
            for k,f in enumerate(ctx.counties):
                true=y[k];pr=p[k];sm=metric(true,pr)
                counties.append(dict(split_seed=ctx.seed,horizon=h,space=space,fipsCode=f,
                    countyName=ctx.names.loc[f,'countyName'],stateAbbr=ctx.names.loc[f,'stateAbbr'],fold=ctx.folds[k],
                    **sm,weighted_N_sse=.35**2*sm['sse']))
                true_peak=true.max();pred_peak=pr.max();tp=int(np.argmax(true));pp=int(np.argmax(pr));active=true_peak>.001
                peaklag.append(dict(kind='peak',split_seed=ctx.seed,horizon=h,space=space,fipsCode=f,
                    low_activity=not active,true_peak=true_peak,pred_peak=pred_peak,
                    true_peak_hour=int(ss[tp]),pred_peak_hour=int(ss[pp]),
                    peak_hour_difference=int(ss[pp]-ss[tp]) if active else np.nan,
                    prediction_at_true_peak=pr[tp],amplitude_ratio=pr[tp]/true_peak if active else np.nan,
                    true_peak_ties=int((true==true_peak).sum()),prediction_peak_ties=int((pr==pred_peak).sum())))
            for lag in range(-3,4):
                yt=y[:,3:-3];pt=p[:,3+lag:y.shape[1]-3+lag]
                peaklag.append(dict(kind='lag',split_seed=ctx.seed,horizon=h,space=space,fipsCode='ALL',
                    lag=lag,common_start=72+h+3,common_end=212,correlation=correlation(yt,pt),**metric(yt,pt)))
                for k,f in enumerate(ctx.counties):
                    peaklag.append(dict(kind='lag',split_seed=ctx.seed,horizon=h,space=space,fipsCode=f,
                        lag=lag,common_start=72+h+3,common_end=212,correlation=correlation(yt[k],pt[k]),**metric(yt[k],pt[k])))
        rowframes.append(g)
        src=grid(ctx,h);src['source_horizon']=h;src['source_origin_hour']=src.hour_idx
        src['source_model_id']=ctx.sources[i,:,sl,1].ravel();src['source_prediction']=ctx.parts[i,:,sl,1].ravel()
        src['candidate_count']=ctx.ctl['count'][:,sl].ravel();src['weight']=1/src.candidate_count
        sourceframes.append(src)
        for scope,start in [('full',72+h),('common_120_215',120)]:
            yy=ctx.truth[:,start:,1];before=ctx.parts[i,:,start:,1];after=ctx.ctl['aligned'][:,start:,1]
            alignment.append(dict(split_seed=ctx.seed,horizon=h,domain='N',scope=scope,**compare(yy,before,after)))
            alignment.append(dict(split_seed=ctx.seed,horizon=h,domain='OSI',scope=scope,
                **compare(ctx.y[:,start:],ctx.ctl['C1'][i,:,start:],ctx.ctl['C3'][i,:,start:])))
        boots.append(dict(split_seed=ctx.seed,horizon=h,scenario='N_aligned_minus_clipped',
            **compare(y,ctx.parts[i,:,sl,1],ctx.ctl['aligned'][:,sl,1]),
            **bootstrap(y,ctx.parts[i,:,sl,1],ctx.ctl['aligned'][:,sl,1])))
    for lo,hi,m in [(73,77,1),(78,95,2),(96,119,3),(120,215,4)]:
        errors=np.stack([ctx.parts[i,:,lo:hi+1,1]-ctx.truth[:,lo:hi+1,1] for i in range(m)])
        av=ctx.ctl['aligned'][:,lo:hi+1,1]-ctx.truth[:,lo:hi+1,1]
        np.testing.assert_allclose(errors.mean(axis=0),av,atol=1e-12,rtol=0)
        reconstructed=0.
        for i in range(m):
            for j in range(m):
                product=float(np.mean(errors[i]*errors[j]));reconstructed+=product/m**2
                cross.append(dict(split_seed=ctx.seed,start=lo,end=hi,candidate_count=m,source_a=HS[i],source_b=HS[j],
                    n=av.size,cross_mean=product,weight=1/m**2,weighted_cross_mean=product/m**2,
                    correlation=correlation(errors[i],errors[j]),source_a_bias=errors[i].mean(),source_b_bias=errors[j].mean(),
                    aligned_mse=float(np.mean(av**2))))
        np.testing.assert_allclose(reconstructed,np.mean(av**2),atol=1e-12,rtol=1e-10)
    frame(ctx.output/'rows/component_diagnostics.parquet',pd.concat(rowframes,ignore_index=True))
    frame(ctx.output/'rows/alignment_sources.parquet',pd.concat(sourceframes,ignore_index=True))
    for name,data in [('component_metrics',metrics),('distribution_and_bins',bins),('peak_and_lag_diagnostics',peaklag),
                      ('county_N_metrics',counties),('alignment_effects',alignment),('source_error_cross_products',cross)]:
        frame(ctx.output/f'tables/{name}.csv',data)
    return boots,pd.DataFrame(counties)


def decompose(ctx):
    frames=[];stats=[]
    for i,h in enumerate(HS):
        sl=slice(72+h,216);y=ctx.y[:,sl];truth=ctx.truth[:,sl]
        for mode in ('C1','C3','main'):
            p=ctx.parts[i,:,sl] if mode=='C1' or (mode=='main' and h>=24) else ctx.ctl['aligned'][:,sl]
            es=(p-truth)*W;z=linear(p);zt=linear(truth);b=post(z)-z+zt-y
            components=np.concatenate([es,b[...,None]],axis=2);err=ctx.ctl[mode][i,:,sl]-y
            np.testing.assert_allclose(components.sum(axis=2),err,atol=1e-12,rtol=1e-10)
            d=grid(ctx,h);d['mode']=mode;d['true_OSI']=y.ravel();d['pred_OSI']=ctx.ctl[mode][i,:,sl].ravel()
            for j,c in enumerate(CS):d[f'e_{c}']=es[...,j].ravel()
            d['b']=b.ravel();d['OSI_error']=err.ravel();d['N_signed_allocation']=(es[...,1]*err).ravel()
            frames.append(d)
            labels=(*CS,'b');reconstructed=0.
            for j,a in enumerate(labels):
                for k in range(j,len(labels)):
                    v=float(np.sum(components[...,j]*components[...,k]))*(1 if j==k else 2)
                    reconstructed+=v
                    stats.append(dict(split_seed=ctx.seed,horizon=h,mode=mode,term_a=a,term_b=labels[k],
                        term_type='square' if j==k else 'twice_cross',sse_term=v,total_osi_sse=float(np.sum(err**2)),
                        N_signed_allocation=float(np.sum(es[...,1]*err))))
            np.testing.assert_allclose(reconstructed,np.sum(err**2),atol=1e-12,rtol=1e-10)
    frame(ctx.output/'rows/osi_error_decomposition.parquet',pd.concat(frames,ignore_index=True))
    frame(ctx.output/'tables/osi_decomposition.csv',stats)


def oracle_diagnostics(ctx,boots):
    metrics=[];countyrows=[];n_oracle=None
    writers={};paths={'oracle':ctx.output/'rows/oracle_predictions.parquet','anchor':ctx.output/'rows/anchor_predictions.parquet'}
    try:
        for name in SCENARIOS:
            p=scenario_parts(ctx,name);ctl=controls(p)
            if name=='O_N_all':n_oracle=ctl['main']
            source_h=int(name.rsplit('_',1)[-1]) if name.startswith('O_N_source_') else 0
            alpha=float(name.rsplit('_',1)[-1]) if name.startswith('O_N_partial_') else 1.
            for i,h in enumerate(HS):
                sl=slice(72+h,216);yy=ctx.y[:,sl];base=ctx.ctl['main'][i,:,sl];pred=ctl['main'][i,:,sl]
                if source_h:
                    unaffected=np.arange(72+h,216)<72+source_h
                    if h>=24 and h!=source_h:unaffected[:]=True
                    np.testing.assert_array_equal(pred[:,unaffected],base[:,unaffected])
                typ='anchor' if name.startswith('A_') else 'oracle'
                result=dict(split_seed=ctx.seed,horizon=h,scenario=name,reference='B_current',
                            source_horizon=source_h,alpha=alpha,uses_future_truth=typ=='oracle',deployable=False,
                            **compare(yy,base,pred))
                metrics.append(result)
                if name.startswith(('O_N_','A_')):boots.append(dict(**result,**bootstrap(yy,base,pred)))
                d=grid(ctx,h);d['scenario']=name;d['source_horizon']=source_h;d['alpha']=alpha
                d['uses_future_truth']=typ=='oracle';d['deployable']=False;d['true_OSI']=yy.ravel()
                d['baseline_OSI']=base.ravel();d['diagnostic_OSI']=pred.ravel()
                d['sse_reduction']=((base-yy)**2-(pred-yy)**2).ravel()
                for j,c in enumerate(CS):
                    d[f'diagnostic_{c}']=(p[i,:,sl,j] if h>=24 else ctl['aligned'][:,sl,j]).ravel()
                table=pa.Table.from_pandas(d,preserve_index=False)
                if typ not in writers:writers[typ]=pq.ParquetWriter(safe(paths[typ]),table.schema,compression='snappy')
                writers[typ].write_table(table)
                for k,f in enumerate(ctx.counties):
                    countyrows.append(dict(split_seed=ctx.seed,horizon=h,scenario=name,group_type='county',group=f,
                        fipsCode=f,countyName=ctx.names.loc[f,'countyName'],stateAbbr=ctx.names.loc[f,'stateAbbr'],
                        fold=ctx.folds[k],**compare(yy[k],base[k],pred[k])))
                ss=np.arange(72+h,216)
                for lo,hi in WINDOWS:
                    hit=(ss>=lo)&(ss<=hi)
                    countyrows.append(dict(split_seed=ctx.seed,horizon=h,scenario=name,group_type='window',group=f'{lo}_{hi}',
                        **compare(yy[:,hit],base[:,hit],pred[:,hit])))
        old=ctx.oldparts.copy()
        for i,h in enumerate(HS):old[i,:,72+h:,1]=ctx.truth[:,72+h:,1]
        oldctl=controls(old)
        for i,h in enumerate(HS):
            sl=slice(72+h,216)
            metrics.append(dict(split_seed=ctx.seed,horizon=h,scenario='O_N_all',reference='B_strict',source_horizon=0,
                alpha=1.,uses_future_truth=True,deployable=False,**compare(ctx.y[:,sl],ctx.oldctl['main'][i,:,sl],oldctl['main'][i,:,sl])))
    finally:
        for w in writers.values():w.close()
    frame(ctx.output/'tables/oracle_metrics.csv',metrics)
    frame(ctx.output/'tables/source_to_output_effects.csv',[r for r in metrics if r['source_horizon']])
    frame(ctx.output/'tables/county_and_window_metrics.csv',countyrows)
    frame(ctx.output/'tables/bootstrap_intervals.csv',boots)
    return n_oracle


def select_cases(ctx,county):
    if ctx.seed==42:
        selections=[]
        hist=csv(ROOT/'review/2026-09-16/similar_counties/focus_match_summary.csv',dtype={'query_fips':str})
        for f,(name,state) in FOCUS.items():
            assert ((hist.query_fips==f)&(hist.query_name==name)&(hist.query_state==state)).any()
            assert ctx.names.loc[f,'countyName']==name and ctx.names.loc[f,'stateAbbr']==state
            selections.append(dict(fipsCode=f,countyName=name,stateAbbr=state,reason='historical_focus',horizon=0,rank=0))
        for h in HS:
            space='aligned' if h<=6 else 'component_clipped'
            ranks=county.loc[(county.horizon==h)&(county.space==space)].sort_values(['weighted_N_sse','fipsCode'],ascending=[False,True])
            for rank,(_,r) in enumerate(ranks.head(3).iterrows(),1):
                selections.append(dict(fipsCode=r.fipsCode,countyName=r.countyName,stateAbbr=r.stateAbbr,
                                       reason='seed42_top_N_weighted_sse',horizon=h,rank=rank))
        frame(OUT/'summary/case_selection.csv',selections)
    selection=csv(OUT/'summary/case_selection.csv',dtype={'fipsCode':str})
    return sorted(selection.fipsCode.unique())


def matched_support(ctx,cases):
    # Ranking phase intentionally accesses no prediction or future-label values.
    rankrows=[];support=[];feature_rows=[]
    for h in HS:
        columns=[c.format(h=h) for c in MATCH]
        x=ctx.features[columns].to_numpy().reshape(ctx.nc,144,8)
        for f in cases:
            k=int(np.flatnonzero(ctx.counties==f)[0]);donors=np.flatnonzero(ctx.folds!=ctx.folds[k])
            xx=x[donors,:144-h];q=x[k,:144-h]
            assert np.isfinite(xx).all() and np.isfinite(q).all()
            means=xx.mean(axis=0);std=xx.std(axis=0);lo=xx.min(axis=0);hi=xx.max(axis=0)
            scales=np.where(std>0,std,1)
            diff=(xx-q[None,:,:])/scales[None,:,:];diff[:,:,]=np.where(std[None,:,:]>0,diff,0)
            distances=np.sqrt(np.sum(diff**2,axis=2))
            for t in range(144-h):
                s=72+t+h
                order=np.lexsort((ctx.counties[donors],distances[:,t]))[:5]
                below=q[t]<lo[t];above=q[t]>hi[t]
                support.append(dict(split_seed=ctx.seed,fipsCode=f,fold=int(ctx.folds[k]),horizon=h,target_hour=s,
                    candidate_count=len(donors),out_of_support_features=int((below|above).sum()),
                    outside_columns=','.join(np.array(columns)[below|above]),zero_variance_columns=','.join(np.array(columns)[std[t]==0]),
                    nearest_distance=distances[order[0],t],fifth_distance=distances[order[-1],t]))
                for j,col in enumerate(columns):
                    feature_rows.append(dict(split_seed=ctx.seed,fipsCode=f,horizon=h,target_hour=s,feature=col,
                        query_value=q[t,j],donor_mean=means[t,j],donor_std=std[t,j],donor_min=lo[t,j],donor_max=hi[t,j],
                        outside_support=bool(below[j]|above[j])))
                for rank,d in enumerate(order,1):
                    j=donors[d]
                    rankrows.append(dict(split_seed=ctx.seed,fipsCode=f,fold=int(ctx.folds[k]),horizon=h,target_hour=s,
                        donor_fipsCode=ctx.counties[j],donor_fold=int(ctx.folds[j]),rank=rank,distance=distances[d,t]))
    rankings=pd.DataFrame(rankrows)
    rankpath=ctx.output/'tables/matching_rankings_without_outcomes.parquet'
    frame(rankpath,rankings)
    # Freeze ranks before accessing future outcomes.
    rankhash=sha(rankpath)
    outcomes=[];support_out=[]
    index={f:i for i,f in enumerate(ctx.counties)}
    for r in rankings.itertuples(index=False):
        k=index[r.fipsCode];j=index[r.donor_fipsCode];i=HS.index(r.horizon);s=r.target_hour
        n=ctx.ctl['aligned'][:,:,1] if r.horizon<=6 else ctx.parts[i,:,:,1]
        outcomes.append(dict(query_true_N=ctx.truth[k,s,1],donor_true_N=ctx.truth[j,s,1],
            query_pred_N=n[k,s],donor_pred_N=n[j,s],query_true_OSI=ctx.y[k,s],donor_true_OSI=ctx.y[j,s],
            query_pred_OSI=ctx.ctl['main'][i,k,s],donor_pred_OSI=ctx.ctl['main'][i,j,s],
            donor_countyName=ctx.names.loc[r.donor_fipsCode,'countyName'],donor_stateAbbr=ctx.names.loc[r.donor_fipsCode,'stateAbbr']))
    for r in support:
        k=index[r['fipsCode']];donors=np.flatnonzero(ctx.folds!=ctx.folds[k]);s=r['target_hour'];h=r['horizon'];i=HS.index(h)
        same=ctx.truth[donors,s,1];all_labels=ctx.truth[donors,72+h:,1].ravel()
        pred=ctx.ctl['aligned'][k,s,1] if h<=6 else ctx.parts[i,k,s,1]
        support_out.append(dict(**r,query_true_N=ctx.truth[k,s,1],query_pred_N=pred,
            donor_same_time_N_max=same.max(),donor_same_time_N_q95=np.quantile(same,.95),
            donor_all_training_N_max=all_labels.max(),donor_all_training_N_q95=np.quantile(all_labels,.95),
            query_truth_above_training_max=bool(ctx.truth[k,s,1]>all_labels.max()),
            query_prediction_above_training_max=bool(pred>all_labels.max()),
            training_N_empirical_cdf_of_query_prediction=float((all_labels<=pred).mean())))
    assert rankhash==sha(rankpath)
    frame(ctx.output/'tables/matched_support.csv',pd.concat([rankings,pd.DataFrame(outcomes)],axis=1))
    frame(ctx.output/'tables/query_feature_support.csv',feature_rows)
    frame(ctx.output/'tables/query_training_support.csv',support_out)
    write_json(ctx.output/'tables/matching_ranking_identity.json',dict(sha256=rankhash,ranked_before_outcome_join=True,
        full_fold_exclusion=True,selected_cases=cases,features=MATCH,k=5,rows=len(rankings)))


def supplemental_rows(ctx,cases,n_oracle):
    rows=[];allcounty=[]
    for i,h in enumerate(HS):
        sl=slice(72+h,216);ss=np.arange(72+h,216);n=ctx.ctl['aligned'][:,sl,1] if h<=6 else ctx.parts[i,:,sl,1]
        for k,f in enumerate(ctx.counties):
            err=ctx.ctl['main'][i,k,sl]-ctx.y[k,sl];ne=.35*(n[k]-ctx.truth[k,sl,1])
            allcounty.append(dict(split_seed=ctx.seed,horizon=h,fipsCode=f,countyName=ctx.names.loc[f,'countyName'],
                stateAbbr=ctx.names.loc[f,'stateAbbr'],fold=ctx.folds[k],N_sse=float(np.sum((ne/.35)**2)),
                N_weighted_sse=float(np.sum(ne**2)),N_signed_allocation=float(ne@err),OSI_sse=float(err@err),
                oracle_N_sse_reduction=float(np.sum(err**2-(n_oracle[i,k,sl]-ctx.y[k,sl])**2))))
            if f not in cases:continue
            for j,s in enumerate(ss):
                row=dict(split_seed=ctx.seed,horizon=h,fipsCode=f,countyName=ctx.names.loc[f,'countyName'],
                    stateAbbr=ctx.names.loc[f,'stateAbbr'],fold=int(ctx.folds[k]),target_hour=int(s),origin_hour=int(s-h),
                    N_true=ctx.truth[k,s,1],N_raw=ctx.rawN[i,k,s],N_clipped=ctx.parts[i,k,s,1],N_aligned=ctx.ctl['aligned'][k,s,1],
                    P_true=ctx.truth[k,s,0],R_true=ctx.truth[k,s,3],OSI_true=ctx.y[k,s],OSI_pred=ctx.ctl['main'][i,k,s],
                    P_pred=ctx.ctl['aligned'][k,s,0] if h<=6 else ctx.parts[i,k,s,0],
                    R_pred=ctx.ctl['aligned'][k,s,3] if h<=6 else ctx.parts[i,k,s,3],
                    gust_target=float(ctx.features.iloc[k*144+s-h-72][f'gust_at_t{h}h']),
                    last_N=ctx.truth[k,71,1],last_P=ctx.truth[k,71,0])
                # These existing F2 descriptors are not input to the current N model.
                xr=ctx.f2features.iloc[k*144+s-h-72]
                for col in ['nbr8_mean_last_N_t','nbr8_max_last_N_t','nbr8_mean_gust_at_t1h','nbr8_max_gust_at_t1h']:
                    row[col]=xr[col]
                row['neighbor_weather_target_hour']=int(s-h+1)
                rows.append(row)
    frame(ctx.output/'tables/county_priority.csv',allcounty)
    frame(ctx.output/'tables/case_trajectories.csv',rows)


def analyze_split(ctx):
    print(f'split{ctx.seed}: distributions, timing and C3',flush=True)
    boots,county=component_diagnostics(ctx);decompose(ctx)
    print(f'split{ctx.seed}: oracle and fixed-anchor sensitivities',flush=True)
    n_oracle=oracle_diagnostics(ctx,boots)
    cases=select_cases(ctx,county)
    print(f'split{ctx.seed}: outcome-blind donor ranking for {len(cases)} focus counties',flush=True)
    matched_support(ctx,cases);supplemental_rows(ctx,cases,n_oracle)
    write_json(ctx.output/'manifest.json',dict(status='ANALYSIS_COMPLETE_PENDING_VERIFICATION',split_seed=ctx.seed,
        reference_identity=IDS[ctx.seed],files={str(p.relative_to(ctx.output)):sha(p) for p in sorted(ctx.output.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json','verification.json')},
        code_sha256={p.name:sha(p) for p in sorted(OUT.glob('*.py'))},config_sha256=sha(OUT/'diagnostic_config.json'),
        input_manifest_sha256=sha(OUT/'reference/initial_input_hashes.json'),training_performed=False))
    print(f'split{ctx.seed}: analysis saved',flush=True)


def summarize():
    summary=[];oracle=[];boot=[];counties=[]
    for seed in SEEDS:
        d=OUT/f'runs/split{seed}/tables'
        m=csv(d/'component_metrics.csv');o=csv(d/'oracle_metrics.csv');b=csv(d/'bootstrap_intervals.csv')
        oracle.append(o);boot.append(b);counties.append(csv(d/'county_priority.csv',dtype={'fipsCode':str}))
        for h in HS:
            row=dict(split_seed=seed,horizon=h)
            for space in ['model_raw','component_clipped','aligned']:
                mm=m.loc[(m.horizon==h)&(m.space==space)&(m.group_type=='all')].iloc[0]
                row[f'N_{space}_rmse']=mm.rmse;row[f'N_{space}_bias']=mm.bias
            for scenario in SCENARIOS:
                oo=o.loc[(o.horizon==h)&(o.scenario==scenario)&(o.reference=='B_current')].iloc[0]
                row['current_OSI_rmse']=oo.baseline_rmse
                row[f'{scenario}_OSI_rmse']=oo.rmse;row[f'{scenario}_change_pct']=oo.rmse_change_pct
            summary.append(row)
    allc=pd.concat(counties,ignore_index=True)
    frame(OUT/'summary/four_horizon_summary.csv',summary)
    frame(OUT/'summary/all_oracle_metrics.csv',pd.concat(oracle,ignore_index=True))
    frame(OUT/'summary/all_bootstrap_intervals.csv',pd.concat(boot,ignore_index=True))
    frame(OUT/'summary/county_priority_all_splits.csv',allc)
    consistency=[]
    for (h,f),g in allc.groupby(['horizon','fipsCode']):
        consistency.append(dict(horizon=h,fipsCode=f,countyName=g.countyName.iloc[0],stateAbbr=g.stateAbbr.iloc[0],
            splits=len(g),oracle_improves_splits=int((g.oracle_N_sse_reduction>0).sum()),
            oracle_worsens_splits=int((g.oracle_N_sse_reduction<0).sum()),
            mean_N_weighted_sse=g.N_weighted_sse.mean(),mean_OSI_sse=g.OSI_sse.mean(),
            mean_oracle_N_sse_reduction=g.oracle_N_sse_reduction.mean()))
    frame(OUT/'summary/split_consistency.csv',consistency)
