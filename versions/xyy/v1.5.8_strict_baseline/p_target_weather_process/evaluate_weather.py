"""Fixed W1-minus-W0 full-county, temporal and process-group evaluations."""
import lightgbm as lgb
from weather_core import *

def metric(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);hit=np.isfinite(y);assert np.isfinite(p[hit]).all() and not np.isinf(y).any()
    e=p[hit]-y[hit];n=len(e)
    return dict(n=n,rmse=float(np.sqrt(np.mean(e**2))) if n else np.nan,sse=float(np.sum(e**2)),mae=float(np.mean(abs(e))) if n else np.nan,bias=float(np.mean(e)) if n else np.nan)
def compare(y,b,p):
    a,c=metric(y,b),metric(y,p)
    return dict(**c,baseline_rmse=a['rmse'],rmse_delta=c['rmse']-a['rmse'],rmse_change_pct=100*(c['rmse']/a['rmse']-1) if a['rmse']>0 else np.nan,sse_reduction=a['sse']-c['sse'])
def bootstrap(fips,y,b,p):
    hit=np.isfinite(y)
    if np.array_equal(b[hit],p[hit]):return dict(ci_low=0.,ci_high=0.,replicates=0)
    codes=pd.Categorical(fips,categories=sorted(set(fips))).codes
    n=np.bincount(codes[hit],minlength=239);a=np.bincount(codes[hit],weights=(b[hit]-y[hit])**2,minlength=239);c=np.bincount(codes[hit],weights=(p[hit]-y[hit])**2,minlength=239)
    draw=np.random.default_rng(20260910).integers(0,239,(2000,239));nn=n[draw].sum(axis=1);assert (nn>0).all()
    difference=np.sqrt(c[draw].sum(axis=1)/nn)-np.sqrt(a[draw].sum(axis=1)/nn);lo,hi=np.quantile(difference,[.025,.975])
    return dict(ci_low=float(lo),ci_high=float(hi),replicates=2000)

def evaluate():
    if (RUN/'EVALUATION_COMPLETE').exists():raise RuntimeError('Completed evaluation is immutable')
    ctx=initialize(snapshot(load_context()));m=read(RUN/'MODEL_CV_COMPLETE');assert m['identity_hash']==ctx.identity_hash
    for p,h in m['files'].items():assert sha(RUN/p)==h,p
    new=norm(pd.read_parquet(RUN/'P24_oof_predictions.parquet'));pd.testing.assert_frame_equal(new[KEYS],ctx.meta[KEYS],check_dtype=False)
    pp={h:{c:v.copy() for c,v in d.items()} for h,d in ctx.parts.items()};pp[H24]['P_t']=np.clip(new.W1_raw_P24.to_numpy(),0,1)
    np.testing.assert_array_equal(pp[H24]['P_t'],new.W1_P24)
    controls,aux=build_controls(ctx.meta,ctx.direct,pp)
    parts={'W0':ctx.parts,'W1':pp};cc={'W0':ctx.controls,'W1':controls};aa={'W0':ctx.aux,'W1':aux}
    pred=ctx.meta[KEYS].copy();comp=pred.copy();pred['identity_hash']=ctx.identity_hash;comp['identity_hash']=ctx.identity_hash
    pframes=[];primary=[];allm=[];comparisons=[];groups=[];counties=[];folds=[];days=[];process=[];boot=[];pmetrics=[];rowchanges=[];decomposition=[]
    fips=ctx.meta.fipsCode.to_numpy();hours=ctx.meta.hour_idx.to_numpy();membership=ctx.membership.set_index(['horizon','target_day','fipsCode']).test_like
    group_definitions=dict(all='Full 239-county valid rows',pre96='Valid target hours below96; unchanged',from96='Valid target hours96-215',
        test_like='Frozen original context test-to-train top5 union per horizon/target day',not_test_like='Complement of frozen test_like',
        all_except_Forest='Diagnostic only; exclude FIPS42053',test_like_except_Forest='Diagnostic only; test_like without FIPS42053')
    process_definitions=dict(no_high='has_high_gust_48h=0',active_high='has_high=1 and age=0',post_high_1_6='has_high=1 and 1<=age<=6',
        post_high_7plus='has_high=1 and 7<=age<=47',multiple_segments='episode_count>=2; overlapping group')
    save(RUN/'group_definitions.json',dict(general=group_definitions,process_24h=process_definitions))
    official=pd.read_csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str},float_precision='round_trip',usecols=['fipsCode','timestamp_et','gust','P_t','N_t','D_t','R_t','osi'])
    official['target_hour']=((pd.to_datetime(official.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int);official=official.set_index(['fipsCode','target_hour'])
    for h in HORIZONS:
        hh=HORIZON_HOURS[h];s=hours+hh;valid=expected_mask(ctx.meta,h);y=ctx.data.y_train[h].to_numpy();yp=ctx.data.component_targets[component_target_name('P_t',h)].to_numpy()
        pred['true_'+h]=y;pred['scoreable_'+h]=valid
        for case in ['W0','W1']:
            for c in COMPONENTS:
                t=component_target_name(c,h);comp[f'pred_{case}_{t}']=parts[case][h][c]
                comp[f'source_{case}_{t}']=new.model_id.to_numpy() if case=='W1' and (h,c)==(H24,'P_t') else ctx.reference_parts['source_NB_P24_'+t].to_numpy()
                if case=='W0' or (h,c)!=(H24,'P_t'):np.testing.assert_array_equal(parts[case][h][c],ctx.parts[h][c])
            for mode in cc[case]:
                pred[f'pred_{case}_{mode}_{h}']=cc[case][mode][h]
                allm.append(dict(case=case,horizon=hh,mode=mode,**metric(y,cc[case][mode][h])))
            primary.append(dict(case=case,horizon=hh,**metric(y,cc[case]['v18_rule'][h])))
            part=ctx.meta[KEYS].copy();part['case']=case;part['horizon']=hh;part['target_hour']=s;part['true_P']=yp;part['P_source']=parts[case][h]['P_t'];part['P_aligned']=aa[case]['aligned_components'][h]['P_t'];pframes.append(part)
        b=ctx.controls['v18_rule'][h];p=controls['v18_rule'][h]
        unchanged=valid&((s<96)|(hh==48));np.testing.assert_array_equal(p[unchanged],b[unchanged])
        row=dict(case='W1',reference='W0',horizon=hh,**compare(y,b,p));comparisons.append(row)
        idx=pd.MultiIndex.from_arrays([np.full(len(fips),hh),s//24,fips]);near=membership.reindex(idx,fill_value=False).to_numpy(bool)
        masks=dict(all=np.ones(len(y),bool),pre96=s<96,from96=s>=96,test_like=near,not_test_like=~near,all_except_Forest=fips!='42053',test_like_except_Forest=near&(fips!='42053'))
        for name,mask in masks.items():
            yy=np.where(mask,y,np.nan);r=dict(horizon=hh,group=name,**compare(yy,b,p));groups.append(r)
            if name in ['all','test_like']:boot.append(dict(**r,**bootstrap(fips,yy,b,p)))
        for f in sorted(set(fips)):
            mask=fips==f;counties.append(dict(horizon=hh,fipsCode=f,countyName=ctx.names.loc[f,'countyName'],stateAbbr=ctx.names.loc[f,'stateAbbr'],fold=int(ctx.meta.loc[mask,'fold'].iloc[0]),**compare(y[mask],b[mask],p[mask])))
        for fold in range(5):
            mask=ctx.meta.fold.to_numpy()==fold;folds.append(dict(horizon=hh,fold=fold,**compare(y[mask],b[mask],p[mask])))
        for day in sorted(set(s[valid]//24)):
            mask=s//24==day;days.append(dict(horizon=hh,target_day=int(day),start=max(72+hh,int(day)*24),end=min(215,int(day)*24+23),**compare(y[mask],b[mask],p[mask])))
        if hh==24:
            has=ctx.X.has_high_gust_48h.to_numpy();age=ctx.X.hours_since_last_high_gust.to_numpy();segments=ctx.X.high_gust_episode_count_48h.to_numpy()
            masks_proc=dict(no_high=has==0,active_high=(has==1)&(age==0),post_high_1_6=(has==1)&(age>=1)&(age<=6),post_high_7plus=(has==1)&(age>=7),multiple_segments=segments>=2)
            for name,mask in masks_proc.items():
                r=compare(np.where(mask,y,np.nan),b,p);hit=mask&valid
                process.append(dict(group=name,**r,truth_mean=float(y[hit].mean()),W0_mean=float(b[hit].mean()),W1_mean=float(p[hit].mean())))
        for space in ['source','aligned']:
            a=parts['W0'][h]['P_t'] if space=='source' else aa['W0']['aligned_components'][h]['P_t']
            c=parts['W1'][h]['P_t'] if space=='source' else aa['W1']['aligned_components'][h]['P_t']
            pmetrics.append(dict(horizon=hh,space=space,**compare(yp,a,c)))
        r=ctx.meta.loc[valid,KEYS].copy();r['horizon']=hh;r['target_hour']=s[valid];r['truth']=y[valid];r['W0']=b[valid];r['W1']=p[valid]
        r['sse_reduction']=(b[valid]-y[valid])**2-(p[valid]-y[valid])**2;r['test_like']=near[valid];rowchanges.append(r)
        # Exact P-only change decomposition on the deployed component space, including nonlinear postprocessing remainder.
        use_aligned=hh<=6
        pa=aa['W0']['aligned_components'][h]['P_t'] if use_aligned else np.clip(parts['W0'][h]['P_t'],0,1)
        pc=aa['W1']['aligned_components'][h]['P_t'] if use_aligned else np.clip(parts['W1'][h]['P_t'],0,1)
        p0=.4*(pa[valid]-yp[valid]);p1=.4*(pc[valid]-yp[valid]);others=np.zeros(valid.sum())
        target=official.reindex(pd.MultiIndex.from_arrays([fips[valid],s[valid]]))
        for c,w in [('N_t',.35),('D_t',.25),('R_t',-.1)]:
            value=aa['W0']['aligned_components'][h][c] if use_aligned else np.clip(parts['W0'][h][c],0,1)
            others+=w*(value[valid]-target[c].to_numpy())
        own=float(np.sum(p0**2-p1**2));cross=float(np.sum(2*(p0-p1)*others));net=float(np.sum((b[valid]-y[valid])**2-(p[valid]-y[valid])**2))
        decomposition.append(dict(horizon=hh,P_squared_reduction=own,P_other_cross_reduction=cross,postprocess_precision_remainder=net-own-cross,OSI_sse_reduction=net))
    frame(RUN/'control_predictions.parquet',pred);frame(RUN/'component_predictions.parquet',comp);frame(RUN/'P_component_predictions.parquet',pd.concat(pframes,ignore_index=True));frame(RUN/'row_changes.parquet',pd.concat(rowchanges,ignore_index=True))
    for name,rows in [('primary_scores',primary),('all_controls',allm),('comparisons',comparisons),('groups',groups),('county_metrics',counties),('fold_metrics',folds),('day_metrics',days),('process_groups_24h',process),('bootstrap',boot),('P_metrics',pmetrics),('P_change_decomposition',decomposition)]:frame(RUN/f'metrics/{name}.csv',rows)
    daily=[];traces=[]
    focus=['54015','54013','54007','39119','54087','54109','42053','39117','39115','18013','54041']
    s=hours+24;hit=ctx.valid;rawmap=official.reindex(pd.MultiIndex.from_arrays([fips[hit],s[hit]]))
    base=pd.DataFrame(dict(fipsCode=fips[hit],target_hour=s[hit],fold=ctx.meta.fold.to_numpy()[hit],true_P=ctx.y[hit],W0_P=ctx.parts[H24]['P_t'][hit],W1_P=pp[H24]['P_t'][hit],
        true_OSI=ctx.data.y_train[H24].to_numpy()[hit],W0_OSI=ctx.controls['v18_rule'][H24][hit],W1_OSI=controls['v18_rule'][H24][hit],gust=rawmap.gust.to_numpy()))
    for col in NEW_COLUMNS:base[col]=ctx.X.loc[hit,col].to_numpy()
    base['target_day']=base.target_hour//24;frame(RUN/'case_trajectories.parquet',base[base.fipsCode.isin(focus)])
    for (f,day),d in base.groupby(['fipsCode','target_day']):
        record=dict(fipsCode=f,countyName=ctx.names.loc[f,'countyName'],stateAbbr=ctx.names.loc[f,'stateAbbr'],target_day=int(day),n=len(d))
        for col in ['true_P','W0_P','W1_P','true_OSI','W0_OSI','W1_OSI']:
            record[col+'_mean']=float(d[col].mean());record[col+'_max']=float(d[col].max());record[col+'_peak_hour']=int(d.loc[d[col].idxmax(),'target_hour'])
        record.update(compare(d.true_OSI.to_numpy(),d.W0_OSI.to_numpy(),d.W1_OSI.to_numpy()));daily.append(record)
    frame(RUN/'metrics/county_day_24h.csv',daily)
    importance=[]
    for rec in read(RUN/'model_manifest.json'):
        if rec['spec']['inner_fold'] is not None:continue
        model=lgb.Booster(model_file=str(RUN/rec['model_path']))
        for name,gain,splits in zip(model.feature_name(),model.feature_importance('gain'),model.feature_importance('split')):
            importance.append(dict(outer_fold=rec['spec']['outer_fold'],feature=name,is_new=name in NEW_COLUMNS,gain=float(gain),split_count=int(splits)))
    frame(RUN/'metrics/feature_importance.csv',importance)
    for p,h in ctx.sources.items():assert sha(p)==h,p
    files={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and 'models' not in p.parts and p.name not in ['MODEL_CV_COMPLETE','EVALUATION_COMPLETE']}
    save(RUN/'EVALUATION_COMPLETE',dict(identity_hash=ctx.identity_hash,evaluation_code_sha256=sha(__file__),files=files))
    print(pd.DataFrame(comparisons)[['horizon','rmse','rmse_change_pct']].to_string(index=False),flush=True)

if __name__=='__main__':evaluate()
