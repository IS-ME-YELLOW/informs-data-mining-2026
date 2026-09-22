"""Target-relative gust-only features. No outage labels are accepted by the builder."""
import math
from weather_core import *

def weather_input(split):
    d=pd.read_csv(ROOT/f'data/DM_{split.title()}.csv',usecols=['fipsCode','timestamp_et','gust'],dtype={'fipsCode':str},float_precision='round_trip')
    d['hour_idx']=((pd.to_datetime(d.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
    d=d[['fipsCode','hour_idx','gust']].sort_values(['fipsCode','hour_idx']).reset_index(drop=True)
    assert np.isfinite(d.gust).all() and not d.duplicated(['fipsCode','hour_idx']).any()
    assert d.groupby('fipsCode').size().eq(216).all();return d

def window_features(gust):
    gust=[float(v) for v in gust];assert len(gust)==48 and all(math.isfinite(v) for v in gust)
    excess=[max(v-30.,0.) for v in gust];high=np.array([v>30 for v in gust]);ages=list(range(47,-1,-1))
    total=math.fsum(excess);decay=[]
    for tau in [6,24]:
        weights=[math.exp(-a/tau) for a in ages];decay.append(math.fsum(e*w for e,w in zip(excess,weights))/math.fsum(weights))
    starts=np.flatnonzero(high&~np.r_[False,high[:-1]]);ends=np.flatnonzero(high&~np.r_[high[1:],False])
    gaps=starts[1:]-ends[:-1]-1
    return decay+[float(47-np.flatnonzero(high)[-1]) if high.any() else 49.,float(high.any()),float(len(starts)),float(gaps.max()) if len(gaps) else 0.,
        math.fsum(a*e for a,e in zip(ages,excess))/total if total else 49.,math.fsum(excess[-6:])/total if total else 0.]

def build_new(weather,meta):
    if set(weather)!=set(['fipsCode','hour_idx','gust']):raise ValueError('Only county/hour/gust allowed')
    if set(meta)!=set(['fipsCode','timestamp_et','hour_idx']):raise ValueError('Only forecast keys allowed')
    assert not meta.duplicated(['fipsCode','hour_idx']).any()
    lookup={f:g.sort_values('hour_idx').gust.to_numpy() for f,g in weather.groupby('fipsCode')}
    rows=np.full((len(meta),8),np.nan)
    for f,indices in meta.groupby('fipsCode',sort=False).groups.items():
        series=lookup[f];assert len(series)==216
        for idx in indices:
            s=int(meta.loc[idx,'hour_idx'])+24
            if s<=215:rows[idx]=window_features(series[s-47:s+1])
    return pd.DataFrame(rows,columns=NEW_COLUMNS)

def build(ctx):
    for split,meta,base in [('train',ctx.data.meta_train,ctx.X0),('test',ctx.data.meta_test,ctx.Xt0)]:
        keys=meta[['fipsCode','timestamp_et','hour_idx']].copy();new=build_new(weather_input(split),keys)
        full=pd.concat([base.reset_index(drop=True),new],axis=1)
        valid=keys.hour_idx+24<=215;assert np.isfinite(new[valid]).all().all() and new[~valid].isna().all().all()
        for name,data in [(f'{split}_191.parquet',full),(f'{split}_new8.parquet',new),(f'{split}_meta.parquet',keys)]:
            path=OUT/'features'/name
            if path.exists():pd.testing.assert_frame_equal(pd.read_parquet(path),data,check_exact=True)
            else:frame(path,data)
    files={p.name:sha(p) for p in sorted((OUT/'features').glob('*.parquet'))}
    manifest=dict(protocol='target_weather_process_features_v1',source_hashes={str(ROOT/f'data/DM_{s.title()}.csv'):sha(ROOT/f'data/DM_{s.title()}.csv') for s in ['train','test']},
        new_columns=NEW_COLUMNS,full_columns=list(ctx.X0)+NEW_COLUMNS,window_points=48,target_horizon=24,threshold=30,age_missing=49,
        numerical_method='float64 math.exp chronological math.fsum',files=files,builder_sha256=sha(__file__))
    path=OUT/'features/feature_manifest.json'
    if path.exists():assert read(path)==manifest
    else:save(path,manifest)
    print('191-column feature bundle built',flush=True)
