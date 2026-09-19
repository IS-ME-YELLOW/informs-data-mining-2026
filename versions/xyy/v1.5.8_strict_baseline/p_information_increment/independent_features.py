"""Scalar geographic/aggregation checks and raw-data reconstruction of legal inputs."""
import importlib.util
import math
import numpy as np
import pandas as pd
from information_protocol import HERE,CONFIG,ROOT
from build_information_features import KEYS,HISTORY,WEATHER,WHITELIST,DEM,DEMCOLS,NBRCOLS


def independent_neighbors(nodes):
    spec=importlib.util.spec_from_file_location('geo_snapshot_reader',HERE/'reference/inputs/spatial_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    points=module.read_dbf_points(HERE/'reference/inputs/c_16ap26.dbf')
    result=[]
    for fi in sorted(nodes):
        lat1,lon1=map(math.radians,points[fi]);ranked=[]
        for fj in sorted(nodes):
            if fj==fi:continue
            lat2,lon2=map(math.radians,points[fj])
            a=math.sin((lat1-lat2)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon1-lon2)/2)**2
            distance=12742.0176*math.asin(math.sqrt(max(0.,min(1.,a))))
            ranked.append((distance,fj))
        for rank,(d,fj) in enumerate(sorted(ranked)[:8],1):result.append((fi,fj,rank,d,.125))
    return pd.DataFrame(result,columns=['target_fips','neighbor_fips','rank','distance_km','weight'])


def aggregate_independently(view,neighbors):
    lookup={f:group.sort_values('hour_idx')[WHITELIST].to_numpy(dtype=float) for f,group in view.groupby('fipsCode')}
    result=[]
    for f,rows in neighbors.groupby('target_fips',sort=True):
        codes=rows.sort_values('rank').neighbor_fips.tolist();assert len(codes)==8 and f not in codes
        columns={}
        for q,c in enumerate(WHITELIST):
            # Separate county-major implementation, not the production tensor indexing.
            mat=np.column_stack([lookup[neighbor][:,q] for neighbor in codes])
            columns['nbr8_mean_'+c]=np.array([np.mean(list(row)) for row in mat])
            columns['nbr8_max_'+c]=np.array([np.max(list(row)) for row in mat])
        columns['self_minus_nbr8_mean_last_P_t']=lookup[f][:,0]-columns['nbr8_mean_last_P_t']
        columns['self_minus_nbr8_mean_gust_at_t1h']=lookup[f][:,6]-columns['nbr8_mean_gust_at_t1h']
        frame=pd.DataFrame(columns);frame['fipsCode']=f;frame['hour_idx']=np.arange(72,216)
        result.append(frame)
    return pd.concat(result,ignore_index=True).set_index(['fipsCode','hour_idx'])


def read_raw():
    columns=['fipsCode','timestamp_et','P_t','N_t','D_t','R_t','osi','gust']
    frames=[]
    for side in ('Train','Test'):
        usecols=columns if side=='Train' else [c for c in columns if c!='osi']
        frame=pd.read_csv(ROOT/f'data/DM_{side}.csv',usecols=usecols,dtype={'fipsCode':str},float_precision='round_trip')
        if side=='Test':
            # DM_Test has no osi column; reproduce the frozen cache's observed-only
            # four-decimal OSI, using only the official past P/N/D/R.
            obs=pd.to_datetime(frame.timestamp_et)<pd.Timestamp('2026-03-14')
            frame['osi']=np.nan;past=frame.loc[obs]
            frame.loc[obs,'osi']=np.maximum(.4*past.P_t+.35*past.N_t+.25*past.D_t-.1*past.R_t,0).round(4)
        frames.append(frame)
    raw=pd.concat(frames,ignore_index=True);raw['timestamp_et']=pd.to_datetime(raw.timestamp_et)
    raw['hour_idx']=((raw.timestamp_et-pd.Timestamp('2026-03-11'))/pd.Timedelta(hours=1)).astype(int)
    return raw


def allowed_from_raw(history,weather):
    # Stop before any future outage values are inspected, even when passed a whole file.
    past=history.loc[history.hour_idx.between(0,71),['fipsCode','hour_idx','P_t','N_t','D_t','R_t','osi']]
    wind=weather[['fipsCode','hour_idx','gust']]
    result=[]
    for f,hist in past.groupby('fipsCode',sort=True):
        hist=hist.sort_values('hour_idx');assert hist.hour_idx.tolist()==list(range(72))
        w=wind.loc[wind.fipsCode==f].sort_values('hour_idx');assert w.hour_idx.tolist()==list(range(216))
        gust=w.gust.to_numpy();last=hist.iloc[-1]
        slope=float(np.polyfit(np.arange(6),hist.osi.iloc[-6:].to_numpy(),1)[0])
        rows=[]
        for t in range(72,216):
            rows.append((f,pd.Timestamp('2026-03-11')+pd.Timedelta(hours=t),t,
                last.P_t,last.N_t,last.D_t,last.R_t,slope,gust[t],gust[t+1] if t<215 else np.nan,
                np.max(gust[t:min(t+7,216)]),np.mean(gust[t:min(t+7,216)])))
        result.append(pd.DataFrame(rows,columns=KEYS+WHITELIST))
    return pd.concat(result,ignore_index=True).sort_values(['hour_idx','fipsCode']).reset_index(drop=True)


def verify_features(ctx,return_features=False):
    folder=HERE/'features/v1';view=pd.read_parquet(folder/'allowed_history_weather.parquet')
    nodes=sorted(view.fipsCode.unique());neighbors=pd.read_csv(folder/'neighbors_k8.csv',dtype={'target_fips':str,'neighbor_fips':str},float_precision='round_trip')
    independent=independent_neighbors(nodes)
    pd.testing.assert_frame_equal(neighbors.drop(columns='distance_km'),independent.drop(columns='distance_km'))
    np.testing.assert_allclose(neighbors.distance_km,independent.distance_km,rtol=0,atol=1e-10)
    raw=read_raw();rv=allowed_from_raw(raw,raw)
    pd.testing.assert_frame_equal(view[KEYS],rv[KEYS],check_dtype=False)
    np.testing.assert_allclose(view[WHITELIST],rv[WHITELIST],rtol=0,atol=1e-12,equal_nan=True)
    agg=aggregate_independently(view,independent)
    terrain=pd.read_csv(HERE/'reference/inputs/county_terrain.csv',dtype={'fipsCode':str},float_precision='round_trip').set_index('fipsCode')
    rebuilt={'F1':{},'F2':{}}
    for side in ('train','test'):
        meta=getattr(ctx.data,f'meta_{side}').copy();meta['fipsCode']=meta.fipsCode.astype(str).str.zfill(5)
        saved_meta=pd.read_parquet(folder/f'meta_{side}.parquet')
        pd.testing.assert_frame_equal(saved_meta,meta[KEYS],check_dtype=False)
        keys=pd.MultiIndex.from_arrays([meta.fipsCode,meta.hour_idx])
        for case,columns in [('F1',DEMCOLS),('F2',NBRCOLS)]:
            saved=ctx.feature_bundle['features'][case][side];original=getattr(ctx.data,f'X_{side}')
            pd.testing.assert_frame_equal(saved[list(original)],original)
            expected=terrain.loc[meta.fipsCode,DEM].to_numpy() if case=='F1' else agg.loc[keys,NBRCOLS].to_numpy()
            np.testing.assert_allclose(saved[columns],expected,rtol=0,atol=1e-10,equal_nan=True)
            rebuilt[case][side]=pd.concat([original.reset_index(drop=True),pd.DataFrame(expected,columns=columns)],axis=1)
            for c in columns:
                nan=(meta.hour_idx==215).to_numpy() if 'gust_at_t1h' in c else np.zeros(len(meta),bool)
                np.testing.assert_array_equal(saved[c].isna(),nan)
    deps=pd.read_parquet(folder/'neighbor_input_dependencies.parquet')
    assert len(deps)==347904 and not deps.duplicated(['fipsCode','hour_idx','neighbor_fips']).any()
    assert deps.history_start_hour.eq(0).all() and deps.history_end_hour.eq(71).all() and deps.last_component_hour.eq(71).all()
    assert deps.trend_start_hour.eq(66).all() and deps.trend_end_hour.eq(71).all()
    assert deps.fipsCode.eq(deps.target_fips).all() and deps.weight.eq(.125).all()
    np.testing.assert_array_equal(deps.weather_origin_hour,deps.hour_idx)
    np.testing.assert_array_equal(deps.weather_target_hour,np.where(deps.hour_idx<215,deps.hour_idx+1,np.nan))
    np.testing.assert_array_equal(deps.weather_window_start,deps.hour_idx)
    np.testing.assert_array_equal(deps.weather_window_end,np.minimum(deps.hour_idx+6,215))
    edge=independent.set_index(['target_fips','rank']).neighbor_fips
    np.testing.assert_array_equal(deps.neighbor_fips,edge.reindex(pd.MultiIndex.from_frame(deps[['target_fips','rank']])))
    report={'status':'PASS','independent_scalar_neighbors':True,'independent_neighbor_aggregates':True,
        'raw_history_and_weather_reconstruction':True,'history_cutoff':71,'nodes':302,'edges':2416,
        'lineage_rows':347904,'feature_counts':{'F1':170,'F2':183},'feature_tolerance':1e-10}
    return (report,rebuilt) if return_features else report
