"""Independent geometry, raw time semantics and bitwise feature reconstruction."""
import math
from neighbor_protocol import *


def raw_inputs():
    pieces=[]
    for split in ('Train','Test'):
        cols=['fipsCode','timestamp_et','P_t','N_t','D_t','R_t','gust']+(['osi'] if split=='Train' else [])
        d=pd.read_csv(ROOT/f'data/DM_{split}.csv',usecols=cols,dtype={'fipsCode':str},float_precision='round_trip')
        d['timestamp_et']=pd.to_datetime(d.timestamp_et)
        d['hour_idx']=((d.timestamp_et-pd.Timestamp('2026-03-11'))/pd.Timedelta(hours=1)).astype(int)
        if split=='Test':
            obs=d.hour_idx<72;d['osi']=np.nan;q=d.loc[obs]
            d.loc[obs,'osi']=np.maximum(.4*q.P_t+.35*q.N_t+.25*q.D_t-.1*q.R_t,0).round(4)
        pieces.append(d)
    return pd.concat(pieces,ignore_index=True)


def raw_view(raw,h):
    past=raw.loc[raw.hour_idx<72,['fipsCode','hour_idx','P_t','N_t','D_t','R_t','osi']]
    weather=raw[['fipsCode','hour_idx','gust']];records=[]
    for f,g in past.groupby('fipsCode',sort=True):
        g=g.sort_values('hour_idx');assert g.hour_idx.tolist()==list(range(72))
        w=weather[weather.fipsCode==f].sort_values('hour_idx');assert w.hour_idx.tolist()==list(range(216))
        wind=w.gust.to_numpy();last=g.iloc[-1];slope=np.polyfit(np.arange(6),g.osi.iloc[-6:],1)[0]
        for t in range(72,216):
            records.append([f,pd.Timestamp('2026-03-11')+pd.Timedelta(hours=t),t,last.P_t,last.N_t,last.D_t,last.R_t,slope,
                wind[t],wind[t+h] if t+h<=215 else np.nan,np.max(wind[t:min(t+7,216)]),np.mean(wind[t:min(t+7,216)])])
    return pd.DataFrame(records,columns=VIEW_KEYS+whitelist(h)).sort_values(['hour_idx','fipsCode']).reset_index(drop=True)


def geometry(points,neighbors):
    coords=points.set_index('fipsCode')[['latitude','longitude']].to_dict('index');rows=[]
    for f in sorted(coords):
        lat,lon=map(math.radians,[coords[f]['latitude'],coords[f]['longitude']]);dists=[]
        for g in sorted(coords):
            if g==f:continue
            lt,ln=map(math.radians,[coords[g]['latitude'],coords[g]['longitude']])
            a=math.sin((lat-lt)/2)**2+math.cos(lat)*math.cos(lt)*math.sin((lon-ln)/2)**2
            dists.append((12742.0176*math.asin(math.sqrt(max(0.,min(1.,a)))),g))
        for rank,(dist,g) in enumerate(sorted(dists)[:8],1):rows.append([f,g,rank,dist,.125])
    expected=pd.DataFrame(rows,columns=neighbors.columns)
    n=neighbors.sort_values(['target_fips','rank']).reset_index(drop=True)
    pd.testing.assert_frame_equal(n.drop(columns='distance_km'),expected.drop(columns='distance_km'),check_dtype=False)
    np.testing.assert_allclose(n.distance_km,expected.distance_km,atol=1e-10,rtol=0)


def aggregate_scalar(view,neighbors,h):
    bycounty={f:g.sort_values('hour_idx').set_index('hour_idx') for f,g in view.groupby('fipsCode')};frames=[]
    for f,g in neighbors.groupby('target_fips',sort=True):
        codes=g.sort_values('rank').neighbor_fips.tolist();columns={}
        for name in whitelist(h):
            total=np.zeros(144);maximum=np.full(144,-np.inf)
            for nb in codes:
                v=bycounty[nb][name].to_numpy();total=total+v;maximum=np.maximum(maximum,v)
            columns['nbr8_mean_'+name]=total/8;columns['nbr8_max_'+name]=maximum
        columns['self_minus_nbr8_mean_last_P_t']=bycounty[f].last_P_t.to_numpy()-columns['nbr8_mean_last_P_t']
        columns[f'self_minus_nbr8_mean_gust_at_t{h}h']=bycounty[f][f'gust_at_t{h}h'].to_numpy()-columns[f'nbr8_mean_gust_at_t{h}h']
        d=pd.DataFrame(columns);d['fipsCode']=f;d['hour_idx']=np.arange(72,216);frames.append(d)
    return pd.concat(frames,ignore_index=True).set_index(['fipsCode','hour_idx'])


def verify_features(data,causality=False):
    neighbors=pd.read_csv(FEATURES/'neighbors_k8.csv',dtype={'target_fips':str,'neighbor_fips':str},float_precision='round_trip')
    points=pd.read_csv(FEATURES/'coordinates.csv',dtype={'fipsCode':str},float_precision='round_trip');geometry(points,neighbors)
    raw=raw_inputs();rebuilt={};max_raw_difference=0.
    for h in (24,):
        view=pd.read_parquet(FEATURES/f'h{h:02d}_allowed_view.parquet')
        actual=raw_view(raw,h)
        pd.testing.assert_frame_equal(view[VIEW_KEYS],actual[VIEW_KEYS],check_dtype=False)
        np.testing.assert_allclose(view[whitelist(h)],actual[whitelist(h)],atol=1e-12,rtol=0,equal_nan=True)
        max_raw_difference=max(max_raw_difference,float(np.nanmax(abs(view[whitelist(h)].to_numpy()-actual[whitelist(h)].to_numpy()))))
        independent=aggregate_scalar(view,neighbors,h)
        for side in ('train','test'):
            meta=getattr(data,'meta_'+side);base=getattr(data,'X_'+side)
            keys=pd.MultiIndex.from_arrays([meta.fipsCode,meta.hour_idx]);extra=independent.loc[keys,added_columns(h)].reset_index(drop=True)
            x=pd.concat([base.reset_index(drop=True),extra],axis=1);saved=pd.read_parquet(FEATURES/f'h{h:02d}_{side}.parquet')
            pd.testing.assert_frame_equal(saved,x,check_exact=True)
            for col in added_columns(h):
                missing=meta.hour_idx.to_numpy()+h>215 if f'gust_at_t{h}h' in col else np.zeros(len(meta),bool)
                np.testing.assert_array_equal(saved[col].isna(),missing)
            rebuilt[h,side]=x
        dep=pd.read_parquet(FEATURES/f'h{h:02d}_dependencies.parquet')
        assert len(dep)==347904 and not dep.duplicated(['fipsCode','hour_idx','neighbor_fips']).any()
        assert dep.history_end.eq(71).all() and dep.weight.eq(.125).all() and dep.fipsCode.eq(dep.target_fips).all()
        np.testing.assert_array_equal(dep.weather_target_hour,dep.hour_idx+h)
        np.testing.assert_array_equal(dep.target_weather_available,dep.hour_idx+h<=215)
        np.testing.assert_array_equal(dep.weather_window_end,np.minimum(dep.hour_idx+6,215))
        np.testing.assert_array_equal(dep.weather_window_start,dep.hour_idx)
        edges=neighbors.set_index(['target_fips','rank']).neighbor_fips
        np.testing.assert_array_equal(dep.neighbor_fips,edges.reindex(pd.MultiIndex.from_frame(dep[['target_fips','rank']])))
        if causality:
            altered=raw.copy();altered.loc[altered.hour_idx>=72,['P_t','N_t','D_t','R_t','osi']]=987654.321
            pd.testing.assert_frame_equal(raw_view(altered,h),actual,check_exact=True)
            from neighbor_features import aggregate
            poison=view.copy();poison['future_target']=987654.321
            try:aggregate(poison,neighbors,h)
            except AssertionError:pass
            else:raise AssertionError('Feature builder accepted forbidden column')
            pulse=view.copy();nb=neighbors.neighbor_fips.iloc[0]
            hit=(pulse.fipsCode==nb)&(pulse.hour_idx==72);pulse.loc[hit,f'gust_at_t{h}h']+=7
            positive=aggregate(pulse,neighbors,h)
            normal=aggregate(view,neighbors,h)
            assert not np.array_equal(positive[added_columns(h)].to_numpy(),normal[added_columns(h)].to_numpy(),equal_nan=True)
    return dict(status='PASS',horizons=[24],nodes=302,edges=2416,feature_count=183,
        raw_view_max_abs_difference=max_raw_difference,augmented_features_bitwise_exact=True,
        independent_geometry=True,rank_order_aggregation=True,lineage_rows=347904,causality_checks=causality),rebuilt
