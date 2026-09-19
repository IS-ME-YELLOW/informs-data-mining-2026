"""Target-free DEM and k=8 geographic-context features on the fixed 302 nodes."""
from pathlib import Path
from types import SimpleNamespace
import struct
import numpy as np
import pandas as pd
from information_protocol import (HERE,CONFIG,ROOT,output_path,write_json,write_frame,read_json,
    sha256_file,digest_object)

HISTORY=['last_P_t','last_N_t','last_D_t','last_R_t','osi_trend_last6h']
WEATHER=['gust_t','gust_at_t1h','gust_max_next_6h','gust_mean_next_6h']
WHITELIST=HISTORY+WEATHER
DEM=['elevation_mean_m','elevation_std_m','elevation_min_m','elevation_max_m','slope_mean_deg','slope_std_deg','terrain_ruggedness']
DEMCOLS=['dem_'+c for c in DEM]
NBRCOLS=[prefix+q for q in WHITELIST for prefix in ('nbr8_mean_','nbr8_max_')]+['self_minus_nbr8_mean_last_P_t','self_minus_nbr8_mean_gust_at_t1h']
KEYS=['fipsCode','timestamp_et','hour_idx']
FEATURE_DIR=HERE/'features/v1'


def feature_inputs(data):
    """Explicitly discard labels, folds, outcomes and all other context members."""
    return SimpleNamespace(X_train=data.X_train,X_test=data.X_test,
        meta_train=data.meta_train[KEYS].copy(),meta_test=data.meta_test[KEYS].copy(),
        loaded_hashes={k:v for k,v in data.loaded_hashes.items() if k in ('features_train','features_test','meta_train','meta_test','feature_names')})


def normalized_fips(series):
    text=pd.Series(series).astype(str)
    if not text.str.fullmatch(r'\d{1,5}').all():raise ValueError('FIPS must be nonmissing integer county codes')
    if (pd.to_numeric(text)<=0).any():raise ValueError('Invalid zero FIPS')
    return text.str.zfill(5)


def coordinates(path,nodes):
    raw=Path(path).read_bytes();n=struct.unpack_from('<I',raw,4)[0]
    head=struct.unpack_from('<H',raw,8)[0];width=struct.unpack_from('<H',raw,10)[0]
    fields={};offset=1
    for pos in range(32,head-1,32):
        name=raw[pos:pos+11].split(b'\x00',1)[0].decode('ascii').strip()
        if name:
            size=raw[pos+16];fields[name]=(offset,size);offset+=size
    def field(rec,key):
        start,size=fields[key];return rec[start:start+size].decode('ascii').strip()
    points={}
    for i in range(n):
        rec=raw[head+i*width:head+(i+1)*width]
        if not rec or rec[:1]==b'*':continue
        f=field(rec,'FIPS').zfill(5)
        if f not in nodes:continue
        if f in points:raise ValueError('Duplicate geographic FIPS')
        points[f]=(float(field(rec,'LAT')),float(field(rec,'LON')))
    if set(points)!=set(nodes):raise ValueError('Missing coordinates')
    result=pd.DataFrame([(f,*points[f]) for f in sorted(nodes)],columns=['fipsCode','latitude','longitude'])
    if not np.isfinite(result.iloc[:,1:].to_numpy()).all() or not result.latitude.between(-90,90).all() or not result.longitude.between(-180,180).all():
        raise ValueError('Invalid geographic coordinates')
    return result


def neighbor_table(points):
    if points.fipsCode.duplicated().any() or len(points)!=302:raise ValueError('Require exactly 302 unique nodes')
    points=points.sort_values('fipsCode').reset_index(drop=True)
    codes=points.fipsCode.to_numpy();rad=np.deg2rad(points[['latitude','longitude']].to_numpy(dtype=float))
    term=np.sin((rad[:,None,0]-rad[None,:,0])/2)**2+np.cos(rad[:,None,0])*np.cos(rad[None,:,0])*np.sin((rad[:,None,1]-rad[None,:,1])/2)**2
    d=2*6371.0088*np.arcsin(np.sqrt(np.clip(term,0,1)));np.fill_diagonal(d,np.inf)
    records=[]
    for i in range(len(codes)):
        nn=np.lexsort((codes,d[i]))[:8]
        for rank,j in enumerate(nn,1):records.append((codes[i],codes[j],rank,float(d[i,j]),.125))
    return pd.DataFrame(records,columns=['target_fips','neighbor_fips','rank','distance_km','weight'])


def allowed_view(data):
    if set(vars(data))!={'X_train','X_test','meta_train','meta_test','loaded_hashes'}:
        raise ValueError('Feature builder accepts only the explicit input-only context')
    frames=[]
    for side in ('train','test'):
        meta=getattr(data,f'meta_{side}')[KEYS].copy();meta['fipsCode']=normalized_fips(meta.fipsCode).to_numpy()
        x=getattr(data,f'X_{side}')
        frames.append(pd.concat([meta.reset_index(drop=True),x[WHITELIST].reset_index(drop=True)],axis=1))
    return validate_view(pd.concat(frames,ignore_index=True))


def validate_view(view):
    if list(view)!=KEYS+WHITELIST:raise ValueError('Only the explicit legal-information whitelist is accepted')
    view=view.copy();view['fipsCode']=normalized_fips(view.fipsCode).to_numpy()
    view['timestamp_et']=pd.to_datetime(view.timestamp_et)
    if len(view)!=43488 or view.fipsCode.nunique()!=302 or view.duplicated(['fipsCode','hour_idx']).any():raise ValueError('Invalid node/hour coverage')
    if not view.timestamp_et.eq(pd.Timestamp('2026-03-11')+pd.to_timedelta(view.hour_idx,unit='h')).all():raise ValueError('Time key mismatch')
    if not view.groupby('fipsCode').hour_idx.apply(lambda s:set(s)==set(range(72,216))).all():raise ValueError('Missing forecast hours')
    for col in WHITELIST:
        expected=(view.hour_idx==215).to_numpy() if col=='gust_at_t1h' else np.zeros(len(view),bool)
        if not np.array_equal(view[col].isna(),expected) or np.isinf(view[col]).any():raise ValueError('Unexpected missing legal input')
    if not (view.groupby('fipsCode')[HISTORY].nunique()==1).all().all():raise ValueError('Known history changed after cutoff')
    return view.sort_values(['hour_idx','fipsCode']).reset_index(drop=True)


def aggregate_neighbors(view,neighbors):
    view=validate_view(view);nodes=sorted(view.fipsCode.unique())
    if len(neighbors)!=2416 or neighbors.duplicated(['target_fips','neighbor_fips']).any():raise ValueError('Invalid neighbor edges')
    if set(neighbors.target_fips)!=set(nodes) or set(neighbors.neighbor_fips)-set(nodes):raise ValueError('Unknown neighbor node')
    if (neighbors.target_fips==neighbors.neighbor_fips).any() or not neighbors.weight.eq(.125).all():raise ValueError('Self edge or changed weights')
    neighbors=neighbors.sort_values(['target_fips','rank'])
    if not neighbors.groupby('target_fips')['rank'].apply(lambda s:s.tolist()==list(range(1,9))).all():raise ValueError('Wrong rank/count')
    node_index={f:i for i,f in enumerate(nodes)}
    indices=np.array([node_index[f] for f in neighbors.neighbor_fips]).reshape(302,8)
    values=view[WHITELIST].to_numpy(dtype=float).reshape(144,302,9)
    nv=values[:,indices,:];means=nv.mean(axis=2);maxima=nv.max(axis=2)
    out=np.stack([means,maxima],axis=-1).reshape(144,302,18)
    out=np.concatenate([out,(values[:,:,0]-means[:,:,0])[:,:,None],(values[:,:,6]-means[:,:,6])[:,:,None]],axis=2)
    return pd.concat([view[KEYS],pd.DataFrame(out.reshape(-1,20),columns=NBRCOLS)],axis=1)


def terrain_table(path,nodes):
    d=pd.read_csv(path,dtype={'fipsCode':str},float_precision='round_trip')
    d['fipsCode']=normalized_fips(d.fipsCode).to_numpy()
    if list(d)!=['fipsCode',*DEM] or d.fipsCode.duplicated().any() or set(d.fipsCode)!=set(nodes):raise ValueError('Terrain schema/coverage mismatch')
    if not np.isfinite(d[DEM].to_numpy()).all():raise ValueError('Nonfinite terrain')
    if not ((d.elevation_min_m<=d.elevation_mean_m)&(d.elevation_mean_m<=d.elevation_max_m)).all():raise ValueError('Terrain elevation order')
    if not (d[['elevation_std_m','slope_std_deg','terrain_ruggedness']]>=0).all().all() or not d.slope_mean_deg.between(0,90).all():raise ValueError('Terrain physical ranges')
    return d.sort_values('fipsCode').reset_index(drop=True)


def produce(data):
    view=allowed_view(data);nodes=sorted(view.fipsCode.unique());geo=CONFIG['geo_sources']
    points=coordinates(geo['c_16ap26.dbf']['snapshot'],nodes);neighbors=neighbor_table(points)
    terrain=terrain_table(geo['county_terrain.csv']['snapshot'],nodes)
    aggregate=aggregate_neighbors(view,neighbors).set_index(['fipsCode','hour_idx'])
    output={}
    for side in ('train','test'):
        meta=getattr(data,f'meta_{side}')[KEYS].copy();meta['fipsCode']=normalized_fips(meta.fipsCode).to_numpy()
        base=getattr(data,f'X_{side}')
        extra1=terrain.set_index('fipsCode').loc[meta.fipsCode,DEM].reset_index(drop=True);extra1.columns=DEMCOLS
        keys=pd.MultiIndex.from_arrays([meta.fipsCode,meta.hour_idx])
        extra2=aggregate.loc[keys,NBRCOLS].reset_index(drop=True)
        for case,extra in [('F1',extra1),('F2',extra2)]:
            full=pd.concat([base.reset_index(drop=True),extra],axis=1)
            assert full.shape[1]==CONFIG['feature_counts'][case]
            pd.testing.assert_frame_equal(full[list(base)],base)
            output[f'{case}_{side}.parquet']=full
        output[f'meta_{side}.parquet']=meta
    # Explicit compact lineage expands every forecast row to its eight input providers.
    deps=view[KEYS].merge(neighbors,left_on='fipsCode',right_on='target_fips',how='left',validate='many_to_many')
    deps['history_start_hour']=0;deps['history_end_hour']=71;deps['last_component_hour']=71
    deps['trend_start_hour']=66;deps['trend_end_hour']=71
    deps['weather_origin_hour']=deps.hour_idx;deps['weather_target_hour']=np.where(deps.hour_idx<215,deps.hour_idx+1,np.nan)
    deps['weather_window_start']=deps.hour_idx;deps['weather_window_end']=np.minimum(deps.hour_idx+6,215)
    assert len(deps)==347904
    output.update({'allowed_history_weather.parquet':view,'coordinates.csv':points,'neighbors_k8.csv':neighbors,
                   'terrain_snapshot.parquet':terrain,'neighbor_input_dependencies.parquet':deps})
    return output


def package_identity(data):
    return {'version':'information_features_v1','config_sha256':sha256_file(HERE/'experiment_config.json'),
        'builder_sha256':sha256_file(__file__),'inputs':{k:v for k,v in data.loaded_hashes.items() if k in ('features_train','features_test','meta_train','meta_test','feature_names')},
        'geo_sha256':{n:x['sha256'] for n,x in CONFIG['geo_sources'].items()},'whitelist':WHITELIST,
        'F1_added_columns':DEMCOLS,'F2_added_columns':NBRCOLS,'neighbors':CONFIG['neighbors']}


def build_features(data):
    expected=package_identity(data)
    if (FEATURE_DIR/'feature_manifest.json').exists():return load_features(data)
    if FEATURE_DIR.exists():raise FileExistsError('Incomplete feature package exists; do not silently replace it')
    frames=produce(data)
    for name,frame in frames.items():write_frame(output_path(FEATURE_DIR/name),frame)
    for case in ('F1','F2'):write_json(FEATURE_DIR/f'{case}_feature_names.json',list(frames[f'{case}_train.parquet']))
    files={str(p.relative_to(FEATURE_DIR)):sha256_file(p) for p in FEATURE_DIR.iterdir() if p.is_file()}
    manifest={'identity':expected,'identity_hash':digest_object(expected),'files':files}
    write_json(FEATURE_DIR/'feature_manifest.json',manifest)
    return load_features(data)


def load_features(data):
    m=read_json(FEATURE_DIR/'feature_manifest.json')
    if m['identity']!=package_identity(data) or m['identity_hash']!=digest_object(m['identity']):raise ValueError('Feature identity mismatch')
    for name,h in m['files'].items():
        path=(FEATURE_DIR/name).resolve()
        if not path.is_relative_to(FEATURE_DIR.resolve()) or sha256_file(path)!=h:raise ValueError('Changed feature artifact')
    features={}
    for case in ('F1','F2'):
        features[case]={}
        names=read_json(FEATURE_DIR/f'{case}_feature_names.json')
        for side in ('train','test'):
            frame=pd.read_parquet(FEATURE_DIR/f'{case}_{side}.parquet');base=getattr(data,f'X_{side}')
            assert list(frame)==names and len(frame)==len(base) and len(names)==CONFIG['feature_counts'][case]
            pd.testing.assert_frame_equal(frame[list(base)],base)
            features[case][side]=frame
    return {'manifest':m,'features':features}
