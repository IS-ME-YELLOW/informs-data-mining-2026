"""Outcome-free, horizon-specific frozen-eight-neighbor aggregation."""
from neighbor_protocol import *


def allowed_view(xtrain,mtrain,xtest,mtest,h):
    rows=[]
    for x,m in ((xtrain,mtrain),(xtest,mtest)):
        keys=m[VIEW_KEYS].copy();keys['fipsCode']=keys.fipsCode.astype(str).str.zfill(5)
        rows.append(pd.concat([keys.reset_index(drop=True),x[whitelist(h)].reset_index(drop=True)],axis=1))
    return validate_view(pd.concat(rows,ignore_index=True),h)


def validate_view(view,h):
    assert list(view)==VIEW_KEYS+whitelist(h),'Unexpected feature-builder field'
    assert len(view)==43488 and view.fipsCode.nunique()==302
    assert not view.duplicated(['fipsCode','hour_idx']).any()
    view=view.sort_values(['hour_idx','fipsCode']).reset_index(drop=True)
    assert view.groupby('fipsCode').hour_idx.apply(lambda x:x.tolist()==list(range(72,216))).all()
    assert (view.groupby('fipsCode')[HISTORY].nunique()==1).all().all()
    for col in whitelist(h):
        missing=view.hour_idx.to_numpy()+h>215 if col==f'gust_at_t{h}h' else np.zeros(len(view),bool)
        assert np.array_equal(view[col].isna(),missing) and not np.isinf(view[col]).any()
    return view


def aggregate(view,neighbors,h):
    view=validate_view(view,h);nodes=sorted(view.fipsCode.unique())
    assert len(neighbors)==2416 and not neighbors.duplicated(['target_fips','neighbor_fips']).any()
    assert neighbors.weight.eq(.125).all() and (neighbors.target_fips!=neighbors.neighbor_fips).all()
    neighbors=neighbors.sort_values(['target_fips','rank'])
    assert neighbors.groupby('target_fips')['rank'].apply(lambda x:x.tolist()==list(range(1,9))).all()
    index={f:i for i,f in enumerate(nodes)}
    ids=np.array([index[f] for f in neighbors.neighbor_fips]).reshape(302,8)
    values=view[whitelist(h)].to_numpy().reshape(144,302,9)
    total=np.zeros_like(values);maximum=np.full_like(values,-np.inf)
    for rank in range(8):
        v=values[:,ids[:,rank],:];total+=v;maximum=np.maximum(maximum,v)
    means=total/8
    extra=np.stack([means,maximum],axis=-1).reshape(144,302,18)
    extra=np.concatenate([extra,(values[:,:,0]-means[:,:,0])[:,:,None],(values[:,:,6]-means[:,:,6])[:,:,None]],axis=2)
    return pd.concat([view[VIEW_KEYS],pd.DataFrame(extra.reshape(-1,20),columns=added_columns(h))],axis=1)


def produce(data):
    neighbors=pd.read_csv(OLD_FEATURES/'neighbors_k8.csv',dtype={'target_fips':str,'neighbor_fips':str},float_precision='round_trip')
    points=pd.read_csv(OLD_FEATURES/'coordinates.csv',dtype={'fipsCode':str},float_precision='round_trip')
    output={'neighbors_k8.csv':neighbors,'coordinates.csv':points}
    for h in (1,6,24,48):
        view=allowed_view(data.X_train,data.meta_train,data.X_test,data.meta_test,h)
        extra=aggregate(view,neighbors,h).set_index(['fipsCode','hour_idx'])
        for side in ('train','test'):
            meta=getattr(data,'meta_'+side);base=getattr(data,'X_'+side)
            index=pd.MultiIndex.from_arrays([meta.fipsCode.astype(str).str.zfill(5),meta.hour_idx])
            add=extra.loc[index,added_columns(h)].reset_index(drop=True)
            full=pd.concat([base.reset_index(drop=True),add],axis=1)
            if h==1:
                old=pd.read_parquet(OLD_FEATURES/f'F2_{side}.parquet')
                pd.testing.assert_frame_equal(full,old,check_exact=True)
            else:
                assert len(full.columns)==183
                for col in added_columns(h):
                    expected=meta.hour_idx.to_numpy()+h>215 if f'gust_at_t{h}h' in col else np.zeros(len(meta),bool)
                    np.testing.assert_array_equal(full[col].isna(),expected)
                output[f'h{h:02d}_{side}.parquet']=full
        if h==1:continue
        output[f'h{h:02d}_allowed_view.parquet']=view
        dep=view[VIEW_KEYS].merge(neighbors,left_on='fipsCode',right_on='target_fips',validate='many_to_many')
        dep['history_end']=71;dep['weather_target_hour']=dep.hour_idx+h
        dep['target_weather_available']=dep.weather_target_hour<=215
        dep['weather_window_start']=dep.hour_idx;dep['weather_window_end']=np.minimum(dep.hour_idx+6,215)
        assert len(dep)==347904
        output[f'h{h:02d}_dependencies.parquet']=dep
    for side in ('train','test'):output[f'meta_{side}.parquet']=getattr(data,'meta_'+side)[VIEW_KEYS]
    return output


def build(ctx):
    """Re-use the immutable seed42 input-only package; never write FEATURES."""
    m=read_json(FEATURES/'feature_manifest.json')
    assert digest_object(m['identity'])==m['identity_hash']
    for path,digest in m['files'].items():assert sha256_file(FEATURES/path)==digest,path
    assert read_json(FEATURES/'h24_schema.json')==list(ctx.data.X_train)+added_columns(24)
