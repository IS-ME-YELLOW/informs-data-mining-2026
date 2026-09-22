"""Independent raw-keyed scalar reconstruction of the eight process features."""
from pathlib import Path
import math,json,hashlib
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
COLS=['exposure_decay_6h','exposure_decay_24h','hours_since_last_high_gust','has_high_gust_48h','high_gust_episode_count_48h','max_lull_between_episodes_48h','exposure_weighted_age_48h','recent6_exposure_fraction']

def scalar(values):
    values=list(map(float,values));events=[j for j,v in enumerate(values) if v>30.]
    excess=[v-30. if v>30. else 0. for v in values]
    runs=0;previous=False;last_end=None;max_gap=0
    for j,v in enumerate(values):
        current=v>30.
        if current and not previous:
            runs+=1
            if last_end is not None:max_gap=max(max_gap,j-last_end-1)
        if previous and not current:last_end=j-1
        previous=current
    out=[]
    for time_constant in [6,24]:
        numer=[];denom=[]
        for j in range(48):
            w=math.exp(-(47-j)/time_constant);denom.append(w);numer.append(excess[j]*w)
        out.append(math.fsum(numer)/math.fsum(denom))
    total=math.fsum(excess)
    return out+[47-events[-1] if events else 49,int(bool(events)),runs,max_gap,
        math.fsum((47-j)*excess[j] for j in range(48))/total if total>0 else 49,
        math.fsum(excess[j] for j in range(42,48))/total if total>0 else 0]

def rebuild(split):
    meta=pd.read_parquet(OUT/f'features/{split}_meta.parquet');raw=pd.read_csv(ROOT/f'data/DM_{split.title()}.csv',usecols=['fipsCode','timestamp_et','gust'],dtype={'fipsCode':str},float_precision='round_trip')
    raw['hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
    assert not raw.duplicated(['fipsCode','hour']).any()
    county_values={f:dict(zip(g.hour,g.gust)) for f,g in raw.groupby('fipsCode')}
    data=np.full((len(meta),8),np.nan)
    for i,row in enumerate(meta.itertuples()):
        target=int(row.hour_idx)+24
        if target<=215:
            values=[county_values[str(row.fipsCode)][hour] for hour in range(target-47,target+1)]
            data[i]=scalar(values)
    old=pd.read_parquet(ROOT/f'versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/features/v1/h24_{split}.parquet')
    return pd.concat([old,pd.DataFrame(data,columns=COLS)],axis=1)

def verify_bundle():
    result={};maximum=0.
    for split in ['train','test']:
        new=rebuild(split);saved=pd.read_parquet(OUT/f'features/{split}_191.parquet')
        assert list(new)==list(saved) and new.shape==saved.shape
        np.testing.assert_allclose(new.to_numpy(),saved.to_numpy(),atol=1e-12,rtol=0,equal_nan=True)
        a=new[COLS].to_numpy();b=saved[COLS].to_numpy();maximum=max(maximum,float(np.nanmax(abs(a-b))))
        result[split]=dict(rows=len(new),feature_columns=len(new.columns),valid_feature_rows=int(np.isfinite(a).all(axis=1).sum()))
    return dict(status='PASS',splits=result,max_absolute_feature_difference=maximum,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
