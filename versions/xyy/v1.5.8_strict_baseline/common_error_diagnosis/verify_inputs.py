"""Independent reconstruction of the allowed matching inputs and joined controls."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];BASE=OUT.parent
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)
def norm(d):d=d.copy();d['fipsCode']=d.fipsCode.astype(str).str.zfill(5);return d

def main():
    x=pd.read_parquet(OUT/'matching/inputs.parquet');blocks=json.loads((OUT/'matching/cache_blocks.json').read_text())
    raw=norm(pd.read_csv(ROOT/'data/DM_Train.csv',float_precision='round_trip'))
    raw['hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
    meta=norm(pd.read_parquet(ROOT/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet'))
    base=pd.read_parquet(ROOT/'versions/xyy/v1.5.6/features_train_v1.5.6.parquet')
    cfg=json.loads((OUT/'diagnostic_config.json').read_text())
    hist=base[cfg['history_columns']+['pop_density','pct_forest','n_utilities']].assign(fipsCode=meta.fipsCode.to_numpy()).groupby('fipsCode').first()
    cut=raw[raw.hour==71].set_index('fipsCode')
    for component in ['P','N','D','R']:close(hist['last_'+component+'_t'],cut.loc[hist.index,component+'_t'])
    count=0
    for h in [1,6,24,48]:
        cache=pd.read_parquet(BASE/'p_information_increment/features/v1/F2_train.parquet') if h==1 else (pd.read_parquet(BASE/'p_neighbor_horizon_increment/features/v1/h24_train.parquet') if h==24 else base)
        for day in range((72+h)//24,9):
            lo=max(72+h,day*24);hi=min(215,day*24+23)
            g=x[(x.horizon==h)&(x.target_day==day)].set_index('fipsCode').sort_index();assert len(g)==239
            assert (g.start==lo).all() and (g.end==hi).all()
            for c in hist:close(g[c],hist.loc[g.index,c])
            close(g.customers_at_cutoff,cut.loc[g.index,'customersTracked'])
            a=raw[raw.hour.between(lo,hi)];b=raw[raw.hour.between(lo-24,lo-1)]
            weather=a.groupby('fipsCode');prior=b.groupby('fipsCode')
            vals={'gust_max':weather.gust.max(),'gust_mean':weather.gust.mean(),'gust_hours_gt30':a.assign(v=a.gust>30).groupby('fipsCode').v.sum(),
                'tp_sum':weather.tp.sum(),'soil_mean':weather.soil_moist.mean(),'prior24_gust_max':prior.gust.max(),'prior24_tp_sum':prior.tp.sum()}
            for c,z in vals.items():close(g[c],z.loc[g.index])
            mask=(meta.hour_idx+h).between(lo,hi)
            z=cache.loc[mask].assign(fipsCode=meta.loc[mask,'fipsCode'].to_numpy()).groupby('fipsCode').mean()
            for c in sum(blocks[str(h)].values(),[]):close(g[c],z.loc[g.index,c.removeprefix('cache__')])
            count+=len(g)
    joined=pd.read_parquet(OUT/'matching/ranked_controls_with_outcomes.parquet');outcome=pd.read_csv(OUT/'matching/window_outcomes.csv',float_precision='round_trip',dtype={'fipsCode':str})
    for side,key in [('query','fipsCode'),('control','matched_fips')]:
        expected=joined[['cv_seed','horizon','target_day',key]].rename(columns={key:'fipsCode'}).merge(outcome,on=['cv_seed','horizon','target_day','fipsCode'],validate='many_to_one')
        for c in outcome.columns.difference(['cv_seed','horizon','target_day','fipsCode']):close(joined[side+'_'+c],expected[c])
    assert np.array_equal(joined.control_low_loss,joined.control_osi_peak<=.01)
    assert np.array_equal(joined.control_high_loss,joined.control_osi_peak>=.05)
    summary=pd.read_csv(OUT/'matching/control_summary.csv',float_precision='round_trip',dtype={'fipsCode':str}).set_index(['cv_seed','horizon','target_day','fipsCode','variant'])
    grouped=joined.groupby(['cv_seed','horizon','target_day','fipsCode','variant'])
    for saved,col in [('low_loss_controls','control_low_loss'),('high_loss_controls','control_high_loss')]:
        sums=grouped[col].sum();close(summary.loc[sums.index,saved],sums)
    result=dict(status='PASS',matching_input_county_windows_rebuilt=count,ranking_outcome_joins_checked=len(joined),
       history_cutoff=71,weather_scope='provided official series',identifier_features_used=False,
       allowed_features_independently_rebuilt=True,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'input_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
