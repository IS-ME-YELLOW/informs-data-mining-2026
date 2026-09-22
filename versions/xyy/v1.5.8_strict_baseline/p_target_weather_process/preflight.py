"""No-fit feature, source-model and train-scope acceptance."""
from types import SimpleNamespace
import lightgbm as lgb
from weather_core import *
from process_features import build,build_new,weather_input,window_features
from verify_features import verify_bundle,scalar

def main():
    ctx=snapshot(load_context());build(ctx);ctx=initialize(ctx)
    verification=verify_bundle();save(OUT/'preflight/feature_verification.json',verification)
    cases=[np.full(48,30.),np.full(48,31.),np.r_[40.,np.full(47,20.)],np.r_[np.full(47,20.),40.],
           np.r_[np.full(2,40.),np.full(5,20.),np.full(2,40.),np.full(39,20.)]]
    for v in cases:np.testing.assert_allclose(window_features(v),scalar(v),atol=1e-12,rtol=0)
    assert window_features(cases[0])[2:]==[49.,0.,0.,0.,49.,0.]
    assert window_features(cases[1])[2:6]==[0.,1.,1.,0.]
    assert window_features(cases[4])[4:6]==[2.,5.]
    weather=weather_input('train');keys=ctx.meta[['fipsCode','timestamp_et','hour_idx']];saved=ctx.X[NEW_COLUMNS]
    for cutoff in [96,144,191]:
        shadow=weather.copy();shadow.loc[shadow.hour_idx>cutoff,'gust']=10000
        built=build_new(shadow,keys);hit=keys.hour_idx+24<=cutoff
        pd.testing.assert_frame_equal(built.loc[hit],saved.loc[hit],check_exact=True)
    try:build_new(weather.assign(P_t=999),keys)
    except ValueError:pass
    else:raise AssertionError('Forbidden outage column accepted')
    descriptions=[];scope_checks=0;max_reload=0.
    oldraw=pd.read_parquet(TREE/'new_P_oof_predictions.parquet');oldraw=norm(oldraw[oldraw.horizon==H24])
    for outer in range(5):
        receipt=read(TREE/f'models/outer{outer}/P_direct_t24h.json');spec=receipt['spec']
        assert spec['identity_hash']==TREE_ID and receipt['model_id']==digest(spec) and spec['params']==params() and receipt['fit']['params']==params()
        assert set(spec['allowed_folds'])==set(range(5))-{outer} and spec['feature_names']==list(ctx.X0)
        assert sha(TREE/receipt['model_path'])==receipt['model_sha256']
        model=lgb.Booster(model_file=str(TREE/receipt['model_path']));hit=ctx.valid&(ctx.meta.fold.to_numpy()==outer)
        pred=model.predict(ctx.X0.loc[hit],num_threads=1);np.testing.assert_allclose(pred,oldraw.loc[hit,'raw_direct'],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(np.clip(pred,0,1),ctx.parts[H24]['P_t'][hit]);max_reload=max(max_reload,float(np.max(abs(pred-oldraw.loc[hit,'raw_direct'].to_numpy()))))
        allowed=set(range(5))-{outer}
        for fs in [allowed]+[allowed-{inner} for inner in sorted(allowed)]:
            real=subset(ctx,fs);forbidden=~(ctx.valid&np.isin(ctx.meta.fold.to_numpy(),list(fs)))
            altered=SimpleNamespace(**vars(ctx));altered.y=ctx.y.copy();altered.y[forbidden]=98765
            altered.X=ctx.X.copy();altered.X.loc[forbidden,:]=99999
            assert describe(subset(altered,fs))==describe(real);scope_checks+=1
            descriptions.append(dict(outer_fold=outer,**describe(real)))
    stats=[];old=ctx.X0.loc[ctx.valid];new=ctx.X.loc[ctx.valid,NEW_COLUMNS]
    for col in NEW_COLUMNS:
        a=new[col];cor=old.corrwith(a).abs().dropna().sort_values(ascending=False);duplicates=[c for c in old if np.array_equal(old[c].to_numpy(),a.to_numpy(),equal_nan=True)]
        assert a.nunique()>1 and not duplicates
        stats.append(dict(feature=col,n=len(a),min=float(a.min()),max=float(a.max()),unique_values=int(a.nunique()),zero_fraction=float((a==0).mean()),
            most_correlated_old_feature=cor.index[0],max_abs_pearson=float(cor.iloc[0]),exact_old_duplicates=','.join(duplicates)))
    frame(OUT/'preflight/new_feature_statistics.csv',stats);save(OUT/'preflight/training_scopes.json',descriptions)
    for p,h in ctx.sources.items():assert sha(p)==h,p
    save(OUT/'preflight/verification.json',dict(status='PASS',identity_hash=ctx.identity_hash,model_fits=0,old_P24_models_reloaded=5,old_prediction_max_abs_difference=max_reload,
        independent_features=verification,scope_poisoning_checks=scope_checks,temporal_cutoff_checks=3,synthetic_window_checks=len(cases),forbidden_label_columns_rejected=True,
        all_eight_nonconstant=True,no_exact_old_feature_duplicates=True,auxiliary_membership_frozen=True,source_files_unchanged=len(ctx.sources)))
    print('Preflight PASS: 191 columns, 5 reference models, 25 training scopes',flush=True)

if __name__=='__main__':main()
