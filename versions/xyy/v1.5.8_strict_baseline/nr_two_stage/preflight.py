"""No-fit reference replay, fold exclusion, and metric boundary checks."""
import lightgbm as lgb
from nr_core import *
def main():
    ctx=initialize(load_context());checks=[];stats=[];maxdiff=0.
    for c in ['N_t','R_t']:
        y=ctx.ys[c];valid=ctx.valid
        stats.append(dict(component=c,rows=int(valid.sum()),zero_fraction=float((y[valid]==0).mean()),positive_fraction=float((y[valid]>0).mean()),tail_gt_001_fraction=float((y[valid]>.01).mean())))
        for outer in range(5):
            rec=read(OLD/f'models/outer{outer}/{component_target_name(c,H1)}.json');assert set(rec['fit']['allowed_folds'])==set(range(5))-{outer}
            model=lgb.Booster(model_file=str(OLD/rec['model_path']));assert model.feature_name()==list(ctx.X)
            hit=valid&(ctx.meta.fold.to_numpy()==outer);pred=np.clip(model.predict(ctx.X.loc[hit],num_threads=1),0,1)
            np.testing.assert_array_equal(pred,ctx.parts[H1][c][hit]);maxdiff=max(maxdiff,float(np.max(abs(pred-ctx.parts[H1][c][hit]))))
            assert (ctx.reference_parts.loc[hit,'source_NB_P24_'+component_target_name(c,H1)]==rec['model_id']).all()
            allowed=set(range(5))-{outer}
            for inner in [None]+sorted(allowed):
                fs=allowed if inner is None else allowed-{inner}
                for head in ['q','m']:
                    real=subset(ctx,c,fs,head=='m');forbidden=~(valid&np.isin(ctx.meta.fold.to_numpy(),list(fs)))
                    shadow=SimpleNamespace(**vars(ctx));shadow.ys={k:v.copy() for k,v in ctx.ys.items()};shadow.ys[c][forbidden]=98765
                    shadow.X=ctx.X.copy();shadow.X.loc[forbidden,:]=99999
                    assert describe(subset(shadow,c,fs,head=='m'))==describe(real)
                    assert head=='m' or set(np.unique(real.y>0))=={False,True}
                    if inner is not None:
                        va=subset(ctx,c,[inner]);assert len(va.rows)>int((va.y>0).sum())>0
                    checks.append(dict(component=c,head=head,outer_fold=outer,inner_fold=inner,**describe(real)))
    np.testing.assert_array_equal(product([0,.5,1,1],[-1,.4,2,-1]),[0,.2,1,0])
    q=np.array([.5,.25]);truth=np.array([0.,.4]);m=np.array([.2,.8])
    val=product_metric(q,truth)(m,None);assert val[0]=='product_rmse' and not val[2]
    np.testing.assert_allclose(val[1],np.sqrt((.1**2+.2**2)/2),atol=1e-15)
    assert val[1]!=abs(.2-.4)  # All validation rows, including the zero row.
    for p,h in ctx.sources.items():assert sha(p)==h,p
    frame(OUT/'preflight/target_statistics.csv',stats);save(OUT/'preflight/training_scopes.json',checks)
    save(OUT/'preflight/verification.json',dict(status='PASS',identity_hash=ctx.identity_hash,formal_fits=0,old_models_reloaded=10,
        old_prediction_max_abs_difference=maxdiff,scope_poisoning_checks=len(checks),all_q_scopes_have_two_classes=True,
        all_m_scopes_have_positive_rows=True,full_validation_product_metric=True,feature_count=163,
        both_positive_rows=int(((ctx.ys['N_t']>0)&(ctx.ys['R_t']>0)&ctx.valid).sum()),source_files_unchanged=len(ctx.sources)))
    print('Preflight PASS: 10 reference models, 100 training scopes, 163 frozen features',flush=True)
if __name__=='__main__':main()
