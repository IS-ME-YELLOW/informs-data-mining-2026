"""No-fit source reload, scope poisoning and reconstruction acceptance."""
from types import SimpleNamespace
import lightgbm as lgb
from growth_core import *

def main():
    ctx=initialize(load_context());checks=0;maxdiff=0.
    valid=expected_mask(ctx.meta,H1)
    for outer in range(5):
        for kind,target in [('a','P_anchor_a_t01h'),('b','P_excess_b_t01h')]:
            receipt=read(F2/f'models/F2/outer{outer}/{target}.json');spec=receipt['spec']
            assert receipt['model_id']==digest(spec) and spec['identity_hash']==F2_ID and spec['candidate']=='F2'
            assert spec['params']==params() and spec['feature_names']==list(ctx.X)
            assert spec['outer_fold']==outer and set(spec['allowed_folds'])==set(range(5))-{outer}
            for probe in spec['probes']:
                q=probe['inner_fold'];assert set(probe['train']['folds'])==set(range(5))-{outer,q}
                assert probe['validation']['folds']==[q]
            path=F2/receipt['model_path'];assert sha(path)==receipt['model_sha256']
            model=lgb.Booster(model_file=str(path));assert model.feature_name()==list(ctx.X)
            mask=valid&(ctx.meta.fold.to_numpy()==outer)&((ctx.p>0) if kind=='a' else (ctx.p<1))
            pred=model.predict(ctx.X.loc[mask],num_threads=1);saved=ctx.branch.loc[mask,'pred_'+kind+'_raw'].to_numpy()
            np.testing.assert_allclose(pred,saved,atol=1e-12,rtol=0);maxdiff=max(maxdiff,float(np.max(abs(pred-saved))))
            for f in ctx.branch.loc[mask,'source_'+kind+'_model_id'].unique():assert f==receipt['model_id']
    # Check scope isolation without additional fitting, across every actual probe/refit.
    for outer in range(5):
        permitted=set(range(5))-{outer}
        for case in NEW:
            for folds in [permitted]+[permitted-{q} for q in sorted(permitted)]:
                s=subset(ctx,folds,case);forbidden=~(ctx.early&np.isin(ctx.meta.fold.to_numpy(),list(folds)))
                altered=SimpleNamespace(**vars(ctx));altered.y=ctx.y.copy();altered.y[forbidden]=987654.321
                altered.X=ctx.X.copy();altered.X.loc[forbidden,:]=99999
                t=subset(altered,folds,case);assert describe(s)==describe(t)
                assert not set(s.meta.fold)&({outer}|(permitted-set(folds)))
                checks+=1
    # Exact endpoint and target back-transformation checks, no fitted model.
    for case in NEW:
        p=np.array([0.,.2,.2,.2,1.]);g=np.array([.02,0.,.03,.01,.02]);y=np.array([.4,0.,.1,.8,.6]);support=p<1
        c=scales(p[support],g[support],case);u=np.maximum(y[support]-p[support],0)/c
        a=np.divide(np.minimum(y,p),p,out=np.zeros(len(p)),where=p>0)
        delta=np.zeros(len(p));delta[support]=np.clip(c*np.clip(u,0,1),0,1-p[support])
        np.testing.assert_allclose(p*a+delta,y,atol=1e-12,rtol=0)
        original=subset(ctx,[0,1,2,3],case)
        np.testing.assert_allclose(original.u*original.scale,np.maximum(ctx.y[original.rows]-ctx.p[original.rows],0),atol=1e-12,rtol=0)
    summary=[]
    for case in NEW:
        for outer in range(5):summary.append(dict(case=case,outer_fold=outer,**describe(subset(ctx,set(range(5))-{outer},case))))
    save(OUT/'preflight/scope_summary.json',summary)
    for p,h in ctx.sources.items():assert sha(p)==h
    save(OUT/'preflight/verification.json',dict(status='PASS',identity_hash=ctx.identity_hash,model_fits=0,reference_models_reloaded=10,
        reference_prediction_max_abs_diff=maxdiff,poisoned_scopes_checked=checks,early_rows=int(ctx.early.sum()),p_one_counties=int(ctx.meta.loc[ctx.p==1,'fipsCode'].nunique()),
        feature_columns=183,aux_membership_frozen=True,source_files_unchanged=len(ctx.sources)))
    print('Preflight PASS',ctx.identity_hash,'scopes',checks,'early rows',ctx.early.sum(),flush=True)

if __name__=='__main__':main()
