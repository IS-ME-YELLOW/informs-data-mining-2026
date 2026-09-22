"""25 strict nested fits for one P24 neighbor confirmation CV."""
import argparse
import time
import lightgbm as lgb
from neighbor_protocol import *
from neighbor_features import build
from independent_features import verify_features


def preflight(ctx):
    report,_=verify_features(ctx.data,causality=True)
    write_json(OUT/'reference/feature_verification.json',report)
    sys.path.insert(0,str(PTRANSFER));import transfer_protocol as old
    for kind in ('a','b'):
        assert params(kind)==old.params()
        p=np.array([.1,.5,.9]);y=np.array([0.,.7,1.]);a,w=transform(y,p,kind);b,v=old.transform(y,p,kind)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(w,v)
    assert params('direct')==make_lgbm_params(42)
    checks=[]
    for h,kind in TASKS:
        for outer in range(5):
            spec=specification(ctx,outer,h,kind,'preflight');allowed=[f for f in range(5) if f!=outer]
            held=set(ctx.meta.loc[ctx.meta.fold==outer,'fipsCode'])
            for pr in spec['probes']:
                assert not held.intersection(pr['train']['counties']+pr['validation']['counties'])
                assert not set(pr['train']['counties']).intersection(pr['validation']['counties'])
            if kind=='direct':
                base=BASE/f'runs/v18_tree_nested_v2_split{SEED}_model42/models/outer{outer}/{component_target_name("P_t",h)}.json'
            else:
                target='P_anchor_a' if kind=='a' else 'P_excess_b'
                base=CURRENT/f'models/outer{outer}/{target}_{h.rsplit("_",1)[-1]}.json'
            receipt=read_json(base);assert receipt['fit']['params']==params(kind)
            assert receipt['spec'].get('allowed_folds',receipt['spec'].get('scope'))==allowed
            altered=SimpleNamespace(**vars(ctx.data));altered.component_targets=ctx.data.component_targets.copy()
            altered.component_targets.loc[ctx.meta.fold==outer,component_target_name('P_t',h)]=987654.321
            shadow=SimpleNamespace(**vars(ctx));shadow.data=altered
            assert specification(shadow,outer,h,kind,'preflight')==spec
            checks.append(dict(horizon=h,kind=kind,outer_fold=outer,scope_invariance=True,unchanged_model_params=True))
    write_json(OUT/'reference/preflight.json',dict(status='PASS',checks=checks,features=report,reference=REFERENCE_ID,training_performed=False))
    print('Preflight PASS: current P6 baseline, exact neighbor features, raw time semantics, 5 allowed scopes',flush=True)


def progress(s):
    print(s,flush=True);path=safe(RUN/'logs/training.log');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(s+'\n')


def fit_one(ctx,outer,h,kind):
    spec=specification(ctx,outer,h,kind,ctx.identity_hash);hh=HORIZON_HOURS[h];prefix=f'models/outer{outer}/P_{kind}_t{hh:02d}h'
    rp=RUN/f'{prefix}.json'
    if rp.exists():
        r=read_json(rp);assert r['spec']==spec and r['model_id']==digest_object(spec)
        assert sha256_file(RUN/r['model_path'])==r['model_sha256'] and sha256_file(RUN/r['probe_path'])==r['probe_sha256']
        model=lgb.Booster(model_file=str(RUN/r['model_path']));assert model.feature_name()==spec['feature_names']
        return model,r
    started=time.monotonic();best=[];curves=[];frames=[]
    for pr in spec['probes']:
        q=pr['inner_fold'];tr=subset(ctx,pr['train']['folds'],h,kind);va=subset(ctx,[q],h,kind);mean=float(tr.w.mean())
        train=lgb.Dataset(tr.X,label=tr.y,weight=None if kind=='direct' else tr.w/mean)
        valid=lgb.Dataset(va.X,label=va.y,weight=None if kind=='direct' else va.w/mean,reference=train);history={}
        feval=None if kind=='direct' else contribution_metric(va.y,va.w,va.total_valid)
        model=lgb.train(params(kind),train,num_boost_round=2000,valid_sets=[valid],valid_names=['validation'],feval=feval,
            callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.record_evaluation(history),lgb.log_evaluation(0)])
        iteration=int(model.best_iteration);assert 1<=iteration<=2000
        pred=model.predict(va.X,num_iteration=iteration,num_threads=1);metric='rmse' if kind=='direct' else 'raw_P_contribution_rmse'
        values=history['validation'][metric];best.append(iteration)
        f=va.meta.copy();f['row_index']=va.rows;f['outer_fold']=outer;f['inner_fold']=q;f['horizon']=h;f['kind']=kind
        f['label']=va.y;f['metric_label']=va.y.astype(np.float32).astype(float) if kind=='direct' else va.y
        f['raw_weight']=va.w;f['normalized_weight']=va.w/mean;f['training_weight_mean']=mean
        f['all_validation_rows']=va.total_valid;f['prediction_at_best']=pred;f['best_iteration']=iteration;frames.append(f)
        curves.append(dict(inner_fold=q,metric=metric,values=values,best_value=values[iteration-1],training_weight_mean=mean))
        progress(f'P{hh} {kind} outer{outer} inner{q}: best={iteration}, score={values[iteration-1]:.8f}')
    full=subset(ctx,spec['allowed_folds'],h,kind);rounds=max(1,int(np.mean(best)));mean=float(full.w.mean())
    model=lgb.train(params(kind),lgb.Dataset(full.X,label=full.y,weight=None if kind=='direct' else full.w/mean),
        num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)])
    mp=RUN/f'{prefix}.txt';atomic_write(safe(mp),lambda q:model.save_model(str(q)))
    probe=f'logs/probes/outer{outer}_P_{kind}_t{hh:02d}h.parquet';write_frame(RUN/probe,pd.concat(frames,ignore_index=True))
    r=dict(spec=spec,model_id=digest_object(spec),model_path=f'{prefix}.txt',model_sha256=sha256_file(mp),
        probe_path=probe,probe_sha256=sha256_file(RUN/probe),fit=dict(best_iterations=best,requested_rounds=rounds,
        trees_saved=model.current_iteration(),curves=curves,refit_weight_mean=mean,fits=5,params=params(kind),seconds=time.monotonic()-started))
    write_json(rp,r);progress(f'P{hh} {kind} outer{outer}: refit={rounds}, seconds={time.monotonic()-started:.1f}')
    return model,r


def train(ctx,resume):
    if (RUN/'MODEL_CV_COMPLETE').exists():raise RuntimeError('Completed model stage is immutable')
    manifest=dict(run_id=RUN.name,identity=ctx.identity,identity_hash=ctx.identity_hash)
    if RUN.exists():assert resume and read_json(RUN/'run_manifest.json')==manifest
    else:
        assert not resume;write_json(RUN/'run_manifest.json',manifest)
    models=[];frames=[];registry=[]
    for h in CHANGED:
        hh=HORIZON_HOURS[h];kinds=['a','b'] if hh==6 else ['direct'];valid=expected_mask(ctx.meta,h)
        values={k:np.full(len(ctx.meta),np.nan) for k in kinds};ids={k:np.full(len(ctx.meta),'',dtype=object) for k in kinds}
        psource=np.full(len(ctx.meta),'',dtype=object)
        for outer in range(5):
            pair={}
            for k in kinds:
                model,r=fit_one(ctx,outer,h,k);models.append(r);pair[k]=r['model_id']
                hit=valid&(ctx.meta.fold.to_numpy()==outer)&support(ctx.p,k)
                values[k][hit]=model.predict(ctx.X[h].loc[hit],num_threads=1);ids[k][hit]=r['model_id']
            definition=dict(identity_hash=ctx.identity_hash,horizon=h,outer_fold=outer,models=pair,p71_sha=array_sha(ctx.p),feature_identity=ctx.feature_identity)
            rid=digest_object(definition) if hh==6 else pair['direct']
            registry.append(dict(source_id=rid,definition=definition));psource[valid&(ctx.meta.fold.to_numpy()==outer)]=rid
        f=ctx.meta.copy();f['horizon']=h;f['target_hour']=ctx.meta.hour_idx+hh;f['scoreable']=valid;f['P71']=ctx.p
        f['true_P']=ctx.data.component_targets[component_target_name('P_t',h)];f['candidate_identity_hash']=ctx.identity_hash
        for k in kinds:f['raw_'+k]=values[k];f['source_'+k]=ids[k]
        if hh==6:
            rebuilt,a,b,ca,cb=reconstruct(ctx.p,values['a'],values['b'],valid)
            f['applied_a']=a;f['applied_b']=b;f['contribution_a']=ca;f['contribution_b']=cb
        else:rebuilt=clip_component(values['direct'])
        f['pred_P']=rebuilt;f['P_source_id']=psource;frames.append(f)
    write_frame(RUN/'new_P_oof_predictions.parquet',pd.concat(frames,ignore_index=True))
    write_json(RUN/'source_registry.json',registry)
    write_json(RUN/'model_manifest.json',dict(identity_hash=ctx.identity_hash,models=models,formal_fits=25,outer_models=5))
    progress('All 25 fits completed: 5 outer models, one P24 neighbor candidate')
    assert_inputs(ctx)
    write_json(RUN/'MODEL_CV_COMPLETE',dict(identity_hash=ctx.identity_hash,
        files={str(p.relative_to(RUN)):sha256_file(p) for p in sorted(RUN.rglob('*')) if p.is_file()}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','train'],required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--split-seed',type=int,choices=(20260917,20260918),required=True);a=ap.parse_args()
    ctx=initialize(load_context());build(ctx);ctx=attach_features(ctx);preflight(ctx)
    if a.stage=='train':train(ctx,a.resume)


if __name__=='__main__':main()
