"""150 scoped fits for three separately evaluated P a/b source migrations."""
import argparse
import time
import lightgbm as lgb
from transfer_protocol import *


def preflight(ctx):
    assert len(ctx.data.X_train.columns)==163
    # Compare the mathematical specification and model parameters to the approved 1h implementation.
    sys.path.insert(0,str(BASE/'p_initial_state_structure'))
    import p_structure_protocol as original
    import weighted_scoped_training as original_training
    checks=[]
    for p0 in (0.,.2,1.):
        p=np.repeat(p0,4);y=np.array([0.,.1,.2,.6]);a=np.full(4,np.nan);b=a.copy()
        for kind,out in [('a',a),('b',b)]:
            hit=support(p,kind)
            if hit.any():
                yy,ww=transform(y[hit],p[hit],kind);oldy,oldw=original.transform(y[hit],p[hit],kind)
                np.testing.assert_array_equal(yy,oldy);np.testing.assert_array_equal(ww,oldw);out[hit]=yy
        np.testing.assert_allclose(reconstruct(p,a,b,np.ones(4,bool))[0],y,atol=1e-12,rtol=0)
    for kind in KINDS:
        assert params()==original.params(kind)
        y=np.array([0.,.5,1.]);w=np.array([.01,.2,.4]);pred=np.array([.1,.4,.8])
        assert contribution_metric(y,w,5)(pred,None)==original_training.contribution_metric(y,w,5)(pred,None)
    for h in TRANSFER:
        valid=expected_mask(ctx.meta,h);y=ctx.data.component_targets[component_target_name('P_t',h)].to_numpy()
        assert np.isfinite(y[valid]).all() and ((y[valid]>=0)&(y[valid]<=1)).all()
        a=np.full(len(y),np.nan);b=a.copy()
        for kind,out in [('a',a),('b',b)]:
            hit=valid&support(ctx.p,kind);out[hit]=transform(y[hit],ctx.p[hit],kind)[0]
        np.testing.assert_allclose(reconstruct(ctx.p,a,b,valid)[0],y,atol=1e-12,rtol=0)
        for outer in range(5):
            held=set(ctx.meta.loc[ctx.meta.fold==outer,'fipsCode'])
            alt=SimpleNamespace(**vars(ctx.data));alt.component_targets=ctx.data.component_targets.copy()
            alt.component_targets.loc[ctx.meta.fold==outer,component_target_name('P_t',h)]=987654.321
            shadow=SimpleNamespace(data=alt,p=ctx.p,meta=ctx.meta)
            for kind in KINDS:
                spec=specification(ctx,outer,h,kind,'preflight')
                assert spec==specification(shadow,outer,h,kind,'preflight')
                for probe in spec['probes']:
                    assert not held.intersection(probe['train']['counties']+probe['validation']['counties'])
                    assert not set(probe['train']['counties']).intersection(probe['validation']['counties'])
                checks.append(dict(horizon=h,outer_fold=outer,kind=kind,scope_and_label_poison='PASS',
                    refit_rows=spec['refit']['rows'],all_valid_refit_rows=spec['refit']['total_valid_rows']))
    write_json(OUT/'reference/preflight.json',dict(status='PASS',checks=checks,P71_matches_history=True,
        original_E2_transform_params_and_metric_match=True,feature_count=163,label_reconstruction='PASS',
        reference_F2_reconstruction='PASS',training_performed=False))
    print('Preflight PASS: exact E2 transfer, 30 branch scopes, endpoints, P71 and F2 checked',flush=True)


def progress(message):
    print(message,flush=True);path=safe(RUN/'logs/training.log');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(message+'\n')


def restore(receipt,spec):
    assert receipt['spec']==spec and receipt['model_id']==digest_object(spec)
    assert sha256_file(RUN/receipt['model_path'])==receipt['model_sha256']
    assert sha256_file(RUN/receipt['probe_path'])==receipt['probe_sha256']
    model=lgb.Booster(model_file=str(RUN/receipt['model_path']))
    assert model.feature_name()==spec['feature_names'] and model.current_iteration()==receipt['fit']['trees_saved']
    return model


def fit_one(ctx,outer,h,kind):
    spec=specification(ctx,outer,h,kind,ctx.identity_hash);prefix=f'models/outer{outer}/{spec["target"]}'
    receipt_path=RUN/f'{prefix}.json'
    if receipt_path.exists():
        receipt=read_json(receipt_path);return restore(receipt,spec),receipt
    started=time.monotonic();best=[];curves=[];frames=[]
    for probe in spec['probes']:
        q=probe['inner_fold'];tr=subset(ctx,probe['train']['folds'],h,kind);va=subset(ctx,[q],h,kind)
        scale=float(tr.w.mean());history={}
        train=lgb.Dataset(tr.X,label=tr.y,weight=tr.w/scale)
        valid=lgb.Dataset(va.X,label=va.y,weight=va.w/scale,reference=train)
        model=lgb.train(params(),train,num_boost_round=2000,valid_sets=[valid],valid_names=['validation'],
            feval=contribution_metric(va.y,va.w,va.total_valid),callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),
                lgb.record_evaluation(history),lgb.log_evaluation(0)])
        iteration=int(model.best_iteration);assert 1<=iteration<=2000
        pred=model.predict(va.X,num_iteration=iteration,num_threads=1)
        f=va.meta.copy();f['row_index']=va.rows;f['outer_fold']=outer;f['inner_fold']=q;f['horizon']=h;f['kind']=kind
        f['branch_label']=va.y;f['P71']=va.p;f['raw_weight']=va.w;f['normalized_weight']=va.w/scale
        f['training_weight_mean']=scale;f['all_validation_rows']=va.total_valid;f['prediction_at_best']=pred;f['best_iteration']=iteration
        frames.append(f);values=history['validation']['raw_P_contribution_rmse'];best.append(iteration)
        curves.append(dict(inner_fold=q,values=values,best_value=values[iteration-1],metric='raw_P_contribution_rmse',training_weight_mean=scale))
        progress(f'{spec["target"]} outer{outer} inner{q}: best={iteration}, contribution_rmse={values[iteration-1]:.8f}')
    full=subset(ctx,spec['allowed_folds'],h,kind);rounds=max(1,int(np.mean(best)));mean=float(full.w.mean())
    model=lgb.train(params(),lgb.Dataset(full.X,label=full.y,weight=full.w/mean),num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)])
    path=RUN/f'{prefix}.txt';atomic_write(safe(path),lambda q:model.save_model(str(q)))
    probe_path=f'logs/probes/outer{outer}_{spec["target"]}.parquet';write_frame(RUN/probe_path,pd.concat(frames,ignore_index=True))
    receipt=dict(spec=spec,model_id=digest_object(spec),model_path=f'{prefix}.txt',model_sha256=sha256_file(path),
        probe_path=probe_path,probe_sha256=sha256_file(RUN/probe_path),
        fit=dict(best_iterations=best,requested_rounds=rounds,trees_saved=model.current_iteration(),curves=curves,
            refit_weight_mean=mean,params=params(),fits=5,seconds=time.monotonic()-started))
    write_json(receipt_path,receipt)
    progress(f'{spec["target"]} outer{outer}: refit={rounds}, trees={model.current_iteration()}, seconds={time.monotonic()-started:.1f}')
    return model,receipt


def train(ctx,resume):
    if (RUN/'MODEL_CV_COMPLETE').exists():raise RuntimeError('Completed training stage is immutable')
    m=dict(run_id=RUN.name,identity=ctx.identity,identity_hash=ctx.identity_hash)
    if RUN.exists():assert resume and read_json(RUN/'run_manifest.json')==m,'Resume requires identical run identity'
    else:
        assert not resume;write_json(RUN/'run_manifest.json',m)
    all_models=[];frames=[];registries=[]
    for h in TRANSFER:
        valid=expected_mask(ctx.meta,h);values={k:np.full(len(ctx.meta),np.nan) for k in KINDS}
        ids={k:np.full(len(ctx.meta),'',dtype=object) for k in KINDS};reconids=np.full(len(ctx.meta),'',dtype=object)
        for outer in range(5):
            pair={}
            for kind in KINDS:
                model,receipt=fit_one(ctx,outer,h,kind);pair[kind]=receipt['model_id'];all_models.append(receipt)
                hit=valid&(ctx.meta.fold.to_numpy()==outer)&support(ctx.p,kind)
                values[kind][hit]=model.predict(ctx.data.X_train.loc[hit],num_threads=1);ids[kind][hit]=receipt['model_id']
            registry=dict(identity_hash=ctx.identity_hash,horizon=h,outer_fold=outer,branch_model_ids=pair,
                p71_source_sha256=ctx.data.loaded_hashes['raw_train'],p71_sha256=array_sha(ctx.p),formula='p*clip(a,0,1)+(1-p)*clip(b,0,1)')
            rid=digest_object(registry);registries.append(dict(reconstruction_id=rid,definition=registry))
            reconids[valid&(ctx.meta.fold.to_numpy()==outer)]=rid
        rebuilt,aa,bb,ca,cb=reconstruct(ctx.p,values['a'],values['b'],valid)
        f=ctx.meta.copy();f['horizon']=h;f['target_hour']=ctx.meta.hour_idx+HORIZON_HOURS[h]
        f['scoreable']=valid;f['P71']=ctx.p;f['candidate_identity_hash']=ctx.identity_hash
        f['true_P']=ctx.data.component_targets[component_target_name('P_t',h)]
        for kind in KINDS:f['raw_'+kind]=values[kind];f['source_'+kind]=ids[kind];f['support_'+kind]=valid&support(ctx.p,kind)
        f['applied_a']=aa;f['applied_b']=bb;f['contribution_a']=ca;f['contribution_b']=cb;f['pred_P']=rebuilt;f['reconstruction_id']=reconids
        frames.append(f)
    write_frame(RUN/'branch_oof_predictions.parquet',pd.concat(frames,ignore_index=True))
    write_json(RUN/'reconstruction_registry.json',registries)
    write_json(RUN/'model_manifest.json',dict(identity_hash=ctx.identity_hash,models=all_models,formal_fits=150,outer_models=30))
    progress('All 150 fits completed; 30 outer models and three single-source P reconstructions saved')
    assert_inputs(ctx)
    files={str(p.relative_to(RUN)):sha256_file(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    write_json(RUN/'MODEL_CV_COMPLETE',dict(identity_hash=ctx.identity_hash,files=files,formal_fits=150,outer_models=30))
    print('MODEL_CV_COMPLETE committed',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','train'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    ctx=initialize(load_context());preflight(ctx)
    if a.stage=='train':train(ctx,a.resume)


if __name__=='__main__':main()
