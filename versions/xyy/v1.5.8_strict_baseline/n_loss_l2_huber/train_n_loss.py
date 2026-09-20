"""Strict nested N-L2 training. Only this entry point performs authorized fits."""
import argparse
import time
import lightgbm as lgb
from loss_protocol import *


def preflight(ctx):
    assert len(ctx.data.X_train.columns)==163
    p=l2_params();old=make_lgbm_params(42)
    assert [k for k in p if p[k]!=old[k]]==['objective']
    checks=[]
    for outer in range(5):
        held=set(ctx.meta.loc[ctx.meta.fold==outer,'fipsCode'])
        for h in HORIZONS:
            spec=specification(ctx,outer,h,'preflight')
            receipt=read_json(HUBER/f'models/outer{outer}/{spec["target"]}.json')
            assert receipt['fit']['params']==old and receipt['spec']['feature_names']==spec['feature_names']
            assert receipt['fit']['allowed_folds']==spec['allowed_folds']
            assert receipt['fit']['refit']['counties']==spec['refit']['counties']
            for q in spec['probes']:
                assert not held.intersection(q['train']['counties']+q['validation']['counties'])
                assert not set(q['train']['counties']).intersection(q['validation']['counties'])
            # No training: poison withheld labels and prove the scoped fit inputs are invariant.
            altered=SimpleNamespace(**vars(ctx.data));altered.component_targets=ctx.data.component_targets.copy()
            target=spec['target'];altered.component_targets.loc[ctx.meta.fold==outer,target]=987654.321
            shadow=SimpleNamespace(data=altered)
            assert specification(shadow,outer,h,'preflight')==spec
            checks.append(dict(outer_fold=outer,horizon=h,heldout_label_poison_invariant=True,
                allowed_counties=len(spec['refit']['counties']),reference_model_id=receipt['model_id']))
    write_json(OUT/'reference/preflight.json',dict(status='PASS',checks=checks,N_models=20,feature_count=163,
        reference_reconstruction_passed=True,only_parameter_changed='objective',inputs=len(ctx.sources),training_performed=False))
    print('Preflight PASS: frozen F2, 20 scoped N specifications, only objective differs',flush=True)


def progress(message):
    print(message,flush=True)
    path=safe(RUN/'logs/training.log');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(message+'\n')


def load_model(receipt,spec):
    assert receipt['spec']==spec and receipt['model_id']==digest_object(spec)
    assert sha256_file(RUN/receipt['model_path'])==receipt['model_sha256']
    assert sha256_file(RUN/receipt['probe_path'])==receipt['probe_sha256']
    fit=receipt['fit'];assert fit['requested_rounds']==max(1,int(np.mean(fit['best_iterations'])))
    m=lgb.Booster(model_file=str(RUN/receipt['model_path']))
    assert m.feature_name()==spec['feature_names'] and m.current_iteration()==fit['trees_saved']
    return m


def fit_one(ctx,outer,h):
    spec=specification(ctx,outer,h,ctx.identity_hash);prefix=f'models/outer{outer}/{spec["target"]}'
    rp=RUN/f'{prefix}.json'
    if rp.exists():
        receipt=read_json(rp);return load_model(receipt,spec),receipt
    best=[];curves=[];rows=[];started=time.monotonic()
    for probe in spec['probes']:
        q=probe['inner_fold'];tr=scope(ctx.data,probe['train']['folds'],spec['target']);va=scope(ctx.data,[q],spec['target'])
        train=lgb.Dataset(tr.X,label=tr.y);valid=lgb.Dataset(va.X,label=va.y,reference=train)
        history={}
        model=lgb.train(l2_params(),train,num_boost_round=2000,valid_sets=[valid],valid_names=['validation'],
            callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0),lgb.record_evaluation(history)])
        iteration=int(model.best_iteration);assert 1<=iteration<=2000
        pred=model.predict(va.X,num_iteration=iteration,num_threads=1)
        d=va.meta.copy();d['row_index']=va.rows;d['outer_fold']=outer;d['inner_fold']=q;d['target']=spec['target']
        d['truth']=va.y;d['metric_label']=va.y.astype(np.float32).astype(float);d['prediction_at_best']=pred;d['best_iteration']=iteration
        rows.append(d);best.append(iteration);curve=history['validation']['rmse']
        curves.append(dict(inner_fold=q,metric='rmse',values=curve,best_value=curve[iteration-1]))
        progress(f'{spec["target"]} outer{outer} inner{q}: best={iteration}, rmse={curve[iteration-1]:.8f}')
    full=scope(ctx.data,spec['allowed_folds'],spec['target']);rounds=max(1,int(np.mean(best)))
    model=lgb.train(l2_params(),lgb.Dataset(full.X,label=full.y),num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)])
    mp=RUN/f'{prefix}.txt';atomic_write(safe(mp),lambda p:model.save_model(str(p)))
    probe_path=f'logs/probes/outer{outer}_{spec["target"]}.parquet'
    write_frame(RUN/probe_path,pd.concat(rows,ignore_index=True))
    receipt=dict(spec=spec,model_id=digest_object(spec),model_path=f'{prefix}.txt',model_sha256=sha256_file(mp),
        probe_path=probe_path,probe_sha256=sha256_file(RUN/probe_path),
        fit=dict(best_iterations=best,requested_rounds=rounds,trees_saved=model.current_iteration(),curves=curves,
                 fits=5,seconds=time.monotonic()-started,params=l2_params()))
    write_json(rp,receipt)
    progress(f'{spec["target"]} outer{outer}: refit={rounds}, saved={model.current_iteration()}, seconds={time.monotonic()-started:.1f}')
    return model,receipt


def train(ctx,resume):
    if (RUN/'MODEL_CV_COMPLETE').exists():raise RuntimeError('Completed model stage is immutable')
    manifest=dict(identity=ctx.identity,identity_hash=ctx.identity_hash,run_id=RUN.name)
    if RUN.exists():
        assert resume and read_json(RUN/'run_manifest.json')==manifest,'Resume requires identical run identity'
    else:
        assert not resume,'Cannot resume absent run'
        write_json(RUN/'run_manifest.json',manifest)
    records=[];frame=ctx.meta.copy();frame['candidate_identity_hash']=ctx.identity_hash
    for h in HORIZONS:
        raw=np.full(len(ctx.meta),np.nan);ids=np.full(len(ctx.meta),'',dtype=object);valid=expected_mask(ctx.meta,h)
        for outer in range(5):
            model,receipt=fit_one(ctx,outer,h)
            mask=valid&(ctx.meta.fold.to_numpy()==outer)
            raw[mask]=model.predict(ctx.data.X_train.loc[mask],num_threads=1)
            ids[mask]=receipt['model_id'];records.append(receipt)
        assert np.isfinite(raw[valid]).all() and np.isnan(raw[~valid]).all()
        tag=component_target_name('N_t',h)
        frame['raw_'+tag]=raw;frame['source_'+tag]=ids;frame['truth_'+tag]=ctx.data.component_targets[tag]
    write_frame(RUN/'n_l2_oof_predictions.parquet',frame)
    write_json(RUN/'model_manifest.json',dict(identity_hash=ctx.identity_hash,models=records,formal_fits=100,outer_models=20))
    progress('All 100 fits completed; 20 N-L2 outer models saved')
    assert_inputs(ctx)
    files={str(p.relative_to(RUN)):sha256_file(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    write_json(RUN/'MODEL_CV_COMPLETE',dict(identity_hash=ctx.identity_hash,files=files,formal_fits=100,outer_models=20))
    print('MODEL_CV_COMPLETE committed',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','train'],required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    ctx=initialize(load_context());preflight(ctx)
    if a.stage=='train':train(ctx,a.resume)


if __name__=='__main__':main()
