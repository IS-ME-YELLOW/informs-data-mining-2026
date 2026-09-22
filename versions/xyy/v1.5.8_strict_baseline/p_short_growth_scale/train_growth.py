"""50 authorized fits, with per-fit immutable receipts and resumable checkpoints."""
import argparse,time
from datetime import datetime,timezone
import lightgbm as lgb
from growth_core import *

def fit_one(ctx,case,outer,inner,train_folds,validation_fold,rounds,attempt):
    tr=subset(ctx,train_folds,case);va=subset(ctx,[validation_fold],case) if validation_fold is not None else None
    prefix=RUN/f'models/{case}/outer{outer}'
    name=f'inner{inner}' if inner is not None else 'refit'
    path=prefix/f'{name}.txt';recpath=prefix/f'{name}.json';predpath=prefix/f'{name}_validation.parquet'
    spec=dict(identity_hash=ctx.identity_hash,case=case,outer_fold=outer,inner_fold=inner,train=describe(tr),
        validation=describe(va) if va else None,params=params(),feature_names=list(ctx.X),requested_rounds=rounds,
        early_stopping_rounds=100 if va else None)
    model_id=digest(spec)
    if recpath.exists():
        receipt=read(recpath);assert receipt['model_id']==model_id and receipt['spec']==spec and sha(path)==receipt['model_sha256']
        if va:assert sha(predpath)==receipt['validation_sha256']
        return lgb.Booster(model_file=str(path)),receipt
    if path.exists():raise RuntimeError(f'Orphan model without receipt: {path}')
    training=lgb.Dataset(tr.X,label=tr.u,weight=tr.w/tr.w.mean())
    history={};start=time.monotonic()
    attempt['fits_started']+=1;save(RUN/'execution.json',attempt)
    if va:
        assert not set(tr.meta.fipsCode)&set(va.meta.fipsCode) and outer not in tr.folds and outer not in va.folds
        validation=lgb.Dataset(va.X,label=va.u,weight=va.w/tr.w.mean(),reference=training)
        def metric(pred,dataset):return 'raw_P_increment_rmse',float(np.sqrt(np.sum(va.w*(pred-va.u)**2)/va.total_valid)),False
        model=lgb.train(params(),training,num_boost_round=2000,valid_sets=[validation],valid_names=['validation'],feval=metric,
            callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.record_evaluation(history),lgb.log_evaluation(0)])
        best=int(model.best_iteration);pred=model.predict(va.X,num_iteration=best,num_threads=1)
        v=va.meta[KEYS].copy();v['row_index']=va.rows;v['label_u']=va.u;v['raw_weight']=va.w;v['scale']=va.scale;v['prediction']=pred
        frame(predpath,v)
    else:
        model=lgb.train(params(),training,num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)]);best=None
    atomic(path,lambda q:model.save_model(str(q),num_iteration=best if va else -1))
    saved=lgb.Booster(model_file=str(path));assert saved.feature_name()==list(ctx.X)
    receipt=dict(model_id=model_id,spec=spec,model_path=str(path.relative_to(RUN)),model_sha256=sha(path),best_iteration=best,
        actual_trees=saved.current_iteration(),curve=history.get('validation',{}).get('raw_P_increment_rmse',[]),
        fit_seconds=time.monotonic()-start,training_weight_mean=float(tr.w.mean()),validation_weight_scale_training_mean=float(tr.w.mean()) if va else None)
    if va:receipt.update(validation_path=str(predpath.relative_to(RUN)),validation_sha256=sha(predpath))
    save(recpath,receipt);attempt['fits_completed']+=1;save(RUN/'execution.json',attempt)
    print(f"[{attempt['fits_completed']}/50] {case} outer{outer} {name}: best={best}, trees={saved.current_iteration()}",flush=True)
    return saved,receipt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    ctx=initialize(load_context());pre=read(OUT/'preflight/verification.json');assert pre['status']=='PASS' and pre['identity_hash']==ctx.identity_hash
    if (RUN/'MODEL_CV_COMPLETE').exists():raise RuntimeError('Completed model run is immutable')
    path=RUN/'run_manifest.json';manifest=dict(identity=ctx.identity,identity_hash=ctx.identity_hash)
    if path.exists():assert args.resume and read(path)==manifest,'Use matching --resume for an existing run'
    else:assert not args.resume;save(path,manifest)
    attempt=read(RUN/'execution.json') if (RUN/'execution.json').exists() else dict(status='running',start_utc=datetime.now(timezone.utc).isoformat(),fits_started=0,fits_completed=0)
    attempt['status']='running';save(RUN/'execution.json',attempt)
    result=ctx.meta[KEYS].copy();result['p71']=ctx.p;result['N71']=ctx.g;result['early_eligible']=ctx.early;result['reference_a_applied']=ctx.a
    result['reference_P1']=ctx.parts[H1]['P_t'];result['identity_hash']=ctx.identity_hash
    manifest_models=[]
    try:
        for case in NEW:
            raw=np.full(len(ctx.meta),np.nan);ids=np.full(len(ctx.meta),'',object)
            for outer in range(5):
                allowed=set(range(5))-{outer};best=[]
                for inner in sorted(allowed):
                    _,receipt=fit_one(ctx,case,outer,inner,allowed-{inner},inner,2000,attempt)
                    best.append(receipt['best_iteration']);manifest_models.append(receipt)
                rounds=max(1,int(np.mean(best)))
                model,receipt=fit_one(ctx,case,outer,None,allowed,None,rounds,attempt);manifest_models.append(receipt)
                rows=np.flatnonzero(ctx.early&(ctx.p<1)&(ctx.meta.fold.to_numpy()==outer))
                raw[rows]=model.predict(ctx.X.iloc[rows],num_threads=1);ids[rows]=receipt['model_id']
            result[case+'_raw_b']=raw;result[case+'_model_id']=ids;result[case+'_P1']=reconstruct(ctx,raw,case)
        assert len(manifest_models)==50 and attempt['fits_completed']==50
        frame(RUN/'branch_predictions.parquet',result);save(RUN/'model_manifest.json',manifest_models)
        for p,h in ctx.sources.items():assert sha(p)==h,p
        attempt.update(status='complete',end_utc=datetime.now(timezone.utc).isoformat());save(RUN/'execution.json',attempt)
        files={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and p.name!='MODEL_CV_COMPLETE'}
        save(RUN/'MODEL_CV_COMPLETE',dict(identity_hash=ctx.identity_hash,formal_fits=50,outer_models=10,inner_models=40,files=files))
        print('All 50 fits completed',flush=True)
    except BaseException as e:
        attempt.update(status='failed',error=repr(e));save(RUN/'execution.json',attempt);raise

if __name__=='__main__':main()
