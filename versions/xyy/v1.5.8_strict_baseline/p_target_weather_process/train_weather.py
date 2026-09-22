"""25 authorized strict fits, with inner checkpoints preserved for verification."""
import argparse,time
from datetime import datetime,timezone
import lightgbm as lgb
from weather_core import *

def fit_one(ctx,outer,inner,train_folds,valid_fold,rounds,execution):
    tr=subset(ctx,train_folds);va=subset(ctx,[valid_fold]) if valid_fold is not None else None
    name=f'inner{inner}' if inner is not None else 'refit';folder=RUN/f'models/outer{outer}'
    mp=folder/f'{name}.txt';rp=folder/f'{name}.json';vp=folder/f'{name}_validation.parquet'
    spec=dict(identity_hash=ctx.identity_hash,case='W1',outer_fold=outer,inner_fold=inner,train=describe(tr),validation=describe(va) if va else None,
        params=params(),feature_names=list(ctx.X),requested_rounds=rounds,metric_label_dtype='float32 Dataset label')
    if rp.exists():
        r=read(rp);assert r['spec']==spec and r['model_id']==digest(spec) and sha(mp)==r['model_sha256']
        if va:assert sha(vp)==r['validation_sha256']
        return lgb.Booster(model_file=str(mp)),r
    if mp.exists():raise RuntimeError('Orphan model without receipt')
    execution['fits_started']+=1;save(RUN/'execution.json',execution);start=time.monotonic()
    training=lgb.Dataset(tr.X,label=tr.y);history={}
    if va:
        assert not set(tr.meta.fipsCode)&set(va.meta.fipsCode)
        validation=lgb.Dataset(va.X,label=va.y,reference=training)
        model=lgb.train(params(),training,num_boost_round=2000,valid_sets=[validation],valid_names=['validation'],
            callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.record_evaluation(history),lgb.log_evaluation(0)])
        best=int(model.best_iteration);pred=model.predict(va.X,num_iteration=best,num_threads=1)
        f=va.meta[KEYS].copy();f['row_index']=va.rows;f['official_P']=va.y;f['metric_P']=va.y.astype(np.float32).astype(float);f['prediction']=pred
        frame(vp,f)
    else:
        model=lgb.train(params(),training,num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)]);best=None
    atomic(mp,lambda q:model.save_model(str(q),num_iteration=best if va else -1));saved=lgb.Booster(model_file=str(mp))
    r=dict(model_id=digest(spec),spec=spec,model_path=str(mp.relative_to(RUN)),model_sha256=sha(mp),best_iteration=best,actual_trees=saved.current_iteration(),
        curve=history.get('validation',{}).get('rmse',[]),fit_seconds=time.monotonic()-start)
    if va:r.update(validation_path=str(vp.relative_to(RUN)),validation_sha256=sha(vp))
    save(rp,r);execution['fits_completed']+=1;save(RUN/'execution.json',execution)
    print(f"[{execution['fits_completed']}/25] outer{outer} {name}: best={best}, trees={saved.current_iteration()}",flush=True)
    return saved,r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    ctx=initialize(snapshot(load_context()));pre=read(OUT/'preflight/verification.json');assert pre['status']=='PASS' and pre['identity_hash']==ctx.identity_hash
    if (RUN/'MODEL_CV_COMPLETE').exists():raise RuntimeError('Completed model stage is immutable')
    manifest=dict(identity=ctx.identity,identity_hash=ctx.identity_hash);path=RUN/'run_manifest.json'
    if path.exists():assert args.resume and read(path)==manifest
    else:assert not args.resume;save(path,manifest)
    execution=read(RUN/'execution.json') if (RUN/'execution.json').exists() else dict(status='running',start_utc=datetime.now(timezone.utc).isoformat(),fits_started=0,fits_completed=0)
    models=[];raw=np.full(len(ctx.meta),np.nan);ids=np.full(len(ctx.meta),'',object)
    try:
        for outer in range(5):
            allowed=set(range(5))-{outer};best=[]
            for inner in sorted(allowed):
                _,r=fit_one(ctx,outer,inner,allowed-{inner},inner,2000,execution);best.append(r['best_iteration']);models.append(r)
            rounds=max(1,int(np.mean(best)));model,r=fit_one(ctx,outer,None,allowed,None,rounds,execution);models.append(r)
            rows=np.flatnonzero(ctx.valid&(ctx.meta.fold.to_numpy()==outer));raw[rows]=model.predict(ctx.X.iloc[rows],num_threads=1);ids[rows]=r['model_id']
        assert len(models)==25 and execution['fits_completed']==25
        out=ctx.meta[KEYS].copy();out['target_hour']=ctx.meta.hour_idx+24;out['scoreable']=ctx.valid;out['true_P24']=ctx.y;out['W1_raw_P24']=raw;out['W1_P24']=np.clip(raw,0,1);out['model_id']=ids;out['identity_hash']=ctx.identity_hash
        frame(RUN/'P24_oof_predictions.parquet',out);save(RUN/'model_manifest.json',models)
        for p,h in ctx.sources.items():assert sha(p)==h,p
        execution.update(status='complete',end_utc=datetime.now(timezone.utc).isoformat());save(RUN/'execution.json',execution)
        save(RUN/'MODEL_CV_COMPLETE',dict(identity_hash=ctx.identity_hash,formal_fits=25,outer_models=5,inner_models=20,
            files={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and p.name!='MODEL_CV_COMPLETE'}))
        print('All 25 fits complete',flush=True)
    except BaseException as e:
        execution.update(status='failed',error=repr(e));save(RUN/'execution.json',execution);raise

if __name__=='__main__':main()
