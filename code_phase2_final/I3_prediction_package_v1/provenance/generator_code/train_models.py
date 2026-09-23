"""Deterministic per-model jobs with immutable receipts and resumable parallel execution."""
import argparse,time,concurrent.futures,multiprocessing,traceback
from package_core import *
_CTX={}
def context(seed):
    if seed not in _CTX:
        _CTX.clear();_CTX[seed]=Context(seed)
    return _CTX[seed]
def load_model(r,path):
    if r['family']=='lightgbm':
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    if r['family']=='xgboost':
        from xgboost import XGBRegressor
        m=XGBRegressor();m.load_model(path);return m
    from catboost import CatBoostRegressor
    m=CatBoostRegressor();m.load_model(path);return m
def prediction(m,r,X,rounds):
    if r['family']=='lightgbm':return np.asarray(m.predict(X,num_iteration=rounds,num_threads=1),float)
    X=X.astype(np.float32)
    if r['family']=='xgboost':return np.asarray(m.predict(X,iteration_range=(0,rounds)),float)
    return np.asarray(m.predict(X,ntree_end=rounds,thread_count=1),float)
def fit(r,tr,va=None,rounds=None):
    cap=maxrounds(r) if va is not None else int(rounds);hist={}
    if r['family']=='lightgbm':
        import lightgbm as lgb
        weighted=r['kind'] in ('a','b');mean=float(tr['w'].mean());ts=lgb.Dataset(tr['X'],label=tr['y'],weight=tr['w']/mean if weighted else None)
        kwargs={}
        if va is not None:
            vs=lgb.Dataset(va['X'],label=va['y'],weight=va['w']/mean if weighted else None,reference=ts)
            def feval(p,d):return 'contribution_rmse',float(np.sqrt(np.sum(va['w']*(p-va['y'])**2)/va['total_valid'])),False
            kwargs=dict(valid_sets=[vs],valid_names=['validation'],feval=feval if weighted else None)
        callbacks=[lgb.log_evaluation(0)]
        if va is not None:callbacks += [lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.record_evaluation(hist)]
        m=lgb.train(params(r),ts,num_boost_round=cap,callbacks=callbacks,**kwargs)
        best=int(m.best_iteration) if va is not None else cap;curve=hist.get('validation',{}).get('contribution_rmse' if weighted else 'rmse',[])
    elif r['family']=='xgboost':
        from xgboost import XGBRegressor
        m=XGBRegressor(**params(r,cap,va is not None));kwargs=dict(eval_set=[(va['X'].astype(np.float32),va['y'])],verbose=False) if va is not None else dict(verbose=False)
        m.fit(tr['X'].astype(np.float32),tr['y'],**kwargs);best=int(m.best_iteration)+1 if va is not None else cap
        curve=m.evals_result()['validation_0']['rmse'] if va is not None else []
    else:
        from catboost import CatBoostRegressor
        m=CatBoostRegressor(**params(r,cap,va is not None));kwargs=dict(eval_set=(va['X'].astype(np.float32),va['y']),use_best_model=True) if va is not None else {}
        m.fit(tr['X'].astype(np.float32),tr['y'],**kwargs);best=int(m.get_best_iteration())+1 if va is not None else cap
        if best<1:best=int(m.tree_count_)
        curve=m.get_evals_result().get('validation',{}).get('RMSE',[]) if va is not None else []
    assert 1<=best<=cap;return m,best,curve
def save_model(m,r,path,best=None):
    atomic(path,lambda p:m.save_model(str(p),num_iteration=best) if r['family']=='lightgbm' and best is not None else m.save_model(str(p)))
def validate_saved(t):
    p=paths(t)
    if not p['receipt'].exists():return None
    rec=read(p['receipt']);assert rec['task']==t and rec['identity_hash']==read(OUT/'identity.json')['identity_hash']
    for k in ['model','prediction']:assert sha(p[k])==rec[k+'_sha256'],str(p[k])
    return rec
def one(t):
    old=validate_saved(t)
    if old:return dict(id=t['id'],status='cached',seconds=0,fit_calls=0)
    start=time.monotonic();ctx=context(t['seed']);r=RECIPES[t['recipe']];s=tuple(t['scope']);pp=paths(t);tr=ctx.subset(r,s);probes=[];fit_calls=0
    spec=dict(task=t,identity_hash=ctx.identity,recipe=r,params=params(r),feature_names=list(ctx.X[r['feature']]),refit=ctx.describe(tr))
    if t['reuse']:
        ref=t['reuse'];rec=read(ref['receipt_path']);assert sha(ref['model_path'])==ref['model_sha256'] and sha(ref['receipt_path'])==ref['receipt_sha256']
        rounds=int(rec['fit']['requested_rounds']);copy_input(ref['model_path'],pp['model']);copy_input(ref['receipt_path'],pp['folder']/'upstream_receipt.json')
        m=load_model(r,pp['model']);model_id=ref['original_model_id']
    else:
        best=[]
        for q,tt,vv in ctx.probes(r,s):
            qp=pp['folder']/'probes'/q;sr=dict(train=ctx.describe(tt),validation=ctx.describe(vv),params=params(r,maxrounds(r),True))
            record_path=qp/'receipt.json';mp=qp/('model'+extension(r));vp=qp/'validation.npz'
            if record_path.exists():
                rec=read(record_path);assert rec['spec']==sr and sha(mp)==rec['model_sha256'] and sha(vp)==rec['validation_sha256'];b=rec['best_iteration']
            else:
                m,b,curve=fit(r,tt,vv);pred=prediction(m,r,vv['X'],b);save_model(m,r,mp,b if r['family']=='lightgbm' else None)
                atomic(vp,lambda p:np.savez_compressed(p,rows=vv['rows'],official_transformed_y=vv['y'],weight=vv['w'],prediction=pred))
                rec=dict(spec=sr,best_iteration=b,curve=curve,model_sha256=sha(mp),validation_sha256=sha(vp));save(record_path,rec);fit_calls+=1
            probes.append(dict(probe=q,receipt=str(record_path.relative_to(OUT)),receipt_sha256=sha(record_path),best_iteration=b));best.append(b)
        rounds=max(1,int(np.mean(best)));m,_,_=fit(r,tr,rounds=rounds);save_model(m,r,pp['model']);fit_calls+=1
        model_id=digest(dict(spec=spec,rounds=rounds));m=load_model(r,pp['model'])
    rows=ctx.predict_rows(r,s);pred=prediction(m,r,ctx.X[r['feature']].iloc[rows],rounds);assert np.isfinite(pred).all()
    atomic(pp['prediction'],lambda p:np.savez_compressed(p,rows=rows,prediction=pred))
    rec=dict(identity_hash=ctx.identity,task=t,model_id=model_id,spec=spec,requested_rounds=rounds,probes=probes,new_fit_calls=(0 if t['reuse'] else len(probes)+1),
        model_sha256=sha(pp['model']),prediction_sha256=sha(pp['prediction']),prediction_rows=len(rows),prediction_row_sha256=ah(rows),seconds=time.monotonic()-start)
    save(pp['receipt'],rec);return dict(id=t['id'],status='reused' if t['reuse'] else 'trained',seconds=rec['seconds'],fit_calls=fit_calls)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--reuse-only',action='store_true');ap.add_argument('--seed',type=int);args=ap.parse_args()
    identity=read(OUT/'identity.json')
    for f,h in identity['identity']['code_hashes'].items():assert sha(OUT/f)==h,f
    tasks=read(OUT/'tasks.json');tasks=[t for t in tasks if (not args.reuse_only or t['reuse']) and (args.seed is None or t['seed']==args.seed)]
    pending=[t for t in tasks if validate_saved(t) is None];start=time.monotonic();done=len(tasks)-len(pending);fitcalls=0
    # Reused models first for early baseline reproducibility. Independent jobs use one native thread each.
    pending.sort(key=lambda t:(not bool(t['reuse']),t['seed'],t['recipe'].startswith('catboost'),len(t['scope']),t['id']))
    print(f'{len(tasks)} required models; {done} completed; {len(pending)} pending',flush=True)
    status_path=OUT/('preflight/reuse_progress.json' if args.reuse_only else 'training_progress.json')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(one,t):t for t in pending}
        for fut in concurrent.futures.as_completed(futures):
            t=futures[fut]
            try:result=fut.result()
            except BaseException:
                save(status_path,dict(status='FAILED',task=t['id'],error=traceback.format_exc(),completed=done,required=len(tasks)))
                for f in futures:f.cancel()
                raise
            done+=1;fitcalls+=result['fit_calls'];print(f'[{done}/{len(tasks)}] {result}',flush=True)
            save(status_path,dict(status='RUNNING' if done<len(tasks) else 'COMPLETE',completed=done,required=len(tasks),fits_this_attempt=fitcalls,elapsed_seconds=time.monotonic()-start,last=result))
    print('Selected model tasks complete',flush=True)
if __name__=='__main__':main()
