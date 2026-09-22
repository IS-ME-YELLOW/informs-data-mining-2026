"""100 authorized fits. Each inner amplitude checkpoint depends on its own inner q."""
import argparse,time
from datetime import datetime,timezone
import lightgbm as lgb
from nr_core import *
def fit_one(ctx,c,head,outer,inner,folds,rounds,execution,q_record=None):
    tr=subset(ctx,c,folds,head=='m');va=subset(ctx,c,[inner]) if inner is not None else None
    name=f'inner{inner}_{head}' if inner is not None else f'refit_{head}';folder=RUN/f'models/{c}/outer{outer}'
    mp=folder/f'{name}.txt';rp=folder/f'{name}.json';vp=folder/f'{name}_validation.parquet';q=None;dependency=None
    if head=='m' and va is not None:
        assert q_record is not None;qs=q_record['spec'];assert (qs['component'],qs['head'],qs['outer_fold'],qs['inner_fold'])==(c,'q',outer,inner)
        assert sha(RUN/q_record['model_path'])==q_record['model_sha256'];assert sha(RUN/q_record['validation_path'])==q_record['validation_sha256']
        qframe=pd.read_parquet(RUN/q_record['validation_path']);np.testing.assert_array_equal(qframe.row_index,va.rows)
        q=qframe.prediction.to_numpy(copy=True);dependency=dict(model_id=q_record['model_id'],model_path=q_record['model_path'],model_sha256=q_record['model_sha256'],
            validation_path=q_record['validation_path'],validation_sha256=q_record['validation_sha256'],q_sha256=arrsha(q))
    train_label=(tr.y>0).astype(float) if head=='q' else tr.y
    spec=dict(identity_hash=ctx.identity_hash,component=c,head=head,outer_fold=outer,inner_fold=inner,train=describe(tr),validation=describe(va) if va else None,
        params=params(head),feature_names=list(ctx.X),requested_rounds=rounds,training_label_sha256=arrsha(train_label),q_dependency=dependency,
        early_stop_metric='binary_logloss' if head=='q' else 'full_validation_product_rmse_float64',metric_label_dtype='binary_exact' if head=='q' else 'official_float64')
    if rp.exists():
        r=read(rp);assert r['spec']==spec and r['model_id']==digest(spec) and sha(mp)==r['model_sha256']
        if va:assert sha(vp)==r['validation_sha256']
        return lgb.Booster(model_file=str(mp)),r
    if mp.exists():raise RuntimeError('Orphan model without receipt')
    execution['fits_started']+=1;save(RUN/'execution.json',execution);start=time.monotonic();history={}
    training=lgb.Dataset(tr.X,label=train_label)
    if va:
        assert not set(tr.meta.fipsCode)&set(va.meta.fipsCode)
        validation=lgb.Dataset(va.X,label=(va.y>0).astype(float) if head=='q' else va.y,reference=training)
        model=lgb.train(params(head),training,num_boost_round=2000,valid_sets=[validation],valid_names=['validation'],
            feval=product_metric(q,va.y) if head=='m' else None,
            callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.record_evaluation(history),lgb.log_evaluation(0)])
        best=int(model.best_iteration);pred=model.predict(va.X,num_iteration=best,num_threads=1)
        f=va.meta[KEYS].copy();f['row_index']=va.rows;f['official_Z']=va.y;f['occurrence']=(va.y>0).astype(int);f['prediction']=pred
        if head=='m':f['q']=q;f['clipped_m']=np.clip(pred,0,1);f['product']=product(q,pred);f['q_model_id']=q_record['model_id']
        frame(vp,f)
    else:
        model=lgb.train(params(head),training,num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)]);best=None
    atomic(mp,lambda p:model.save_model(str(p),num_iteration=best if va else -1));saved=lgb.Booster(model_file=str(mp))
    r=dict(model_id=digest(spec),spec=spec,model_path=str(mp.relative_to(RUN)),model_sha256=sha(mp),best_iteration=best,actual_trees=saved.current_iteration(),
        curve=history.get('validation',{}).get('binary_logloss' if head=='q' else 'product_rmse',[]),fit_seconds=time.monotonic()-start)
    if va:r.update(validation_path=str(vp.relative_to(RUN)),validation_sha256=sha(vp))
    save(rp,r);execution['fits_completed']+=1;save(RUN/'execution.json',execution)
    print(f"[{execution['fits_completed']}/100] {c} outer{outer} {name}: best={best}, trees={saved.current_iteration()}",flush=True)
    return saved,r
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    ctx=initialize(load_context());pre=read(OUT/'preflight/verification.json');assert pre['status']=='PASS' and pre['identity_hash']==ctx.identity_hash
    if (RUN/'MODEL_CV_COMPLETE').exists():raise RuntimeError('Completed model stage is immutable')
    path=RUN/'run_manifest.json';manifest=dict(identity=ctx.identity,identity_hash=ctx.identity_hash)
    if path.exists():assert args.resume and read(path)==manifest
    else:assert not args.resume;save(path,manifest)
    execution=read(RUN/'execution.json') if (RUN/'execution.json').exists() else dict(status='running',start_utc=datetime.now(timezone.utc).isoformat(),fits_started=0,fits_completed=0)
    records=[];out=ctx.meta[KEYS].copy();out['target_hour']=ctx.meta.hour_idx+1;out['scoreable']=ctx.valid;out['identity_hash']=ctx.identity_hash
    try:
        for c in ['N_t','R_t']:
            arrays={k:np.full(len(ctx.meta),np.nan) for k in ['q','m','prior_q']};ids={k:np.full(len(ctx.meta),'',object) for k in ['q','m']}
            for outer in range(5):
                allowed=set(range(5))-{outer};bests={h:[] for h in ['q','m']}
                for inner in sorted(allowed):
                    _,qr=fit_one(ctx,c,'q',outer,inner,allowed-{inner},2000,execution);records.append(qr);bests['q'].append(qr['best_iteration'])
                    _,mr=fit_one(ctx,c,'m',outer,inner,allowed-{inner},2000,execution,qr);records.append(mr);bests['m'].append(mr['best_iteration'])
                rows=np.flatnonzero(ctx.valid&(ctx.meta.fold.to_numpy()==outer))
                for head in ['q','m']:
                    rounds=max(1,int(np.mean(bests[head])));model,r=fit_one(ctx,c,head,outer,None,allowed,rounds,execution);records.append(r)
                    arrays[head][rows]=model.predict(ctx.X.iloc[rows],num_threads=1);ids[head][rows]=r['model_id']
                arrays['prior_q'][rows]=float((subset(ctx,c,allowed).y>0).mean())
            out['true_'+c]=ctx.ys[c]
            for k,v in arrays.items():out[c+'_'+k]=v
            out[c+'_clipped_m']=np.clip(arrays['m'],0,1);out[c+'_product']=product(arrays['q'],arrays['m'])
            for k,v in ids.items():out[c+'_'+k+'_model_id']=v
            out[c+'_product_id']=[digest(dict(q=q,m=m,formula='q*clip(m,0,1)')) if q else '' for q,m in zip(ids['q'],ids['m'])]
        assert len(records)==100 and execution['fits_started']==execution['fits_completed']==100
        frame(RUN/'NR1_oof_predictions.parquet',out);save(RUN/'model_manifest.json',records)
        for p,h in ctx.sources.items():assert sha(p)==h,p
        execution.update(status='complete',end_utc=datetime.now(timezone.utc).isoformat());save(RUN/'execution.json',execution)
        save(RUN/'MODEL_CV_COMPLETE',dict(identity_hash=ctx.identity_hash,formal_fits=100,outer_models=20,inner_models=80,
            files={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and p.name!='MODEL_CV_COMPLETE'}))
        print('All 100 fits complete',flush=True)
    except BaseException as e:
        execution.update(status='failed',error=repr(e));save(RUN/'execution.json',execution);raise
if __name__=='__main__':main()
