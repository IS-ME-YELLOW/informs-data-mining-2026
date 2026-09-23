"""Independent row/label/weight/scope reconstruction and every-checkpoint replay."""
import argparse,time,concurrent.futures,multiprocessing,math
from package_core import *
from train_models import load_model,prediction
_CACHE={}
def get(seed):
    if seed not in _CACHE:
        _CACHE.clear();meta=pd.read_parquet(INPUT/'meta.parquet');fold=pd.read_parquet(INPUT/f'folds_{seed}.parquet').fold.to_numpy()
        xs={k:pd.read_parquet(INPUT/f'features_{k}.parquet') for k in ['base','p1','p24']};ys=pd.read_parquet(INPUT/'labels.parquet');kk=pd.read_parquet(INPUT/'known_history.parquet').K1.to_numpy()
        _CACHE[seed]=(meta,fold,xs,ys,kk)
    return _CACHE[seed]
def check_one(t):
    seed=t['seed'];s=t['scope'];r=RECIPES[t['recipe']];p=paths(t);rec=read(p['receipt']);identity=read(OUT/'identity.json')['identity_hash']
    cache_path=OUT/'audit/model_cache'/t['id']/'verification.json';verifier_hash=sha(__file__)
    if cache_path.exists():
        cached=read(cache_path)
        if cached['verifier_sha256']==verifier_hash:
            assert cached['identity_hash']==identity and cached['task_sha256']==digest(t)
            for name,hsh in cached['artifact_hashes'].items():assert sha(OUT/name)==hsh,name
            return dict(cached['result'],cache_reused=True)
    assert rec['identity_hash']==identity and rec['task']==t and rec['spec']['task']==t and rec['spec']['identity_hash']==identity
    assert rec['spec']['recipe']==r and rec['spec']['params']==params(r)
    for key in ['model','prediction']:assert sha(p[key])==rec[key+'_sha256']
    meta,fold,xs,ys,kk=get(seed);hours=meta.hour_idx.to_numpy();fips=meta.fipsCode.to_numpy();n=len(meta);X=xs[r['feature']];initial=xs['base'].last_P_t.to_numpy();h=r['horizon']
    def selection(allowed,counties=None):
        hit=(fold>=0)&np.isin(fold,allowed)&(hours+h<=215)
        if counties is not None:hit &= np.isin(fips,list(counties))
        total=int(hit.sum())
        if r['kind']=='a':hit &= initial>0
        if r['kind']=='b':hit &= initial<1
        rows=np.flatnonzero(hit);label=ys[r['target']].iloc[rows].to_numpy(copy=True);w=np.ones(len(rows));pp=initial[rows]
        if r['kind']=='a':label=np.minimum(label,pp)/pp;w=pp*pp
        if r['kind']=='b':label=np.maximum(label-pp,0)/(1-pp);w=(1-pp)**2
        if r['kind']=='u':label=label-kk[rows]
        desc=dict(rows=len(rows),total_valid=total,counties=sorted(set(fips[rows])),folds=sorted(set(fold[rows])),row_sha=ah(rows),feature_sha=ah(X.iloc[rows].to_numpy()),label_sha=ah(label),weight_sha=ah(w))
        return rows,label,w,total,desc
    refit=selection(s);assert rec['spec']['refit']==clean(refit[-1]);assert rec['spec']['feature_names']==list(X)
    assert set(fold[refit[0]])==set(s)
    model=load_model(r,p['model']);rounds=rec['requested_rounds'];assert 1<=rounds<=maxrounds(r)
    if r['family']=='lightgbm':assert model.feature_name()==list(X) and model.current_iteration()<=rounds
    stored=np.load(p['prediction']);rows=np.flatnonzero((~np.isin(fold,s))&(hours+h<=215));np.testing.assert_array_equal(stored['rows'],rows)
    pred=prediction(model,r,X.iloc[rows],rounds);np.testing.assert_array_equal(pred,stored['prediction']);maxdiff=0.
    assert rec['prediction_rows']==len(rows) and rec['prediction_row_sha256']==ah(rows)
    probes=0;old_prediction_rows=0;native_replays=0;native_max_delta=0.
    if t['reuse']:
        assert rec['new_fit_calls']==0 and not rec['probes'];up=read(p['folder']/'upstream_receipt.json');assert up['model_id']==t['reuse']['original_model_id']==rec['model_id']
        assert rounds==up['fit']['requested_rounds']
        if 'task' in up:
            assert up['task']['scope']==s and up['task']['target']==r['target']
            oldp=Path(t['reuse']['receipt_path'])
            while not (oldp/'run_manifest.json').exists():
                if oldp==oldp.parent:raise ValueError('No upstream root')
                oldp=oldp.parent
            op=oldp/up['prediction_path'];assert sha(op)==up['prediction_sha256'];ov=np.load(op)
            pp=prediction(model,r,X.iloc[ov['rows']],rounds);np.testing.assert_allclose(pp,ov['predictions'],atol=1e-12,rtol=0);old_prediction_rows=len(pp)
        else:
            oldscope=up['spec'].get('scope',up['spec'].get('allowed_folds'));assert oldscope==s
    else:
        best=[];seen=set()
        for pr in rec['probes']:
            q=pr['probe'];rp=OUT/pr['receipt'];assert sha(rp)==pr['receipt_sha256'];saved=read(rp);seen.add(q)
            if len(s)>1:
                qf=int(q);tr=selection([v for v in s if v!=qf]);va=selection([qf])
            else:
                counties=sorted(set(fips[np.isin(fold,s)]));group=int(q.removeprefix('county_group'))
                tr=selection(s,[c for i,c in enumerate(counties) if i%3!=group]);va=selection(s,[c for i,c in enumerate(counties) if i%3==group])
            assert saved['spec']==clean(dict(train=tr[-1],validation=va[-1],params=params(r,maxrounds(r),True)))
            assert not set(fips[tr[0]])&set(fips[va[0]]) and set(fold[tr[0]])|set(fold[va[0]])<=set(s)
            mp=rp.parent/('model'+extension(r));vp=rp.parent/'validation.npz'
            assert sha(mp)==saved['model_sha256'] and sha(vp)==saved['validation_sha256']
            v=np.load(vp);np.testing.assert_array_equal(v['rows'],va[0]);np.testing.assert_array_equal(v['official_transformed_y'],va[1]);np.testing.assert_array_equal(v['weight'],va[2])
            b=saved['best_iteration'];assert b==pr['best_iteration'];pm=load_model(r,mp);a=prediction(pm,r,X.iloc[va[0]],b);np.testing.assert_array_equal(a,v['prediction']);curve=np.asarray(saved['curve'])
            assert np.isfinite(curve).all() and b==int(np.argmin(curve))+1 and 1<=b<=maxrounds(r)
            if r['family']=='lightgbm':
                truth=va[1] if r['kind'] in ('a','b') else va[1].astype(np.float32).astype(float)
                value=math.sqrt(math.fsum(float(w)*float(e)*float(e) for w,e in zip(va[2],a-truth))/va[3])
                difference=abs(value-curve[b-1]);native_max_delta=max(native_max_delta,float(difference))
                if difference>1e-12:
                    # Best-round native caches can differ from the saved prefix after training.
                    # Rebuild this allowed probe with the same full early-stop schedule, then require BOTH
                    # raw prediction equality and exact native metric equality. Never relax either.
                    audit_path=OUT/'audit/native_metric_replays'/t['id']/f'{q}.json'
                    key=dict(checkpoint_sha256=sha(mp),probe_receipt_sha256=sha(rp),best=b,identity_hash=identity)
                    if audit_path.exists():
                        ar=read(audit_path);assert ar['key']==key and ar['status']=='PASS'
                    else:
                        import lightgbm as lgb
                        weighted=r['kind'] in ('a','b');mean=tr[2].mean()
                        ts=lgb.Dataset(X.iloc[tr[0]],label=tr[1],weight=tr[2]/mean if weighted else None)
                        vs=lgb.Dataset(X.iloc[va[0]],label=va[1],weight=va[2]/mean if weighted else None,reference=ts)
                        cache={}
                        def capture(env):
                            if env.iteration==b-1:cache['native']=env.model._Booster__inner_predict(data_idx=1).copy()
                        capture.order=25
                        def feval(pred,dataset):return 'contribution_rmse',float(np.sqrt(np.sum(va[2]*(pred-va[1])**2)/va[3])),False
                        replay=lgb.train(params(r),ts,num_boost_round=maxrounds(r),valid_sets=[vs],feval=feval if weighted else None,
                            callbacks=[lgb.log_evaluation(0),capture,lgb.early_stopping(100,first_metric_only=True,verbose=False)])
                        assert replay.best_iteration==b
                        rawp=replay.predict(X.iloc[va[0]],num_iteration=b,num_threads=1);np.testing.assert_array_equal(rawp,a)
                        native_metric=math.sqrt(math.fsum(float(w)*float(e)*float(e) for w,e in zip(va[2],cache['native']-truth))/va[3])
                        np.testing.assert_allclose(native_metric,curve[b-1],atol=1e-12,rtol=0)
                        ar=dict(status='PASS',key=key,diagnostic_fit_calls=1,raw_predictions_exact=True,native_metric=native_metric,
                            curve_metric=float(curve[b-1]),raw_inference_metric=value,raw_native_max_prediction_difference=float(abs(a-cache['native']).max()))
                        save(audit_path,ar)
                    native_replays+=1
            else:
                # Native metrics accumulate with backend-specific precision; checkpoint equality is exact above.
                value=np.sqrt(np.mean((a-va[1])**2));np.testing.assert_allclose(value,curve[b-1],atol=2e-7,rtol=1e-5)
            probes+=1;best.append(b)
        assert seen==({str(v) for v in s} if len(s)>1 else {'county_group0','county_group1','county_group2'})
        assert rounds==max(1,sum(best)//len(best)) and rec['new_fit_calls']==len(best)+1
        assert rec['model_id']==digest(dict(spec=rec['spec'],rounds=rounds))
    result=dict(id=t['id'],scope=s,seed=seed,family=r['family'],reused=bool(t['reuse']),checkpoints=probes+1,prediction_rows=len(rows),upstream_rows_reproduced=old_prediction_rows,max_abs_difference=maxdiff,
        native_metric_diagnostic_replays=native_replays,native_raw_metric_max_difference=native_max_delta)
    artifact_paths=[f for f in p['folder'].rglob('*') if f.is_file()]
    native_folder=OUT/'audit/native_metric_replays'/t['id']
    if native_folder.exists():artifact_paths+=list(native_folder.glob('*.json'))
    save(cache_path,dict(status='PASS',identity_hash=identity,task_sha256=digest(t),verifier_sha256=verifier_hash,
        artifact_hashes={str(f.relative_to(OUT)):sha(f) for f in artifact_paths},result=result))
    return dict(result,cache_reused=False)
def verify_inputs():
    identity=read(OUT/'identity.json');assert digest(identity['identity'])==identity['identity_hash']
    assert environment()==identity['identity']['environment']
    for name,h in identity['identity']['code_hashes'].items():assert sha(OUT/name)==h
    for name,h in identity['identity']['input_hashes'].items():assert sha(PORT/name)==h
    for name,h in identity['identity']['source_hashes'].items():assert sha(name)==h
    meta=pd.read_parquet(INPUT/'meta.parquet');labels=pd.read_parquet(INPUT/'labels.parquet');fips=meta.fipsCode.to_numpy();hours=meta.hour_idx.to_numpy()
    raw=pd.read_csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str});raw['hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int);raw=raw.set_index(['fipsCode','hour'])
    for h in HS:
        keys=pd.MultiIndex.from_arrays([fips,hours+h])
        for c in ['osi',*CS]:np.testing.assert_array_equal(labels[target(c,h)],raw[c].reindex(keys).to_numpy())
    history=pd.read_parquet(INPUT/'observed_P_history.parquet');assert history.hour.between(0,71).all()
    lookup=history.set_index(['fipsCode','hour']).P_t;K=np.zeros(len(meta))
    for i,(f,t) in enumerate(zip(fips,hours+1)):
        if t>215:K[i]=np.nan
        elif t<=76:K[i]=sum(float(lookup.loc[(f,k)]) for k in range(t-5,72))/6
    np.testing.assert_allclose(K,pd.read_parquet(INPUT/'known_history.parquet').K1,atol=1e-15,rtol=0,equal_nan=True)
    return dict(raw_labels_exact=True,known_history_independent=True,source_files_unchanged=len(identity['identity']['source_hashes']))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--completed-only',action='store_true');args=ap.parse_args()
    inputs=verify_inputs();tasks=read(OUT/'tasks.json');selected=[t for t in tasks if paths(t)['receipt'].exists()] if args.completed_only else tasks
    start=time.monotonic();rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,r in enumerate(pool.map(check_one,selected),1):
            rows.append(r)
            if i%50==0:print(f'Independently verified {i}/{len(selected)} model jobs',flush=True)
    suffix='_partial' if args.completed_only else '';frame(OUT/f'audit/model_replay{suffix}.csv',rows)
    result=dict(status='PASS',complete=len(selected)==len(tasks),models=len(rows),checkpoints_reloaded=sum(r['checkpoints'] for r in rows),maximum_prediction_difference=max(r['max_abs_difference'] for r in rows),
                prediction_rows=sum(r['prediction_rows'] for r in rows),old_prediction_rows_reproduced=sum(r['upstream_rows_reproduced'] for r in rows),inputs=inputs,seconds=time.monotonic()-start,verifier_sha256=sha(__file__),
                accepted_cached_models=sum(r['cache_reused'] for r in rows),newly_replayed_models=sum(not r['cache_reused'] for r in rows),
                checkpoint_count_semantics='unique accepted checkpoints across this audit and hash-verified cached audits')
    save(OUT/f'audit/model_verification{suffix}.json',result);print(result,flush=True)
if __name__=='__main__':main()
