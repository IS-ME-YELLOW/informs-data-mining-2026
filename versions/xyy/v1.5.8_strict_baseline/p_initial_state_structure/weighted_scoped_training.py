"""Allowed-county first transforms, train-only weight normalization and nested rounds."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import lightgbm as lgb
from p_structure_protocol import (CONFIG, PT, KINDS, TARGETS, support, transform, params, array_sha,
    digest_object, read_json, sha256_file, safe_artifact, write_json, write_frame)
from run_artifacts import atomic_write


def subset(ctx, folds, kind):
    folds = tuple(sorted(set(folds)))
    if not folds or not set(folds).issubset(range(5)) or len(folds) >= 5 or kind not in KINDS:
        raise ValueError('Invalid fold scope or target')
    eligible = ctx.valid & np.isin(ctx.data.row_folds, folds)
    rows = np.flatnonzero(eligible & support(ctx.p,kind))
    if not len(rows):
        raise ValueError('Empty informative branch support')
    # Index the county/time/support subset BEFORE reading any future target values.
    official = ctx.data.component_targets[PT].iloc[rows].to_numpy(dtype=float,copy=True)
    p = ctx.p[rows].copy()
    y,w = transform(official,p,kind)
    if not np.isfinite(w).all() or (w<=0).any() or w.sum()<=0:
        raise ValueError('Invalid positive contribution weights')
    return SimpleNamespace(X=ctx.data.X_train.iloc[rows].reset_index(drop=True), y=y, w=w, p=p,
        rows=rows, meta=ctx.meta.iloc[rows].reset_index(drop=True), folds=folds, kind=kind,
        total_valid=int(eligible.sum()), excluded_rows=np.flatnonzero(eligible & ~support(ctx.p,kind)))


def describe(s):
    keys=(s.meta.fipsCode+'|'+s.meta.timestamp_et.astype(str)).tolist()
    return {'folds':list(s.folds),'counties':sorted(s.meta.fipsCode.unique().tolist()),
            'rows':len(s.y),'total_valid_rows':s.total_valid,'omitted_rows':len(s.excluded_rows),
            'row_indices_sha256':array_sha(s.rows),'row_keys_sha256':digest_object(keys),
            'omitted_indices_sha256':array_sha(s.excluded_rows),'label_sha256':array_sha(s.y),
            'feature_sha256':array_sha(s.X.to_numpy()),'p_sha256':array_sha(s.p),
            'weight_sha256':array_sha(s.w),'weight_mean':float(s.w.mean()),
            'weight_min':float(s.w.min()),'weight_max':float(s.w.max()),
            'normalized_weight_sha256':array_sha(s.w/s.w.mean()),
            'normalized_mean':float(np.mean(s.w/s.w.mean())),
            'row_weight_ess':float(s.w.sum()**2/np.square(s.w).sum())}


def specification(ctx, outer, kind, identity):
    scope=tuple(f for f in range(5) if f!=outer)
    probes=[]
    for q in scope:
        tr=subset(ctx,[f for f in scope if f!=q],kind)
        va=subset(ctx,[q],kind)
        probes.append({'inner_fold':q,'train':describe(tr),'validation':describe(va)})
    return {'identity_hash':identity,'outer_fold':outer,'target':TARGETS[kind],'kind':kind,
            'allowed_folds':list(scope),'feature_names':list(ctx.data.X_train.columns),
            'probes':probes,'refit':describe(subset(ctx,scope,kind)),'params':params(kind)}


def contribution_metric(y, w, n_all):
    y,w=np.asarray(y).copy(),np.asarray(w).copy()
    if n_all<len(y) or not len(y) or w.sum()<=0:
        raise ValueError('Invalid validation support/denominator')
    def evaluate(pred, dataset):
        return 'raw_P_contribution_rmse',float(np.sqrt(np.sum(w*(pred-y)**2)/n_all)),False
    return evaluate


def load_model(run, receipt, spec):
    if receipt['spec']!=spec or receipt['model_id']!=digest_object(spec):
        raise ValueError('Model target/scope/weights/identity mismatch')
    fit=receipt['fit']; best=fit['best_iterations']
    if len(best)!=4 or any(type(b) is not int or not 1<=b<=2000 for b in best):
        raise ValueError('Invalid nested early-stopping rounds')
    if fit['requested_rounds']!=max(1,int(np.mean(best))) or not 1<=fit['actual_trees']<=fit['requested_rounds']:
        raise ValueError('Invalid final rounds/tree count')
    for name in ('model_path','probe_predictions_path'):
        path=safe_artifact(run,receipt[name])
        if sha256_file(path)!=receipt[name.replace('_path','_sha256')]:
            raise ValueError('Saved fit artifact changed')
    model=lgb.Booster(model_file=str(safe_artifact(run,receipt['model_path'])))
    if model.feature_name()!=spec['feature_names'] or model.current_iteration()!=fit['actual_trees']:
        raise ValueError('Saved model schema/tree mismatch')
    return model


def get_or_fit(ctx,run,outer,kind,identity,progress):
    spec=specification(ctx,outer,kind,identity)
    prefix=f'models/outer{outer}/{TARGETS[kind]}'
    receipt_path=run/f'{prefix}.json'
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if receipt['model_path']!=f'{prefix}.txt' or receipt['receipt_path']!=f'{prefix}.json':
            raise ValueError('Unexpected receipt location')
        return load_model(run,receipt,spec),receipt
    best=[]; curves=[]; probe_frames=[]
    for record in spec['probes']:
        q=record['inner_fold']
        tr=subset(ctx,record['train']['folds'],kind); va=subset(ctx,[q],kind)
        mean=float(tr.w.mean())
        train=lgb.Dataset(tr.X,label=tr.y,weight=None if kind=='direct' else tr.w/mean)
        valid=lgb.Dataset(va.X,label=va.y,weight=None if kind=='direct' else va.w/mean,reference=train)
        curve={}
        feval=None if kind=='direct' else contribution_metric(va.y,va.w,va.total_valid)
        model=lgb.train(params(kind),train,num_boost_round=2000,valid_sets=[valid],valid_names=['validation'],
            feval=feval,callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),
                                  lgb.record_evaluation(curve),lgb.log_evaluation(0)])
        iteration=int(model.best_iteration)
        if not 1<=iteration<=2000:
            raise RuntimeError('Invalid best iteration')
        pred=model.predict(va.X,num_iteration=iteration,num_threads=1)
        metric_name='rmse' if kind=='direct' else 'raw_P_contribution_rmse'
        values=curve['validation'][metric_name]
        frame=va.meta.copy();frame['outer_fold']=outer;frame['inner_fold']=q;frame['kind']=kind
        frame['official_transformed_label']=va.y
        frame['metric_label']=va.y.astype(np.float32).astype(float) if kind=='direct' else va.y
        frame['raw_weight']=va.w;frame['raw_prediction_at_best']=pred
        frame['all_validation_rows']=va.total_valid;frame['best_iteration']=iteration
        probe_frames.append(frame);best.append(iteration)
        curves.append({'inner_fold':q,'metric':metric_name,'values':values,'best_value':float(values[iteration-1]),
                       'validation_weight_scale_training_mean':mean})
        progress(f'outer{outer} {kind} inner{q}: best={iteration}',1)
    refit=subset(ctx,spec['allowed_folds'],kind)
    rounds=max(1,int(np.mean(best)))
    model=lgb.train(params(kind),lgb.Dataset(refit.X,label=refit.y,
        weight=None if kind=='direct' else refit.w/refit.w.mean()),num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)])
    atomic_write(run/f'{prefix}.txt',lambda tmp:model.save_model(str(tmp)))
    probe_name=f'logs/probes/outer{outer}_{kind}.parquet'
    write_frame(run/probe_name,pd.concat(probe_frames,ignore_index=True))
    receipt={'spec':spec,'model_id':digest_object(spec),'model_path':f'{prefix}.txt',
        'receipt_path':f'{prefix}.json','model_sha256':sha256_file(run/f'{prefix}.txt'),
        'probe_predictions_path':probe_name,'probe_predictions_sha256':sha256_file(run/probe_name),
        'fit':{'best_iterations':best,'requested_rounds':rounds,'actual_trees':model.current_iteration(),
               'curves':curves,'formal_fit_count':5,'refit_weight_mean':float(refit.w.mean())}}
    write_json(receipt_path,receipt)
    progress(f'outer{outer} {kind}: refit={rounds}, trees={model.current_iteration()}',1)
    return model,receipt
