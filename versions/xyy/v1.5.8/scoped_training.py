"""LightGBM fits whose gradient training AND round selection exclude held-out counties."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import make_lgbm_params, HORIZONS
from protocol import expected_mask, target_horizon, sha256_file
from run_artifacts import atomic_write, write_json, digest_object, safe_artifact

@dataclass
class ScopedData:
    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame
    row_folds: np.ndarray
    scope: tuple
    target: str

def scope_data(data, scope, target):
    scope = tuple(sorted(set(scope)))
    if len(scope) not in (4,5) or not set(scope).issubset(range(5)):
        raise ValueError("Tree fits require four outer-training folds or all five folds")
    h = target_horizon(target)
    rows = np.flatnonzero(np.isin(data.row_folds,scope) & expected_mask(data.meta_train,h))
    # Index before inspecting labels: no held-out label enters this object.
    source = data.y_train[target] if target in HORIZONS else data.component_targets[target]
    labels = source.iloc[rows].to_numpy(dtype=float, copy=True)
    if not len(rows) or not np.isfinite(labels).all():
        raise ValueError("Invalid required labels inside the allowed scope")
    return ScopedData(data.X_train.iloc[rows].reset_index(drop=True), labels,
                      data.meta_train.iloc[rows].reset_index(drop=True),
                      data.row_folds[rows].copy(), scope, target)

def provenance(scoped):
    def describe(mask):
        m = scoped.meta.loc[mask]
        keys = (m.fipsCode.astype(str)+"|"+m.timestamp_et.astype(str)).tolist()
        return {"rows":len(m), "counties":sorted(m.fipsCode.unique().tolist()), "row_keys_hash":digest_object(keys)}
    probes = []
    for q in scoped.scope:
        train, valid = scoped.row_folds != q, scoped.row_folds == q
        if not train.any() or not valid.any():
            raise ValueError("Empty inner county split")
        probes.append({"inner_fold":q, "train_folds":[f for f in scoped.scope if f!=q],
                       "early_stop_folds":[q], "train":describe(train), "early_stop":describe(valid)})
    return {"allowed_folds":list(scoped.scope), "refit":describe(np.ones(len(scoped.y),dtype=bool)), "probes":probes}

def fit_scoped_base(scoped, settings):
    if not np.isin(scoped.row_folds, scoped.scope).all() or not np.isfinite(scoped.y).all():
        raise ValueError("Scoped fit received invalid supervision")
    params = make_lgbm_params(settings["model_seed"])
    record = provenance(scoped)
    best = []
    for q in scoped.scope:
        tr, va = scoped.row_folds != q, scoped.row_folds == q
        train_set = lgb.Dataset(scoped.X.loc[tr], label=scoped.y[tr])
        valid_set = lgb.Dataset(scoped.X.loc[va], label=scoped.y[va], reference=train_set)
        probe = lgb.train(params, train_set, num_boost_round=settings["max_rounds"], valid_sets=[valid_set],
                          callbacks=[lgb.early_stopping(settings["early_stopping_rounds"],verbose=False),lgb.log_evaluation(0)])
        iteration = int(probe.best_iteration)
        if iteration < 1:
            raise RuntimeError("Early stopping did not produce a positive round count")
        best.append(iteration)
    rounds = max(1,int(np.mean(best)))
    model = lgb.train(params,lgb.Dataset(scoped.X,label=scoped.y),num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)])
    record.update(best_iterations=best, requested_rounds=rounds, trees_saved=model.current_iteration(), params=params)
    return model, record

def model_spec(scoped, identity_hash):
    return {"run_identity_hash":identity_hash,"target":scoped.target,"scope":list(scoped.scope),
            "feature_names":list(scoped.X.columns),"provenance":provenance(scoped)}

def validate_receipt(receipt, scoped, identity_hash, settings):
    spec = model_spec(scoped,identity_hash)
    if receipt["spec"] != spec or receipt["model_id"] != digest_object(spec):
        raise ValueError("Model scope/target/run identity differs")
    fit = receipt["fit"]
    if {k:fit[k] for k in ("allowed_folds","refit","probes")} != provenance(scoped):
        raise ValueError("Model dependency record differs from required county scope")
    best = fit["best_iterations"]
    if len(best)!=len(scoped.scope) or any(type(n) is not int or not 1<=n<=settings["max_rounds"] for n in best):
        raise ValueError("Invalid early stopping record")
    if fit["requested_rounds"] != max(1,int(np.mean(best))) or fit["params"] != make_lgbm_params(settings["model_seed"]):
        raise ValueError("Invalid refit rounds or parameters")
    if not 1<=fit["trees_saved"]<=fit["requested_rounds"]:
        raise ValueError("Invalid saved tree count")

def load_fit(run_dir, receipt, scoped, identity_hash, settings):
    validate_receipt(receipt,scoped,identity_hash,settings)
    path = safe_artifact(run_dir,receipt["model_path"])
    if not path.is_file() or sha256_file(path)!=receipt["model_sha256"]:
        raise ValueError("Missing or modified saved model")
    model = lgb.Booster(model_file=str(path))
    if model.feature_name()!=list(scoped.X.columns) or model.current_iteration()!=receipt["fit"]["trees_saved"]:
        raise ValueError("Model feature schema/tree count mismatch")
    return model

def get_or_fit(run_dir, data, scope, target, settings, identity_hash):
    scoped = scope_data(data,scope,target)
    role = "final" if len(scoped.scope)==5 else f"outer{next(iter(set(range(5))-set(scoped.scope)))}"
    relative = f"models/{role}/{target}.txt"
    receipt_path = Path(run_dir)/f"models/{role}/{target}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["model_path"] != relative:
            raise ValueError("Unexpected model path in receipt")
        return load_fit(run_dir,receipt,scoped,identity_hash,settings),receipt
    model, fit = fit_scoped_base(scoped,settings)
    model_path = Path(run_dir)/relative
    atomic_write(model_path,lambda tmp:model.save_model(str(tmp)))
    spec = model_spec(scoped,identity_hash)
    receipt = {"spec":spec,"model_id":digest_object(spec),"fit":fit,"model_path":relative,
               "receipt_path":str(receipt_path.relative_to(run_dir)),"model_sha256":sha256_file(model_path)}
    write_json(receipt_path,receipt)  # commit marker, only after the entire fit is saved
    return model,receipt
