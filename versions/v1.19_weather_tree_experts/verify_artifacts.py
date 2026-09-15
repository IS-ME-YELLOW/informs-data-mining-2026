from __future__ import annotations
import json
import numpy as np
import pandas as pd
import joblib
from run_weather_tree_experts import ARTIFACTS_DIR, DEPTHS, HORIZONS, INPUTS, MODELS_DIR, PROJECT_ROOT, feature_names, load_experiment_data, path, predict, sha256_file, source_dict, write_json, FEATURE_VERSION

def compare(a, b, label):
    x, y = np.asarray(a, float), np.asarray(b, float)
    if not np.array_equal(np.isfinite(x), np.isfinite(y)): raise ValueError(f"{label}: masks")
    mask = np.isfinite(x); maximum = float(np.max(np.abs(x[mask]-y[mask]))) if mask.any() else 0.0
    if maximum > 1e-12: raise ValueError(f"{label}: {maximum}")
    return maximum

def main():
    data=load_experiment_data(FEATURE_VERSION); v=pd.read_parquet(INPUTS['v112_oof']); e=pd.read_parquet(INPUTS['et_oof']); r=pd.read_parquet(INPUTS['rf_oof'])
    vt=pd.read_csv(INPUTS['v112_test'],dtype={'fipsCode':str}); et=pd.read_parquet(INPUTS['et_test']); rt=pd.read_parquet(INPUTS['rf_test']); v18t=pd.read_parquet(INPUTS['v18_test'])
    saved=pd.read_parquet(ARTIFACTS_DIR/'oof_predictions.parquet'); savedt=pd.read_parquet(ARTIFACTS_DIR/'test_predictions.parquet'); folds=v.fold.to_numpy(int)
    maximum=0.0; loaded=0
    for h in HORIZONS:
        y=v[f'actual_{h}'].to_numpy(float); s=source_dict(v,e,r,h); st={'v1.8':v18t[f'pred_C1_component_osi_{h}'].to_numpy(float),'v1.12':vt[f'pred_tail_protected_{h}'].to_numpy(float),'ET-A':et[f'pred_ET-A_{h}'].to_numpy(float),'ET-B':et[f'pred_ET-B_{h}'].to_numpy(float),'RF':rt[f'pred_RF_{h}'].to_numpy(float)}
        for d in DEPTHS:
            name=f'weather_gate_depth{d}'; oof=np.full(len(y),np.nan)
            for fold in range(5):
                outer=(folds==fold)&np.isfinite(y); b=joblib.load(path(h,d,f'fold{fold}')); loaded+=1; oof[outer]=predict(b,data.X_train.loc[outer],{k:z[outer] for k,z in s.items()})
            maximum=max(maximum,compare(saved[f'pred_{name}_{h}'],oof,f'OOF/{name}/{h}'))
            b=joblib.load(path(h,d,'final')); loaded+=1; maximum=max(maximum,compare(savedt[f'pred_{name}_{h}'],predict(b,data.X_test,st),f'test/{name}/{h}'))
    m=json.loads((ARTIFACTS_DIR/'run_metadata.json').read_text(encoding='utf-8')); badin=[n for n,p in INPUTS.items() if sha256_file(p)!=m['input_hashes'][n]]; bada=[rel for rel,d in m['artifact_hashes'].items() if not (PROJECT_ROOT/rel).exists() or sha256_file(PROJECT_ROOT/rel)!=d]
    expected=len(HORIZONS)*len(DEPTHS)*6; result={'status':'pass' if loaded==expected and not badin and not bada else 'fail','loaded_models':loaded,'expected_models':expected,'max_prediction_difference':maximum,'input_hashes_valid':not badin,'artifact_hashes_valid':not bada,'bad_inputs':badin,'bad_artifacts':bada}; write_json(ARTIFACTS_DIR/'verification.json',result)
    if result['status']!='pass': raise ValueError(str(result))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
