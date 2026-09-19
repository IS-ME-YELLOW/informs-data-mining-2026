"""No production training. Tiny synthetic fits test leakage, resume and artifacts."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import PROJECT_ROOT, HORIZONS, HORIZON_HOURS, COMPONENTS, RUNS_DIR
from protocol import (ExperimentData, TARGETS, load_data, load_cv_folds, expected_mask,
                      metrics, build_controls, component_target_name)
from scoped_training import scope_data, fit_scoped_base, get_or_fit
from run_artifacts import prepare_run, sha256_file, data_identity, digest_object
from train_v18 import run_stage, generate_predictions, parse_args
from verify_artifacts import verify_loaded_run

SETTINGS={"model_seed":42,"max_rounds":2,"early_stopping_rounds":1,"diagnostic":True,
          "bootstrap_replicates":20,"bootstrap_seed":20260910}

def synthetic(directory):
    def rows(count,start):
        meta=pd.DataFrame({"fipsCode":np.repeat([str(start+i) for i in range(count)],144),
                           "hour_idx":np.tile(np.arange(72,216),count)})
        meta["timestamp_et"]=pd.Timestamp("2026-03-11")+pd.to_timedelta(meta.hour_idx,unit="h")
        meta["stateAbbr"]="IN"
        meta["severity_tier"]=0
        number=np.repeat(np.arange(count),144)
        x=pd.DataFrame({"weather":np.sin(meta.hour_idx/13),"observed":.01+.005*number,
                        "time":meta.hour_idx/216})
        return x,meta,number
    x,meta,number=rows(10,18000)
    xt,mt,_=rows(2,19000)
    targets=meta[["fipsCode","timestamp_et","hour_idx"]].copy()
    y=pd.DataFrame(index=meta.index)
    for h in HORIZONS:
        s=meta.hour_idx.to_numpy()+HORIZON_HOURS[h]
        parts={"P_t":.035+.025*np.sin(s/13)+.003*number,
               "D_t":.03+.02*np.sin((s-2)/13)+.002*number,
               "N_t":.004*(1+np.cos(s/13)),"R_t":.003*(1-np.cos(s/13))}
        for c in COMPONENTS:
            values=parts[c].copy();values[~expected_mask(meta,h)]=np.nan
            targets[component_target_name(c,h)]=values
        values=.4*parts["P_t"]+.35*parts["N_t"]+.25*parts["D_t"]-.1*parts["R_t"]
        values[~expected_mask(meta,h)]=np.nan
        y[h]=values
    assignment=meta[["fipsCode","stateAbbr","severity_tier"]].drop_duplicates().reset_index(drop=True)
    assignment["fold"]=np.arange(10)%5
    row_fold=meta.fipsCode.map(assignment.set_index("fipsCode").fold).to_numpy()
    folds=[(np.flatnonzero(row_fold!=f),np.flatnonzero(row_fold==f)) for f in range(5)]
    template=mt[["fipsCode","stateAbbr","timestamp_et"]].copy()
    template["countyName"]="Synthetic county"
    template["timestamp_et"]=template.timestamp_et.dt.strftime("%m/%d/%Y %H:%M")
    for h in HORIZONS:
        template[h]=np.where(expected_mask(mt,h),0.,np.nan)
    template_file=directory/"template.csv";template.to_csv(template_file,index=False)
    fixture_file=directory/"fixture.parquet";pd.concat([meta,y,targets.drop(columns=["fipsCode","timestamp_et","hour_idx"])],axis=1).to_parquet(fixture_file,index=False)
    return ExperimentData(x,y,meta,xt,mt,folds,row_fold,targets,assignment,
                          {"submission":template_file,"fixture":fixture_file})

class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real=load_data()

    def test_all_frozen_folds_and_readonly_preflight(self):
        import train_v18
        paths=list(self.real.input_paths.values())
        before={str(p):(sha256_file(p),p.stat().st_mtime_ns) for p in paths}
        runs_before=sorted(str(p) for p in RUNS_DIR.rglob('*')) if RUNS_DIR.exists() else []
        with patch.object(lgb,"train",side_effect=AssertionError("preflight tried training")):
            for seed in [42,20260917,20260918]:
                with redirect_stdout(io.StringIO()):
                    train_v18.main(["--cv-file",f"cv/cv_assignments_balanced_v1_seed{seed}.csv"])
        self.assertEqual(before,{str(p):(sha256_file(p),p.stat().st_mtime_ns) for p in paths})
        self.assertEqual(runs_before,sorted(str(p) for p in RUNS_DIR.rglob('*')) if RUNS_DIR.exists() else [])
        cp=subprocess.run([sys.executable,"-B",str(HERE/'train_v18.py'),"--validate-only"],cwd="/tmp",capture_output=True,text=True)
        self.assertEqual(cp.returncode,0,cp.stderr)
        self.assertFalse(json.loads(cp.stdout)["training_performed"])

    def test_invalid_cv_and_missing_predictions_fail(self):
        for value in [np.nan,np.inf,-np.inf]:
            with self.assertRaises(ValueError): metrics([0.,1.],[0.,value])
        with tempfile.TemporaryDirectory() as tmp:
            a=self.real.assignment.copy();a['fold']=a.fold.astype(float);a.loc[0,'fold']=1.5
            p=Path(tmp)/'bad.csv';a.to_csv(p,index=False)
            with self.assertRaises(ValueError):load_cv_folds(self.real.meta_train,p)
            # Global counts remain balanced, but a swap breaks per-stratum balance.
            meta=pd.DataFrame({'fipsCode':[str(18000+i) for i in range(10)],
                               'stateAbbr':['IN']*10,'severity_tier':[0]*5+[1]*5})
            a=meta.copy();a['fold']=list(range(5))*2
            a.loc[0,'fold'],a.loc[6,'fold']=1,0
            a.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'within state'):load_cv_folds(meta,p)
        with redirect_stdout(io.StringIO()), patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit):parse_args(['--stage','cv'])
            with self.assertRaises(SystemExit):parse_args(['--max-rounds','1'])

    def test_existing_C1_C3_and_fixed_rule_are_preserved(self):
        saved=pd.read_parquet(HERE/'oof_predictions.parquet')
        parts=pd.read_parquet(HERE/'oof_component_predictions.parquet')
        direct={h:saved[f'pred_C0_direct_osi_{h}'].to_numpy() for h in HORIZONS}
        components={h:{c:parts[f'pred_{component_target_name(c,h)}'].to_numpy() for c in COMPONENTS} for h in HORIZONS}
        controls,_=build_controls(self.real.meta_train,direct,components)
        for h in HORIZONS:
            for mode in ['C1_component_osi','C3_aligned_component']:
                np.testing.assert_allclose(controls[mode][h],saved[f'pred_{mode}_{h}'],atol=1e-12,rtol=0)
            source='C3_aligned_component' if HORIZON_HOURS[h]<=6 else 'C1_component_osi'
            np.testing.assert_array_equal(controls['v18_rule'][h],controls[source][h])
        broken=copy.deepcopy(components);broken[HORIZONS[0]]['P_t'][0]=np.nan
        with self.assertRaises(ValueError):build_controls(self.real.meta_train,direct,broken)

class ScopedTrainingTests(unittest.TestCase):
    def test_all_targets_outer_label_perturbation_real_lightgbm(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=synthetic(Path(tmp));changed=copy.deepcopy(data)
            held=data.row_folds==0
            # Alter all held-out OSI AND component labels, keeping tail masks unchanged.
            for frame in [changed.y_train,changed.component_targets]:
                for target in TARGETS:
                    if target in frame:
                        mask=held & np.isfinite(frame[target])
                        frame.loc[mask,target]=.8
            for target in TARGETS:
                a=scope_data(data,(1,2,3,4),target);b=scope_data(changed,(1,2,3,4),target)
                self.assertFalse(set(a.meta.fipsCode)&set(data.meta_train.loc[held,'fipsCode']))
                model_a,record_a=fit_scoped_base(a,SETTINGS)
                model_b,record_b=fit_scoped_base(b,SETTINGS)
                self.assertEqual(record_a,record_b)
                self.assertEqual(model_a.model_to_string(),model_b.model_to_string())
                np.testing.assert_allclose(model_a.predict(data.X_train.loc[held],num_threads=1),
                                           model_b.predict(data.X_train.loc[held],num_threads=1),atol=1e-12,rtol=0)
                self.assertEqual(len(record_a['probes']),4)
                for probe in record_a['probes']:
                    self.assertEqual(len(probe['train_folds']),3)
                    self.assertNotIn(0,probe['train_folds']+probe['early_stop_folds'])

    def test_real_estimator_callbacks_only_receive_allowed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=synthetic(Path(tmp));scope=scope_data(data,(0,1,2,3),HORIZONS[0])
            calls=[];original=lgb.train
            def record(params,dataset,**kwargs):
                self.assertTrue(np.isfinite(dataset.label).all())
                calls.append((len(dataset.label),[len(v.label) for v in kwargs.get('valid_sets',[])]))
                return original(params,dataset,**kwargs)
            with patch.object(lgb,'train',side_effect=record):fit_scoped_base(scope,SETTINGS)
            self.assertEqual(len(calls),5)
            self.assertEqual(calls[:4],[(6*143,[2*143])]*4)
            self.assertEqual(calls[4],(8*143,[]))

class ArtifactTests(unittest.TestCase):
    def test_partial_resume_cv_final_reload_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=synthetic(root);run=root/'run'
            manifest=prepare_run(run,'synthetic',data,SETTINGS)
            identity=manifest['identity_hash']
            # Interrupt after one fully saved scope; its model must be reused without fit.
            first,receipt=get_or_fit(run,data,(1,2,3,4),HORIZONS[0],SETTINGS,identity)
            with patch.object(lgb,'train',side_effect=AssertionError('resume retrained complete model')):
                second,again=get_or_fit(run,data,(1,2,3,4),HORIZONS[0],SETTINGS,identity)
            self.assertEqual(receipt,again)
            self.assertEqual(first.model_to_string(),second.model_to_string())
            prepare_run(run,'synthetic',data,SETTINGS,resume=True)
            relocated=copy.deepcopy(data)
            alternate=root/'alternate.parquet';alternate.write_bytes(data.input_paths['fixture'].read_bytes())
            relocated.input_paths['fixture']=alternate
            with self.assertRaisesRegex(ValueError,'locations differ'):prepare_run(run,'synthetic',relocated,SETTINGS,resume=True)
            modified=copy.deepcopy(data);modified.assignment['fold']=(modified.assignment.fold+1)%5
            with self.assertRaises(ValueError):prepare_run(run,'synthetic',modified,SETTINGS,resume=True)
            with redirect_stdout(io.StringIO()):run_stage(run,data,manifest,'cv')
            self.assertTrue((run/'CV_COMPLETE').is_file())
            self.assertFalse((run/'models/final').exists())
            cv_before={str(p.relative_to(run)):sha256_file(p) for p in run.rglob('*') if p.is_file()}
            with patch.object(lgb,'train',side_effect=AssertionError('verification trained')):
                result=verify_loaded_run(run,data,'cv')
                self.assertEqual(result['models_reloaded'],100)
                with redirect_stdout(io.StringIO()):generate_predictions(run,data,SETTINGS,identity,'cv')
            with redirect_stdout(io.StringIO()):run_stage(run,data,manifest,'final')
            self.assertTrue((run/'COMPLETE').is_file())
            for p,digest in cv_before.items():self.assertEqual(sha256_file(run/p),digest,p)
            with patch.object(lgb,'train',side_effect=AssertionError('verification trained')):
                self.assertEqual(verify_loaded_run(run,data,'final')['models_reloaded'],20)
            # Even without relying on a completion marker, recomputation rejects a missing prediction.
            table=run/'oof_predictions.parquet';original=table.read_bytes()
            frame=pd.read_parquet(table);frame.loc[0,f'pred_v18_rule_{HORIZONS[0]}']=np.nan
            frame.to_parquet(table,index=False)
            with self.assertRaises((ValueError,AssertionError)):verify_loaded_run(run,data,'cv',require_marker=False)
            table.write_bytes(original)
            model_path=run/receipt['model_path'];model_path.write_bytes(model_path.read_bytes()+b'\n')
            with self.assertRaises(ValueError):get_or_fit(run,data,(1,2,3,4),HORIZONS[0],SETTINGS,identity)

if __name__=='__main__':
    unittest.main(verbosity=2)
