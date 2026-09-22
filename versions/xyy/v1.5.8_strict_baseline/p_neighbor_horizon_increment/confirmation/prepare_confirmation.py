"""Audited adaptation of completed seed42 code; no parent mutation or fitting."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent;PARENT=HERE.parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def once(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new)


def funcs(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}


def main():
    (HERE/'reference').mkdir(exist_ok=True)
    complete=json.loads((PARENT/'completion.json').read_text());assert complete['status']=='COMPLETE'
    history={str(PARENT/n):h for n,h in complete['files'].items()};history[str(PARENT/'completion.json')]=sha(PARENT/'completion.json')
    for p,h in history.items():assert sha(p)==h,p
    hp=HERE/'reference/parent_input_hashes.json'
    if hp.exists():assert json.loads(hp.read_text())==history
    else:hp.write_text(json.dumps(history,indent=2)+'\n')
    audit={}
    for name in ('neighbor_protocol.py','neighbor_features.py','independent_features.py','train_neighbors.py','evaluate_neighbors.py','verify_neighbors.py'):
        old=(PARENT/name).read_text();s=old
        if name=='neighbor_protocol.py':
            s=once(s,'OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]',
                'from confirmation_runtime import CONFIRMATION,PARENT,OUT,BASE,ROOT,SEED,CV_FILE,REFERENCE_ID')
            s=once(s,'from protocol import load_data,expected_mask,build_controls,clip_component,component_target_name',
                'from protocol import load_data as _baseline_load_data,expected_mask,build_controls,clip_component,component_target_name')
            s=once(s,"RUN=OUT/'runs/v18_pneighbor_v1_split42_model42'","RUN=OUT/f'runs/v18_pneighbor_v1_split{SEED}_model42'")
            s=once(s,"CURRENT=PTRANSFER/'runs/v18_ptransfer_ab_v1_split42_model42'","CURRENT=PTRANSFER/f'confirmation/split{SEED}/runs/v18_ptransfer_ab_v1_split{SEED}_model42'")
            s=once(s,"F2=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42'","F2=BASE/f'p_information_increment/runs/v18_pinfo_v1_split{SEED}_model42'")
            s=once(s,"FEATURES=OUT/'features/v1'","FEATURES=PARENT/'features/v1'")
            s=once(s,"REFERENCE_ID='7c971d4269ff6d23fb520c28cfe0e4d365f8cea7237ced5369f0b1f226d3b6a0'\n",'')
            s=once(s,'CHANGED=HORIZONS[1:]',"CHANGED=('osi_target_t24h',)")
            s=once(s,"CASES={'B0':None,'NB_P06':CHANGED[0],'NB_P24':CHANGED[1],'NB_P48':CHANGED[2]}","CASES={'B0':None,'NB_P24':CHANGED[0]}")
            s=once(s,"TASKS=((CHANGED[0],'a'),(CHANGED[0],'b'),(CHANGED[1],'direct'),(CHANGED[2],'direct'))","TASKS=((CHANGED[0],'direct'),)")
            s=once(s,"TRAIN_CODE=('neighbor_protocol.py','neighbor_features.py','independent_features.py','train_neighbors.py')",
                "TRAIN_CODE=('confirmation_runtime.py','neighbor_protocol.py','neighbor_features.py','independent_features.py','train_neighbors.py')\n\ndef load_data():\n    return _baseline_load_data(cv_file=CV_FILE)")
            s=once(s,"if v['split_seed']==42)","if v['split_seed']==SEED)")
            s=once(s,"    add(SOURCE_MANIFEST)","    add(SOURCE_MANIFEST)\n    add(CONFIRMATION/'reference/adaptation_verification.json')\n    add(CONFIRMATION/'Confirmation_Protocol_2026-09-20.md')\n    add(FEATURES/'feature_manifest.json')\n    for path,digest in read_json(FEATURES/'feature_manifest.json')['files'].items():add(FEATURES/path,digest)")
            s=once(s,'split_seed=42,model_seed=42','split_seed=SEED,model_seed=42')
            s=once(s,'formal_fits=100,new_outer_models=20','formal_fits=25,new_outer_models=5')
            s=once(s,"protocol_sha256=sha256_file(OUT/'Experiment_Protocol_2026-09-20.md')","protocol_sha256=sha256_file(CONFIRMATION/'Confirmation_Protocol_2026-09-20.md')")
            s=s.replace('sha256_file(OUT/p)','sha256_file(CONFIRMATION/p)')
            adapted={'load_data','load_context','snapshot_sources','initialize','attach_features','assert_inputs'}
        elif name=='neighbor_features.py':
            index=s.index('\ndef build(ctx):')
            s=s[:index]+'''\ndef build(ctx):
    """Re-use the immutable seed42 input-only package; never write FEATURES."""
    m=read_json(FEATURES/'feature_manifest.json')
    assert digest_object(m['identity'])==m['identity_hash']
    for path,digest in m['files'].items():assert sha256_file(FEATURES/path)==digest,path
    assert read_json(FEATURES/'h24_schema.json')==list(ctx.data.X_train)+added_columns(24)
'''
            adapted={'build'}
        elif name=='independent_features.py':
            s=once(s,'for h in (6,24,48):','for h in (24,):')
            s=once(s,'horizons=[6,24,48]','horizons=[24]')
            s=once(s,'lineage_rows=1043712','lineage_rows=347904')
            adapted={'verify_features'}
        elif name=='train_neighbors.py':
            s=s.replace('100 strict nested fits: P6 a/b and P24/P48 direct Huber with neighbor inputs.','25 strict nested fits for one P24 neighbor confirmation CV.')
            s=once(s,"v18_tree_nested_v2_split42_model42/models/outer{outer}","v18_tree_nested_v2_split{SEED}_model42/models/outer{outer}")
            s=s.replace('20 allowed scopes','5 allowed scopes')
            s=s.replace('formal_fits=100,outer_models=20','formal_fits=25,outer_models=5')
            s=s.replace('All 100 fits completed: 20 outer models, three independent neighbor candidates','All 25 fits completed: 5 outer models, one P24 neighbor candidate')
            s=once(s,"ap.add_argument('--resume',action='store_true');a=ap.parse_args()",
                "ap.add_argument('--resume',action='store_true');ap.add_argument('--split-seed',type=int,choices=(20260917,20260918),required=True);a=ap.parse_args()")
            adapted={'preflight','train','main'}
        elif name=='evaluate_neighbors.py':
            s=once(s,"('branch_loss_decomposition',decompositions)","('branch_loss_decomposition',pd.DataFrame(decompositions,columns=['horizon','a_mse','b_mse','twice_cross_mean','P_mse']))")
            adapted={'evaluate'}
        else:
            s=s.replace('sha256_file(OUT/name)','sha256_file(CONFIRMATION/name)')
            s=s.replace("sha256_file(OUT/'evaluate_neighbors.py')","sha256_file(CONFIRMATION/'evaluate_neighbors.py')")
            s=s.replace('20 reloaded models, 80 probes, 80 control scores, 18 intervals','5 reloaded models, 20 probes, 40 control scores, 6 intervals')
            adapted={'verify'}
        before,after=funcs(old),funcs(s);same=[f for f in before if f not in adapted]
        for f in same:assert before[f]==after[f],(name,f)
        target=HERE/name
        if target.exists():assert target.read_text()==s,'Refuse to overwrite changed adapted engine'
        else:target.write_text(s)
        audit[name]=dict(parent_sha256=sha(PARENT/name),adapted_sha256=sha(target),
                         unchanged_function_AST=same,allowed_adapted_functions=sorted(adapted))
    result=dict(status='PASS',scope='CV/source/output identities; P24 only; immutable feature package reuse; counts',
        files=audit,parent_frozen_files=len(history),explicit_CV_loader=True,model_seed_unchanged=42,
        runtime_sha256=sha(HERE/'confirmation_runtime.py'),preparation_sha256=sha(Path(__file__)))
    output=HERE/'reference/adaptation_verification.json'
    if output.exists():assert json.loads(output.read_text())==result
    else:output.write_text(json.dumps(result,indent=2)+'\n')
    print('Adaptation PASS: same numeric fit and metrics, explicit CV, P24 only, frozen features read-only')


if __name__=='__main__':main()
