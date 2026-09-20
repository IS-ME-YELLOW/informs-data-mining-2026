"""Narrow, audited adaptation of completed seed42 engines; never edits parent files."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace_once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def functions(text):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


def main():
    (HERE/'reference').mkdir(exist_ok=True)
    completed=json.loads((PARENT/'completion.json').read_text());assert completed['status']=='COMPLETE'
    historical={str(PARENT/path):digest for path,digest in completed['files'].items()}
    historical[str(PARENT/'completion.json')]=sha(PARENT/'completion.json')
    for path,digest in historical.items():assert sha(path)==digest,path
    history=HERE/'reference/parent_input_hashes.json'
    if history.exists():assert json.loads(history.read_text())==historical
    else:history.write_text(json.dumps(historical,indent=2)+'\n')
    audit={}
    for name in ('transfer_protocol.py','train_transfer.py','evaluate_transfer.py','verify_transfer.py'):
        old=(PARENT/name).read_text();text=old
        if name=='transfer_protocol.py':
            text=replace_once(text,'OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]',
                'from confirmation_runtime import CONFIRMATION,OUT,BASE,ROOT,SEED,F2_ID,CV_FILE')
            text=replace_once(text,'from protocol import load_data,expected_mask,build_controls,clip_component,component_target_name',
                'from protocol import load_data as _baseline_load_data,expected_mask,build_controls,clip_component,component_target_name')
            text=replace_once(text,"RUN=OUT/'runs/v18_ptransfer_ab_v1_split42_model42'", "RUN=OUT/f'runs/v18_ptransfer_ab_v1_split{SEED}_model42'")
            text=replace_once(text,"REF=BASE/'p_information_increment/runs/v18_pinfo_v1_split42_model42'", "REF=BASE/f'p_information_increment/runs/v18_pinfo_v1_split{SEED}_model42'")
            text=replace_once(text,"F2_ID='e39b76d75f191891abda1939dc47933815cc4fa15796227b0c200eb1b96bfcac'\n",'')
            text=replace_once(text,'TRANSFER=tuple(h for h in HORIZONS if HORIZON_HOURS[h]>1)',"TRANSFER=('osi_target_t06h',)")
            text=replace_once(text,"CASES={'B0':None,'AB_P06':TRANSFER[0],'AB_P24':TRANSFER[1],'AB_P48':TRANSFER[2]}","CASES={'B0':None,'AB_P06':TRANSFER[0]}")
            text=replace_once(text,"TRAIN_CODE=('transfer_protocol.py','train_transfer.py')", "TRAIN_CODE=('confirmation_runtime.py','transfer_protocol.py','train_transfer.py')\n\ndef load_data():\n    return _baseline_load_data(cv_file=CV_FILE)")
            text=replace_once(text,"    add(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv')", "    add(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv')\n    add(CONFIRMATION/'reference/adaptation_verification.json')\n    add(CONFIRMATION/'Confirmation_Protocol_2026-09-19.md')")
            text=replace_once(text,"split_seed=42,model_seed=42", "split_seed=SEED,model_seed=42")
            text=replace_once(text,'formal_fits=150,new_outer_models=30','formal_fits=50,new_outer_models=10')
            text=replace_once(text,"authorized_scope='seed42; three single-source AB migrations only; no joint or feature increment'", "authorized_scope='P6 only; selected confirmation CV; no joint or feature increment'")
            text=replace_once(text,"protocol_sha256=sha256_file(OUT/'Experiment_Protocol_2026-09-19.md')", "protocol_sha256=sha256_file(CONFIRMATION/'Confirmation_Protocol_2026-09-19.md')")
            text=text.replace('sha256_file(OUT/p)','sha256_file(CONFIRMATION/p)')
            changed={'load_data','sources','initialize','assert_inputs'}
        elif name=='train_transfer.py':
            text=text.replace('150 scoped fits for three separately evaluated P a/b source migrations.','50 scoped fits for one P6 confirmation CV.')
            text=text.replace('30 branch scopes','10 branch scopes')
            text=text.replace('formal_fits=150,outer_models=30','formal_fits=50,outer_models=10')
            text=text.replace('All 150 fits completed; 30 outer models and three single-source P reconstructions saved',
                              'All 50 fits completed; 10 outer models and one P6 reconstruction saved')
            text=replace_once(text,"p.add_argument('--resume',action='store_true');a=p.parse_args()", "p.add_argument('--resume',action='store_true');p.add_argument('--split-seed',type=int,choices=(20260917,20260918),required=True);a=p.parse_args()")
            changed={'main','preflight','train'}
        elif name=='evaluate_transfer.py':
            text=text.replace('Three predeclared single-source P transfers; no joint or adaptive blending.','Frozen P6 confirmation; no joint or adaptive blending.')
            changed=set()
        else:
            text=replace_once(text,'import lightgbm as lgb','import lightgbm as lgb\nfrom confirmation_runtime import CONFIRMATION')
            text=text.replace('sha256_file(OUT/name)','sha256_file(CONFIRMATION/name)')
            text=text.replace("sha256_file(OUT/'evaluate_transfer.py')", "sha256_file(CONFIRMATION/'evaluate_transfer.py')")
            text=text.replace('new_models_reloaded=30','new_models_reloaded=10')
            text=text.replace('30 real models, 120 probes, reconstruction, 80 control scores, 18 intervals',
                              '10 real models, 40 probes, reconstruction, 40 control scores, 6 intervals')
            changed={'verify'}
        before,after=functions(old),functions(text)
        same=[k for k in before if k not in changed]
        for k in same:assert before[k]==after[k],(name,k)
        target=HERE/name
        if target.exists():assert target.read_text()==text,'Refuse to replace altered adaptation'
        else:target.write_text(text)
        audit[name]={'parent_sha256':sha(PARENT/name),'adapted_sha256':sha(target),
            'unchanged_function_AST':same,'allowed_adapted_functions':sorted(changed)}
    result={'status':'PASS','adaptation_scope':'CV, F2 identity, output paths, P6 candidate restriction, counts',
        'numeric_training_and_evaluation_functions_identical':True,'files':audit,
        'runtime_sha256':sha(HERE/'confirmation_runtime.py'),'preparation_sha256':sha(Path(__file__)),
        'explicit_cv_loader':True,'model_seed_unchanged':42,'parent_frozen_files':len(historical)}
    output=HERE/'reference/adaptation_verification.json'
    if output.exists():assert json.loads(output.read_text())==result
    else:output.write_text(json.dumps(result,indent=2)+'\n')
    print('Adaptation PASS: numeric functions unchanged; explicit CV loader; parent frozen files intact')


if __name__=='__main__':main()
