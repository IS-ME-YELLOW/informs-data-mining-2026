"""Reviewable, allowlisted seed/path-only adaptation of immutable seed42 runners."""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import json

CONF=Path(__file__).resolve().parent
EXP=CONF.parent
ADAPTED=('p_structure_protocol.py','train_p_structure.py','verify_p_structure.py')
SHARED=('weighted_scoped_training.py','evaluate_p_structure.py','preflight_checks.py','metric_helpers.py','independent_numerics.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def adapt(name,original):
    text=original
    def replace(old,new,count=1):
        nonlocal text
        assert text.count(old)==count,(name,old,text.count(old),count)
        text=text.replace(old,new)
    if name=='p_structure_protocol.py':
        replace('Frozen seed42 P experiment','Frozen repeated-split P experiment')
        replace("HERE = Path(__file__).resolve().parent\nCONFIG = json.loads((HERE / 'experiment_config.json').read_text())",
                'from _runtime import HERE, CODE_ROOT, CONFIG_PATH, META_ROOT, RUN_ID, CONFIG, SEED')
        replace('def load_context(seed=42):','def load_context(seed=SEED):')
        replace("if seed != 42 or CONFIG['authorized_split_seeds'] != [42]:","if seed != SEED or seed not in CONFIG['authorized_split_seeds']:")
        replace("raise ValueError('Only seed42 authorized at this stage')","raise ValueError('Confirmation split does not match the selected source package')")
        replace("ROOT/'cv/cv_assignments_balanced_v1_seed42.csv'","ROOT/f'cv/cv_assignments_balanced_v1_seed{SEED}.csv'")
        replace("'split_seed':42, 'config_sha256':sha256_file(HERE/'experiment_config.json')",
                "'split_seed':SEED, 'config_sha256':sha256_file(CONFIG_PATH)")
        replace("'code_sha256':{n:sha256_file(HERE/n) for n in CODE_FILES}",
                "'code_sha256':{str(p):sha256_file(p) for p in [*(CODE_ROOT/n if n in ('p_structure_protocol.py','train_p_structure.py','verify_p_structure.py') else HERE/n for n in CODE_FILES), CODE_ROOT/'_runtime.py']}")
    elif name=='train_p_structure.py':
        replace('Authorized seed42-only P comparison.','Authorized repeated-split P confirmation.')
        replace("RUN_ID='v18_pstate_nested_v1_split42_model42'",'from _runtime import CODE_ROOT, META_ROOT, RUN_ID, SEED')
        replace("choices=(42,),type=int,default=42","choices=(20260917,20260918),type=int,required=True")
        replace("HERE/'preflight/seed42.json'","META_ROOT/'preflight.json'",count=2)
        replace("HERE/'preflight/fit_support_and_weights.json'","META_ROOT/'fit_support_and_weights.json'")
        replace("HERE/'reference/source_manifest.json'","META_ROOT/'source_manifest.json'",count=2)
        replace("checks.update(identity_hash=identity_hash,resume_identity_rejection='PASS')",
                "checks.update(identity_hash=identity_hash,split_seed=SEED,resume_identity_rejection='PASS')")
        replace("str(HERE/'verify_p_structure.py'),'--split-seed','42'","str(CODE_ROOT/'verify_p_structure.py'),'--split-seed',str(SEED)")
    elif name=='verify_p_structure.py':
        replace("CASES=('E0','E1','E2')","from _runtime import SEED, RUN_ID\n\nCASES=('E0','E1','E2')")
        replace('ctx=load_context(42)','ctx=load_context(SEED)')
        replace("'split_seed':42,","'split_seed':SEED,")
        replace('choices=(42,),default=42','choices=(20260917,20260918),required=True')
        replace("run=HERE/'runs/v18_pstate_nested_v1_split42_model42'","run=HERE/'runs'/RUN_ID")
    else:raise ValueError(name)
    ast.parse(text)
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();cfg=json.loads((CONF/'experiment_config.json').read_text())
    source=EXP/'runs/v18_pstate_nested_v1_split42_model42/run_manifest.json'
    manifest=json.loads(source.read_text())
    assert manifest['identity_hash']==cfg['source_seed42_identity_hash']
    assert sha(EXP/'experiment_config.json')==cfg['source_seed42_config_sha256']
    diff=[];hashes={}
    for name in (*ADAPTED,*SHARED):
        original=(EXP/name).read_text()
        assert sha(EXP/name)==manifest['identity']['code_sha256'][name],name
        if name in ADAPTED:
            expected=adapt(name,original)
            if args.check:assert (CONF/name).read_text()==expected,name
            else:
                assert not (CONF/name).exists(),name
                (CONF/name).write_text(expected)
            diff.extend(difflib.unified_diff(original.splitlines(True),expected.splitlines(True),fromfile=f'seed42/{name}',tofile=f'confirmation/{name}'))
            hashes[name]={'source':sha(EXP/name),'adapted':sha(CONF/name)}
        else:hashes[name]={'source':sha(EXP/name),'reuse':'same file; not copied or modified'}
    # All mathematical functions in the adapted protocol remain AST-identical.
    def funcs(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old,new=funcs(EXP/ADAPTED[0]),funcs(CONF/ADAPTED[0])
    identical=[n for n in old if n not in ('load_context','experiment_identity')]
    for name in identical:assert old[name]==new[name],name
    report={'status':'PASS','training_performed':False,'authorized_split_seeds':[20260917,20260918],
            'source_seed42_identity_hash':cfg['source_seed42_identity_hash'],'files':hashes,
            'AST_identical_protocol_functions':identical,'model_seed':42,
            'changes':'split selection, source identities, config/code identity paths, preflight destinations and run IDs only'}
    if not args.check:
        (CONF/'reference/runner_adaptation.diff').write_text(''.join(diff))
        (CONF/'reference/adaptation_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    else:assert json.loads((CONF/'reference/adaptation_verification.json').read_text())==report
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
