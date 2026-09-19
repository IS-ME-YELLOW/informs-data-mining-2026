"""Check that confirmation changes only split/path bookkeeping, not the experiment."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
OLD=HERE.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path):
    return {node.name:node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}


def main():
    original=json.loads((OLD/'experiment_config.json').read_text())
    identical_files=('known_history.py','metric_helpers.py','independent_numerics.py','unknown_d_evaluation.py')
    core=('output_path','json_read','array_sha','source_files','validate_targets','make_scoped','reconstructed_d','candidate','make_source_manifest')
    all_hashes={};checks=[]
    for seed in (20260917,20260918):
        folder=HERE/f'split{seed}';cfg=json.loads((folder/'experiment_config.json').read_text())
        assert cfg['split_seed']==seed and cfg['authorized_split_seeds']==[seed]
        for key in ('protocol','definition','reconstruction','epsilon','settings','target','reconstructed_target','model_seed','primary','source_projector_sha256','approved_plan_sha256'):
            assert cfg[key]==original[key],key
        for name in identical_files:assert sha(folder/name)==sha(OLD/name),name
        a,b=functions(OLD/'unknown_d_protocol.py'),functions(folder/'unknown_d_protocol.py')
        for name in core:assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
        old_main=functions(OLD/'train_unknown_d.py')['main']
        new_main=functions(folder/'train_unknown_d.py')['main']
        old_fit=next(node for node in old_main.body if isinstance(node,ast.With))
        new_fit=next(node for node in new_main.body if isinstance(node,ast.With))
        assert ast.dump(old_fit,include_attributes=False)==ast.dump(new_fit,include_attributes=False)
        text=(folder/'verify_unknown_d.py').read_text().replace('ctx=load_context(CONFIG["split_seed"])','ctx=load_context(42)').replace('"split_seed":CONFIG["split_seed"]','"split_seed":42')
        normalized=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name=='verify_run')
        assert ast.dump(normalized,include_attributes=False)==ast.dump(functions(OLD/'verify_unknown_d.py')['verify_run'],include_attributes=False)
        for path in [*folder.glob('*.py'),folder/'experiment_config.json']:
            if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
            all_hashes[str(path.relative_to(HERE))]=sha(path)
        checks.append({'split_seed':seed,'formula_and_scoped_U_target':'AST_IDENTICAL','training_loop':'AST_IDENTICAL',
                       'evaluation_and_independent_numerics':'BYTE_IDENTICAL','independent_verifier':'IDENTICAL_EXCEPT_SPLIT_METADATA'})
    result={'status':'PASS','checks':checks,'files':all_hashes,'checker_sha256':sha(Path(__file__))}
    (HERE/'reference/adaptation_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':checks},indent=2))


if __name__=='__main__':main()
