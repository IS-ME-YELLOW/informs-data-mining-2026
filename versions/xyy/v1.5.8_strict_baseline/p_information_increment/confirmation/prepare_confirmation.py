"""Generate audited scope/path adapters; keep frozen fit and feature mathematics."""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import json

CONF=Path(__file__).resolve().parent;EXP=CONF.parent
ADAPTED=('information_preflight.py','evaluate_information.py','train_information.py','verify_information.py','verify_information_exact.py')
SHARED=('information_training.py','build_information_features.py','independent_features.py')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def adapt(name,original):
    s=original
    def r(a,b,count=1):
        nonlocal s
        assert s.count(a)==count,(name,a,s.count(a),count)
        s=s.replace(a,b)
    if name=='information_preflight.py':
        r('from independent_features import verify_features,allowed_from_raw,read_raw',
          'from independent_features import allowed_from_raw,read_raw\nfrom verify_information_exact import exact_feature_check as verify_features')
        r("build_candidates(ctx,{'F1':zeros,'F2':zeros})","build_candidates(ctx,{'F2':zeros})")
        r('case_count=2','case_count=1')
    elif name=='evaluate_information.py':
        r("PAIRS=(('F1_minus_F0','F0','F1'),('F2_minus_F0','F0','F2'),('F2_minus_F1','F1','F2'))","PAIRS=(('F2_minus_F0','F0','F2'),)")
        r("CASES=('F0','F1','F2')","CASES=('F0','F2')")
        r("for c in ('F1','F2')","for c in ('F2',)")
    elif name=='train_information.py':
        r('Approved first-round F1/F2 experiment; only seed42','Authorized F2-only confirmation on fixed splits')
        r("RUN_ID='v18_pinfo_v1_split42_model42'",'from confirmation_runtime import CODE_ROOT,META_ROOT,RUN_ID,SEED')
        r('type=int,choices=(42,),default=42','type=int,choices=(20260917,20260918),required=True')
        r("    if args.stage=='preflight':build_features(feature_inputs(ctx.data))\n",'    # Reuse the frozen input package; no regeneration in confirmation.\n')
        r("pre=HERE/'preflight/split42'",'pre=META_ROOT')
        r('split_seed=42,planned_formal_fits=100,planned_new_outer_models=20','split_seed=SEED,planned_formal_fits=50,planned_new_outer_models=10')
        r("HERE/'reference/source_manifest.json'","META_ROOT/'source_manifest.json'",count=2)
        r('03d}/100]','03d}/50]')
        r('All 100 fits complete','All 50 F2 fits complete')
        r("str(HERE/'verify_information.py'),'--split-seed','42'","str(CODE_ROOT/'verify_information_exact.py'),'--split-seed',str(SEED)")
        r("HERE/'reference/initial_workspace_snapshot.json'","CODE_ROOT/'reference/initial_workspace_snapshot.json'")
    elif name=='verify_information.py':
        r("PAIRS={'F1_minus_F0':('F0','F1'),'F2_minus_F0':('F0','F2'),'F2_minus_F1':('F1','F2')}",
          "from confirmation_runtime import RUN_ID,SEED\n\nPAIRS={'F2_minus_F0':('F0','F2')}")
        r('load_reference(42)','load_reference(SEED)')
        r("assert len(records)==20 and len({r['model_id'] for r in records})==20","assert len(records)==10 and len({r['model_id'] for r in records})==10")
        r('len(rounds)==80','len(rounds)==40')
        r('len(branch)==3*34416','len(branch)==2*34416')
        r('len(unique)==3*4*34177','len(unique)==2*4*34177')
        r("len(tables['pooled'])==84 and len(tables['bootstrap'])==16","len(tables['pooled'])==28 and len(tables['bootstrap'])==8")
        r("len(tables['branch_diagnostics'])==36 and len(tables['branch_loss_decomposition'])==18", "len(tables['branch_diagnostics'])==24 and len(tables['branch_loss_decomposition'])==12")
        r("'split_seed':42,","'split_seed':SEED,")
        r("'new_outer_models':20,'inner_probes':80,'formal_fits':100","'new_outer_models':10,'inner_probes':40,'formal_fits':50")
        r('type=int,choices=(42,),default=42','type=int,choices=(20260917,20260918),required=True')
        r("run=HERE/'runs/v18_pinfo_v1_split42_model42'","run=HERE/'runs'/RUN_ID")
    elif name=='verify_information_exact.py':
        r('from information_protocol import HERE,read_json,sha256_file,digest_object,write_json',
          'from information_protocol import HERE,read_json,sha256_file,digest_object,write_json\nfrom confirmation_runtime import CODE_ROOT,RUN_ID,SEED')
        r("sha256_file(HERE/'verify_information.py')","sha256_file(CODE_ROOT/'verify_information.py')")
        r('type=int,choices=(42,),default=42','type=int,choices=(20260917,20260918),required=True')
        r("run=HERE/'runs/v18_pinfo_v1_split42_model42'","run=HERE/'runs'/RUN_ID")
    ast.parse(s)
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    source=json.loads((EXP/'runs/v18_pinfo_v1_split42_model42/run_manifest.json').read_text())
    audit=json.loads((EXP/'runs/v18_pinfo_v1_split42_model42/logs/independent_verification.json').read_text())
    diff=[];hashes={}
    for name in (*ADAPTED,*SHARED):
        expected_hash=audit['audit_identity']['adapter_sha256'] if name=='verify_information_exact.py' else source['identity']['code_sha256'][name]
        assert sha(EXP/name)==expected_hash,name
        if name in ADAPTED:
            original=(EXP/name).read_text();new=adapt(name,original)
            if args.check:assert (CONF/name).read_text()==new,name
            else:
                assert not (CONF/name).exists(),name
                (CONF/name).write_text(new)
            diff.extend(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile=f'frozen/{name}',tofile=f'confirmation/{name}'))
            hashes[name]={'source':expected_hash,'adapted':sha(CONF/name)}
        else:hashes[name]={'source':expected_hash,'reuse':'unmodified frozen file'}
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old=functions(EXP/'information_protocol.py');new=functions(CONF/'information_protocol.py')
    for n in ('output_path','build_candidates'):assert old[n]==new[n],n
    old=functions(EXP/'verify_information_exact.py');new=functions(CONF/'verify_information_exact.py')
    for n in ('aggregate_rank_order','exact_feature_check'):assert old[n]==new[n],n
    report={'status':'PASS','training_performed':False,'trained_cases':['F2'],'authorized_splits':[20260917,20260918],
        'files':hashes,'F0_assembly_and_rank_order_audit_AST':'identical',
        'new_reference_loader':'explicit same-split frozen E2; no new labels or fitted preprocessing',
        'fitting_and_feature_math':'unchanged; shared training and immutable input package'}
    if args.check:assert json.loads((CONF/'reference/adaptation_verification.json').read_text())==report
    else:
        (CONF/'reference/adaptation.diff').write_text(''.join(diff))
        (CONF/'reference/adaptation_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
