"""Read-only, cross-platform consumer validation. No ML training libraries required."""
from pathlib import Path
import sys,json,hashlib,argparse
sys.dont_write_bytecode=True
import numpy as np
from prediction_reader import PredictionPackage,file_sha256
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hashes-only',action='store_true');ap.add_argument('--producer-check',action='store_true');args=ap.parse_args();root=Path(__file__).resolve().parent
    marker=json.loads((root/'PACKAGE_COMPLETE.json').read_text());assert marker['status']==('VALIDATION_PENDING' if args.producer_check else 'COMPLETE')
    for name,h in marker['files'].items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and p.is_file(),name
        assert file_sha256(p)==h,name
    if not args.hashes_only:
        pkg=PredictionPackage(root)
        for seed in [42,20260917,20260918]:
            for scope in [(1,2,3,4),(2,3,4),(0,1,2,3,4)]:
                for h in [1,6,24,48]:
                    d=pkg.graph_context(seed,scope,h);r,m=pkg.residual_supervision(seed,scope,h)
                    assert np.isfinite(d.base_graph_input).all() and np.isfinite(r[m]).all()
                    for source,q in zip(d.source_scope,d.fold):
                        assert set(map(int,source))<=set(scope)
                        if q in scope:assert q not in set(map(int,source))
        print('Representative outer/inner/final contexts, four horizons, all CVs: PASS')
    print(json.dumps(dict(status='PASS',identity_hash=marker['identity_hash'],files_checked=len(marker['files']),training_performed=False),ensure_ascii=False))
if __name__=='__main__':main()
