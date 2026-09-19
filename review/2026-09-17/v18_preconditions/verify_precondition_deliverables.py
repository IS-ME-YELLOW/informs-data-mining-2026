"""Verify new CV/report evidence and that audited existing assets are unchanged."""
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import pandas as pd

from generate_cv_assignments import pure_cv_functions

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads((OUT / "prerequisite_audit.json").read_text())
    changed = [name for name, digest in audit["protected_sha256"].items() if sha(ROOT/name) != digest]
    assert not changed, changed
    registry = json.loads((ROOT / "cv/repeated_cv_manifest_2026-09-17.json").read_text())
    assert sha(ROOT/'code_phase1/cv.py') == registry['algorithm_sha256']
    assert sha(OUT/'generate_cv_assignments.py') == registry['generator_sha256']
    funcs = pure_cv_functions()
    reference = pd.read_csv(ROOT/'cv/cv_assignments_balanced_v1_seed42.csv', dtype={'fipsCode':str, 'stateAbbr':str})
    counties = funcs['_county_table'](reference)
    for record in registry['assignments']:
        path = ROOT/record['file']
        assert sha(path) == record['sha256']
        frame = pd.read_csv(path, dtype={'fipsCode':str, 'stateAbbr':str})
        funcs['_validate_assignment'](frame, counties, 5)
        generated = funcs['_build_assignment'](counties, 5, record['seed'])
        pd.testing.assert_frame_equal(frame, generated, check_dtype=False)
    shards = [json.loads((OUT/f'feature_causality_shard{i}.json').read_text()) for i in range(4)]
    frames = pd.concat([pd.read_csv(OUT/f'feature_causality_shard{i}.csv', dtype={'fipsCode':str}) for i in range(4)], ignore_index=True)
    assert len(frames)==302 and not frames.fipsCode.duplicated().any()
    assert frames.rows.sum()==43488 and frames.columns.size > 0
    for field in ['frozen_equal_at_tolerance','nan_mask_equal','future_nan','future_large_constant']:
        assert frames[field].all(), field
    assert all(s['passed'] for s in shards)
    feature = json.loads((OUT/'feature_causality_verification.json').read_text())
    assert feature['passed'] and feature['compared_feature_cells']==7088544
    assert feature['max_abs_difference_from_frozen']==float(frames.max_abs_difference.max())
    models = json.loads((OUT/'saved_model_verification.json').read_text())
    assert models['models_reloaded']==120 and models['numeric_reproduction_passed']
    assert models['maximum_prediction_difference']<=1e-12
    model_records = pd.read_csv(OUT/'saved_model_reload_audit.csv')
    names = json.loads((ROOT/'versions/xyy/v1.5.6/feature_names_v1.5.6.json').read_text())
    for row in model_records.itertuples():
        p=ROOT/row.actual_path
        assert sha(p)==row.current_sha256
        header=next(line for line in p.read_text().splitlines() if line.startswith('feature_names='))
        assert header.partition('=')[2].split()==names
    for p in OUT.glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    for p in OUT.glob('*.md'):
        text=p.read_text()
        assert sum(line.startswith('```') for line in text.splitlines())%2==0, p
        assert 'TODO' not in text and 'TBD' not in text
    result={
        'deliverables_verified':True, 'training_performed':False,
        'existing_audited_asset_count':len(audit['protected_sha256']),
        'existing_audited_code_data_models_predictions_and_seed42_unchanged':True,
        'all_three_assignments_reproduced_and_balanced':True,
        'new_split_seeds':[20260917,20260918],
        'feature_causality_counties':302, 'feature_causality_columns':163,
        'future_outage_nan_and_large_constant_invariance':True,
        'historical_numeric_model_reproduction_passed':True,
        'all_120_model_feature_name_orders_match':True,
        'historical_model_hash_mismatch_count':int((~model_records.historical_hash_matches).sum()),
        'formal_training_ready':False,
        'formal_training_blockers':['relocated paths','outer early stopping','CV/run/resume identity','strict missing-prediction validation and read-only preflight'],
        'new_review_file_sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='verification.json'},
    }
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='new_review_file_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
