"""Versioned verifier adapter: match rank-ordered IEEE-754 sums, no model changes.

The original independent implementation used a contiguous np.mean reduction;
the production tensor sums the eight ranked neighbors sequentially. Both are
mathematically equivalent, but bit-level differences can cross tree thresholds.
This adapter preserves the original training identity and all trained artifacts,
and records a separate audit identity. All original verification checks still run.
"""
import argparse
import json
import numpy as np
import pandas as pd
from information_protocol import HERE,read_json,sha256_file,digest_object,write_json
import independent_features as feature_verifier
from build_information_features import WHITELIST,NBRCOLS


def aggregate_rank_order(view,neighbors):
    """Independent county-major loop; deterministic addition in neighbor rank order."""
    lookup={f:g.sort_values('hour_idx')[WHITELIST].to_numpy(dtype=float) for f,g in view.groupby('fipsCode')}
    frames=[]
    for f,rows in neighbors.groupby('target_fips',sort=True):
        codes=rows.sort_values('rank').neighbor_fips.tolist()
        assert len(codes)==8 and len(set(codes))==8 and f not in codes
        cols={}
        for q,c in enumerate(WHITELIST):
            total=np.zeros(144,dtype=np.float64)
            maximum=np.full(144,-np.inf,dtype=np.float64)
            for neighbor in codes:
                total=total+lookup[neighbor][:,q]
                maximum=np.maximum(maximum,lookup[neighbor][:,q])
            cols['nbr8_mean_'+c]=total/8.
            cols['nbr8_max_'+c]=maximum
        cols[NBRCOLS[-2]]=lookup[f][:,0]-cols['nbr8_mean_last_P_t']
        cols[NBRCOLS[-1]]=lookup[f][:,6]-cols['nbr8_mean_gust_at_t1h']
        frame=pd.DataFrame(cols);frame['fipsCode']=f;frame['hour_idx']=np.arange(72,216);frames.append(frame)
    return pd.concat(frames,ignore_index=True).set_index(['fipsCode','hour_idx'])


original_feature_check=feature_verifier.verify_features


def exact_feature_check(ctx,return_features=False):
    report,rebuilt=original_feature_check(ctx,return_features=True)
    for case in ('F1','F2'):
        for side in ('train','test'):
            pd.testing.assert_frame_equal(rebuilt[case][side],ctx.feature_bundle['features'][case][side],check_exact=True)
    report.update(augmented_features_bitwise_exact=True,aggregation_reduction='float64 addition in fixed neighbor rank order')
    return (report,rebuilt) if return_features else report


# Explicit dependency replacement limited to independently rebuilding predictors.
# The production builder, trained models, saved OOF, metric tolerances and checks
# are not replaced or modified.
feature_verifier.aggregate_independently=aggregate_rank_order
feature_verifier.verify_features=exact_feature_check
import verify_information as core


def verify(run,unfrozen=False):
    report=core.verify(run,unfrozen=unfrozen)
    audit={'training_identity_hash':report['identity_hash'],
        'adapter_sha256':sha256_file(__file__),
        'core_verifier_sha256':sha256_file(HERE/'verify_information.py'),
        'original_feature_verifier_sha256':sha256_file(HERE/'independent_features.py'),
        'reason':'Only independent summation order corrected; all reconstructed predictors now bitwise exact.',
        'model_or_saved_prediction_changes':False,'tolerance_relaxed':False,'additional_fits':0}
    report.update(core_verifier_sha256=report['verifier_sha256'],verifier_sha256=sha256_file(__file__),
        audit_identity=audit,audit_identity_hash=digest_object(audit))
    if not unfrozen:
        saved=read_json(run/'logs/independent_verification.json')
        assert report['audit_identity_hash']==saved['audit_identity_hash']
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split-seed',type=int,choices=(42,),default=42)
    p.add_argument('--unfrozen',action='store_true');args=p.parse_args()
    run=HERE/'runs/v18_pinfo_v1_split42_model42';report=verify(run,args.unfrozen)
    if args.unfrozen:write_json(run/'logs/independent_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
