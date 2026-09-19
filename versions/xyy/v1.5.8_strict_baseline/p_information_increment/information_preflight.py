"""Meaningful no-fit checks for allowed-input sharing and held-out-label isolation."""
from types import SimpleNamespace
import ast
import numpy as np
import pandas as pd
from information_protocol import HERE,PBASE,PT,NEW_CASES,KINDS,output_path,build_candidates,sha256_file,read_json
from information_training import subset,specification,contribution_metric
from build_information_features import (KEYS,WHITELIST,HISTORY,NBRCOLS,DEMCOLS,neighbor_table,
    aggregate_neighbors,validate_view,normalized_fips)
from independent_features import verify_features,allowed_from_raw,read_raw


def reject(call):
    try:call()
    except (ValueError,AssertionError,KeyError):return
    raise AssertionError('Invalid operation accepted')


def run_checks(ctx):
    result=verify_features(ctx)
    folder=HERE/'features/v1';view=pd.read_parquet(folder/'allowed_history_weather.parquet')
    points=pd.read_csv(folder/'coordinates.csv',dtype={'fipsCode':str},float_precision='round_trip')
    nn=pd.read_csv(folder/'neighbors_k8.csv',dtype={'target_fips':str,'neighbor_fips':str},float_precision='round_trip')
    original=aggregate_neighbors(view,nn)
    reordered=aggregate_neighbors(view.sample(frac=1,random_state=41),nn.sample(frac=1,random_state=19))
    pd.testing.assert_frame_equal(original,reordered)
    pd.testing.assert_frame_equal(neighbor_table(points),neighbor_table(points.sample(frac=1,random_state=8)))
    tied=points.copy();tied['latitude']=40.;tied['longitude']=-80.
    ties=neighbor_table(tied)
    for f,g in ties.groupby('target_fips'):
        assert g.neighbor_fips.tolist()==sorted(set(points.fipsCode)-{f})[:8]
    reject(lambda:neighbor_table(points.iloc[:-1]))
    reject(lambda:neighbor_table(pd.concat([points.iloc[:1],points.iloc[:-1]])))
    reject(lambda:validate_view(view.iloc[:-1]))
    reject(lambda:validate_view(pd.concat([view.iloc[:-1],view.iloc[:1]])))
    reject(lambda:normalized_fips(pd.Series([1.5])))
    constant=view.copy()
    for k,q in enumerate(WHITELIST):constant[q]=np.where(view[q].isna(),np.nan,(k+1)/10)
    c=aggregate_neighbors(constant,nn);valid=c.hour_idx<215
    for k,q in enumerate(WHITELIST):
        np.testing.assert_allclose(c.loc[valid,'nbr8_mean_'+q],(k+1)/10,rtol=0,atol=1e-12)
        np.testing.assert_allclose(c.loc[valid,'nbr8_max_'+q],(k+1)/10,rtol=0,atol=1e-12)
    np.testing.assert_allclose(c.loc[valid,NBRCOLS[-2:]],0,rtol=0,atol=1e-12)
    # Single provider perturbation: changes legal context across fold/test boundaries.
    provider=str(ctx.data.meta_test.fipsCode.iloc[0]).zfill(5)
    mutated=view.copy();mutated.loc[mutated.fipsCode==provider,'last_P_t']+=.125
    changed=aggregate_neighbors(mutated,nn)
    recipients=set(nn.loc[nn.neighbor_fips==provider,'target_fips']);assert recipients
    delta=changed.nbr8_mean_last_P_t-original.nbr8_mean_last_P_t
    np.testing.assert_allclose(delta,np.where(view.fipsCode.isin(recipients),.125/8,0),rtol=0,atol=1e-12)
    assert provider not in recipients
    w=view.copy();w.loc[(w.fipsCode==provider)&(w.hour_idx==100),'gust_t']+=8.
    wc=aggregate_neighbors(w,nn)
    np.testing.assert_allclose(wc.nbr8_mean_gust_t-original.nbr8_mean_gust_t,
        np.where(view.fipsCode.isin(recipients)&(view.hour_idx==100),1.,0),rtol=0,atol=1e-12)
    raw=read_raw();future=raw.hour_idx>=72;past=allowed_from_raw(raw,raw)
    for replacement in (np.nan,999.):
        bad=raw.copy();bad.loc[future,['P_t','N_t','D_t','R_t','osi']]=replacement
        rv=allowed_from_raw(bad,raw)
        pd.testing.assert_frame_equal(rv,past)
    pd.testing.assert_frame_equal(allowed_from_raw(raw.loc[~future],raw),past)
    # Independent raw reconstruction already equals the frozen whitelist; no future labels enter its builder.
    provenance=[]
    for case in NEW_CASES:
        cc=ctx.cases[case]
        for outer in range(5):
            changed_ctx=SimpleNamespace(**vars(cc));data=SimpleNamespace(**vars(cc.data))
            data.component_targets=cc.data.component_targets.copy();data.y_train=cc.data.y_train.copy()
            excluded=cc.data.row_folds==outer
            data.component_targets.loc[excluded,PT]=999.;data.y_train.loc[excluded,:]=999.
            changed_ctx.data=data
            for kind in KINDS:
                spec=specification(cc,outer,kind,'preflight')
                assert spec==specification(changed_ctx,outer,kind,'preflight')
                for probe in spec['probes']:
                    tr=set(probe['train']['counties']);va=set(probe['validation']['counties']);out=set(cc.meta.loc[excluded,'fipsCode'])
                    assert not tr&va and not (tr|va)&out
                provenance.append(spec)
        reject(lambda:subset(cc,range(5),'a'))
        reject(lambda:subset(cc,[0],'direct'))
    zeros=np.where(ctx.valid,0.,np.nan);build_candidates(ctx,{'F1':zeros,'F2':zeros})
    reject(lambda:output_path(HERE.parent/'forbidden'))
    # The actual training algorithm and weighted metric retain the original AST.
    old=ast.parse((PBASE/'weighted_scoped_training.py').read_text());new=ast.parse((HERE/'information_training.py').read_text())
    oldf={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
    newf={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
    for n in ('subset','describe','contribution_metric','load_model'):assert oldf[n]==newf[n]
    adaptation=read_json(HERE/'reference/adaptation_manifest.json')
    assert sha256_file(HERE/'information_training.py')==adaptation['adapted_sha256']
    result.update(training_performed=False,case_count=2,fold_label_isolation='PASS',
        future_outage_perturbation_deletion='PASS',legal_test_history_positive_control='PASS',
        node_row_permutation_ties_constant_fields='PASS',source_training_math='unchanged',
        source_F0_replay='exact',empty_duplicate_missing_and_path_rejection='PASS')
    return result,provenance
