"""No-fit numerical and dependency-boundary tests for the approved experiment."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from p_structure_protocol import (HERE,ROOT,PT,H1,HORIZONS,KINDS,output_path,p_from_history,support,
    transform,reconstruct,build_candidates,expected_mask,require_unit)
from weighted_scoped_training import subset,describe,specification,contribution_metric


def reject(call):
    try:
        call()
    except (ValueError,AssertionError):
        return
    raise AssertionError('Invalid input accepted')


def run_checks(ctx):
    raw=ctx.raw; p=ctx.p; valid=ctx.valid
    y=ctx.data.component_targets[PT].to_numpy(copy=True)
    require_unit(y[valid])
    labels={};weights={}
    for kind in ('a','b'):
        take=valid & support(p,kind)
        labels[kind]=np.full(len(p),np.nan);weights[kind]=np.full(len(p),np.nan)
        labels[kind][take],weights[kind][take]=transform(y[take],p[take],kind)
    rebuilt=reconstruct(p,labels['a'],labels['b'],valid)[0]
    np.testing.assert_allclose(rebuilt,y,rtol=0,atol=1e-12,equal_nan=True)
    # Explicit endpoints, tiny positive p, nextafter(1,0), recovery and growth.
    pe=np.array([0.,1.,1e-12,np.nextafter(1.,0.),.2,.2,.2,.2])
    ye=np.array([.6,.3,0.,1.,0.,.1,.2,.6])
    ab=[]
    for kind in ('a','b'):
        take=support(pe,kind);a=np.full(len(pe),np.nan)
        a[take]=transform(ye[take],pe[take],kind)[0];ab.append(a)
    np.testing.assert_allclose(reconstruct(pe,*ab,np.ones(len(pe),bool))[0],ye,rtol=0,atol=1e-12)
    np.testing.assert_allclose(ab[0][-4:],[0,.5,1,1],rtol=0,atol=1e-12)
    np.testing.assert_allclose(ab[1][-4:],[0,0,0,.5],rtol=0,atol=1e-12)
    reject(lambda:transform(np.array([.2]),np.array([0.]),'a'))
    reject(lambda:transform(np.array([.2]),np.array([1.]),'b'))
    reject(lambda:require_unit(np.array([-1e-15])))
    reject(lambda:reconstruct(np.array([0.]),np.array([0.]),np.array([.2]),np.array([True])))
    reject(lambda:contribution_metric(np.array([]),np.array([]),3))
    test_metric=contribution_metric(np.array([.2,.4]),np.array([.01,.25]),3)
    metric=test_metric(np.array([.1,.7]),None)[1]
    np.testing.assert_allclose(metric,np.sqrt((.01*.1**2+.25*.3**2)/3),rtol=0,atol=1e-12)
    future=pd.to_datetime(raw.timestamp_et)>=pd.Timestamp('2026-03-14')
    for replacement in (np.nan,999.,np.random.default_rng(17).normal(size=int(future.sum()))):
        changed=raw.copy();changed.loc[future,'P_t']=replacement
        np.testing.assert_array_equal(p_from_history(changed,ctx.meta),p)
    np.testing.assert_array_equal(p_from_history(raw.loc[~future],ctx.meta),p)
    changed=raw.copy();h71=pd.to_datetime(changed.timestamp_et)==pd.Timestamp('2026-03-13 23:00')
    changed.loc[h71,'P_t']=.123456
    assert not np.array_equal(p_from_history(changed,ctx.meta),p)
    reject(lambda:p_from_history(raw.loc[~h71],ctx.meta))
    reject(lambda:p_from_history(pd.concat([raw,raw.loc[h71].iloc[:1]]),ctx.meta))
    reject(lambda:p_from_history(raw,pd.concat([ctx.meta.iloc[:1]]*2)))
    ordered=ctx.meta.sample(frac=1,random_state=42)
    np.testing.assert_array_equal(p_from_history(raw.sample(frac=1,random_state=13),ordered),p[ordered.index])
    raw_test=pd.read_csv(ROOT/'data/DM_Test.csv',usecols=['fipsCode','timestamp_et','P_t'],dtype={'fipsCode':str},float_precision='round_trip')
    pt=p_from_history(raw_test,ctx.data.meta_test)
    np.testing.assert_array_equal(pt,ctx.data.X_test.last_P_t)
    provenance=[]
    for outer in range(5):
        data=SimpleNamespace(**vars(ctx.data));data.component_targets=ctx.data.component_targets.copy()
        data.y_train=ctx.data.y_train.copy()
        excluded=ctx.data.row_folds==outer
        data.component_targets.loc[excluded,PT]=999.;data.y_train.loc[excluded,:]=999.
        mutated=SimpleNamespace(**vars(ctx));mutated.data=data
        for kind in KINDS:
            before=specification(ctx,outer,kind,'test');after=specification(mutated,outer,kind,'test')
            assert before==after
            provenance.append(before)
            ext=set(ctx.meta.loc[excluded,'fipsCode'])
            for probe in before['probes']:
                tr,va=set(probe['train']['counties']),set(probe['validation']['counties'])
                assert not tr&va and not (tr|va)&ext
                np.testing.assert_allclose(probe['train']['normalized_mean'],1,rtol=0,atol=1e-12)
    zeros=np.where(valid,0.,np.nan)
    build_candidates(ctx,zeros,zeros)
    reject(lambda:output_path(HERE.parent/'forbidden'))
    reject(lambda:subset(ctx,range(5),'a'))
    reject(lambda:subset(ctx,[0],'unknown'))
    empty=SimpleNamespace(**vars(ctx));empty.p=np.zeros_like(p)
    reject(lambda:subset(empty,[0,1,2],'a'))
    return {'status':'PASS','training_performed':False,'features':163,'train_counties':239,'test_counties':63,
        'scoreable_rows':{h:int(expected_mask(ctx.meta,h).sum()) for h in HORIZONS},
        'p_zero_counties':int(ctx.meta.loc[p==0,'fipsCode'].nunique()),
        'a_support_rows':int((valid&(p>0)).sum()),'b_support_rows':int((valid&(p<1)).sum()),
        'target_reconstruction_max_abs':float(np.max(np.abs(rebuilt[valid]-y[valid]))),
        'checks':['source hashes and E0 replay','label reconstruction and endpoints','weight normalization and custom metric',
        'future-history perturbation/deletion and positive control','all outer labels excluded before transform',
        'all inner county sets isolated','wrong support/key/path rejection','same-horizon and long-horizon invariants',
        'test history only, no test forecasts']},provenance
