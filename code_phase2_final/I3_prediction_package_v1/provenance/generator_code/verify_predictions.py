"""Independent arithmetic, gate dependencies, OOF scores, and GAT adapter verification."""
import sys,math,argparse
from functools import lru_cache
from package_core import *
sys.path.insert(0,str(PORT))
from prediction_reader import PredictionPackage
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)
def own_post(a):
    v=np.minimum(np.maximum(a,0),.65);v[v<.001]=0;return v
def own_comp(d):return own_post(.4*np.clip(d['P_t'],0,1)+.35*np.clip(d['N_t'],0,1)+.25*np.clip(d['D_t'],0,1)-.1*np.clip(d['R_t'],0,1))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=SEEDS);args=ap.parse_args();seeds=(args.seed,) if args.seed is not None else SEEDS
    pkg=PredictionPackage(PORT);meta=pkg.meta;hours=meta.hour_idx.to_numpy();fips=meta.fipsCode.to_numpy();n=len(meta)
    summaries=[];contexts=[];gates=0;checked_rows=0;scope_count=0
    for seed in seeds:
        fold=pkg.folds(seed);ctx=Context(seed);task_lookup={t['id']:t for t in read(OUT/'tasks.json') if t['seed']==seed}
        pkg.assert_cv_assignment(seed,fips,fold)
        wrong=fold.copy();wrong[0]=(wrong[0]+1)%5
        try:pkg.assert_cv_assignment(seed,fips,wrong)
        except ValueError:pass
        else:raise AssertionError('Wrong consumer CV accepted')
        @lru_cache(maxsize=180)
        def scalar(s,name):
            t=task_lookup[task_id(seed,s,name)];p=paths(t);r=read(p['receipt']);a=np.load(p['prediction']);out=np.full(n,np.nan);out[a['rows']]=a['prediction'];return out,r['model_id']
        @lru_cache(maxsize=30)
        def get_scope(s):return pkg.base_scope(seed,s)
        for s in SCOPES:
            d=get_scope(s);receipt=read(PORT/f'splits/seed{seed}/scopes/S{scope_id(s)}.json');assert receipt['identity_hash']==ctx.identity and receipt['scope']==list(s)
            live=~np.isin(fold,s);np.testing.assert_array_equal(d.eligible_prediction,live);assert d.source_scope.eq(scope_id(s)).all()
            for name,ref in receipt['models'].items():assert ref['model_id']==scalar(s,name)[1]
            long=[]
            for h in HS:
                valid=live&(hours+h<=215);np.testing.assert_array_equal(d[f'valid_{h:02d}'],valid);pp={}
                for c in CS:
                    if c=='P_t' and h in [1,6]:
                        pa=scalar(s,f'P{h:02d}_a')[0].copy();pb=scalar(s,f'P{h:02d}_b')[0].copy();pa[valid&(ctx.p==0)]=0;pb[valid&(ctx.p==1)]=0
                        pp[c]=ctx.p*np.clip(pa,0,1)+(1-ctx.p)*np.clip(pb,0,1);close(d[f'raw_P_a_{h:02d}'],pa);close(d[f'raw_P_b_{h:02d}'],pb)
                    else:pp[c]=np.clip(scalar(s,'P24_neighbor' if c=='P_t' and h==24 else f'lgb_{c}_{h:02d}')[0],0,1)
                    if c=='D_t' and h==1:
                        u=scalar(s,'D01_unknown')[0];hit=valid&(ctx.K>0);pp[c][hit]=np.clip(ctx.K[hit]+np.maximum(u[hit],0),0,1);close(d.raw_U_01,u)
                    pp[c][~valid]=np.nan;close(d[f'T_source_{c}_{h:02d}'],pp[c]);assert np.array_equal(np.isfinite(pp[c]),valid)
                ff=pd.DataFrame({c:pp[c][valid] for c in CS});ff['fipsCode']=fips[valid];ff['target_hour']=hours[valid]+h;long.append(ff)
            unique=pd.concat(long,ignore_index=True).groupby(['fipsCode','target_hour'])[list(CS)].mean()
            for h in HS:
                valid=live&(hours+h<=215);idx=pd.MultiIndex.from_arrays([fips,hours+h]);al=unique.reindex(idx)
                for c in CS:close(d[f'T_aligned_{c}_{h:02d}'],al[c].to_numpy())
                p={c:d[f'T_source_{c}_{h:02d}'].to_numpy() for c in CS};T=own_comp({c:al[c].to_numpy() for c in CS} if h<=6 else p);close(d[f'T_OSI_{h:02d}'],T)
                if h>=24:
                    sources={}
                    for prefix in ['lgb','xgboost','catboost']:
                        raw={c:scalar(s,f'{prefix}_{c}_{h:02d}')[0] for c in CS};direct=own_post(scalar(s,f'{prefix}_osi_{h:02d}')[0]);comp=own_comp(raw)
                        close(d[f'S_{prefix}_direct_{h:02d}'],direct);close(d[f'S_{prefix}_component_{h:02d}'],comp)
                        for c,a in raw.items():close(d[f'S_raw_{prefix}_{c}_{h:02d}'],a)
                        sources[prefix+'_direct']=direct;sources[prefix+'_component']=comp
                    anchors=[];source_ids=[]
                    for q in s:
                        sub=tuple(k for k in s if k!=q);hit=(fold==q)&(hours+h<=215)
                        if len(sub)==1:
                            ap=PORT/f'splits/seed{seed}/anchors/S{sub[0]}.parquet';ar=read(ap.with_suffix('.json'));assert sha(ap)==ar['prediction_sha256'];aa=pd.read_parquet(ap)
                            av=own_comp({c:scalar(sub,f'lgb_{c}_{h:02d}')[0] for c in CS});close(aa[f'L0_{h:02d}'],av)
                        else:av=get_scope(sub)[f'S_L0_{h:02d}'].to_numpy()
                        anchors.extend(av[hit].tolist());source_ids.append(dict(predicted_fold=q,source_scope=list(sub),rows=int(hit.sum()),model_ids={c:scalar(sub,f'lgb_{c}_{h:02d}')[1] for c in CS}))
                    values=np.asarray(anchors);positive=sorted(values[values>0]);pos=.95*(len(positive)-1);lo=int(np.floor(pos));hi=int(np.ceil(pos));theta=positive[lo]+(pos-lo)*(positive[hi]-positive[lo])
                    gr=next(g for g in receipt['gates'] if g['horizon']==h);assert gr['sources']==source_ids and gr['anchor_sha256']==ah(values)
                    assert gr['valid_count']==len(values) and gr['positive_count']==len(positive);close(gr['theta'],theta);close(d[f'S_theta_{h:02d}'],theta)
                    l0=sources['lgb_component'];order=['lgb_direct','lgb_component','xgboost_direct','catboost_direct','xgboost_component','catboost_component']
                    l1=own_post(np.mean(np.stack([sources[k] for k in order]),axis=0));l2=np.where(l0>theta,l0,l1)
                    close(d[f'S_L0_{h:02d}'],l0);close(d[f'S_L1_{h:02d}'],l1);close(d[f'S_L2_{h:02d}'],l2);gates+=1
                    final=own_post((T+l2)/2) if h==24 else l2
                else:final=T
                close(d[f'I3_OSI_{h:02d}'],final);assert np.array_equal(np.isfinite(final),valid);checked_rows+=int(valid.sum())
            scope_count+=1
        # Read all 192 graph contexts and prove the *whole* base recipe excludes each supervised row's fold.
        for s in [s for s in SCOPES if len(s)>=3]:
            for h in HS:
                d=pkg.graph_context(seed,s,h);valid=hours+h<=215;expected_sup=valid&np.isin(fold,s)
                np.testing.assert_array_equal(d.supervised,expected_sup);np.testing.assert_array_equal(d.valid,valid)
                for q in [-1,*FOLDS]:
                    hit=fold==q;sub=tuple(k for k in s if k!=q) if q in s else s
                    assert set(sub)<=set(s) and q not in sub;assert d.loc[hit,'source_scope'].eq(scope_id(sub)).all()
                    close(d.loc[hit,'base_prediction'],get_scope(sub).loc[hit,f'I3_OSI_{h:02d}'])
                res,mask=pkg.residual_supervision(seed,s,h);close(res[mask],ctx.labels[hname(h)].to_numpy()[mask]-d.base_prediction.to_numpy()[mask]);assert (res[~mask]==0).all()
                bad=np.flatnonzero(~expected_sup)[:1]
                try:pkg.labels_for_rows(seed,s,h,bad)
                except PermissionError:pass
                else:raise AssertionError('Out-of-scope label read allowed')
                assert np.isfinite(d.base_graph_input).all();np.testing.assert_array_equal(d.loc[~valid,'base_graph_input'],0)
                contexts.append(dict(seed=seed,scope=scope_id(s),horizon=h,nodes=302,valid_rows=int(valid.sum()),supervised_rows=int(mask.sum()),test_rows=int((valid&(fold<0)).sum())))
        # Original frozen OOF equality and pooled scores; no new candidate selection.
        original=pd.read_parquet(PORT/f'reference/split{seed}/I3_candidates.parquet')
        for h in HS:
            pred=np.full(34416,np.nan)
            for outer in FOLDS:
                s=tuple(k for k in FOLDS if k!=outer);hit=fold[:34416]==outer;pred[hit]=get_scope(s)[f'I3_OSI_{h:02d}'].to_numpy()[:34416][hit]
            col=next(c for c in original if 'I3' in c and c.endswith(hname(h)));close(pred,original[col]);y=ctx.labels[hname(h)].to_numpy()[:34416];hit=np.isfinite(y)
            e=pred[hit]-y[hit];summaries.append(dict(seed=seed,horizon=h,n=int(hit.sum()),rmse=float(np.sqrt(np.mean(e**2))),sse=float(np.sum(e**2)),max_reference_difference=float(np.max(abs(pred[hit]-original[col].to_numpy()[hit])))))
        print(f'All scope arithmetic and GAT contexts independently verified seed{seed}',flush=True)
    suffix=f'_seed{args.seed}' if args.seed is not None else ''
    frame(OUT/f'audit/context_checks{suffix}.csv',contexts);frame(OUT/f'audit/frozen_I3_scores{suffix}.csv',summaries)
    assert scope_count==26*len(seeds) and gates==52*len(seeds) and len(contexts)==64*len(seeds)
    result=dict(status='PASS',complete=len(seeds)==3,scoped_recipes=scope_count,independent_gate_checks=gates,GAT_contexts=len(contexts),valid_scoped_prediction_rows=checked_rows,
        final_OOF_max_difference=max(r['max_reference_difference'] for r in summaries),all_component_sources_verified=True,all_I3_recipes_verified=True,
        adapter_scope_exclusion=True,forbidden_label_reads_rejected=True,invalid_tail_zero_only_for_graph_input=True,verifier_sha256=sha(__file__))
    save(OUT/f'audit/prediction_verification{suffix}.json',result);print(result,flush=True)
if __name__=='__main__':main()
