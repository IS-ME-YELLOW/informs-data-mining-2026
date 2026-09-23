"""Assemble complete B(S) recipes, then strictly cross-fit them into GAT contexts."""
import argparse,time
from functools import lru_cache
from package_core import *
from protocol import build_controls
class Assembler:
    def __init__(self,seed):
        self.ctx=Context(seed);self.seed=seed;self.tasks={t['id']:t for t in read(OUT/'tasks.json') if t['seed']==seed};self.n=len(self.ctx.meta)
    @lru_cache(maxsize=160)
    def scalar(self,s,name):
        task=self.tasks[task_id(self.seed,s,name)];p=paths(task);rec=read(p['receipt']);assert sha(p['prediction'])==rec['prediction_sha256']
        data=np.load(p['prediction']);expected=self.ctx.predict_rows(RECIPES[name],s);np.testing.assert_array_equal(data['rows'],expected)
        a=np.full(self.n,np.nan);a[data['rows']]=data['prediction'];return a,rec['model_id'],str(p['receipt'].relative_to(OUT))
    def anchor(self,s,h):return compose({c:self.scalar(s,f'lgb_{c}_{h:02d}')[0] for c in CS})
    def threshold(self,s,h):
        values=[];sources=[]
        for q in s:
            sub=tuple(f for f in s if f!=q);hit=(self.ctx.fold==q)&(self.ctx.hours+h<=215);a=self.anchor(sub,h)[hit];assert np.isfinite(a).all()
            values.append(a);sources.append(dict(predicted_fold=q,source_scope=list(sub),rows=int(hit.sum()),model_ids={c:self.scalar(sub,f'lgb_{c}_{h:02d}')[1] for c in CS}))
        v=np.concatenate(values);pos=v[v>0];assert len(pos)>0
        return float(np.quantile(pos,.95,method='linear')),dict(horizon=h,scope=list(s),positive_count=len(pos),valid_count=len(v),anchor_sha256=ah(v),sources=sources)
    def build(self,s):
        meta=self.ctx.meta;fold=self.ctx.fold;live=~np.isin(fold,s);parts={};columns={k:meta[k] for k in [*KEYS,'split']};columns['fold']=fold
        columns['source_scope']=scope_id(s);columns['eligible_prediction']=live;dependencies={};gate_records=[]
        def scalar(name):
            a,mid,receipt=self.scalar(s,name);dependencies[name]=dict(model_id=mid,receipt=receipt);return a.copy()
        for h in HS:
            valid=live&(self.ctx.hours+h<=215);pp={}
            for c in CS:
                if c=='P_t' and h in (1,6):
                    a=scalar(f'P{h:02d}_a');b=scalar(f'P{h:02d}_b');p=self.ctx.p
                    a[valid&(p==0)]=0;b[valid&(p==1)]=0;value=p*np.clip(a,0,1)+(1-p)*np.clip(b,0,1)
                    for kind,raw in [('a',a),('b',b)]:columns[f'raw_P_{kind}_{h:02d}']=raw
                else:
                    name='P24_neighbor' if c=='P_t' and h==24 else f'lgb_{c}_{h:02d}';value=np.clip(scalar(name),0,1)
                if c=='D_t' and h==1:
                    u=scalar('D01_unknown');K=self.ctx.K;switch=valid&(K>0);value[switch]=np.clip(K[switch]+np.maximum(u[switch],0),0,1);columns['raw_U_01']=u
                value[~valid]=np.nan;assert np.isfinite(value[valid]).all();pp[c]=value;columns[f'T_source_{c}_{h:02d}']=value
            parts[hname(h)]=pp;columns[f'valid_{h:02d}']=valid
        # Reuse the frozen C3 arithmetic, only on counties excluded from S.
        m=meta.loc[live,KEYS].reset_index(drop=True);p={hn:{c:a[live] for c,a in pc.items()} for hn,pc in parts.items()}
        dummy={hname(h):np.where(m.hour_idx.to_numpy()+h<=215,0.,np.nan) for h in HS};controls,aux=build_controls(m,dummy,p)
        for h in HS:
            a=np.full(self.n,np.nan);a[live]=controls['v18_rule'][hname(h)];columns[f'T_OSI_{h:02d}']=a
            for c in CS:
                a=np.full(self.n,np.nan);a[live]=aux['aligned_components'][hname(h)][c];columns[f'T_aligned_{c}_{h:02d}']=a
        for h in (24,48):
            valid=columns[f'valid_{h:02d}'];processed=[]
            for family,prefix in [('lightgbm','lgb'),('xgboost','xgboost'),('catboost','catboost')]:
                direct=post(scalar(f'{prefix}_osi_{h:02d}'));pc={c:scalar(f'{prefix}_{c}_{h:02d}') for c in CS};comp=compose(pc)
                columns[f'S_{prefix}_direct_{h:02d}']=direct;columns[f'S_{prefix}_component_{h:02d}']=comp
                for c,v in pc.items():columns[f'S_raw_{prefix}_{c}_{h:02d}']=v
                processed.extend([direct,comp])
            # Preserve Stella SOURCE_ORDER exactly: summation order is part of frozen arithmetic.
            processed=[columns[f'S_{x}_{h:02d}'] for x in ['lgb_direct','lgb_component','xgboost_direct','catboost_direct','xgboost_component','catboost_component']]
            l0=columns[f'S_lgb_component_{h:02d}'];l1=post(np.mean(np.stack(processed),axis=0));theta,g=self.threshold(s,h);g['theta']=theta;gate_records.append(g)
            stella=np.where(l0>theta,l0,l1);stella[~valid]=np.nan
            columns[f'S_L0_{h:02d}']=l0;columns[f'S_L1_{h:02d}']=l1;columns[f'S_L2_{h:02d}']=stella;columns[f'S_theta_{h:02d}']=theta
        for h in HS:
            if h<=6:value=columns[f'T_OSI_{h:02d}'].copy()
            elif h==24:value=post((columns['T_OSI_24']+columns['S_L2_24'])/2)
            else:value=columns['S_L2_48'].copy()
            columns[f'I3_OSI_{h:02d}']=value
        return pd.DataFrame(columns),dict(seed=self.seed,scope=list(s),identity_hash=self.ctx.identity,models=dependencies,gates=gate_records)
def reproduce(seed):
    engine=Assembler(seed);n=engine.ctx.ntrain;oldtree=pd.read_parquet(PORT/f'reference/split{seed}/tree_component_predictions.parquet')
    oldosi=pd.read_parquet(PORT/f'reference/split{seed}/tree_control_predictions.parquet');oldstella=pd.read_parquet(PORT/f'reference/split{seed}/stella_candidate_predictions.parquet')
    oldi3=pd.read_parquet(PORT/f'reference/split{seed}/I3_candidates.parquet');rows=[]
    for outer in FOLDS:
        s=tuple(f for f in FOLDS if f!=outer);d,g=engine.build(s);hit=(engine.ctx.fold==outer)
        for h in HS:
            mask=hit&(engine.ctx.hours+h<=215);ix=np.flatnonzero(mask);assert (ix<n).all()
            pairs=[(d[f'T_OSI_{h:02d}'].to_numpy()[ix],oldosi[f'pred_NB_P24_v18_rule_{hname(h)}'].to_numpy()[ix],'T_OSI')]
            for c in CS:pairs.append((d[f'T_source_{c}_{h:02d}'].to_numpy()[ix],oldtree[f'pred_NB_P24_{target(c,h)}'].to_numpy()[ix],c))
            if h>=24:pairs.append((d[f'S_L2_{h:02d}'].to_numpy()[ix],oldstella[f'pred_L2_{hname(h)}'].to_numpy()[ix],'Stella_L2'))
            col=next(c for c in oldi3 if 'I3' in c and c.endswith(hname(h)))
            pairs.append((d[f'I3_OSI_{h:02d}'].to_numpy()[ix],oldi3[col].to_numpy()[ix],'I3'))
            for a,b,name in pairs:
                np.testing.assert_allclose(a,b,atol=1e-12,rtol=0);rows.append(dict(seed=seed,outer_fold=outer,horizon=h,field=name,n=len(ix),max_abs_difference=float(np.max(abs(a-b)))))
    frame(OUT/f'preflight/reproduction_seed{seed}.csv',rows);return rows
def assemble(seeds=SEEDS):
    tasks=[t for t in read(OUT/'tasks.json') if t['seed'] in seeds]
    for t in tasks:
        p=paths(t);r=read(p['receipt']);assert r['task']==t and sha(p['prediction'])==r['prediction_sha256'] and sha(p['model'])==r['model_sha256']
    for seed in seeds:
        engine=Assembler(seed)
        for f in FOLDS:
            s=(f,);a=engine.ctx.meta.copy();models={}
            for h in (24,48):
                raw={}
                for c in CS:
                    value,mid,receipt=engine.scalar(s,f'lgb_{c}_{h:02d}');raw[c]=value;a[f'raw_{c}_{h:02d}']=value;models[f'{c}_{h:02d}']=dict(model_id=mid,receipt=receipt)
                a[f'L0_{h:02d}']=compose(raw)
            dest=PORT/f'splits/seed{seed}/anchors/S{f}.parquet';frame(dest,a)
            save(dest.with_suffix('.json'),dict(seed=seed,scope=[f],models=models,prediction_sha256=sha(dest)))
        for s in SCOPES:
            dest=PORT/f'splits/seed{seed}/scopes/S{scope_id(s)}.parquet';receipt=dest.with_suffix('.json')
            if receipt.exists():assert sha(dest)==read(receipt)['prediction_sha256'];continue
            d,g=engine.build(s);frame(dest,d);g['prediction_sha256']=sha(dest);save(receipt,g)
            print(f'Assembled split{seed} scope{s}',flush=True)
        reproduce(seed)
    print(f'All {26*len(seeds)} selected scoped I3 predictions assembled',flush=True)
    barrier=OUT/'audit/await_partial_audit.json'
    if len(seeds)==3 and barrier.exists():
        requirement=read(barrier);started=time.monotonic()
        while True:
            result=read(OUT/'audit/model_verification_partial.json')
            if result.get('status')=='PASS' and result.get('verifier_sha256')==requirement['verifier_sha256'] and result.get('models',0)>=requirement['models']:break
            log=(OUT/'partial_verification_cached.log').read_text()
            if 'Traceback (most recent call last)' in log:raise RuntimeError('Background model audit failed; inspect partial_verification_cached.log')
            if time.monotonic()-started>3600:raise RuntimeError('Background model audit did not finish within the assembly barrier')
            time.sleep(10)
        print('Background audit complete; final audit may safely reuse immutable caches',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reproduce-only',action='store_true');ap.add_argument('--seed',type=int,choices=SEEDS);args=ap.parse_args()
    seeds=(args.seed,) if args.seed is not None else SEEDS
    if args.reproduce_only:
        for seed in seeds:reproduce(seed);print(f'Frozen OOF reproduced split{seed}',flush=True)
    else:assemble(seeds)
