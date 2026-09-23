"""Portable I3 source package: frozen recipes, scoped inputs, safe immutable outputs."""
from pathlib import Path
import os,sys,json,hashlib,itertools,platform,importlib.metadata,shutil
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(OUT/'runtime_packages'));sys.path.insert(0,str(BASE))
import numpy as np
import pandas as pd
from config import make_lgbm_params
from protocol import load_data,expected_mask,component_target_name
SEEDS=(42,20260917,20260918);HS=(1,6,24,48);CS=('P_t','N_t','D_t','R_t');FOLDS=tuple(range(5))
SCOPES=[s for k in (2,3,4,5) for s in itertools.combinations(FOLDS,k)]
PORT=OUT/'portable';INPUT=PORT/'inputs';JOBS=OUT/'training';KEYS=['fipsCode','timestamp_et','hour_idx']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ah(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def digest(v):return hashlib.sha256(json.dumps(clean(v),sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def atomic(p,writer):
    p=Path(p).resolve();assert p.is_relative_to(OUT) and p!=OUT;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.stem+'.partial'+p.suffix);writer(q);q.replace(p)
def save(p,v):atomic(p,lambda q:q.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n'))
def frame(p,v):
    d=v if isinstance(v,pd.DataFrame) else pd.DataFrame(v)
    atomic(p,lambda q:d.to_parquet(q,index=False) if Path(p).suffix=='.parquet' else d.to_csv(q,index=False))
def immutable_json(p,v):
    if Path(p).exists():assert read(p)==clean(v),str(p)
    else:save(p,v)
def copy_input(src,dst):
    if Path(dst).exists():assert sha(src)==sha(dst)
    else:atomic(dst,lambda q:shutil.copyfile(src,q))
def hname(h):return f'osi_target_t{h:02d}h'
def target(c,h):return hname(h) if c=='osi' else f'{c}_target_t{h:02d}h'
def scope_id(s):return ''.join(map(str,s))
def post(a):
    a=np.clip(a,0,.65);return np.where(a<.001,0,a)
def compose(a):return post(np.maximum(sum(w*np.clip(a[c],0,1) for c,w in zip(CS,[.4,.35,.25,-.1])),0))
def recipes():
    rr=[]
    def add(name,family,c,h,feature='base',kind='raw'):
        rr.append(dict(name=name,family=family,component=c,horizon=h,feature=feature,kind=kind,target=target(c,h)))
    for h in HS:
        for c in CS:
            if c=='P_t' and h in (1,6):continue
            add(f'lgb_{c}_{h:02d}','lightgbm',c,h)
    for h in (24,48):add(f'lgb_osi_{h:02d}','lightgbm','osi',h)
    for h,feat in [(1,'p1'),(6,'base')]:
        for kind in ['a','b']:add(f'P{h:02d}_{kind}','lightgbm','P_t',h,feat,kind)
    add('P24_neighbor','lightgbm','P_t',24,'p24');add('D01_unknown','lightgbm','D_t',1,'base','u')
    for family in ['xgboost','catboost']:
        for h in (24,48):
            for c in ['osi',*CS]:add(f'{family}_{c}_{h:02d}',family,c,h)
    assert len(rr)==42;return {r['name']:r for r in rr}
RECIPES=recipes()
def params(r,rounds=None,probe=False):
    if r['family']=='lightgbm':
        p=make_lgbm_params(42)
        if r['kind'] in ('a','b'):p.update(objective='regression',metric='None')
        return p
    if r['family']=='xgboost':
        p=dict(objective='reg:pseudohubererror',huber_slope=.01,eval_metric='rmse',n_estimators=rounds,learning_rate=.05,max_depth=6,min_child_weight=50,
               subsample=.8,colsample_bytree=.8,reg_alpha=.1,reg_lambda=1.,tree_method='hist',random_state=42,n_jobs=1,verbosity=0)
    else:p=dict(loss_function='Huber:delta=0.01',eval_metric='RMSE',iterations=rounds,learning_rate=.05,depth=6,l2_leaf_reg=3.,random_strength=1.,rsm=.8,random_seed=42,thread_count=1,allow_writing_files=False,verbose=False)
    if probe:p['early_stopping_rounds']=100
    return p
def maxrounds(r):return 2000 if r['family']=='lightgbm' else (8000 if r['family']=='xgboost' and r['component']!='osi' else 4000)
def extension(r):return {'lightgbm':'.txt','xgboost':'.json','catboost':'.cbm'}[r['family']]
def task_id(seed,s,name):return f'split{seed}/S{scope_id(s)}/{name}'
def paths(task):
    d=JOBS/task['id'];return dict(folder=d,model=d/('model'+extension(RECIPES[task['recipe']])),receipt=d/'receipt.json',prediction=d/'prediction.npz')
def environment():
    import lightgbm,xgboost,catboost,scipy,sklearn,pyarrow
    return dict(python=platform.python_version(),platform=platform.platform(),libraries={m.__name__:m.__version__ for m in [np,pd,lightgbm,xgboost,catboost,scipy,sklearn,pyarrow]})
def tree_root(seed,sub,run):
    folder=BASE/sub
    if sub in ['p_structure_horizon_transfer','p_neighbor_horizon_increment','d_unknown_contribution'] and seed!=42:folder=folder/f'confirmation/split{seed}'
    return folder/'runs'/f'{run}_split{seed}_model42'
def reference_model(seed,s,r):
    name=r['name'];family=r['family'];h=r['horizon'];c=r['component'];kind=r['kind']
    st=ROOT/'versions/stella_v112_strict/runs'/f'stella_v112_nested_v1_split{seed}_model42'
    if len(s)==3 and name.startswith('lgb_') and c in CS and h in (24,48):
        p=st/'models/lightgbm'/('folds_'+'_'.join(map(str,s)))/(r['target']+'.receipt.json');return st,p
    if len(s)!=4:return None
    outer=next(iter(set(FOLDS)-set(s)))
    if family in ('xgboost','catboost'):
        return st,st/'models'/family/('folds_'+'_'.join(map(str,s)))/(r['target']+'.receipt.json')
    if name.startswith('lgb_'):root=BASE/'runs'/f'v18_tree_nested_v2_split{seed}_model42';p=root/f'models/outer{outer}/{r["target"]}.json'
    elif name.startswith('P01_'):
        root=tree_root(seed,'p_information_increment','v18_pinfo_v1');p=root/f'models/F2/outer{outer}/P_{"anchor_a" if kind=="a" else "excess_b"}_t01h.json'
    elif name.startswith('P06_'):
        root=tree_root(seed,'p_structure_horizon_transfer','v18_ptransfer_ab_v1');p=root/f'models/outer{outer}/P_{"anchor_a" if kind=="a" else "excess_b"}_t06h.json'
    elif name=='P24_neighbor':
        root=tree_root(seed,'p_neighbor_horizon_increment','v18_pneighbor_v1');p=root/f'models/outer{outer}/P_direct_t24h.json'
    else:
        root=tree_root(seed,'d_unknown_contribution','v18_ud01_nested_v1');p=root/f'models/outer{outer}/U_D_t_target_t01h.json'
    return root,p
class Context:
    def __init__(self,seed):
        self.seed=seed;self.meta=pd.read_parquet(INPUT/'meta.parquet');self.ntrain=34416
        self.X={k:pd.read_parquet(INPUT/f'features_{k}.parquet') for k in ['base','p1','p24']}
        self.fold=pd.read_parquet(INPUT/f'folds_{seed}.parquet').fold.to_numpy();self.labels=pd.read_parquet(INPUT/'labels.parquet')
        self.K=pd.read_parquet(INPUT/'known_history.parquet').K1.to_numpy();self.p=self.X['base'].last_P_t.to_numpy()
        self.hours=self.meta.hour_idx.to_numpy();self.train=self.meta.split.eq('train').to_numpy();self.config=read(OUT/'config.json');self.identity=read(OUT/'identity.json')['identity_hash']
    def subset(self,r,folds,county_group=None):
        eligible=self.train&np.isin(self.fold,folds)&(self.hours+r['horizon']<=215)
        if county_group is not None:eligible &= self.meta.fipsCode.isin(county_group).to_numpy()
        alln=int(eligible.sum());hit=eligible.copy()
        if r['kind']=='a':hit &= self.p>0
        if r['kind']=='b':hit &= self.p<1
        rows=np.flatnonzero(hit);y=self.labels[r['target']].iloc[rows].to_numpy(float,copy=True);w=np.ones(len(rows))
        if r['kind']=='a':y=np.minimum(y,self.p[rows])/self.p[rows];w=self.p[rows]**2
        if r['kind']=='b':y=np.maximum(y-self.p[rows],0)/(1-self.p[rows]);w=(1-self.p[rows])**2
        if r['kind']=='u':y-=self.K[rows];assert (y>=-1e-6).all()
        assert len(rows)>0 and np.isfinite(y).all() and (w>0).all()
        return dict(rows=rows,X=self.X[r['feature']].iloc[rows],y=y,w=w,total_valid=alln)
    def describe(self,x):
        return dict(rows=len(x['rows']),total_valid=x['total_valid'],counties=sorted(self.meta.fipsCode.iloc[x['rows']].unique()),
            folds=sorted(set(self.fold[x['rows']])),row_sha=ah(x['rows']),feature_sha=ah(x['X'].to_numpy()),label_sha=ah(x['y']),weight_sha=ah(x['w']))
    def probes(self,r,s):
        if len(s)>1:
            for q in s:yield str(q),self.subset(r,[v for v in s if v!=q]),self.subset(r,[q])
        else:
            counties=sorted(self.meta.loc[self.train&np.isin(self.fold,s),'fipsCode'].unique())
            for q in range(3):yield f'county_group{q}',self.subset(r,s,[v for i,v in enumerate(counties) if i%3!=q]),self.subset(r,s,[v for i,v in enumerate(counties) if i%3==q])
    def predict_rows(self,r,s):return np.flatnonzero((~np.isin(self.fold,s))&(self.hours+r['horizon']<=215))
