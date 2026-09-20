"""No-training CPU reconstruction of the downloaded immutable old GAT package."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import sys
import time
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;PKG=OUT.parents[1];REPO=PKG.parent
CONFIG=json.loads((OUT/'diagnostic_config.json').read_text());RUN=Path(CONFIG['source_run'])
CACHE=REPO/'versions/xyy/v1.5.6'
sys.path.insert(0,str(PKG))
from gat_model import ResidualGAT,predict_gat,block_edges,block_edge_attributes
from data import add_neighbor_feature_aggregates,neighbor_source_names
from base_model import scoped_seed

HOURS=(1,6,24,48);COMPONENTS=('P_t','N_t','D_t','R_t');WEIGHTS=(.4,.35,.25,-.1)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)


def forbidden_fit(*args,**kwargs):raise PermissionError('This diagnostic forbids model fitting')
lgb.train=forbidden_fit
import gat_model
gat_model.fit_gat=forbidden_fit


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def output(path):
    p=(OUT/path).resolve()
    if not p.is_relative_to(OUT) or p==OUT:raise ValueError('Output outside diagnostic directory')
    p.parent.mkdir(parents=True,exist_ok=True);return p
def save_json(path,obj):output(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def save_csv(path,df):df.to_csv(output(path),index=False)
def post(x):
    a=np.clip(np.asarray(x,dtype=float),0,.65);return np.where(a<.001,0.,a)
def base_id(S,h,c):return f'base:{c}:{h}:S{",".join(map(str,S))}'
def stack_id(S,h):return f'stack:component_v158:{h}:S{",".join(map(str,S))}'


class Replay:
    """Prediction inputs are only frozen features, metadata, trees and checkpoint state."""
    def __init__(self):
        self.manifest=read(RUN/'run_manifest.json');self.identity=self.manifest['identity']['digest']
        assert self.identity==CONFIG['source_identity']
        for name,h in self.manifest['identity']['definition']['sources'].items():assert sha(PKG/name)==h,name
        for name,entry in self.manifest['inputs']['package_manifest']['files'].items():assert sha(CACHE/name)==entry['sha256'].lower(),name
        self.names=read(CACHE/'feature_names_v1.5.6.json');assert len(self.names)==163
        self.schema=read(RUN/'feature_schema.json')['graph_schemas']
        self.X_train=pd.read_parquet(CACHE/'features_train_v1.5.6.parquet');self.X_test=pd.read_parquet(CACHE/'features_test_v1.5.6.parquet')
        self.meta_train=pd.read_parquet(CACHE/'meta_train_v1.5.6.parquet');self.meta_test=pd.read_parquet(CACHE/'meta_test_v1.5.6.parquet')
        for meta in (self.meta_train,self.meta_test):meta['fipsCode']=meta.fipsCode.astype(str).str.zfill(5)
        self.X=pd.concat([self.X_train,self.X_test],ignore_index=True)
        self.meta=pd.concat([self.meta_train,self.meta_test],ignore_index=True)
        self.ntrain=len(self.X_train)
        folds=pd.read_csv(RUN/'folds.csv',dtype={'fipsCode':str},float_precision='round_trip')
        self.foldmap=folds.groupby('fipsCode').fold.first();self.rowfold=self.meta_train.fipsCode.map(self.foldmap).to_numpy(dtype=int)
        with np.load(RUN/'graph.npz',allow_pickle=False) as g:
            self.nodes=g['all_fips'].tolist();self.edges=g['edge_index'].copy();self.attrs=g['edge_attr'].copy();self.coords=g['coords'].copy()
        assert self.nodes==self.manifest['inputs']['all_fips']
        self.node_of={f:i for i,f in enumerate(self.nodes)}
        self.nodefold=np.array([self.foldmap.get(f,-1) for f in self.nodes],dtype=int)
        self.time=self.meta.hour_idx.to_numpy(dtype=int)-72;self.node=self.meta.fipsCode.map(self.node_of).to_numpy(dtype=int)
        assert len(self.meta)==43488 and len(set(zip(self.time,self.node)))==43488
        self.raw173=np.zeros((144,302,173),np.float32)
        self.raw173[self.time,self.node,1:164]=self.X.fillna(0).to_numpy(dtype=np.float32)
        self.raw173[:,:,164:166]=self.coords
        terrain=pd.read_csv(REPO/'data/geo/county_terrain.csv',dtype={'fipsCode':str},float_precision='round_trip').set_index('fipsCode')
        raw_terrain=(REPO/'data/geo/county_terrain.csv').read_bytes()
        assert hashlib.sha256(raw_terrain.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()==self.manifest['inputs']['terrain_sha256']
        self.raw173[:,:,166:173]=terrain.loc[self.nodes].to_numpy(np.float32)
        self.terrain_names=list(terrain)
        self.nonbase={}
        for hour in HOURS:
            h=f'osi_target_t{hour:02d}h';assert 'base' not in neighbor_source_names(h)
            raw,extra=add_neighbor_feature_aggregates(self.raw173,self.names,self.edges,h)
            assert ['base',*self.names,'latitude','longitude',*self.terrain_names,*extra]==self.schema[h]
            self.nonbase[h]=raw
        self.base_manifest=pd.read_parquet(RUN/'base_fit_manifest_final.parquet').set_index('model_id')
        self.prediction_cache={};self.base_hashes={};self.loaded_base_count=0
        for mid,row in self.base_manifest.iterrows():
            p=RUN/row.model_path.replace('\\','/');r=read(Path(str(p)+'.complete.json'))
            assert r['record_version']==2 and r['run_identity']==self.identity and r['model_id']==mid and r['model_file']==p.name
            assert sha(p)==r['sha256']==row.model_sha256;self.base_hashes[mid]=r['sha256']
        self.gat_paths=sorted((RUN/'models/gat').glob('*.pt'));assert len(self.gat_paths)==64

    def tree_prediction(self,S,h,c):
        mid=base_id(S,h,c)
        if mid not in self.prediction_cache:
            row=self.base_manifest.loc[mid];path=RUN/row.model_path.replace('\\','/')
            model=lgb.Booster(model_file=str(path));assert model.feature_name()==self.names and model.current_iteration()==row.actual_rounds
            pred=np.asarray(model.predict(self.X,num_iteration=int(row.actual_rounds),num_threads=1),dtype=float)
            assert np.isfinite(pred).all();self.prediction_cache[mid]=np.clip(pred,0,1)
            self.loaded_base_count+=1
        return self.prediction_cache[mid]

    def base_values(self,S,h):
        parts={}
        for c in COMPONENTS:
            values=np.empty(len(self.X))
            for fold in range(5):
                rows=np.flatnonzero(self.rowfold==fold)
                T=tuple(f for f in S if f!=fold) if fold in S else S
                values[rows]=self.tree_prediction(T,h,c)[rows]
            values[self.ntrain:]=self.tree_prediction(S,h,c)[self.ntrain:]
            parts[c]=values
        combined=np.zeros(len(self.X))
        for c,w in zip(COMPONENTS,WEIGHTS):combined+=w*parts[c]
        return post(combined),parts

    def load_checkpoint(self,path):
        receipt=read(Path(str(path)+'.complete.json'))
        assert receipt['record_version']==2 and receipt['run_identity']==self.identity and receipt['sha256']==sha(path)
        # User's own hash-matched checkpoint; CPU map handles its CUDA storages.
        p=torch.load(path,map_location='cpu',weights_only=False)
        S=tuple(p['scope']);h=p['horizon'];assert p['stack_id']==receipt['model_id']==stack_id(S,h)
        assert p['protocol']=='dem_v158_nested_v2' and p['mode']=='component_v158' and p['run_identity']==self.identity
        assert p['feature_names']==self.schema[h] and p['all_fips']==self.nodes and p['time_order']==list(range(72,216))
        assert p['feature_schema_hash']==hashlib.sha256(json.dumps(p['feature_names'],separators=(',',':')).encode()).hexdigest()
        for key in ('feature_side_hash','feature_package_hash','graph_hash'):assert p[key]==self.manifest['inputs'][key]
        np.testing.assert_array_equal(p['edge_index'],self.edges);np.testing.assert_array_equal(p['edge_attr'],self.attrs)
        expected={base_id(T,h,c):self.base_hashes[base_id(T,h,c)] for T in [S]+[tuple(f for f in S if f!=q) for q in S] for c in COMPONENTS}
        assert p['base_dependencies']==receipt['details']['base_dependencies']==expected
        for c in COMPONENTS:
            wanted=np.array([base_id(tuple(k for k in S if k!=f) if f in S else S,h,c) for f in self.rowfold],dtype=object)
            np.testing.assert_array_equal(p['base_source_by_component'][c],wanted)
            assert p['test_base_source_by_component'][c]==base_id(S,h,c)
        assert p['seed']==scoped_seed(42,'gat',S,h,'component_v158','fit')
        assert p['architecture']=={'in_dim':205,'hidden':24,'heads':4,'dropout':.12,'edge_dim':4}
        assert p['training_config']=={'epochs':220,'patience':35,'time_stride':1,'loss_mode':'mse','lr':.003,'weight_decay':.0002}
        assert 1<=p['fit_info']['best_epoch']<=p['fit_info']['epochs']<=220 and np.isfinite(p['fit_info']['best_loss'])
        for arr in (p['feature_mean'],p['feature_std']):assert np.asarray(arr).shape==(205,) and np.isfinite(arr).all()
        assert np.min(p['feature_std'])>=1e-6 and p['residual_scale']>=.003
        model=ResidualGAT(**{'in_dim':205,'hidden':24,'heads':4,'dropout':.12,'edge_dim':4})
        model.load_state_dict(p['state_dict'],strict=True);model.eval();model.requires_grad_(False)
        assert all(torch.isfinite(v).all().item() for v in p['state_dict'].values())
        return model,p

    def inputs(self,S,h,payload):
        base,parts=self.base_values(S,h)
        raw=self.nonbase[h].copy();raw[self.time,self.node,0]=base.astype(np.float32)
        hour=int(h[-3:-1]);mask=(np.arange(144)+72+hour<=215)[:,None]&np.isin(self.nodefold,S)[None,:]
        values=raw[mask]
        mean=values.mean(axis=0,dtype=np.float64);std=values.std(axis=0,dtype=np.float64);std[std<1e-6]=1.
        mean_error=float(np.max(np.abs(mean-payload['feature_mean'])));std_error=float(np.max(np.abs(std-payload['feature_std'])))
        np.testing.assert_allclose(mean,payload['feature_mean'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(std,payload['feature_std'],rtol=0,atol=1e-10)
        scaled=((raw.astype(np.float64)-payload['feature_mean'])/payload['feature_std']).astype(np.float32)
        assert np.isfinite(scaled).all()
        return raw,scaled,parts,mask,mean_error,std_error

    def predict(self,S,h,path=None):
        if path is None:path=RUN/'models/gat'/f'gat_component_v158_{h.replace("osi_target_","")}_S{"-".join(map(str,S))}.pt'
        model,p=self.load_checkpoint(path)
        raw,scaled,parts,mask,me,se=self.inputs(S,h,p)
        norm=predict_gat(model,scaled,self.edges,'cpu',self.attrs)
        corr=norm*float(p['residual_scale'])
        assert np.isfinite(corr).all()
        return {'model':model,'payload':p,'raw':raw,'scaled':scaled,'parts':parts,'mask':mask,
            'normalised_prediction':norm,'correction':corr,'feature_mean_error':me,'feature_std_error':se}


def metadata(engine):
    rows=[]
    for path in engine.gat_paths:
        model,p=engine.load_checkpoint(path)
        rows.append({'stack_id':p['stack_id'],'horizon':p['horizon'],'scope':','.join(map(str,p['scope'])),
            'seed':p['seed'],'epochs':p['fit_info']['epochs'],'best_epoch':p['fit_info']['best_epoch'],
            'best_loss':p['fit_info']['best_loss'],'residual_scale':p['residual_scale'],
            'feature_mean_pre_event_mean_osi':float(p['feature_mean'][p['feature_names'].index('pre_event_mean_osi')]),
            'feature_std_pre_event_mean_osi':float(p['feature_std'][p['feature_names'].index('pre_event_mean_osi')]),
            'parameter_count':sum(v.numel() for v in model.parameters()),'dependency_count':len(p['base_dependencies'])})
    save_csv('tables/checkpoint_metadata.csv',pd.DataFrame(rows));return rows


def replay_one(engine,S,h,outer,inner,test,targets):
    start=time.monotonic();result=engine.predict(S,h);p=result['payload'];raw=result['raw'];corr=result['correction']
    rows=[];detail=[];norm=result['normalised_prediction'];label_grid=np.zeros((144,302),float)
    valid_train=(engine.meta_train.hour_idx.to_numpy()+int(h[-3:-1])<=215)&np.isin(engine.rowfold,S)
    ids=np.flatnonzero(valid_train)
    base=raw[engine.time[:engine.ntrain],engine.node[:engine.ntrain],0].astype(float)
    residual=targets[h].to_numpy()[ids]-base[ids]
    scale=max(float(np.std(residual,ddof=0)),.003)
    np.testing.assert_allclose(scale,p['residual_scale'],rtol=0,atol=1e-10)
    label_grid[engine.time[ids],engine.node[ids]]=residual/p['residual_scale']
    eval_mse=float(np.mean((norm[result['mask']].astype(float)-label_grid[result['mask']])**2))
    # Best loss was CUDA float32 training MSE; record the CPU difference separately.
    groups=[]
    if len(S)==4:groups.append(('outer',outer.loc[(outer.stack_id==p['stack_id'])]))
    if len(S)==3:groups.append(('inner',inner.loc[inner.stack_id==p['stack_id']]))
    if len(S)==5:groups.append(('test',test.loc[test.stack_id==p['stack_id']]))
    for kind,frame in groups:
        assert len(frame)
        ti=frame.hour_idx.to_numpy(dtype=int)-72;ni=frame.fipsCode.map(engine.node_of).to_numpy(dtype=int)
        predicted=corr[ti,ni].astype(float);b=raw[ti,ni,0].astype(float)
        base_error=float(np.max(np.abs(b-frame.base_prediction.to_numpy())))
        c_error=float(np.max(np.abs(predicted-frame.correction_osi.to_numpy())))
        assert base_error<=1e-12,(p['stack_id'],'base',base_error)
        assert c_error<=CONFIG['tolerances']['CPU_vs_CUDA_raw_correction_absolute'],(p['stack_id'],'correction',c_error)
        prediction_error=None;flips=0
        if kind!='inner':
            valid=frame.is_scoreable.to_numpy(dtype=bool);before=b+frame.alpha.to_numpy()*predicted;cpu=post(before)
            old_before=frame.prediction_before_postprocess.to_numpy();old_pred=frame.prediction.to_numpy()
            pre_error=float(np.max(np.abs(before[valid]-old_before[valid])))
            prediction_error=float(np.max(np.abs(cpu[valid]-old_pred[valid])))
            flips=int(np.sum(valid&((before<.001)!=(old_before<.001))))
            assert pre_error<=1e-5
            # Threshold flips are explicitly retained, not dismissed as a tight replay.
            effect=frame.copy();effect['CPU_base']=b;effect['CPU_correction']=predicted;effect['CPU_prediction']=np.where(valid,cpu,np.nan)
            effect['CPU_raw_difference']=predicted-frame.correction_osi.to_numpy();detail.append(effect)
        rows.append({'stack_id':p['stack_id'],'horizon':h,'scope':','.join(map(str,S)),'evidence':kind,'compared_rows':len(frame),
            'base_max_absolute_error':base_error,'correction_max_absolute_error':c_error,'postprediction_max_absolute_error':prediction_error,
            'zero_threshold_flips':flips,'feature_mean_max_absolute_error':result['feature_mean_error'],
            'feature_std_max_absolute_error':result['feature_std_error'],'residual_scale_error':abs(scale-p['residual_scale']),
            'saved_best_loss':p['fit_info']['best_loss'],'CPU_training_mask_MSE':eval_mse,'training_mask_MSE_difference':eval_mse-p['fit_info']['best_loss'],
            'wall_seconds':time.monotonic()-start})
    return result,rows,detail


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['metadata','newton','all'],default='metadata');args=parser.parse_args()
    start=time.monotonic();engine=Replay();meta=metadata(engine)
    save_json('metadata_verification.json',{'status':'PASS','base_models_hashed':416,'GAT_checkpoints_loaded':64,
        'scope_dependencies_schemas_weights_verified':True,'torch_version':torch.__version__,'device':'cpu','training_performed':False})
    if args.stage=='metadata':print('Metadata PASS: all 64 checkpoints loaded and 416 base hashes checked');return
    # Saved outcomes are evaluation-only. Prediction methods above never accept them.
    outer=pd.read_parquet(RUN/'outer_oof.parquet');inner=pd.read_parquet(RUN/'inner_oof.parquet');test=pd.read_parquet(RUN/'test_predictions.parquet')
    targets=pd.read_parquet(CACHE/'targets_train_v1.5.6.parquet')
    todo=[((0,1,3,4),'osi_target_t01h')]
    if args.stage=='all':
        todo += [(S,f'osi_target_t{h:02d}h') for size in (4,3,5) for S in itertools.combinations(range(5),size) for h in HOURS if (S,f'osi_target_t{h:02d}h') not in todo]
    metrics=[];details=[]
    for n,(S,h) in enumerate(todo,1):
        result,records,frames=replay_one(engine,S,h,outer,inner,test,targets);metrics.extend(records);details.extend(frames)
        if S==(0,1,3,4) and h=='osi_target_t01h':
            np.savez_compressed(output('Newton_outer2_inputs.npz'),raw=result['raw'],scaled=result['scaled'],correction=result['correction'])
            repeated=predict_gat(result['model'],result['scaled'],engine.edges,'cpu',engine.attrs)*float(result['payload']['residual_scale'])
            np.testing.assert_array_equal(repeated,result['correction'])
        print(f'[{n}/{len(todo)}] {stack_id(S,h)} correction max diff={records[0]["correction_max_absolute_error"]:.3g}',flush=True)
        save_csv('tables/CPU_replay_metrics.csv',pd.DataFrame(metrics))
    if details:pd.concat(details,ignore_index=True).to_parquet(output('tables/CPU_replayed_outer_and_test.parquet'),index=False)
    frame=pd.DataFrame(metrics)
    summary={'status':'CPU_REPLAY_PASS_WITH_DECLARED_TOLERANCE','stage':args.stage,'checkpoints_replayed':len(todo),
        'all_checkpoint_metadata_verified':64,'base_models_predicted':engine.loaded_base_count,
        'max_correction_absolute_error':float(frame.correction_max_absolute_error.max()),
        'max_base_absolute_error':float(frame.base_max_absolute_error.max()),
        'max_postprediction_absolute_error':float(frame.postprediction_max_absolute_error.max()),
        'zero_threshold_flips':int(frame.zero_threshold_flips.sum()),'Newton_CPU_repeat_bitwise_equal':True,
        'correction_tolerance':1e-5,'wall_seconds':time.monotonic()-start,'torch':torch.__version__,
        'original_device':'Windows/CUDA','current_device':'Linux/CPU','original_run_identity':engine.identity,
        'formal_original_CUDA_acceptance_claimed':False,'training_performed':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    save_json('replay_verification.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
