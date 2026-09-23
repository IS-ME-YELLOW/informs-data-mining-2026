"""Wendy-side reader: pure NumPy/Pandas; never needs training libraries or model weights."""
from pathlib import Path
from functools import lru_cache
import json,hashlib,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def read_frame(path):
    # Arrow's physical schema is portable; avoid pandas-3-only dtype metadata on pandas 2.
    return pq.read_table(path).to_pandas(ignore_metadata=True)
class PredictionPackage:
    def __init__(self,root=None):
        self.root=Path(root or Path(__file__).resolve().parent)
        self.meta=read_frame(self.root/'inputs/meta.parquet');self.n=len(self.meta)
        self._labels=None
    @staticmethod
    def scope(scope,minimum=2):
        s=tuple(sorted(set(int(v) for v in scope)))
        if len(s)<minimum or len(s)>5 or not set(s)<=set(range(5)):raise ValueError('Invalid allowed fold scope')
        return s
    @lru_cache(maxsize=8)
    def folds(self,seed):
        if seed not in [42,20260917,20260918]:raise ValueError('Unknown frozen CV')
        return read_frame(self.root/f'inputs/folds_{seed}.parquet').fold.to_numpy()
    @lru_cache(maxsize=8)
    def base_scope(self,seed,scope):
        s=self.scope(scope);p=self.root/f'splits/seed{seed}/scopes/S{"".join(map(str,s))}.parquet';rec=json.loads(p.with_suffix('.json').read_text())
        actual=file_sha256(p)
        if actual!=rec['prediction_sha256']:raise ValueError('Scoped prediction hash mismatch')
        d=read_frame(p)
        pd.testing.assert_frame_equal(d[['fipsCode','timestamp_et','hour_idx','split']],self.meta)
        if not np.array_equal(d.fold,self.folds(seed)):raise ValueError('CV row mismatch')
        return d
    def graph_context(self,seed,scope,horizon):
        """B(S minus r) for supervised r in S; B(S) for held-out and test nodes.

        Return row-keyed data in train-then-test source order. Reindex explicitly
        to the graph's county/time node order; do not assume its array order.
        base_prediction preserves NaN tails. base_graph_input fills only invalid
        tails with 0 and is never a scoring target/prediction.
        """
        s=self.scope(scope,3);h=int(horizon)
        if h not in [1,6,24,48]:raise ValueError('Unsupported horizon')
        fold=self.folds(seed);out=self.meta.copy();out['fold']=fold;out['source_scope']='';out['horizon']=h;out['target_hour']=out.hour_idx+h
        names=[f'I3_OSI_{h:02d}',f'T_OSI_{h:02d}']+[f'T_{space}_{c}_{k:02d}' for space in ['source','aligned'] for c in ['P_t','N_t','D_t','R_t'] for k in [1,6,24,48]]
        if h>=24:names += [f'S_L2_{h:02d}',f'S_theta_{h:02d}']
        arrays={c:np.full(self.n,np.nan) for c in names}
        for r in [-1,0,1,2,3,4]:
            hit=fold==r;sub=tuple(f for f in s if f!=r) if r in s else s;d=self.base_scope(seed,sub)
            out.loc[hit,'source_scope']=''.join(map(str,sub))
            for c in names:arrays[c][hit]=d.loc[hit,c]
            if r in sub:raise AssertionError('A node used its own fold to fit its baseline')
            if not set(sub)<=set(s):raise AssertionError('Baseline dependency escapes GAT scope')
        for c,v in arrays.items():out[c]=v
        valid=out.target_hour.to_numpy()<=215;out['valid']=valid
        out['supervised']=valid&np.isin(fold,s);out['outer_or_inner_holdout']=valid&(fold>=0)&~np.isin(fold,s)
        out['base_prediction']=arrays[f'I3_OSI_{h:02d}'];out['base_graph_input']=np.where(valid,out.base_prediction,0.)
        if not np.array_equal(np.isfinite(out.base_prediction),valid):raise ValueError('Missing valid node prediction or non-NaN tail')
        return out
    def labels_for_rows(self,seed,scope,horizon,rows):
        """Explicitly reject label reads outside the allowed training county scope."""
        s=self.scope(scope,3);rows=np.asarray(rows,dtype=int);h=int(horizon)
        if h not in [1,6,24,48] or rows.ndim!=1 or ((rows<0)|(rows>=self.n)).any():raise ValueError('Invalid target/row selection')
        allowed=np.isin(self.folds(seed)[rows],s)&(self.meta.hour_idx.to_numpy()[rows]+h<=215)
        if not allowed.all():raise PermissionError('Requested labels outside supervised scope or valid horizon')
        if self._labels is None:self._labels=read_frame(self.root/'inputs/labels.parquet')
        return self._labels[f'osi_target_t{h:02d}h'].iloc[rows].to_numpy(float,copy=True)
    def residual_supervision(self,seed,scope,horizon):
        d=self.graph_context(seed,scope,horizon);mask=d.supervised.to_numpy();rows=np.flatnonzero(mask)
        residual=np.zeros(self.n);residual[rows]=self.labels_for_rows(seed,scope,horizon,rows)-d.base_prediction.to_numpy()[rows]
        return residual,mask
    def model_features(self,kind='base'):
        if kind not in ['base','p1','p24']:raise ValueError('Unknown frozen feature set')
        return read_frame(self.root/f'inputs/features_{kind}.parquet')
    def assert_cv_assignment(self,seed,fips_codes,row_folds):
        """Call at the integration boundary to reject a legacy/default CV mapping."""
        mapping=dict(zip(self.meta.fipsCode,self.folds(seed)))
        keys=pd.Series(fips_codes).astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
        expected=keys.map(mapping);actual=np.asarray(row_folds)
        if expected.isna().any() or actual.shape!=expected.shape or not np.array_equal(actual,expected.to_numpy()):
            raise ValueError('Consumer CV mapping differs from the selected prediction package split')
    def align_context(self,context,fips_codes,hour_indices):
        """Align to caller graph/feature order by unique keys, returning original row indices."""
        fips=pd.Series(fips_codes).astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
        hour=np.asarray(hour_indices)
        if hour.ndim!=1 or len(hour)!=len(fips) or not np.isfinite(hour.astype(float)).all() or not np.array_equal(hour.astype(float),hour.astype(int)):
            raise ValueError('Invalid requested county/hour keys')
        keys=pd.MultiIndex.from_arrays([fips.to_numpy(),hour.astype(int)])
        if keys.has_duplicates:raise ValueError('Duplicate requested county/hour keys')
        source=pd.MultiIndex.from_frame(context[['fipsCode','hour_idx']]);assert not source.has_duplicates
        idx=source.get_indexer(keys)
        if (idx<0).any():raise ValueError('Requested county/hour is absent from the package context')
        package_index=pd.MultiIndex.from_frame(self.meta[['fipsCode','hour_idx']]);original_rows=package_index.get_indexer(keys)
        if (original_rows<0).any():raise ValueError('Context contains a county/hour absent from package metadata')
        out=context.iloc[idx].copy();out['package_row_index']=original_rows
        return out.reset_index(drop=True)
