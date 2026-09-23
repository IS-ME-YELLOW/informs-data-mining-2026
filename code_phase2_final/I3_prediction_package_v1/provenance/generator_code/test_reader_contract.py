"""Synthetic consumer checks; no fitting and no dependency on incomplete model jobs."""
import sys,types
from package_core import *
sys.path.insert(0,str(PORT))
from prediction_reader import PredictionPackage
def main():
    rows=[]
    for fold in [-1,0,1,2,3,4]:
        for hour in [72,215]:rows.append(dict(fipsCode=f'{10000+fold:05d}',timestamp_et=pd.Timestamp('2026-03-11')+pd.Timedelta(hours=hour),hour_idx=hour,split='test' if fold<0 else 'train',fold=fold))
    allmeta=pd.DataFrame(rows);pkg=PredictionPackage.__new__(PredictionPackage);pkg.meta=allmeta.drop(columns='fold');pkg.n=len(allmeta);pkg._labels=pd.DataFrame({hname(h):np.where(allmeta.fold>=0,.1,np.nan) for h in HS})
    pkg.folds=lambda seed:allmeta.fold.to_numpy()
    used=[]
    def mock_scope(seed,s):
        used.append(tuple(s));d=allmeta.copy();live=~allmeta.fold.isin(s)
        for h in HS:
            valid=live&(allmeta.hour_idx+h<=215);value=sum(2**q for q in s)/100
            d[f'I3_OSI_{h:02d}']=np.where(valid,value,np.nan);d[f'T_OSI_{h:02d}']=d[f'I3_OSI_{h:02d}']
            for c in CS:
                for space in ['source','aligned']:d[f'T_{space}_{c}_{h:02d}']=np.where(valid,.01,np.nan)
            if h>=24:d[f'S_L2_{h:02d}']=d[f'I3_OSI_{h:02d}'];d[f'S_theta_{h:02d}']=.05
        return d
    pkg.base_scope=mock_scope;checks=0
    pkg.assert_cv_assignment(42,allmeta.fipsCode,allmeta.fold)
    wrong=allmeta.fold.to_numpy(copy=True);wrong[2]=(wrong[2]+1)%5
    try:pkg.assert_cv_assignment(42,allmeta.fipsCode,wrong)
    except ValueError:pass
    else:raise AssertionError('Mismatched CV accepted')
    for s in [(1,2,3),(1,2,3,4),(0,1,2,3,4)]:
        for h in HS:
            d=pkg.graph_context(42,s,h)
            reverse=np.arange(len(d))[::-1];aligned=pkg.align_context(d,d.fipsCode.iloc[reverse],d.hour_idx.iloc[reverse])
            np.testing.assert_array_equal(aligned.package_row_index,reverse);np.testing.assert_array_equal(aligned.base_prediction,d.base_prediction.to_numpy()[reverse])
            for i,row in d.iterrows():
                sub=set(s)-{row.fold} if row.fold in s else set(s)
                assert set(map(int,row.source_scope))==sub
                if row.valid:assert row.base_prediction==sum(2**q for q in sub)/100
                else:assert np.isnan(row.base_prediction) and row.base_graph_input==0
                assert row.supervised==bool(row.valid and row.fold in s)
            r,m=pkg.residual_supervision(42,s,h);np.testing.assert_allclose(r[m],.1-d.loc[m,'base_prediction']);assert (r[~m]==0).all();checks+=1
            for bad in [0,1,3]:
                if not d.supervised.iloc[bad]:
                    try:pkg.labels_for_rows(42,s,h,[bad])
                    except PermissionError:pass
                    else:raise AssertionError('Forbidden label accepted')
    for bad in [(0,1),(),(0,1,8)]:
        try:pkg.graph_context(42,bad,1)
        except ValueError:pass
        else:raise AssertionError('Invalid GAT scope accepted')
    for ff,hh in [(['99999'],[72]),([d.fipsCode.iloc[0]]*2,[72,72])]:
        try:pkg.align_context(d,ff,hh)
        except ValueError:pass
        else:raise AssertionError('Invalid alignment accepted')
    save(OUT/'preflight/reader_contract.json',dict(status='PASS',synthetic_contexts=checks,scope_routing=True,invalid_tail_isolation=True,label_access_rejection=True,formal_fits=0,reader_sha256=sha(PORT/'prediction_reader.py')))
    print('Synthetic reader contract PASS',flush=True)
if __name__=='__main__':main()
