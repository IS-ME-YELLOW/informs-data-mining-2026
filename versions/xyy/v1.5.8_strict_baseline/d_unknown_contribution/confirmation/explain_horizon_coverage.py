"""Explain the fixed C3 scoring-window relationship; diagnostic only, no new candidate."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    runs=[(42,HERE.parent/'runs/v18_ud01_nested_v1_split42_model42')]+[(s,HERE/f'split{s}/runs/v18_ud01_nested_v1_split{s}_model42') for s in (20260917,20260918)]
    rows=[];checks=[];sources={}
    for seed,run in runs:
        marker=json.loads((run/'U_CV_COMPLETE').read_text())
        path=run/'control_predictions.parquet'
        assert sha(path)==marker['files']['control_predictions.parquet']
        frame=pd.read_parquet(path);tables={}
        for h in (1,6):
            horizon=f'osi_target_t{h:02d}h';valid=frame[f'scoreable_{horizon}']
            data=frame.loc[valid,['fipsCode','hour_idx']].copy();data['target_hour']=data.hour_idx+h
            data['actual']=frame.loc[valid,f'actual_{horizon}'].to_numpy()
            for case in ('A','B','C'):data[case]=frame.loc[valid,f'pred_{case}_v18_rule_{horizon}'].to_numpy()
            tables[h]=data.set_index(['fipsCode','target_hour']).sort_index()
        overlap=tables[1].loc[tables[6].index]
        for key in ('actual','A','B','C'):np.testing.assert_array_equal(overlap[key],tables[6][key])
        first=tables[1].reset_index();reductions=[]
        for label,low,high in [('target73_76',73,76),('target77',77,77),('target78_215',78,215)]:
            subset=first.loc[first.target_hour.between(low,high)]
            b=(subset.B-subset.actual).to_numpy();c=(subset.C-subset.actual).to_numpy()
            change=float(np.sum(b*b)-np.sum(c*c));reductions.append(change)
            rows.append({'split_seed':seed,'window':label,'n':len(subset),'B_sse':float(np.sum(b*b)),
                         'C_sse':float(np.sum(c*c)),'sse_reduction_C_vs_B':change,
                         'B_rmse':float(np.sqrt(np.mean(b*b))),'C_rmse':float(np.sqrt(np.mean(c*c)))})
        six=tables[6]
        six_gain=float(((six.B-six.actual)**2-(six.C-six.actual)**2).sum())
        total_gain=float(((first.B-first.actual)**2-(first.C-first.actual)**2).sum())
        np.testing.assert_allclose(reductions[-1],six_gain,atol=1e-12,rtol=0)
        np.testing.assert_allclose(sum(reductions),total_gain,atol=1e-12,rtol=0)
        checks.append({'split_seed':seed,'1h_and_6h_predictions_equal_on_common_targets':True,
                       'common_scoreable_rows':len(six),'additive_SSE_identity':'PASS'})
        sources[str(path)]=sha(path)
    out=HERE/'summary';out.mkdir(exist_ok=True)
    table=pd.DataFrame(rows);table.to_csv(out/'horizon_coverage.csv',index=False)
    result={'status':'PASS','purpose':'Explanatory partition determined by fixed horizon coverage; not a new primary metric or post-hoc routed candidate',
            'checks':checks,'source_sha256':sha(Path(__file__)),'input_sha256':sources,
            'output_sha256':{'horizon_coverage.csv':sha(out/'horizon_coverage.csv')}}
    (out/'horizon_coverage_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(table.to_string(index=False))


if __name__=='__main__':main()
