"""Independent verification of the diagnostic tables; never writes source artifacts."""
from pathlib import Path
import json
import hashlib
import subprocess
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;PKG=OUT.parent;REPO=PKG.parent
RUN=PKG/'outputs/runs/dem_v158_p2_component_seed42_cuda_log'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def csv(p):return pd.read_csv(p,dtype={'fipsCode':str,'scope':str},float_precision='round_trip')
def close(a,b,tol=1e-12):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
def post(values):return np.array([0. if min(.65,max(0.,float(v)))<.001 else min(.65,max(0.,float(v))) for v in values])


def compare(table,y,b,g):
    be=b-y;ge=g-y
    for key,value in {'n':len(y),'base_sse':np.dot(be,be),'gat_sse':np.dot(ge,ge),
        'base_rmse':np.sqrt(np.dot(be,be)/len(y)),'gat_rmse':np.sqrt(np.dot(ge,ge)/len(y)),
        'delta_sse':np.dot(ge,ge)-np.dot(be,be),'rmse_change_pct':100*(np.sqrt(np.dot(ge,ge)/np.dot(be,be))-1) if np.dot(be,be) else np.nan}.items():
        if key=='rmse_change_pct':
            # Newton's percentage exceeds 5,000%; independent reductions differ
            # at a few final bits. Absolute RMSE/SSE retain the 1e-12 check.
            np.testing.assert_allclose(getattr(table,key),value,rtol=1e-12,atol=1e-12,equal_nan=True)
        else:close(getattr(table,key),value)


def main():
    o=pd.read_parquet(RUN/'outer_oof.parquet');o=o.loc[o.is_scoreable].copy()
    i=pd.read_parquet(RUN/'inner_oof.parquet');a=pd.read_parquet(RUN/'alpha_selection.parquet')
    for table_name in ('overall_metrics','fold_effects','county_effects','window_effects'):
        t=csv(OUT/f'tables/{table_name}.csv')
        for row in t.itertuples(index=False):
            g=o.loc[o.horizon==row.horizon]
            if table_name=='fold_effects':g=g.loc[g.outer_fold==row.outer_fold]
            elif table_name=='county_effects':g=g.loc[g.fipsCode==row.fipsCode]
            elif table_name=='window_effects':
                if row.population=='Newton':g=g.loc[g.fipsCode=='18111']
                elif row.population=='others':g=g.loc[g.fipsCode!='18111']
                lo,hi=map(int,row.target_window.split('-'));hour=int(row.horizon[-3:-1]);g=g.loc[(g.hour_idx+hour).between(lo,hi)]
            compare(row,g.y_true.to_numpy(),post(g.base_prediction),g.prediction.to_numpy())
    rec=csv(OUT/'tables/inner_alpha_recomputed.csv')
    for row in rec.itertuples(index=False):
        g=i.loc[(i.outer_fold==row.outer_fold)&(i.horizon==row.horizon)]
        e=post(g.base_prediction+row.alpha*g.correction_osi)-g.y_true.to_numpy()
        close(row.sse,np.dot(e,e));close(row.rmse,np.sqrt(np.dot(e,e)/len(e)))
    for (q,h),g in rec.groupby(['outer_fold','horizon']):
        selected=g.loc[g.selected].iloc[0];assert selected.alpha==g.sort_values(['rmse','alpha']).iloc[0].alpha
        assert a.loc[(a.outer_fold==q)&(a.horizon==h)&a.selected,'alpha'].tolist()==[selected.alpha]
    outer_grid=csv(OUT/'tables/outer_alpha_hindsight_diagnostic.csv')
    for row in outer_grid.itertuples(index=False):
        g=o.loc[(o.horizon==row.horizon)&(o.outer_fold==row.outer_fold)]
        compare(row,g.y_true.to_numpy(),post(g.base_prediction),post(g.base_prediction+row.alpha*g.correction_osi))
    sensitivity=csv(OUT/'tables/diagnostic_exclusions_and_rollbacks.csv');h1=o.loc[o.horizon=='osi_target_t01h']
    for row in sensitivity.itertuples(index=False):
        g=h1.copy()
        if row.scenario=='exclude_Newton_hindsight':g=g.loc[g.fipsCode!='18111']
        elif row.scenario=='exclude_fold2_hindsight':g=g.loc[g.outer_fold!=2]
        elif row.scenario=='Newton_only':g=g.loc[g.fipsCode=='18111']
        b=post(g.base_prediction);pred=g.prediction.to_numpy(copy=True)
        if row.scenario=='Newton_alpha0_hindsight':pred[g.fipsCode.eq('18111').to_numpy()]=b[g.fipsCode.eq('18111').to_numpy()]
        elif row.scenario=='fold2_alpha0_hindsight':pred[g.outer_fold.eq(2).to_numpy()]=b[g.outer_fold.eq(2).to_numpy()]
        compare(row,g.y_true.to_numpy(),b,pred)
    # Independently validate the strongest input-support finding using only the
    # original Phase-1 column, not the graph feature construction helper.
    x=pd.read_parquet(REPO/'versions/xyy/v1.5.6/features_train_v1.5.6.parquet',columns=['pre_event_mean_osi'])
    meta=pd.read_parquet(REPO/'versions/xyy/v1.5.6/meta_train_v1.5.6.parquet');codes=meta.fipsCode.astype(str).str.zfill(5)
    folds=csv(RUN/'folds.csv').groupby('fipsCode').fold.first();allowed=codes.map(folds).ne(2)&meta.hour_idx.lt(215)
    values=x.pre_event_mean_osi.to_numpy(dtype=np.float32);train=values[allowed].astype(float);newton=values[codes=='18111'].astype(float)
    table=csv(OUT/'tables/Newton_feature_support_by_scope.csv')
    r=table.loc[(table.scope=='0,1,3,4')&(table.feature=='pre_event_mean_osi')].iloc[0]
    close(r.training_mean,train.mean());close(r.training_std,train.std(ddof=0));close(r.training_max,train.max())
    close(r.Newton_max_abs_z,abs((newton[0]-train.mean())/train.std(ddof=0)),tol=1e-9)
    official=csv(REPO/'data/DM_Train.csv');official=official.loc[official.fipsCode=='18111'].copy()
    hours=((pd.to_datetime(official.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta(hours=1)).astype(int)
    close(float(np.float32(official.loc[hours.between(0,47),'osi'].mean())),r.Newton_max)
    # Stored final alpha=0 is checked directly on the saved test rows.
    test=pd.read_parquet(RUN/'test_predictions.parquet');t=test.loc[test.horizon.eq('osi_target_t01h')&test.is_scoreable]
    assert len(t)==9009 and t.alpha.eq(0).all();close(t.prediction,post(t.base_prediction))
    final=csv(OUT/'tables/final_alpha_and_test_replay.csv');assert final.loc[final.horizon=='osi_target_t01h','test_changed_vs_base'].iloc[0]==0
    before=read(OUT/'reference/before_diagnosis.json');changed=[]
    for p,h in before['sha256'].items():
        if not Path(p).is_file() or sha(p)!=h:changed.append(p)
    assert not changed,changed
    status=subprocess.run(['git','--no-optional-locks','status','--porcelain','--untracked-files=all'],cwd=REPO,text=True,capture_output=True,check=True).stdout
    prefix='code_phase2_dem_eval_v158/diagnostics/'
    old_lines={l for l in before['git_status_before'].splitlines() if prefix not in l}
    new_lines={l for l in status.splitlines() if prefix not in l}
    assert old_lines==new_lines,'Changes outside diagnostics detected by git'
    report={'status':'PASS','saved_prediction_metric_replay':True,'inner_alpha_selection_recomputed':True,
        'Newton_loss_and_history_support_independently_checked':True,'saved_final_test_1h_exactly_base':True,
        'source_files_unchanged':len(before['sha256']),'git_changes_outside_diagnostics':False,
        'model_reload_performed':False,'model_training_performed':False,
        'scope':'independent diagnosis verification; not full model acceptance'}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
