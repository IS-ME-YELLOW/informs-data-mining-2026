"""Adapt the frozen single-source evaluator; retain its scoring and bootstrap functions."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'p_structure_horizon_transfer'


def main():
    original=(PARENT/'evaluate_transfer.py').read_text();s=original
    s=s.replace('from transfer_protocol import *','from neighbor_protocol import *\nTRANSFER=CHANGED')
    s=s.replace('ctx=initialize(load_context());m=', 'ctx=attach_features(initialize(load_context()));m=')
    s=s.replace("RUN/'branch_oof_predictions.parquet'","RUN/'new_P_oof_predictions.parquet'")
    s=s.replace("close=reconstruct(ctx.p,d.raw_a.to_numpy(),d.raw_b.to_numpy(),expected_mask(ctx.meta,h))[0]",
        "close=(reconstruct(ctx.p,d.raw_a.to_numpy(),d.raw_b.to_numpy(),expected_mask(ctx.meta,h))[0] if HORIZON_HOURS[h]==6 else clip_component(d.raw_direct.to_numpy()))")
    s=s.replace("'reference_F2_identity_hash']=F2_ID","'reference_current_identity_hash']=REFERENCE_ID")
    s=s.replace('byh[h].reconstruction_id.to_numpy()','byh[h].P_source_id.to_numpy()')
    s=s.replace("ctx.reference_parts['source_F2_'+tag]","ctx.reference_parts['source_AB_P06_'+tag]")
    first=s.index('    branch_stats=[];decompositions=[];audits=[]')
    last=s.index("    write_frame(RUN/'component_predictions.parquet'",first)
    replacement='''    branch_stats=[];decompositions=[];audits=[]
    for h,d in byh.items():
        y=d.true_P.to_numpy();valid=d.scoreable.to_numpy(bool);p=ctx.p
        audit=ctx.meta.copy();audit['horizon']=h;audit['P71']=p;audit['true_P']=y;audit['scoreable']=valid
        for kind in (('a','b') if HORIZON_HOURS[h]==6 else ('direct',)):
            active=valid&support(p,kind);labels=np.full(len(p),np.nan);weights=np.where(valid,0.,np.nan)
            labels[active],weights[active]=transform(y[active],p[active],kind)
            raw=d['raw_'+kind].to_numpy();applied=np.clip(raw,0,1)
            audit['label_'+kind]=labels;audit['weight_'+kind]=weights;audit['support_'+kind]=active
            branch_stats.append(dict(horizon=h,kind=kind,supported_rows=int(active.sum()),all_valid_rows=int(valid.sum()),
                raw_label_rmse=metric(labels,raw)['rmse'],clipped_label_rmse=metric(labels,applied)['rmse'],
                raw_P_contribution_rmse=float(np.sqrt(np.sum(weights[active]*(raw[active]-labels[active])**2)/valid.sum())),
                clipped_P_contribution_rmse=float(np.sqrt(np.sum(weights[active]*(applied[active]-labels[active])**2)/valid.sum())),
                clip_low_rows=int((raw[active]<0).sum()),clip_high_rows=int((raw[active]>1).sum())))
        if HORIZON_HOURS[h]==6:
            ea=d.contribution_a.to_numpy()[valid]-np.minimum(y[valid],p[valid]);eb=d.contribution_b.to_numpy()[valid]-np.maximum(y[valid]-p[valid],0)
            mse=np.mean((d.pred_P.to_numpy()[valid]-y[valid])**2)
            np.testing.assert_allclose(np.mean(ea**2)+np.mean(eb**2)+2*np.mean(ea*eb),mse,atol=1e-12,rtol=0)
            decompositions.append(dict(horizon=h,a_mse=np.mean(ea**2),b_mse=np.mean(eb**2),twice_cross_mean=2*np.mean(ea*eb),P_mse=mse))
        audits.append(audit)
'''
    s=s[:first]+replacement+s[last:]
    before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
    after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    for name in ('metric','comparison','bootstrap'):assert before[name]==after[name]
    target=HERE/'evaluate_neighbors.py'
    if target.exists():assert target.read_text()==s
    else:target.write_text(s)
    result={'status':'PASS','source':str(PARENT/'evaluate_transfer.py'),
        'parent_sha256':hashlib.sha256(original.encode()).hexdigest(),'adapted_sha256':hashlib.sha256(s.encode()).hexdigest(),
        'unchanged_functions':['metric','comparison','bootstrap'],'adaptations':['new baseline prefix','three horizon-specific feature candidates','P24/P48 direct diagnostics']}
    (HERE/'reference').mkdir(exist_ok=True)
    (HERE/'reference/evaluation_adaptation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Evaluation adaptation PASS')


if __name__=='__main__':main()
