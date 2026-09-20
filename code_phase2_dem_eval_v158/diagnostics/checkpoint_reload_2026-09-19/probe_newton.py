"""Fixed-weight computational interventions, not retrained model comparisons."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from replay_models import (OUT,RUN,Replay,post,save_json,save_csv,output,predict_gat,
    block_edges,block_edge_attributes,add_neighbor_feature_aggregates)


def main():
    start=time.monotonic();engine=Replay();S=(0,1,3,4);h='osi_target_t01h'
    path=RUN/'models/gat/gat_component_v158_t01h_S0-1-3-4.pt'
    model,p=engine.load_checkpoint(path)
    with np.load(OUT/'Newton_outer2_inputs.npz',allow_pickle=False) as z:
        raw=z['raw'].copy();scaled=z['scaled'].copy();expected=z['correction'].copy()
    np.testing.assert_array_equal(scaled,((raw.astype(float)-p['feature_mean'])/p['feature_std']).astype(np.float32))
    node=engine.node_of['18111'];rows=torch.arange(144)*302+node
    edges=block_edges(engine.edges,302,144,'cpu');attrs=block_edge_attributes(engine.attrs,144,'cpu')
    x=torch.from_numpy(scaled.reshape(-1,205))
    scale=float(p['residual_scale']);idx=p['feature_names'].index('pre_event_mean_osi')
    def messages(values):return F.elu(model.gat2(F.elu(model.gat1(values,edges,attrs)),edges,attrs))
    def head(values):return (model.head(values).squeeze(-1).reshape(144,302).cpu().numpy()*scale)
    with torch.no_grad():
        msg=messages(x);skip=model.skip(x)
        full=head(msg+skip);np.testing.assert_array_equal(full,expected)
        no_skip=head(msg);no_message=head(skip);zero=head(torch.zeros_like(skip))
        x_mean=x.clone();x_mean[rows,idx]=0.
        msg_mean=messages(x_mean);skip_mean=model.skip(x_mean)
        remove_both=head(msg_mean+skip_mean)
        remove_skip_feature=head(msg+skip_mean)
        remove_message_feature=head(msg_mean+skip)
        baseline_skip_norm=torch.linalg.vector_norm(skip[rows],dim=1).cpu().numpy()
        baseline_msg_norm=torch.linalg.vector_norm(msg[rows],dim=1).cpu().numpy()
        pre_skip=x[rows,idx,None]*model.skip.weight[:,idx][None,:]
        pre_skip_norm=torch.linalg.vector_norm(pre_skip,dim=1).cpu().numpy()
    variants={'original':full,'zero_skip_path':no_skip,'zero_message_path':no_message,'head_at_zero_hidden':zero,
        'Newton_pre_event_to_train_mean_both_paths':remove_both,
        'Newton_pre_event_removed_only_from_skip':remove_skip_feature,
        'Newton_pre_event_removed_only_from_message':remove_message_feature}
    scope_nodes=np.isin(engine.nodefold,S);train=raw[:143,scope_nodes,:].reshape(-1,205)
    mins=train.min(axis=0);maxs=train.max(axis=0)
    def raw_intervention(name,fields):
        altered=raw[:,:,:173].copy()
        for field in fields:
            j=p['feature_names'].index(field);assert 1<=j<164
            altered[:,node,j]=np.clip(altered[:,node,j],mins[j],maxs[j])
        graph,_=add_neighbor_feature_aggregates(altered,engine.names,engine.edges,h)
        z=((graph.astype(float)-p['feature_mean'])/p['feature_std']).astype(np.float32)
        variants[name]=predict_gat(model,z,engine.edges,'cpu',engine.attrs)*scale
    raw_intervention('Newton_pre_event_to_training_max',['pre_event_mean_osi'])
    raw_intervention('Newton_four_extreme_history_to_training_bounds',
        ['pre_event_mean_osi','osi_mean_72h','outage_pct_mean_72h','pre_event_max_outage_pct'])
    variants['all_nodes_all_inputs_z_clipped_5']=predict_gat(model,np.clip(scaled,-5,5),engine.edges,'cpu',engine.attrs)*scale
    old=pd.read_parquet(RUN/'outer_oof.parquet')
    focus=old.loc[(old.horizon==h)&(old.fipsCode=='18111')].sort_values('hour_idx').reset_index(drop=True)
    assert focus.hour_idx.tolist()==list(range(72,216))
    valid=focus.is_scoreable.to_numpy();base=raw[:,node,0].astype(float);truth=focus.y_true.to_numpy();alpha=float(focus.alpha.iloc[0]);assert alpha==.2
    timeline=focus[['fipsCode','hour_idx','target_timestamp','is_scoreable','y_true','base_prediction','alpha']].copy()
    summaries=[]
    for name,all_corr in variants.items():
        corr=all_corr[:,node].astype(float);pred=post(base+alpha*corr)
        timeline[name+'_correction']=corr;timeline[name+'_prediction']=np.where(valid,pred,np.nan)
        summaries.append({'intervention':name,'fixed_weight_sensitivity_only':True,
            'Newton_mean_correction':float(corr[valid].mean()),'Newton_max_correction':float(corr[valid].max()),
            'Newton_mean_prediction':float(pred[valid].mean()),'Newton_1h_RMSE_diagnostic_only':float(np.sqrt(np.mean((pred[valid]-truth[valid])**2))),
            'mean_correction_change_vs_original':float((corr[valid]-full[valid,node]).mean())})
        print(name,'mean correction',corr[valid].mean(),flush=True)
    full64=full[:,node].astype(float);m=no_skip[:,node].astype(float);s=no_message[:,node].astype(float);z=zero[:,node].astype(float)
    path_skip=.5*((s-z)+(full64-m));path_message=.5*((m-z)+(full64-s))
    np.testing.assert_allclose(path_skip+path_message,full64-z,rtol=0,atol=1e-12)
    f00=remove_both[:,node].astype(float);f10=remove_skip_feature[:,node].astype(float);f01=remove_message_feature[:,node].astype(float)
    feature_skip=.5*((f01-f00)+(full64-f10));feature_message=.5*((f10-f00)+(full64-f01))
    np.testing.assert_allclose(feature_skip+feature_message,full64-f00,rtol=0,atol=1e-12)
    decomposition=pd.DataFrame({'origin_hour':np.arange(72,216),'scoreable':valid,'original_correction':full64,'head_zero_correction':z,
        'skip_path_two_player_contribution':path_skip,'message_path_two_player_contribution':path_message,
        'pre_event_feature_removed_correction':f00,'pre_event_via_skip_two_player_contribution':feature_skip,
        'pre_event_via_message_two_player_contribution':feature_message,
        'skip_hidden_L2':baseline_skip_norm,'message_hidden_L2':baseline_msg_norm,'pre_event_skip_hidden_L2':pre_skip_norm})
    weight=pd.DataFrame({'feature':p['feature_names'],'skip_weight_L2':torch.linalg.vector_norm(model.skip.weight,dim=0).cpu().numpy(),
        'Newton_mean_standardized_value':scaled[:143,node].mean(axis=0,dtype=np.float64)})
    weight['Newton_skip_contribution_L2']=np.abs(weight.Newton_mean_standardized_value)*weight.skip_weight_L2
    save_csv('tables/Newton_interventions.csv',pd.DataFrame(summaries));save_csv('tables/Newton_intervention_timeline.csv',timeline)
    save_csv('tables/Newton_path_decomposition.csv',decomposition);save_csv('tables/skip_feature_weights.csv',weight.sort_values('Newton_skip_contribution_L2',ascending=False))
    findings={'status':'FIXED_WEIGHT_DIAGNOSTICS_COMPLETE','source_checkpoint':str(path),'scope':list(S),'horizon':h,
        'saved_residual_scale':scale,'saved_best_epoch':p['fit_info']['best_epoch'],'saved_epochs':p['fit_info']['epochs'],
        'Newton_pre_event_standardized_value':float(scaled[0,node,idx]),
        'original_mean_correction':float(full64[valid].mean()),'head_zero_mean_correction':float(z[valid].mean()),
        'skip_path_mean_contribution':float(path_skip[valid].mean()),'message_path_mean_contribution':float(path_message[valid].mean()),
        'pre_event_feature_total_mean_effect':float((full64-f00)[valid].mean()),
        'pre_event_via_skip_mean_contribution':float(feature_skip[valid].mean()),
        'pre_event_via_message_mean_contribution':float(feature_message[valid].mean()),
        'training_performed':False,'weights_modified':False,'original_predictions_modified':False,
        'interpretation':'Exact interventions in this frozen computational graph; not physical causality, not retrained ablations, not new unbiased CV scores.',
        'wall_seconds':time.monotonic()-start}
    save_json('Newton_findings.json',findings);print(json.dumps(findings,indent=2))


if __name__=='__main__':main()
