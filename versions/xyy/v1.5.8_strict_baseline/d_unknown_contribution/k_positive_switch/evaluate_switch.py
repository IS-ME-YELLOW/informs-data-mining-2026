"""Replay one frozen CV split with the fixed K>0 model switch; never trains."""
from datetime import datetime,timezone
import argparse
import json
import numpy as np
import pandas as pd
from switch_protocol import (HERE,CONFIG,HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,H1,KEYS,
    load_sources,identity_for,check_prediction_boundaries,output_path,digest_object,
    write_json,write_frame,read_json,sha256_file,expected_mask)
from gated_evaluation import evaluate_frames,PAIRS
from metric_helpers import comparison,bootstrap
from run_artifacts import run_lock,freeze_stage


def frames_for(ctx,identity_hash):
    ids=ctx.meta.copy();ids['gated_identity_hash']=identity_hash
    ids['source_u_identity_hash']=ctx.item['u_identity_hash'];ids['baseline_identity_hash']=ctx.item['baseline_identity_hash']
    valid=expected_mask(ctx.meta,H1)
    gate=ids.copy();gate['target_timestamp']=gate.timestamp_et+pd.Timedelta(hours=1)
    gate['scoreable']=valid;gate['K']=ctx.bounds[H1];gate['use_U']=ctx.gate;gate['pred_U_raw']=ctx.raw_u
    for case in ('A','B','C','G'):gate[f'{case}_D1']=ctx.parts[case][H1]['D_t']
    original_id=ctx.source_part.source_A_D_t_target_t01h.to_numpy()
    u_id=ctx.source_u.source_U_model_id.to_numpy()
    selected=np.where(ctx.gate,u_id,original_id)
    gate['original_D_model_id']=original_id;gate['U_model_id']=u_id;gate['selected_model_id']=selected
    gate['selected_rule']=np.where(valid,np.where(ctx.gate,'U_reconstruction','original_D'),'unscoreable')
    for before in ('A','B','C'):gate[f'D_changed_vs_{before}']=valid&(gate.G_D1.to_numpy()!=gate[f'{before}_D1'].to_numpy())
    components={k:ids[k] for k in ids};controls={k:ids[k] for k in ids}
    for h in HORIZONS:
        controls[f'actual_{h}']=ctx.data.y_train[h].to_numpy()
        controls[f'scoreable_{h}']=expected_mask(ctx.meta,h)
        controls[f'raw_direct_{h}']=ctx.direct[h]
        controls[f'source_direct_{h}']=ctx.source_control[f'source_direct_{h}'].to_numpy()
        for case in ('A','B','C','G'):
            for mode in MODES:controls[f'pred_{case}_{mode}_{h}']=ctx.controls[case][mode][h]
        for pair,before,after in PAIRS:
            for mode in MODES:controls[f'changed_{pair}_{mode}_{h}']=expected_mask(ctx.meta,h)&(ctx.controls[after][mode][h]!=ctx.controls[before][mode][h])
        for c in COMPONENTS:
            target=f'{c}_target_{h.rsplit("_",1)[-1]}'
            components[f'pred_G_{target}']=ctx.parts['G'][h][c]
            components[f'source_G_{target}']=selected if (h,c)==(H1,'D_t') else ctx.source_part[f'source_A_{target}'].to_numpy()
    frames={'gate_decisions.parquet':gate,'component_predictions.parquet':pd.DataFrame(components),
            'control_predictions.parquet':pd.DataFrame(controls),'aligned_unique_components.parquet':ctx.aux['G']['unique_components']}
    frames.update(evaluate_frames(ctx,ctx.parts,ctx.controls))
    d_rows=[];d_boot=[]
    for space,h in (('raw_D1',H1),('aligned_D1',H1),('aligned_D6','osi_target_t06h')):
        y=ctx.data.component_targets[f'D_t_target_{h.rsplit("_",1)[-1]}'].to_numpy();mask=np.isfinite(y)
        for pair,before,after in PAIRS:
            parts=ctx.parts if space=='raw_D1' else {case:aux['aligned_components'] for case,aux in ctx.aux.items()}
            b,a=parts[before][h]['D_t'],parts[after][h]['D_t']
            d_rows.append({'space':space,'horizon':h,'comparison':pair,**comparison(y[mask],b[mask],a[mask])})
            for population,take in (('all_counties',mask),('exclude_Morrow_diagnostic_only',mask&(ctx.meta.fipsCode.to_numpy()!='39117'))):
                d_boot.append({'space':space,'horizon':h,'comparison':pair,'population':population,
                    **comparison(y[take],b[take],a[take]),**bootstrap(y[take],b[take],a[take],ctx.meta.loc[take,'fipsCode'],CONFIG['bootstrap']['seed'],CONFIG['bootstrap']['replicates'])})
    frames['metrics/D_spaces.csv']=pd.DataFrame(d_rows);frames['metrics/D_bootstrap.csv']=pd.DataFrame(d_boot)
    windows=[]
    y=ctx.data.y_train[H1].to_numpy();early=ctx.meta.hour_idx.to_numpy()<=75
    for pair,before,after in PAIRS:
        b,a=ctx.controls[before]['v18_rule'][H1],ctx.controls[after]['v18_rule'][H1]
        for label,take in (('K_positive',ctx.gate),('early_K_zero',valid&early&~ctx.gate),('later_K_zero',valid&~early)):
            windows.append({'comparison':pair,'window':label,**comparison(y[take],b[take],a[take])})
    frames['metrics/gate_windows.csv']=pd.DataFrame(windows)
    changed=valid&(ctx.controls['G']['v18_rule'][H1]!=ctx.controls['B']['v18_rule'][H1])
    detail=gate.loc[ctx.gate].copy()
    detail['actual_D']=ctx.data.component_targets.D_t_target_t01h.to_numpy()[ctx.gate]
    detail['actual_osi']=y[ctx.gate]
    for case in ('A','B','C','G'):detail[f'{case}_osi']=ctx.controls[case]['v18_rule'][H1][ctx.gate]
    detail['sse_reduction_G_vs_B']=(detail.B_osi-detail.actual_osi)**2-(detail.G_osi-detail.actual_osi)**2
    frames['eligible_rows.csv']=detail
    stats={'eligible_rows':int(ctx.gate.sum()),'eligible_counties':int(ctx.meta.loc[ctx.gate,'fipsCode'].nunique()),
        'D_changed_vs_B':int(gate.D_changed_vs_B.sum()),'OSI_changed_vs_B':int(changed.sum()),
        'OSI_changed_counties':int(ctx.meta.loc[changed,'fipsCode'].nunique()),'training_performed':False}
    return frames,stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-seed',required=True,type=int,choices=(42,20260917,20260918))
    args=parser.parse_args();ctx=load_sources(args.split_seed)
    identity=identity_for(ctx);hash_=digest_object(identity)
    out=output_path(HERE/'runs'/f'split{args.split_seed}')
    if (out/'G_COMPLETE').exists():raise FileExistsError('Frozen candidate; use verifier')
    if (out/'manifest.json').exists() and read_json(out/'manifest.json')['identity_hash']!=hash_:raise ValueError('Candidate identity differs; do not overwrite')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        preflight=check_prediction_boundaries(ctx)
        write_json(out/'preflight.json',preflight)
        write_json(out/'manifest.json',{'identity':identity,'identity_hash':hash_,'created_at_utc':datetime.now(timezone.utc).isoformat()})
        frames,stats=frames_for(ctx,hash_)
        for name,frame in frames.items():write_frame(output_path(out/name),frame)
        write_json(out/'routing_stats.json',stats)
        from verify_switch import verify_run
        verified=verify_run(out,require_marker=False)
        write_json(out/'verification.json',verified)
        names=[*frames,'preflight.json','manifest.json','routing_stats.json','verification.json']
        freeze_stage(out,'G_COMPLETE',names,hash_)
        print(frames['metrics/pooled.csv'].query("comparison=='G_minus_B' and model=='v18_rule'").to_string(index=False),flush=True)
        print(json.dumps(verified,indent=2),flush=True)


if __name__=='__main__':main()
