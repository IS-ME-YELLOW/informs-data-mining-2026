"""Full OOF artifacts and predeclared P/OSI comparisons; no model fitting."""
import numpy as np
import pandas as pd
from information_protocol import (CONFIG,HERE,H1,PT,HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,
    KINDS,KEYS,expected_mask,support,transform,reconstruct,build_candidates,build_controls,digest_object)
from metric_helpers import metrics,comparison,bootstrap

PAIRS=(('F2_minus_F0','F0','F2'),)
CASES=('F0','F2')
WINDOWS=((73,76),(77,95),(96,143),(144,215))


def p_bin(p):
    return np.select([p==0,p<.01,p<.1,p<.5],['p=0','0<p<0.01','0.01<=p<0.1','0.1<=p<0.5'],default='0.5<=p<=1')


def evaluation_arrays(ctx,parts,controls,aux):
    arrays={}
    for h in HORIZONS:
        y=ctx.data.y_train[h].to_numpy()
        for mode in MODES:
            arrays['OSI',mode,h]=(y,{case:controls[case][mode][h] for case in CASES})
        p=ctx.data.component_targets[f'P_t_target_{h.rsplit("_",1)[-1]}'].to_numpy()
        arrays['P','raw',h]=(p,{case:parts[case][h]['P_t'] for case in CASES})
        arrays['P','aligned',h]=(p,{case:aux[case]['aligned_components'][h]['P_t'] for case in CASES})
    return arrays


def make_metrics(ctx,parts,controls,aux):
    arrays=evaluation_arrays(ctx,parts,controls,aux)
    rows={n:[] for n in ('pooled','slices','bootstrap','error_interactions')}
    codes=ctx.meta.fipsCode.to_numpy();folds=ctx.meta.fold.to_numpy();origins=ctx.meta.hour_idx.to_numpy()
    bins=p_bin(ctx.p)
    for (domain,mode,h),(y,preds) in arrays.items():
        valid=expected_mask(ctx.meta,h);target=origins+HORIZON_HOURS[h]
        for pair,before,after in PAIRS:
            ids={'domain':domain,'mode':mode,'horizon':h,'comparison':pair}
            b,a=preds[before],preds[after]
            rows['pooled'].append({**ids,**comparison(y[valid],b[valid],a[valid])})
            detailed=(domain=='OSI' and mode=='v18_rule') or (domain=='P' and h==H1)
            if detailed:
                slices=[('fold',str(f),folds==f) for f in range(5)]
                slices += [('county',f,codes==f) for f in sorted(set(codes))]
                slices += [('window',f'{lo}-{hi}',(target>=lo)&(target<=hi)) for lo,hi in WINDOWS]
                slices += [('p_bin',group,bins==group) for group in sorted(set(bins))]
                slices += [('population','exclude_Morrow',codes!='39117')]
                if h in (H1,'osi_target_t06h'):
                    slices += [('target_hour',str(s),target==s) for s in sorted(set(target[valid]))]
                for typ,value,mask in slices:
                    take=valid&mask
                    if not take.any():continue
                    rows['slices'].append({**ids,'slice_type':typ,'slice_value':value,
                        'counties':int(len(set(codes[take]))),
                        'countyName':ctx.names.loc[value] if typ=='county' else '',
                        **comparison(y[take],b[take],a[take])})
            boot=(domain=='OSI' and mode=='v18_rule' and HORIZON_HOURS[h]<=6) or (domain=='P' and h==H1)
            if boot and pair!='F2_minus_F1':
                for pop,take in (('all_counties',valid),('exclude_Morrow',valid&(codes!='39117'))):
                    rows['bootstrap'].append({**ids,'population':pop,**comparison(y[take],b[take],a[take]),
                        **bootstrap(y[take],b[take],a[take],codes[take],20260910,2000)})
    # Error cross terms before lower/upper/zero OSI processing, plus actual final error.
    for h in (H1,'osi_target_t06h'):
        valid=expected_mask(ctx.meta,h);yosi=ctx.data.y_train[h].to_numpy()[valid]
        truth={c:ctx.data.component_targets[f'{c}_target_{h.rsplit("_",1)[-1]}'].to_numpy()[valid] for c in COMPONENTS}
        rounding=.4*truth['P_t']+.35*truth['N_t']+.25*truth['D_t']-.1*truth['R_t']-yosi
        for case in CASES:
            for space,part,mode in (('raw',parts[case][h],'C1_component_osi'),('aligned',aux[case]['aligned_components'][h],'C3_aligned_component')):
                err={c:part[c][valid]-truth[c] for c in COMPONENTS}
                ep=.4*err['P_t'];eo=.35*err['N_t']+.25*err['D_t']-.1*err['R_t']
                linear=ep+eo+rounding;final=controls[case][mode][h][valid]-yosi
                cov=float(np.mean((ep-ep.mean())*(eo-eo.mean())))
                rows['error_interactions'].append({'case':case,'space':space,'horizon':h,'n':int(valid.sum()),
                    'P_weighted_rmse':float(np.sqrt(np.mean(ep**2))),
                    'other_weighted_rmse':float(np.sqrt(np.mean(eo**2))),
                    'two_mean_P_other_product':float(2*np.mean(ep*eo)), 'P_other_covariance':cov,
                    'official_truth_rounding_rmse':float(np.sqrt(np.mean(rounding**2))),
                    'linear_osi_rmse':float(np.sqrt(np.mean(linear**2))),
                    'postprocessed_osi_rmse':float(np.sqrt(np.mean(final**2))),
                    'postprocessing_mse_change':float(np.mean(final**2)-np.mean(linear**2))})
    return {f'metrics/{name}.csv':pd.DataFrame(table) for name,table in rows.items()}


def artifact_frames(ctx,new_raw,new_ids,identity):
    valid=ctx.valid;p=ctx.p;y=ctx.data.component_targets[PT].to_numpy()
    raw={'F0':{k:ctx.f0_branch[f'pred_{k}_raw'].to_numpy(copy=True) for k in KINDS},**new_raw}
    ids={'F0':{k:ctx.f0_branch[f'source_{k}_model_id'].to_numpy(copy=True) for k in KINDS},**new_ids}
    reconstruction={case:reconstruct(p,raw[case]['a'],raw[case]['b'],valid) for case in CASES}
    np.testing.assert_array_equal(reconstruction['F0'][0],ctx.parts[H1]['P_t'])
    parts,controls,aux=build_candidates(ctx,{c:reconstruction[c][0] for c in ('F2',)})
    meta=ctx.meta.copy();meta['candidate_identity_hash']=identity
    meta['reference_E2_identity_hash']=CONFIG['reference_identity_hash']
    audit=meta.copy();audit['target_timestamp']=meta.timestamp_et+pd.Timedelta(hours=1)
    audit['scoreable']=valid;audit['p71']=p;audit['official_P']=y
    for k in KINDS:
        mask=valid&support(p,k);label=np.full(len(p),np.nan);weight=np.full(len(p),np.nan)
        label[mask],weight[mask]=transform(y[mask],p[mask],k)
        audit[k+'_supported']=mask;audit[k+'_label']=label;audit[k+'_raw_weight']=weight
    branches=[];recon_ids={};diagnostics=[];decomp=[]
    for case in CASES:
        rebuilt,aa,bb,ca,cb=reconstruction[case]
        if case=='F0':rid=ctx.f0_branch.reconstruction_id.to_numpy(copy=True)
        else:
            rid=np.full(len(p),'',object)
            for fold in range(5):
                take=valid&(ctx.meta.fold.to_numpy()==fold)
                spec={'run':identity,'candidate':case,'outer_fold':fold,'feature_package':ctx.feature_bundle['manifest']['identity_hash'],
                    'a_model':ids[case]['a'][take][0],'b_model':ids[case]['b'][take][0],
                    'p_source':ctx.data.loaded_hashes['raw_train'],'formula':CONFIG['reconstruction']}
                rid[take]='reconstruction:'+digest_object(spec)
        recon_ids[case]=rid
        b=meta.copy();b['case']=case;b['target_timestamp']=audit.target_timestamp;b['scoreable']=valid;b['p71']=p
        b['feature_package_identity']=ctx.feature_bundle['manifest']['identity_hash'] if case!='F0' else 'frozen_163_E2'
        b['p71_source_sha256']=ctx.data.loaded_hashes['raw_train'];b['reconstruction_id']=rid
        for k in KINDS:
            b[k+'_supported']=audit[k+'_supported'];b['pred_'+k+'_raw']=raw[case][k];b['source_'+k+'_model_id']=ids[case][k]
            b[k+'_clipped_low']=np.isfinite(raw[case][k])&(raw[case][k]<0)
            b[k+'_clipped_high']=np.isfinite(raw[case][k])&(raw[case][k]>1)
        b['a_applied']=aa;b['b_applied']=bb;b['a_P_contribution']=ca;b['b_P_contribution']=cb;b['pred_P']=rebuilt
        b['reconstruction_numeric_clipped']=valid&((ca+cb<0)|(ca+cb>1));branches.append(b)
        for population,take in [('all',valid),*[(f'fold{f}',valid&(ctx.meta.fold.to_numpy()==f)) for f in range(5)]]:
            for k in KINDS:
                mask=take&support(p,k);pred=raw[case][k][mask];target=audit[k+'_label'].to_numpy()[mask];w=audit[k+'_raw_weight'].to_numpy()[mask]
                diagnostics.append({'case':case,'population':population,'kind':k,'support_rows':int(mask.sum()),
                    'all_rows':int(take.sum()),'support_counties':ctx.meta.loc[mask,'fipsCode'].nunique(),
                    'raw_label_rmse':float(np.sqrt(np.mean((pred-target)**2))),
                    'clipped_label_rmse':float(np.sqrt(np.mean((np.clip(pred,0,1)-target)**2))),
                    'raw_P_contribution_rmse':float(np.sqrt(np.sum(w*(pred-target)**2)/take.sum())),
                    'clipped_P_contribution_rmse':float(np.sqrt(np.sum(w*(np.clip(pred,0,1)-target)**2)/take.sum())),
                    'clipped_low':int((pred<0).sum()),'clipped_high':int((pred>1).sum())})
            ea=ca[take]-np.minimum(y[take],p[take]);eb=cb[take]-np.maximum(y[take]-p[take],0)
            decomp.append({'case':case,'population':population,'n':int(take.sum()),'a_mse':float(np.mean(ea**2)),
                'b_mse':float(np.mean(eb**2)),'twice_cross_mean':float(2*np.mean(ea*eb)),'P_mse':float(np.mean((rebuilt[take]-y[take])**2))})
    component={k:meta[k] for k in meta};oof={k:meta[k] for k in meta};aligned=[]
    for case in CASES:
        table=aux[case]['unique_components'].copy();table.insert(0,'case',case);table['candidate_identity_hash']=identity;aligned.append(table)
    for h in HORIZONS:
        truth=ctx.data.y_train[h].to_numpy();oof[f'actual_{h}']=truth;oof[f'scoreable_{h}']=expected_mask(ctx.meta,h)
        oof[f'raw_direct_{h}']=ctx.direct[h];oof[f'source_direct_{h}']=ctx.f0_saved[f'source_direct_{h}'].to_numpy()
        for case in CASES:
            for mode in MODES:
                oof[f'pred_{case}_{mode}_{h}']=controls[case][mode][h]
                oof[f'error_{case}_{mode}_{h}']=controls[case][mode][h]-truth
            for c in COMPONENTS:
                target=f'{c}_target_{h.rsplit("_",1)[-1]}'
                component[f'pred_{case}_{target}']=parts[case][h][c]
                source=ctx.f0_component[f'source_E2_{target}'].to_numpy()
                if (h,c)==(H1,'P_t'):source=recon_ids[case]
                component[f'source_{case}_{target}']=source
        for pair,b,a in PAIRS:oof[f'sse_reduction_{pair}_{h}']=(controls[b]['v18_rule'][h]-truth)**2-(controls[a]['v18_rule'][h]-truth)**2
    frames={'target_audit.parquet':audit,'branch_oof_predictions.parquet':pd.concat(branches,ignore_index=True),
        'component_predictions.parquet':pd.DataFrame(component),'control_predictions.parquet':pd.DataFrame(oof),
        'aligned_unique_components.parquet':pd.concat(aligned,ignore_index=True),
        'metrics/branch_diagnostics.csv':pd.DataFrame(diagnostics),'metrics/branch_loss_decomposition.csv':pd.DataFrame(decomp)}
    frames.update(make_metrics(ctx,parts,controls,aux))
    return frames
