"""Full OOF artifacts and predeclared P/OSI comparisons; no model fitting."""
import numpy as np
import pandas as pd
from p_structure_protocol import (CONFIG,HERE,H1,PT,HORIZONS,HORIZON_HOURS,COMPONENTS,MODES,
    KINDS,KEYS,expected_mask,support,transform,reconstruct,build_candidates,build_controls,digest_object)
from metric_helpers import metrics,comparison,bootstrap

PAIRS=(('E2_minus_E1','E1','E2'),('E2_minus_E0','E0','E2'),('E1_minus_E0','E0','E1'))
CASES=('E0','E1','E2')
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
    rows={n:[] for n in ('pooled','slices','bootstrap','d_interaction','error_interactions')}
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
            if boot:
                for pop,take in (('all_counties',valid),('exclude_Morrow',valid&(codes!='39117'))):
                    rows['bootstrap'].append({**ids,'population':pop,**comparison(y[take],b[take],a[take]),
                        **bootstrap(y[take],b[take],a[take],codes[take],20260910,2000)})
    # Fixed D_B sensitivity; the primary reference remains D_G.
    b_controls={}
    for case in CASES:
        part={h:{c:v.copy() for c,v in q.items()} for h,q in parts[case].items()}
        for h in HORIZONS:part[h]['D_t']=ctx.b_d[h]
        b_controls[case],_=build_controls(ctx.meta,ctx.direct,part)
    for h in HORIZONS:
        y=ctx.data.y_train[h].to_numpy();valid=expected_mask(ctx.meta,h)
        for mode in MODES:
            for pair,before,after in PAIRS:
                b,a=b_controls[before][mode][h],b_controls[after][mode][h]
                rows['d_interaction'].append({'D_reference':'B','domain':'OSI','mode':mode,'horizon':h,
                    'comparison':pair,**comparison(y[valid],b[valid],a[valid])})
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
    return {f'metrics/{name}.csv':pd.DataFrame(table) for name,table in rows.items()},b_controls


def artifact_frames(ctx,raw,model_ids,identity):
    valid=ctx.valid;p=ctx.p;y=ctx.data.component_targets[PT].to_numpy()
    structured,aa,bb,ca,cb=reconstruct(p,raw['a'],raw['b'],valid)
    direct=np.clip(raw['direct'],0,1)
    parts,controls,aux=build_candidates(ctx,direct,structured)
    meta=ctx.meta.copy();meta['candidate_identity_hash']=identity
    meta['baseline_identity_hash']=CONFIG['baseline_identity_hash'];meta['D_G_identity_hash']=CONFIG['g_identity_hash']
    audit=meta.copy();audit['target_timestamp']=meta.timestamp_et+pd.Timedelta(hours=1)
    audit['scoreable']=valid;audit['p71']=p;audit['official_P']=y
    branch=meta.copy();branch['target_timestamp']=audit.target_timestamp;branch['scoreable']=valid;branch['p71']=p
    branch['p71_source_sha256']=ctx.data.loaded_hashes['raw_train']
    for kind in ('a','b'):
        mask=valid&support(p,kind);label=np.full(len(p),np.nan);weight=np.full(len(p),np.nan)
        label[mask],weight[mask]=transform(y[mask],p[mask],kind)
        audit[f'{kind}_supported']=mask;audit[f'{kind}_label']=label;audit[f'{kind}_raw_weight']=weight
        branch[f'{kind}_supported']=mask
    for kind in KINDS:
        branch[f'pred_{kind}_raw']=raw[kind];branch[f'source_{kind}_model_id']=model_ids[kind]
        branch[f'{kind}_clipped_low']=np.isfinite(raw[kind])&(raw[kind]<0)
        branch[f'{kind}_clipped_high']=np.isfinite(raw[kind])&(raw[kind]>1)
    branch['a_applied']=aa;branch['b_applied']=bb
    branch['a_P_contribution']=ca;branch['b_P_contribution']=cb
    branch['pred_E1_P']=direct;branch['pred_E2_P']=structured
    branch['reconstruction_numeric_clipped']=valid&((ca+cb<0)|(ca+cb>1))
    reconstruction_ids=np.full(len(p),'',object)
    for fold in range(5):
        mask=valid&(ctx.meta.fold.to_numpy()==fold)
        spec={'run':identity,'outer_fold':fold,'formula':CONFIG['reconstruction'],
            'a_model':model_ids['a'][mask][0],'b_model':model_ids['b'][mask][0],
            'p71_source':ctx.data.loaded_hashes['raw_train']}
        reconstruction_ids[mask]='reconstruction:'+digest_object(spec)
    branch['reconstruction_id']=reconstruction_ids
    component={k:meta[k] for k in meta};oof={k:meta[k] for k in meta};aligned=[]
    for case in CASES:
        table=aux[case]['unique_components'].copy();table.insert(0,'case',case);table['candidate_identity_hash']=identity
        aligned.append(table)
    for h in HORIZONS:
        oof[f'actual_{h}']=ctx.data.y_train[h].to_numpy();oof[f'scoreable_{h}']=expected_mask(ctx.meta,h)
        oof[f'raw_direct_{h}']=ctx.direct[h];oof[f'source_direct_{h}']=ctx.saved[f'source_direct_{h}']
        for case in CASES:
            for mode in MODES:
                pred=controls[case][mode][h]
                oof[f'pred_{case}_{mode}_{h}']=pred
                oof[f'error_{case}_{mode}_{h}']=pred-ctx.data.y_train[h].to_numpy()
            for c in COMPONENTS:
                target=f'{c}_target_{h.rsplit("_",1)[-1]}'
                component[f'pred_{case}_{target}']=parts[case][h][c]
                source=ctx.component[f'source_G_{target}'].to_numpy()
                if (h,c)==(H1,'P_t'):
                    if case=='E1':source=model_ids['direct']
                    if case=='E2':source=reconstruction_ids
                component[f'source_{case}_{target}']=source
        for pair,b,a in PAIRS:
            oof[f'sse_reduction_{pair}_{h}']=(controls[b]['v18_rule'][h]-ctx.data.y_train[h].to_numpy())**2-(controls[a]['v18_rule'][h]-ctx.data.y_train[h].to_numpy())**2
    frames={'target_audit.parquet':audit,'p_branch_oof_predictions.parquet':branch,
            'component_predictions.parquet':pd.DataFrame(component),'control_predictions.parquet':pd.DataFrame(oof),
            'aligned_unique_components.parquet':pd.concat(aligned,ignore_index=True)}
    metric_frames,b_controls=make_metrics(ctx,parts,controls,aux);frames.update(metric_frames)
    secondary=meta.copy()
    for case in CASES:
        for mode in MODES:
            for h in HORIZONS:secondary[f'pred_{case}_{mode}_{h}']=b_controls[case][mode][h]
    frames['D_B_control_predictions.parquet']=secondary
    diagnostics=[];decomposition=[]
    for name,take in [('all',valid),*[(f'fold{f}',valid&(ctx.meta.fold.to_numpy()==f)) for f in range(5)]]:
        for kind in KINDS:
            mask=take&support(p,kind)
            target=y[mask] if kind=='direct' else audit[f'{kind}_label'].to_numpy()[mask]
            w=np.ones(mask.sum()) if kind=='direct' else audit[f'{kind}_raw_weight'].to_numpy()[mask]
            pred=raw[kind][mask]
            diagnostics.append({'population':name,'kind':kind,'support_rows':int(mask.sum()),'all_rows':int(take.sum()),
                'support_counties':ctx.meta.loc[mask,'fipsCode'].nunique(),
                'raw_label_rmse':float(np.sqrt(np.mean((pred-target)**2))),
                'clipped_label_rmse':float(np.sqrt(np.mean((np.clip(pred,0,1)-target)**2))),
                'raw_P_contribution_rmse':float(np.sqrt(np.sum(w*(pred-target)**2)/take.sum())),
                'clipped_P_contribution_rmse':float(np.sqrt(np.sum(w*(np.clip(pred,0,1)-target)**2)/take.sum())),
                'clipped_low':int((pred<0).sum()),'clipped_high':int((pred>1).sum())})
        true_a=np.minimum(y[take],p[take]);true_b=np.maximum(y[take]-p[take],0)
        ea=ca[take]-true_a;eb=cb[take]-true_b
        decomposition.append({'population':name,'n':int(take.sum()),'a_mse':float(np.mean(ea**2)),
            'b_mse':float(np.mean(eb**2)),'twice_cross_mean':float(2*np.mean(ea*eb)),
            'P_mse':float(np.mean((structured[take]-y[take])**2))})
    frames['metrics/branch_diagnostics.csv']=pd.DataFrame(diagnostics)
    frames['metrics/branch_loss_decomposition.csv']=pd.DataFrame(decomposition)
    return frames
