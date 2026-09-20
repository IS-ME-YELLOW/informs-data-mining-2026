"""Summarize verified single-source results and explain component/OSI differences."""
from transfer_protocol import *


def read_csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])


def component_effects(ctx,primary):
    ptable=pd.read_parquet(RUN/'P_component_diagnostics.parquet')
    pred=normalize(pd.read_parquet(RUN/'control_predictions.parquet'))
    tables=[];rows=[]
    weights=np.array([.4,.35,.25,-.1])
    for case,changed in CASES.items():
        if changed is None:continue
        for h in HORIZONS:
            hh=HORIZON_HOURS[h];valid=expected_mask(ctx.meta,h)
            q=normalize(ptable[(ptable['case']==case)&(ptable.horizon==h)])
            truth=np.column_stack([ctx.data.component_targets[component_target_name(c,h)].to_numpy() for c in COMPONENTS])
            baseline_parts=ctx.parts[h] if hh>=24 else ctx.aux['aligned_components'][h]
            pp0=np.column_stack([baseline_parts[c] for c in COMPONENTS])
            pp1=pp0.copy();pp1[:,0]=q['component_clipped' if hh>=24 else 'aligned'].to_numpy()
            y=ctx.data.y_train[h].to_numpy();y0=pred[f'pred_B0_v18_rule_{h}'].to_numpy();y1=pred[f'pred_{case}_v18_rule_{h}'].to_numpy()
            e0=.4*(pp0[:,0]-truth[:,0]);e1=.4*(pp1[:,0]-truth[:,0]);other=(pp0[:,1:]-truth[:,1:])@weights[1:]
            b0=y0-pp0@weights+truth@weights-y;b1=y1-pp1@weights+truth@weights-y
            np.testing.assert_allclose(e0+other+b0,y0-y,atol=1e-12,rtol=0)
            np.testing.assert_allclose(e1+other+b1,y1-y,atol=1e-12,rtol=0)
            self_gain=e0**2-e1**2;cross_gain=2*(e0-e1)*other
            processing_gain=b0**2-b1**2+2*((e0+other)*b0-(e1+other)*b1)
            actual=(y0-y)**2-(y1-y)**2
            np.testing.assert_allclose(self_gain+cross_gain+processing_gain,actual,atol=1e-12,rtol=0)
            d=ctx.meta.loc[valid].copy();d['case']=case;d['horizon']=h;d['target_hour']=d.hour_idx+hh
            for name,v in [('e_P_before',e0),('e_P_after',e1),('other_component_error',other),('b_before',b0),('b_after',b1),
                ('P_self_sse_gain',self_gain),('P_other_cross_sse_gain',cross_gain),('postprocessing_and_precision_sse_gain',processing_gain),('actual_OSI_sse_gain',actual)]:d[name]=v[valid]
            rows.append(d)
            row=dict(case=case,horizon=h,n=int(valid.sum()),P_self_sse_gain=float(np.nansum(self_gain)),
                P_other_cross_sse_gain=float(np.nansum(cross_gain)),postprocessing_and_precision_sse_gain=float(np.nansum(processing_gain)),actual_OSI_sse_gain=float(np.nansum(actual)))
            official=primary[(primary['case']==case)&(primary.horizon==h)].iloc[0]
            np.testing.assert_allclose(row['actual_OSI_sse_gain'],official.sse_reduction,atol=1e-12,rtol=0)
            tables.append(row)
    write_frame(OUT/'summary/OSI_change_decomposition.csv',tables)
    write_frame(OUT/'summary/OSI_change_decomposition.parquet',pd.concat(rows,ignore_index=True))
    write_json(OUT/'summary/decomposition_identity_check.json',dict(status='PASS',rowwise_additive_identity=True,
        total_matches_verified_OSI_SSE=True,absolute_tolerance=1e-12,method='explicit squared-error expansion with postprocessing remainder'))
    return pd.DataFrame(tables)


def main():
    checked=read_json(RUN/'logs/independent_verification.json');assert checked['status']=='PASS'
    ctx=load_context()
    names=['primary_scores','P_component_metrics','P71_bins','county_metrics','fold_metrics','window_metrics','bootstrap_intervals','branch_diagnostics','branch_loss_decomposition']
    for name in names:
        d=read_csv(RUN/f'metrics/{name}.csv',dtype={'fipsCode':str} if name=='county_metrics' else {})
        write_frame(OUT/f'summary/{name}.csv',d)
    primary=read_csv(OUT/'summary/primary_scores.csv');pm=read_csv(OUT/'summary/P_component_metrics.csv')
    boots=read_csv(OUT/'summary/bootstrap_intervals.csv');county=read_csv(OUT/'summary/county_metrics.csv',dtype={'fipsCode':str})
    folds=read_csv(OUT/'summary/fold_metrics.csv');win=read_csv(OUT/'summary/window_metrics.csv');bins=read_csv(OUT/'summary/P71_bins.csv')
    focus=read_csv(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv',dtype={'fipsCode':str})
    write_frame(OUT/'summary/frozen_focus_counties.csv',county[county.fipsCode.isin(focus.fipsCode.unique())])
    effect=component_effects(ctx,primary)
    rounds=[]
    modelm=read_json(RUN/'model_manifest.json')
    for m in modelm['models']:
        rounds.append(dict(horizon=m['spec']['horizon'],outer_fold=m['spec']['outer_fold'],kind=m['spec']['kind'],
            best_iterations=','.join(map(str,m['fit']['best_iterations'])),requested_rounds=m['fit']['requested_rounds'],
            trees_saved=m['fit']['trees_saved'],fit_seconds=m['fit']['seconds']))
    write_frame(OUT/'summary/round_selection.csv',rounds)
    priorities=[];matrix=[];absrows=[];prows=[];intervals=[];countyrows=[];windowrows=[];group_rows=[];toprows=[]
    for case,h in CASES.items():
        if h is None:continue
        g=primary[primary['case']==case].set_index('horizon')
        matrix.append([case]+[f'{g.loc[z,"rmse_change_pct"]:+.3f}%' for z in HORIZONS])
        for z in HORIZONS:absrows.append([case,f'{HORIZON_HOURS[z]}h',f'{g.loc[z,"baseline_rmse"]:.12f}',f'{g.loc[z,"rmse"]:.12f}'])
        for space in ['component_clipped','aligned']:
            r=pm[(pm['case']==case)&(pm.horizon==h)&(pm.space==space)&(pm['group']=='full')].iloc[0]
            prows.append([case,space,f'{r.baseline_rmse:.9f}',f'{r.rmse:.9f}',f'{r.rmse_change_pct:+.3f}%'])
        for z in HORIZONS:
            r=boots[(boots['case']==case)&(boots.horizon==z)&(boots.domain=='OSI')].iloc[0]
            if z in (HORIZONS[0],HORIZONS[1],h):intervals.append([case,f'{HORIZON_HOURS[z]}h',f'{r.rmse_delta:+.9f}',f'[{r.ci_low:+.9f}, {r.ci_high:+.9f}]'])
        c=county[(county['case']==case)&(county.horizon==h)];f=folds[(folds['case']==case)&(folds.horizon==h)]
        countyrows.append([case,f'{int((c.sse_reduction>0).sum())}/{int((c.sse_reduction<0).sum())}',
            f'{int((f.sse_reduction>0).sum())}/{int((f.sse_reduction<0).sum())}',f'{c.sse_reduction.sum():+.6f}'])
        rank=c.sort_values(['sse_reduction','fipsCode'],ascending=[False,True])
        for direction,selected in [('改善',rank.head(5)),('恶化',rank.tail(5).sort_values('sse_reduction'))]:
            for r in selected.itertuples():toprows.append(dict(case=case,horizon=h,direction=direction,fipsCode=r.fipsCode,
                countyName=r.countyName,stateAbbr=r.stateAbbr,sse_reduction=r.sse_reduction))
        for r in win[(win['case']==case)&(win.horizon==h)&(~win.common_target)&(win.n>0)].itertuples():
            windowrows.append([case,f'{max(int(r.start),72+HORIZON_HOURS[h])}–{int(r.end)}',int(r.n),f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
        for r in bins[(bins['case']==case)&(bins.horizon==h)&(bins.space=='component_clipped')].itertuples():
            group_rows.append([case,r.P71_group,int(r.n),f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
        priorities.append(dict(case=case,source_horizon=HORIZON_HOURS[h],decision='confirm on two frozen CVs' if case=='AB_P06' else 'do not adopt at present',
            evidence='seed42 single-source result only',joint_evaluated=False,other_CVs_trained=False))
    write_frame(OUT/'summary/top_county_changes.csv',toprows);write_frame(OUT/'summary/next_step_recommendation.csv',priorities)
    e=effect[(effect['case']=='AB_P06')&(effect.horizon==HORIZONS[1])].iloc[0]
    p6w=win[(win['case']=='AB_P06')&(win.horizon==HORIZONS[1])&(~win.common_target)]
    early=float(p6w.loc[p6w.start==77,'sse_reduction'].iloc[0]);total=float(primary[(primary['case']=='AB_P06')&(primary.horizon==HORIZONS[1])].sse_reduction.iloc[0])
    input_count=len(read_json(OUT/'reference/initial_input_hashes.json'))
    report=['# P a/b结构向6/24/48h迁移：单来源结果（2026-09-19）','',
        '**结论：seed42支持优先确认P6迁移；P24、P48直接迁移本轮均退化，暂不采用。三个来源的效果不同，不能把P1h的结构收益直接推广到全部horizon。**','',
        '本轮新增150次拟合、30个外层模型，只使用原163列。当前P1h＋F2、D_G、N/R及其他来源保持固定。分别评估三个单来源替换，没有训练直接P-L2对照、没有新增邻县特征，也没有联合替换或其他CV训练。主参照为当前完整F2，不是最初strict基线。','',
        '## 1. 整体OSI：单来源 → 四个最终输出','',
        '下表为相对当前完整F2的pooled RMSE变化，负数改善、正数恶化。','',
        table(['替换来源','输出1h','输出6h','输出24h','输出48h'],matrix),'',
        'P6迁移改善1h/6h，24h/48h逐行不变；P24和P48既恶化各自长时输出，也通过C3小幅恶化短时输出。没有根据结果改权重、删来源或按县选择方案。','',
        table(['候选','输出h','B0 RMSE','候选RMSE'],absrows),'',
        '1h与6h的SSE增减在各候选中相同：两者的C3预测在重叠目标小时一致，新增来源覆盖前的早期目标不变；两项RMSE百分比不同来自样本范围和基准误差不同。不能把这两项当作独立重复证据。','',
        '## 2. P分量：需分清来源预测与C3后预测','',
        table(['候选（同名来源P）','空间','B0 P RMSE','候选P RMSE','变化'],prows),'',
        '**P6来源自身的P RMSE改善2.735%，但C3后P6误差反而小幅上升0.184%；最终6h OSI仍改善1.554%。** 分量误差与最终加权误差存在交叉项，不能将整体收益全部解释成“P更准确”。','',
        'P6对应最终6h的SSE变化拆解如下，正值表示减少误差：','',
        table(['项','SSE改善'],[
            ['P自身加权平方误差变化',f'{e.P_self_sse_gain:+.9f}'],
            ['P与固定N/D/R误差的两倍交叉项变化',f'{e.P_other_cross_sse_gain:+.9f}'],
            ['后处理与官方精度余项变化',f'{e.postprocessing_and_precision_sse_gain:+.9f}'],
            ['最终OSI SSE净改善',f'{e.actual_OSI_sse_gain:+.9f}']]),'',
        '这里的整体改善主要来自P与其他分量误差的关系变得更有利。这是实际OSI指标下的收益，但需要重复CV确认稳定性。P24/P48的来源P误差分别增加约7.42%/7.77%，本轮没有支持直接采用。','',
        '## 3. 区间、县与时段','',
        '县整条轨迹配对bootstrap（2,000次）的整体ΔRMSE：','',
        table(['候选','输出h','ΔRMSE','95%区间'],intervals),'',
        'P6的1h/6h整体区间均低于0，且5折中4折改善；但P6自身和C3后分量的区间均跨0。P24/P48整体退化的区间跨0，不能称为已经显著恶化；其来源P分量退化区间均高于0。所有结论仍是已参与开发的seed42证据。','',
        table(['候选（同名最终输出）','改善县/恶化县','改善折/恶化折','净SSE改善'],countyrows),'',
        table(['候选','有效目标小时窗口','行数','整体RMSE变化','SSE改善'],windowrows),'',
        f'P6约{100*early/total:.1f}%的净SSE改善来自目标小时78–95；144–215仅有很小净改善。P24在96–143与144–215都退化；P48后期144–215退化更明显。此处是时段诊断，没有据此增加时间门控。','',
        'P6改善较大的县包括Muskingum、Monroe、Morrow、Tuscarawas、Licking；Forest本轮略有恶化，因此收益并非单靠修复Forest。P24/P48的恶化较集中于Clay、Calhoun等县，P24还包括Morrow。完整239县表和原先冻结的7县表均保留，没有删除大误差县。','',
        '## 4. 初始状态分组说明了什么','',
        '以下是被替换来源P在C3前的分量指标，不是最终OSI：','',
        table(['候选','P71组','行数','P RMSE变化','P SSE改善'],group_rows),'',
        'P24的五个P71组在点估计上均退化，P71≥0.5的小组退化尤其明显；P48的损失主要出现在P71较低组和0.1–0.5组。因而不能把长时失败简单归因于两个极端初始状态县。','',
        'a/b重建本身允许P恢复到0，并没有P≥P71下界，也没有强制单调恢复。当前证据说明该分解、权重和两分支容量的组合没有在较晚目标窗口复现同样收益；不能仅凭此轮认定初始历史已经无用，或归因于某个单一公式。','',
        '## 5. 下一步建议与采用边界','',
        '1. **优先只确认AB_P06**：在seed20260917、seed20260918复用各自完整F2作为B0，仍用原163列、同a/b规则，分别评估四个输出。由于C3后P分量没有同步改善，应特别检查整体收益是否稳定。','2. **P24/P48暂保留原strict来源**。本轮不支持直接迁移，更没有证据支持把三个新来源联合加入。以后若增加信息、改长时结构，应单独提出假设，不能把本轮失败掩盖在联合组合里。','3. **当前模型尚未自动替换**。P6为待跨CV确认候选；本轮没有运行联合版本、F2迁移、其他分折或正式提交。','',
        '## 6. 执行和验收','',
        f'- 完整F2复现、P71、p端点、监督精确重建、30个允许范围及外折标签扰动检查均PASS；{input_count}个来源文件核验一致。',
        '- 120次内层早停＋30次四折refit，共150次拟合；分支权重仅按该次训练支持集均值归一化。早停沿用原E2的P贡献RMSE，分母包括该早停折全部有效行。',
        '- 独立进程重载30个真实模型，复算120个probe的标签、权重、指标与最佳轮数；独立实现P重建及pandas同目标小时组合，核对80组控制指标、18组bootstrap及全部县/折/时段表。',
        '- 正式评分行数仍为34,177 / 32,982 / 28,680 / 22,944。有效行没有缺失预测；p=0分支不支持不等于删除该县评分。P1、N/D/R和直接OSI保持原来源，单来源作用范围全部通过验收。',
        '- 本轮没有新增直接P-L2或等容量对照，结论限于a/b整体候选的净增量；不是单独识别数学分解的因果作用。三套冻结CV已用于开发，本轮区间未消除空间依赖或候选筛选偏差。',
        '- 所有新文件都在本目录；原代码、模型、分折、缓存与历史报告只读。','',
        '## 7. 阅读入口','',
        f'- [整体指标]({OUT}/summary/primary_scores.csv)',
        f'- [P分量指标]({OUT}/summary/P_component_metrics.csv)',
        f'- [bootstrap区间]({OUT}/summary/bootstrap_intervals.csv)',
        f'- [县级结果]({OUT}/summary/county_metrics.csv)',
        f'- [冻结关注县]({OUT}/summary/frozen_focus_counties.csv)',
        f'- [P与整体误差变化拆解]({OUT}/summary/OSI_change_decomposition.csv)',
        f'- [模型重载验收]({RUN}/logs/independent_verification.json)',
        f'- [执行说明]({OUT}/README.md)','']
    safe(OUT/'Results_2026-09-19.md').write_text('\n'.join(report))
    safe(OUT/'README.md').write_text(f'''# P a/b结构三来源独立迁移

**状态：seed42完成，150次拟合、30个新模型，独立核验PASS。** [结果报告]({OUT}/Results_2026-09-19.md)。

候选仅B0、AB_P06、AB_P24、AB_P48；没有联合替换、直接P-L2对照或邻县特征增量。B0是完整F2，P1h已有的F2仍保留，新训练P6/P24/P48只用原163列。

P6迁移使整体1h/6h RMSE下降约1.26%/1.55%，但C3后P6分量RMSE略升；P24/P48的同名整体RMSE分别上升约2.73%/3.95%。建议只优先确认P6，尚未替换当前模型。

首次执行命令如下；完成stage拒绝覆盖：

```bash
cd {ROOT}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/train_transfer.py --stage preflight
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/train_transfer.py --stage train
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/evaluate_transfer.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/verify_transfer.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/summarize_results.py
```

`--resume`只适用于同代码/配置/来源身份的未完成train。单独运行verify_transfer.py可重新验收，不拟合模型。

- `reference/`：源文件哈希、P71/端点/外折隔离和原E2一致性预检。
- `runs/v18_ptransfer_ab_v1_split42_model42/models/`：各外折、各horizon的a/b模型和receipt。
- 同run的`logs/probes/`：内层验证逐行分支标签、权重及最佳轮次预测；完整曲线保存在模型receipt。
- 同run的`branch_oof_predictions.parquet`：原始/截断a/b、贡献、支持、P重建与模型来源；`reconstruction_registry.json`关联两个分支与P71来源。
- 同run的`component_predictions.parquet`、`control_predictions.parquet`和`aligned_unique_components.parquet`：全体239县的完整单来源候选结果。
- `summary/`：所有指标、固定7县及误差交叉项；`Results_2026-09-19.md`为结论。
- 训练、评估、验收完成标记与顶层`completion.json`记录本轮身份；没有全量提交模型。

统计与逐行预测均使用项目venv。没有外部下载或新增依赖。全部新增写入限定本目录，旧产物只读。
''')
    print('Results, scope documentation and component/OSI decomposition saved',flush=True)


if __name__=='__main__':main()
