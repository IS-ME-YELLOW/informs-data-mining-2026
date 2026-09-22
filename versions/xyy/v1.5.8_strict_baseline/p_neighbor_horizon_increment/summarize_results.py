"""Report all three single-source information increments against the current P6 baseline."""
from neighbor_protocol import *


def csv(path,**kwargs):return pd.read_csv(path,float_precision='round_trip',**kwargs)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])


def explain_covariance(ctx,primary):
    ptable=pd.read_parquet(RUN/'P_component_diagnostics.parquet');saved=normalize(pd.read_parquet(RUN/'control_predictions.parquet'))
    w=np.array([.4,.35,.25,-.1]);rows=[];summary=[]
    for case,hsource in CASES.items():
        if hsource is None:continue
        for h in HORIZONS:
            hh=HORIZON_HOURS[h];valid=expected_mask(ctx.meta,h)
            d=normalize(ptable[(ptable['case']==case)&(ptable.horizon==h)])
            t=np.column_stack([ctx.data.component_targets[component_target_name(c,h)].to_numpy() for c in COMPONENTS])
            old=ctx.parts[h] if hh>=24 else ctx.aux['aligned_components'][h]
            old=np.column_stack([old[c] for c in COMPONENTS]);new=old.copy();new[:,0]=d['component_clipped' if hh>=24 else 'aligned']
            y=ctx.data.y_train[h].to_numpy();p0=saved[f'pred_B0_v18_rule_{h}'].to_numpy();p1=saved[f'pred_{case}_v18_rule_{h}'].to_numpy()
            e0=.4*(old[:,0]-t[:,0]);e1=.4*(new[:,0]-t[:,0]);other=(old[:,1:]-t[:,1:])@w[1:]
            b0=p0-old@w+t@w-y;b1=p1-new@w+t@w-y
            sg=e0**2-e1**2;cg=2*(e0-e1)*other;bg=b0**2-b1**2+2*((e0+other)*b0-(e1+other)*b1)
            gain=(p0-y)**2-(p1-y)**2
            np.testing.assert_allclose(sg+cg+bg,gain,atol=1e-12,rtol=0)
            d=ctx.meta.loc[valid].copy();d['case']=case;d['horizon']=h
            for key,v in [('P_self_sse_gain',sg),('P_other_cross_sse_gain',cg),('postprocessing_sse_gain',bg),('OSI_sse_gain',gain)]:d[key]=v[valid]
            rows.append(d);r=dict(case=case,horizon=h,P_self_sse_gain=float(np.nansum(sg)),P_other_cross_sse_gain=float(np.nansum(cg)),
                postprocessing_sse_gain=float(np.nansum(bg)),OSI_sse_gain=float(np.nansum(gain)))
            expect=primary[(primary['case']==case)&(primary.horizon==h)].sse_reduction.iloc[0]
            np.testing.assert_allclose(r['OSI_sse_gain'],expect,atol=1e-12,rtol=0);summary.append(r)
    write_frame(OUT/'summary/OSI_change_decomposition.csv',summary)
    write_frame(OUT/'summary/OSI_change_decomposition.parquet',pd.concat(rows,ignore_index=True))
    return pd.DataFrame(summary)


def main():
    verified=read_json(RUN/'logs/independent_verification.json');assert verified['status']=='PASS'
    ctx=load_context()
    for name in ('primary_scores','P_component_metrics','P71_bins','county_metrics','fold_metrics','window_metrics',
                 'bootstrap_intervals','branch_diagnostics','branch_loss_decomposition'):
        df=csv(RUN/f'metrics/{name}.csv',dtype={'fipsCode':str} if name=='county_metrics' else {})
        write_frame(OUT/f'summary/{name}.csv',df)
    primary=csv(OUT/'summary/primary_scores.csv');pm=csv(OUT/'summary/P_component_metrics.csv');ci=csv(OUT/'summary/bootstrap_intervals.csv')
    counties=csv(OUT/'summary/county_metrics.csv',dtype={'fipsCode':str});folds=csv(OUT/'summary/fold_metrics.csv')
    windows=csv(OUT/'summary/window_metrics.csv');bins=csv(OUT/'summary/P71_bins.csv')
    focus=csv(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv',dtype={'fipsCode':str}).fipsCode.unique()
    write_frame(OUT/'summary/frozen_focus_counties.csv',counties[(counties['case']!='B0')&counties.fipsCode.isin(focus)])
    decomposition=explain_covariance(ctx,primary)
    matrix=[];absrows=[];prows=[];intervals=[];crows=[];wrows=[];brows=[];top=[];recommendations=[]
    for case,h in CASES.items():
        if h is None:continue
        g=primary[primary['case']==case].set_index('horizon')
        matrix.append([case]+[f'{g.loc[k,"rmse_change_pct"]:+.3f}%' for k in HORIZONS])
        for k in HORIZONS:
            absrows.append([case,f'{HORIZON_HOURS[k]}h',f'{g.loc[k,"baseline_rmse"]:.12f}',f'{g.loc[k,"rmse"]:.12f}'])
            if k in (HORIZONS[0],HORIZONS[1],h):
                r=ci[(ci['case']==case)&(ci.horizon==k)&(ci.domain=='OSI')].iloc[0]
                intervals.append([case,f'{HORIZON_HOURS[k]}h',f'{r.rmse_delta:+.9f}',f'[{r.ci_low:+.9f}, {r.ci_high:+.9f}]'])
        for space in ('component_clipped','aligned'):
            r=pm[(pm['case']==case)&(pm.horizon==h)&(pm.space==space)&(pm['group']=='full')].iloc[0]
            prows.append([case,space,f'{r.baseline_rmse:.9f}',f'{r.rmse:.9f}',f'{r.rmse_change_pct:+.3f}%'])
        c=counties[(counties['case']==case)&(counties.horizon==h)];f=folds[(folds['case']==case)&(folds.horizon==h)]
        crows.append([case,f'{int((c.sse_reduction>0).sum())}/{int((c.sse_reduction<0).sum())}',
            f'{int((f.sse_reduction>0).sum())}/{int((f.sse_reduction<0).sum())}',f'{c.sse_reduction.sum():+.6f}'])
        ranked=c.sort_values('sse_reduction')
        for label,z in [('worsened',ranked.head(5)),('improved',ranked.tail(5))]:
            for r in z.itertuples():top.append(dict(case=case,horizon=h,direction=label,fipsCode=r.fipsCode,
                countyName=r.countyName,stateAbbr=r.stateAbbr,sse_reduction=r.sse_reduction))
        for r in windows[(windows['case']==case)&(windows.horizon==h)&(~windows.common_target)&(windows.n>0)].itertuples():
            wrows.append([case,f'{max(72+HORIZON_HOURS[h],int(r.start))}–{int(r.end)}',f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
        for r in bins[(bins['case']==case)&(bins.horizon==h)&(bins.space=='component_clipped')].itertuples():
            brows.append([case,r.P71_group,int(r.n),f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
        target_delta=float(g.loc[h,'rmse_change_pct']);short_max=float(g.loc[list(HORIZONS[:2]),'rmse_change_pct'].max())
        if target_delta<0 and short_max<=0:
            decision='优先重复CV确认' if case=='NB_P24' else '保留候选，确认暂缓'
        else:decision='有取舍，需要进一步判断' if target_delta<0 else '本轮暂不采用'
        recommendations.append(dict(case=case,horizon=h,target_rmse_change_pct=target_delta,
            maximum_short_horizon_change_pct=short_max,recommendation=decision,adopted=False))
    write_frame(OUT/'summary/top_county_changes.csv',top);write_frame(OUT/'summary/next_step_recommendation.csv',recommendations)
    models=read_json(RUN/'model_manifest.json')['models'];rounds=[]
    for m in models:rounds.append(dict(horizon=m['spec']['horizon'],kind=m['spec']['kind'],outer_fold=m['spec']['outer_fold'],
        best_iterations=','.join(map(str,m['fit']['best_iterations'])),requested_rounds=m['fit']['requested_rounds'],seconds=m['fit']['seconds']))
    write_frame(OUT/'summary/round_selection.csv',rounds)
    report=['# P6/P24/P48邻县信息增量：seed42结果（2026-09-20）','',
        '**结论：三个单来源的点估计均有小幅改善，但所有非零收益比较的区间均跨零；优先确认P24，P6/P48邻县增量暂缓采用。** 三个实验及独立验收均完成。本轮参照是已加入P6 a/b的完整组合：P1 a/b＋F2、P6 a/b（163列）、D_G，其余原strict。每个候选只给对应来源P增加20列邻县摘要，其他来源固定。','',
        '## 1. 整体OSI：相对当前组合的净增量','',
        '负数改善、正数恶化；未将三个候选合并。','',
        table(['候选','输出1h','输出6h','输出24h','输出48h'],matrix),'',
        table(['候选','输出h','B0 RMSE','候选RMSE'],absrows),'',
        table(['候选','本轮建议'],[[r['case'],r['recommendation']] for r in recommendations]),'',
        'P24的同名整体改善约0.683%，并有4/5折改善，短时输出没有退化；其确认每套只需25次拟合，比P6的两分支成本低。P6/P48的同名改善仅约0.157%/0.175%，暂不投入同等确认优先级。三个结果仍只是seed42开发证据，不能直接当作三套CV确认或隐藏测试集收益；没有自动替换当前模型。','',
        '## 2. 修改范围与可解释性','',
        '- NB_P06：原a/b结构、L2、贡献权重和早停规则不变，只将其输入163→183列。当前P1已有F2邻县信息，因此这是条件于已有P1空间信息后的额外收益。',
        '- NB_P24/NB_P48：原直接P树及Huber损失不变，只将输入163→183列；没有复用此前退化的长horizon a/b。',
        '- 三套增强输入都沿用8个冻结地理近邻。五项历史固定在0–71，目标阵风分别对齐t+6/t+24/t+48。起点天气与原[t,min(t+6,215)]最多7点暴露窗口保持原样，没有扩成24/48小时窗口。',
        '- 302县中包含测试县已提供的历史和天气，邻居不按fold筛选；未来停电、标签、残差和OOF预测均不进入聚合。没有测试标签使用、测试模型推理或提交。',
        '- 长horizon来源会通过C3影响最终短时输出；NB_P06的24/48h、NB_P24的48h、NB_P48的24h逐行不变，最早未被来源覆盖的目标小时不变。','',
        '## 3. P分量与统计区间','',
        table(['候选（同名来源P）','空间','B0 P RMSE','新P RMSE','变化'],prows),'',
        'P6来源自身的P误差下降约1.39%，C3后只下降约0.144%，最终6h下降约0.157%；不能把来源模型的分量改善等同于最终整体收益。P24在直接P树上获得小幅改善，也说明此前a/b迁移失败并不等于邻县信息无用。以上都仍需重复CV确认。','',
        'component_clipped为来源P进入组合前的值，aligned为C3后的值。最终24/48h使用C1，不能把aligned当成它们的主路线。来源P精度与最终OSI都必须评价，不能以其中之一替代另一项。','',
        table(['候选','输出h','整体ΔRMSE','县轨迹95%区间'],intervals),'',
        '所有非零变化的OSI与P分量区间均跨零，不能宣称显著提分。县轨迹bootstrap固定2000次、seed20260910；区间未消除县间空间相关和此前模型选择偏差。四项原始RMSE不取平均冒充官方分数。','',
        '## 4. 县、时段与初始状态','',
        table(['候选（同名最终输出）','改善县/恶化县','改善折/恶化折','净SSE改善'],crows),'',
        table(['候选','有效目标小时','整体RMSE变化','SSE改善'],wrows),'',
        '收益并非覆盖全部时段：P6在96–143小时改善、早期78–95和后期144–215略差；P24主要改善96–143小时，144–215略差；P48两个有效窗口都小幅改善，但最终48h仅2/5折改善。没有据此追加时间门控。','',
        '大误差县也不是一致改善。P6的Forest/Morrow有收益，但Clay略差；P24的Muskingum和Brown改善，而Clay的SSE增加约0.02785，几乎相当于该候选全体县净SSE改善0.02814。这是确认P24时需重点检查的稳定性问题，不能据此对Clay单独切回旧模型。','',
        '全部239县均保留。上述聚合是描述性分组，没有事后删县、逐县规则或按时段启用新模型。P71分组的来源P指标为：','',
        table(['候选','P71组','行数','P RMSE变化','P SSE改善'],brows),'',
        '完整县表、七个冻结关注县及改善/恶化前五县均保存在summary。OSI变化的P自身平方项、P与N/D/R交叉项及后处理余项另存，并逐行验证与实际SSE变化闭合。','',
        '下一步建议只先完成NB_P24在seed20260917、seed20260918的同协议确认。当前已确认基线保持不变；若P24方向不能复现，应保留这次弱增量结论，不再围绕同一批OOF反复搜索邻居数/窗口/县名单。没有理由直接把三个候选同时纳入。','',
        '## 5. 核验与交付','',
        '- 新增80次内层早停＋20次完整四折refit，共100次拟合、20个外层模型。每个外折都排除该县集合的训练和轮数选择，分箱仅参考相应训练子集。',
        '- 三个183列包均完成独立几何/排名与原始时间口径核对、逐位一致聚合重建；未来停电扰动不改变输入，天气正对照能改变输入。h=1回放与原F2逐位一致，原163列未改变。',
        '- 独立验收用重建的183列重新加载20个模型，检查80个probe的作用域、标签、权重、指标和最佳轮数；独立pandas目标小时对齐，复算80组控制指标、18组区间及全部县/折/时段指标。',
        '- 有效行数为34,177 / 32,982 / 28,680 / 22,944；所有有效预测finite，无效尾部和分支支持掩码严格核对。所有特征/模型/预测均带来源身份和哈希。',
        '- 原代码、分折、模型、缓存和报告只读。新文件均在本目录；没有训练另外两套CV、联合候选或全量提交模型。','',
        '## 6. 文件入口','',
        f'- [整体指标]({OUT}/summary/primary_scores.csv)',f'- [P分量指标]({OUT}/summary/P_component_metrics.csv)',
        f'- [区间]({OUT}/summary/bootstrap_intervals.csv)',f'- [县级结果]({OUT}/summary/county_metrics.csv)',
        f'- [固定关注县]({OUT}/summary/frozen_focus_counties.csv)',f'- [整体误差变化拆解]({OUT}/summary/OSI_change_decomposition.csv)',
        f'- [独立模型与特征验收]({RUN}/logs/independent_verification.json)',f'- [执行说明]({OUT}/README.md)','']
    safe(OUT/'Results_2026-09-20.md').write_text('\n'.join(report))
    safe(OUT/'README.md').write_text(f'''# P6/P24/P48邻县信息单来源实验

状态：seed42三候选完成，100次拟合、20个模型；独立核验PASS。[结果]({OUT}/Results_2026-09-20.md)。点估计均小幅改善、收益区间均跨零；优先确认NB_P24，NB_P06/NB_P48暂缓，当前已确认组合保持不变。

参照是已确认P6 a/b的完整组合。NB_P06仍为a/b加权L2，NB_P24/NB_P48仍为原直接P-Huber。仅增加对应时间口径的20列邻县特征，不联合候选、不训练其他CV。

首次执行顺序：

```bash
cd {ROOT}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/prepare_evaluation.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/train_neighbors.py --stage preflight
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/train_neighbors.py --stage train
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/evaluate_neighbors.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/verify_neighbors.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/summarize_results.py
```

完成stage拒绝覆盖；--resume只允许同身份未完成训练。单独运行verify_neighbors.py不拟合模型。

- features/v1：三套183列训练/测试矩阵、白名单视图、固定邻居/坐标和1,043,712条时段依赖；测试矩阵仅供特征覆盖核验。
- reference：冻结输入哈希、时间/因果/作用域预检和评价代码适配记录。
- runs/v18_pneighbor_v1_split42_model42：20个模型及回执、内层预测/曲线、完整新P OOF、候选分量和控制预测、全部指标与独立验收。
- summary：四输出矩阵、P指标、县/折/时段/初始状态、bootstrap和误差变化拆解。
- completion.json、final_integrity_check.json：本轮完成身份与旧来源不变证明。

独立特征验证使用与生产张量不同的按县逐邻居累加，并固定IEEE-754加法顺序；从原始数据核对历史/天气语义，再用冻结缓存的精确值重建183列进行模型重载。原163列继承既有因果审查，不宣称本轮重做全部外部静态数据提取。
''')
    print(primary[primary['case']!='B0'][['case','horizon','baseline_rmse','rmse','rmse_change_pct']].to_string(index=False))
    print(recommendations)


if __name__=='__main__':main()
