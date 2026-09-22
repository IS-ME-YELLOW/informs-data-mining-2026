"""Report the fixed-route experiment, including mixed evidence and tail tradeoffs."""
from integration_protocol import *


def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    for seed in SEEDS:assert read_json(OUT/f'runs/split{seed}/verification.json')['status']=='PASS'
    names=('primary_scores','comparisons','paired_county_bootstrap','county_metrics','fold_metrics','target_window_metrics',
           'target_day_metrics','severity_metrics','original_gate_metrics','complementarity')
    data={name:pd.concat([csv(OUT/f'runs/split{s}/metrics/{name}.csv',dtype={'fipsCode':str} if name=='county_metrics' else {}) for s in SEEDS],ignore_index=True) for name in names}
    for name,d in data.items():write_frame(OUT/f'summary/{"three_cv_scores" if name=="primary_scores" else name}.csv',d)
    comp=data['comparisons'];boot=data['paired_county_bootstrap'];primary=data['primary_scores'];county=data['county_metrics'];fold=data['fold_metrics']
    gate=data['original_gate_metrics'];complement=data['complementarity']
    consistency=[]
    for (pair,h),g in comp.groupby(['pair','horizon']):
        consistency.append(dict(pair=pair,horizon=h,improved_splits=int((g.rmse_delta < -1e-12).sum()),
            worsened_splits=int((g.rmse_delta > 1e-12).sum()),equivalent_splits=int((g.rmse_delta.abs()<=1e-12).sum()),
            min_rmse_change_pct=float(g.rmse_change_pct.min()),max_rmse_change_pct=float(g.rmse_change_pct.max())))
    write_frame(OUT/'summary/comparison_consistency.csv',consistency)
    cc=[]
    for (pair,h,f),g in county.groupby(['pair','horizon','fipsCode']):
        assert len(g)==3
        cc.append(dict(pair=pair,horizon=h,fipsCode=f,countyName=g.countyName.iloc[0],stateAbbr=g.stateAbbr.iloc[0],
            improved_splits=int((g.sse_reduction>0).sum()),worsened_splits=int((g.sse_reduction<0).sum()),
            **{f'split{s}_sse_reduction':float(g.loc[g.cv_seed==s,'sse_reduction'].iloc[0]) for s in SEEDS}))
    cc=pd.DataFrame(cc);write_frame(OUT/'summary/county_consistency.csv',cc)
    write_frame(OUT/'summary/fixed_focus_counties.csv',county[county.fipsCode.isin(FOCUS)])
    top=[]
    for (seed,pair,h),g in county.groupby(['cv_seed','pair','horizon']):
        if not (g.sse_reduction!=0).any():continue
        g=g.sort_values(['sse_reduction','fipsCode'])
        for direction,selected in [('worsened',g[g.sse_reduction<0].head(10)),('improved',g[g.sse_reduction>0].tail(10).sort_values(['sse_reduction','fipsCode'],ascending=[False,True]))]:
            for rank,r in enumerate(selected.itertuples(),1):top.append(dict(cv_seed=seed,pair=pair,horizon=h,direction=direction,rank=rank,
                fipsCode=r.fipsCode,countyName=r.countyName,stateAbbr=r.stateAbbr,sse_reduction=r.sse_reduction))
    write_frame(OUT/'summary/top_county_changes.csv',top)
    c1=comp[(comp.pair=='I3_minus_I1')&(comp.horizon==24)];c2=comp[(comp.pair=='I3_minus_I2')&(comp.horizon==24)]
    beats_tree=bool((c1.rmse_delta < -1e-12).all());beats_both=bool((c1.rmse_delta<=1e-12).all() and (c2.rmse_delta<=1e-12).all())
    if beats_both:category='stronger_integration_candidate'
    elif beats_tree:category='tradeoff_requires_stress_test'
    else:category='no_uniform_increment'
    decision=dict(category=category,I3_better_than_tree24_all_CVs=beats_tree,I3_not_worse_than_both_all_CVs=beats_both,
        I3_better_than_Stella24_splits=int((c2.rmse_delta < -1e-12).sum()),
        next_stage_reference='I1',next_stage_candidate='I3',retain_comparator='I2',
        final_winner_declared=False,automatic_deployment=False,source_models_changed=False,new_fits=0,
        source_models_reloaded=0,weights_unchanged=[.5,.5],source_gate_unchanged=True,
        interpretation='I3 improves current tree24 in all three CVs but trades off against Stella24; pressure testing remains separate')
    write_json(OUT/'summary/decision.json',decision)
    # Make all fixed options reviewable without silently choosing a different one per CV.
    artifacts=[]
    for seed in SEEDS:
        run=OUT/f'runs/split{seed}';m=read_json(run/'run_manifest.json')
        artifacts.append(dict(cv_seed=seed,identity_hash=m['identity_hash'],run_directory=str(run),
            files={name:dict(path=str(run/name),sha256=sha(run/name)) for name in
                ('source_predictions.parquet','integrated_predictions.parquet','source_lineage.json','run_manifest.json','verification.json')}))
    write_json(OUT/'summary/route_candidates.json',dict(status='CV_route_comparison_not_final_submission',cases=list(CASES),
        candidate_to_validate='I3',reference_for_next_stage='I1',fits=0,splits=artifacts))
    overall=[];absolutes=[];short=[];pairs=[];counties=[];tail=[];decomp=[];window=[]
    for seed in SEEDS:
        g=primary[primary.cv_seed==seed].set_index(['candidate','horizon'])
        full=comp[(comp.cv_seed==seed)&(comp.pair=='I3_minus_B0')].set_index('horizon')
        overall.append([seed]+[f'{full.loc[h,"rmse_change_pct"]:+.3f}%' for h in HS])
        absolutes.append([seed,f'{g.loc[("I1",24),"rmse"]:.12f}',f'{g.loc[("I2",24),"rmse"]:.12f}',
            f'{g.loc[("I3",24),"rmse"]:.12f}',f'{g.loc[("B0",48),"rmse"]:.12f}',f'{g.loc[("I1",48),"rmse"]:.12f}'])
        short.append([seed,f'{g.loc[("B0",1),"rmse"]:.12f}',f'{g.loc[("B0",6),"rmse"]:.12f}','四候选逐行一致'])
        for pair in ('I3_minus_I1','I3_minus_I2'):
            row=boot[(boot.cv_seed==seed)&(boot.pair==pair)&(boot.horizon==24)].iloc[0]
            pairs.append([seed,pair,f'{row.rmse_change_pct:+.3f}%',f'{row.rmse_delta:+.9f}',f'[{row.ci_low:+.9f}, {row.ci_high:+.9f}]'])
            c=county[(county.cv_seed==seed)&(county.pair==pair)&(county.horizon==24)]
            f=fold[(fold.cv_seed==seed)&(fold.pair==pair)&(fold.horizon==24)]
            counties.append([seed,pair,f'{int((c.sse_reduction>0).sum())}/{int((c.sse_reduction<0).sum())}',
                f'{int((f.sse_reduction>0).sum())}/{int((f.sse_reduction<0).sum())}',f'{c.sse_reduction.sum():+.6f}'])
            for r in gate[(gate.cv_seed==seed)&(gate.pair==pair)].itertuples():
                tail.append([seed,pair,'回退L0' if r.original_gate else '使用L1',int(r.n),f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
        r=complement[(complement.cv_seed==seed)&(complement['group']=='all')].iloc[0]
        decomp.append([seed,f'{r.error_correlation:.4f}',f'{r.correction_correlation:.4f}',f'{r.diversity_sse_reduction:.6f}',
            f'{r.postprocessing_sse_increase:+.6f}',int(r.post_changed_rows)])
        d=data['target_window_metrics'];d=d[(d.cv_seed==seed)&(d.horizon==24)&d.pair.isin(['I3_minus_I1','I3_minus_I2'])&(~d.common_target)&(~d.accounting_only)]
        for r in d.itertuples():window.append([seed,r.pair,f'{int(r.start)}–{int(r.end)}',f'{r.rmse_change_pct:+.3f}%'])
    result=['# 最新树与Stella路线整合结果（2026-09-20）','',
        '**结论：固定1:1组合I3对当前树24h在三套均改善，但对Stella L2只在前两套改善，第三套退化；属于计划第6节的“存在互补、仍有取舍”情形，不能宣布它统一胜过两条来源。**',
        '建议I1作为下一阶段整套参照，I3保留为待压力测试的主候选，I2继续作24h对照。没有调整权重或门控，没有按不同CV挑不同赢家。','',
        '本轮新增拟合0次、模型0个。三套来源身份/对齐、组合算术和独立复算全部PASS；源模型采用其既有验收记录，本轮没有重新加载源模型。','',
        '## 1. 整套输出与来源','',
        '| 方案 | 1h/6h | 24h | 48h |\n|---|---|---|---|\n| B0 | 最新树T | 最新树T | 最新树T |\n| I1 | 最新树T | 最新树T | Stella L2 |\n| I2 | 最新树T | Stella L2 | Stella L2 |\n| I3 | 最新树T | H((T＋Stella L2)/2) | Stella L2 |','',
        'T包含P1 a/b＋F2、P6 a/b、P24直接树＋邻县、D_G及其他strict来源。L2是Stella候选名。所有组合在最终OSI层进行，未重新计算C3或改动P/N/D/R。','',
        'I3相对整套B0的RMSE变化（负数改善）：','',
        table(['CV seed','1h','6h','24h','48h'],overall),'',
        '**48h的收益来自已保存Stella L2，I1/I2/I3共用，并不是1:1融合的新发现。** 新增研究问题仅为24h的固定混合。','',
        table(['CV seed','树24h / I1','Stella24h / I2','混合24h / I3','B0的48h','I1/I2/I3的48h'],absolutes),'',
        table(['CV seed','全部候选1h RMSE','全部候选6h RMSE','核验'],short),'',
        'Stella文件中较早的F2短时预测未进入候选，因而没有丢掉此前P6结构和P24邻县对短时的收益。','',
        '## 2. 固定平均是否胜过两个24h来源','',
        table(['CV seed','比较','RMSE变化','ΔRMSE','县轨迹95%区间'],pairs),'',
        'I3−I1在三套均为负，但只有seed20260918的区间完全低于0。I3−I2前两套为负、第三套为正，三套区间均跨零。没有达到“每套都不劣于两个来源”的较强判断条件。','',
        '因此可以说I3提供了相对当前树的同向点估计增量，不能说它已确定优于Stella或是最终最优组合。48h的复用收益不能用来掩盖24h对I2的取舍；四个RMSE也未被平均成所谓官方总分。','',
        '## 3. 尾部保护确有取舍','',
        '以下按Stella原有24h门控分组。标签只用于评分，组别完全由已冻结的原预测/阈值决定。','',
        table(['CV seed','比较','原门控区域','行数','RMSE变化','SSE改善'],tail),'',
        'I3相对最新树T，在seed20260917/20260918的原门控区域分别恶化约1.14%/0.52%，由非门控区域收益补偿；seed42门控区域略有改善。相对Stella S，seed42门控区略差，而另两套有所改善。','',
        '**“保留原Stella门控结果作为一个输入”不等于“平均后还保留其原保护性质”。** 两个单独来源的尾部优势也会随CV变化。本轮不追加新的门控或为个别县回退；任何更新锚点/阈值的方案都应另立严格内层实验。','',
        '## 4. 互补与后处理','',
        table(['CV seed','来源误差相关','相对L0修正相关','平均相对两来源SSE均值的减少','后处理SSE增加','置零改变行数'],decomp),'',
        '两个来源的误差仍高度相关，但修正部分并不完全重合。恒等式 `SSE(mean)=0.5*SSE(T)+0.5*SSE(S)-0.25*sum((T-S)^2)` 逐行和汇总均闭合。它保证原始平均不差于两个来源SSE的均值，**不保证优于较好的那个来源**。','',
        '原有<0.001置零使约1,939–2,024行平均值改变，并在三套都增加少量SSE，抵消部分平均收益。按照批准计划，原始平均仅作为诊断，不作为第五个候选，也没有因看到该结果而改变阈值。','',
        '## 5. 县、外折与时段','',
        table(['CV seed','比较','改善县/恶化县','改善折/恶化折','净SSE改善'],counties),'',
        '相对树T，I3在约62%–67%的县改善，但前两套并非每折都改善；相对Stella S，县数多数并不改善，整体取决于误差量级，不能用改善县比例替代pooled RMSE。','',
        table(['CV seed','比较','目标小时','RMSE变化'],window),'',
        'seed42的I3对树在96–119小时变差，后续变好；seed20260917对树的净收益很小，多个时段/尾部区域的损益相互抵消；seed20260918虽对树所有固定窗口都改善，却对Stella三个窗口都略差。','',
        'Brown、Muskingum等县在部分分折相对树受损；Tuscarawas（39157）在三套均是I3相对Stella的较大损失县。这些都是结果诊断，未进入任何权重、阈值或路由选择。全239县、预定11县和前10损益县完整保留。','',
        '## 6. 下一步判断','',
        '1. **I1可作为下一阶段整套参照**：保留最新树短时/24h，48h复用已有跨CV方向一致的Stella L2。该48h收益仍保留原有不确定性。',
        '2. **I3进入待压力测试候选，暂不宣布为最终赢家**。它对树三套同向，但对Stella存在取舍，且六个主要比较仅一项区间不含0。保留I2，不能按CV分别挑最小RMSE拼成绩。',
        '3. 接下来可另立冻结候选后的地理留出等压力测试，或严格设计新的锚点/门控实验；本轮不追加这些操作。不要在当前OOF继续扫描融合权重或逐县规则。','',
        '三套同一批县、同一个事件已经被多次用于开发，区间没有修正完整模型选择过程，也没有消除空间相关。此次检验的是固定组合的算术和开发集证据，不是外部泛化证明。','',
        '## 7. 验收与文件','',
        '- 59个只读输入文件通过来源和哈希检查；两套源预测在三个CV均与县/时间键、fold、官方真值和有效掩码一致；已保存门控与阈值匹配，原指标复现。',
        '- 每套独立核验5,215条总体/分组指标、6组非恒定bootstrap区间、593,915条逐行变化；合计1,781,745条变化记录。复制输出精确一致，I3和各指标容差1e-12。',
        '- 原计划的三个固定窗口覆盖96–215；额外存了96以前短时不变量的accounting_only行，仅用于核对全量行数/SSE加总，没有增加候选或事后评分窗口。',
        '- 针对0、.001边界、一零一正、相等来源、有效行缺失/Inf及无效尾部的检查通过；平均MSE恒等式、后处理差额及各分组加总通过。',
        '- 没有新训练、源模型重载、源阈值修改、测试集推理或提交。没有改写上游验收文件；源模型的严格训练声明来自其已有记录，本轮只验证保存预测与整合层。','',
        f'- [三CV全部候选分数]({OUT}/summary/three_cv_scores.csv)',f'- [主要比较与区间]({OUT}/summary/paired_county_bootstrap.csv)',
        f'- [原门控区域]({OUT}/summary/original_gate_metrics.csv)',f'- [11个预定关注县]({OUT}/summary/fixed_focus_counties.csv)',
        f'- [全县一致性]({OUT}/summary/county_consistency.csv)',f'- [互补与后处理拆解]({OUT}/summary/complementarity.csv)',
        f'- [候选来源清单]({OUT}/summary/route_candidates.json)',f'- [独立验收与旧输入完整性]({OUT}/final_integrity_check.json)',
        f'- [执行说明]({OUT}/README.md)','']
    safe(OUT/'Results_2026-09-20.md').write_text('\n'.join(result))
    safe(OUT/'README.md').write_text(f'''# 最新树与Stella路线整合

状态：三套CV的B0/I1/I2/I3全部完成，独立核验PASS。新增训练0次、模型0个；[结果]({OUT}/Results_2026-09-20.md)。

1h/6h全部沿用最新树T。I1为T24＋Stella48；I2为Stella24/48；I3为固定1:1平均24h（再按原规则后处理）＋Stella48。B0保留完整T。

I3相对树24h三套改善，相对Stella24h为两套改善/一套恶化。建议I1作为下一阶段参照，I3作为待压力测试候选，保留I2；没有最终赢家或自动部署声明。

首次执行顺序：

```bash
cd {ROOT}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/integrate_routes.py --stage preflight --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/integrate_routes.py --stage evaluate --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/verify_integration.py --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/summarize_integration.py
```

已完成评价不覆盖；单独运行verify_integration.py可重验，不训练、不重载源模型。

- reference/approved_plan.md：批准原文；根计划保留编写时状态，本README和completion为当前状态。
- reference/source_manifest.json：实际读取文件路径、哈希匹配方式与parquet schema；没有复制大模型。
- runs/split*/source_predictions.parquet：T四输出、S长期及原门控；S旧短时不进入组合。
- runs/split*/integrated_predictions.parquet：完整34,416行、四输出、四候选及真值/掩码。
- runs/split*/row_changes.parquet：全有效行的五个固定比较；blend_postprocessing_audit.parquet：平均、阈值和误差恒等式。
- runs/split*/metrics/、verification.json：全县/折/日期/严重度/尾部、互补、bootstrap及独立验收。
- summary/：三套分别汇总、判断和候选来源；completion.json：本轮完成身份。

本轮只组合最终OSI，不重算C3、不改Stella q95、不更换其内部L0，不学习权重。原始平均只用于诊断；无第五候选。地理压力测试、随机邻居负对照、重新训练或提交均不在此次执行范围。
''')
    print('Summary saved:',decision)


if __name__=='__main__':main()
