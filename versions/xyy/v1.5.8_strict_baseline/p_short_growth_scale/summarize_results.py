"""Seed42 results with explicit denominators and concentration limits; no plots."""
from growth_core import *

def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    ver=read(RUN/'verification.json');assert ver['status']=='PASS'
    primary=csv(RUN/'metrics/primary_scores.csv');comp=csv(RUN/'metrics/comparisons.csv');group=csv(RUN/'metrics/group_metrics.csv')
    boots=csv(RUN/'metrics/bootstrap.csv');counties=csv(RUN/'metrics/county_metrics.csv',dtype={'fipsCode':str});folds=csv(RUN/'metrics/fold_metrics.csv')
    pm=csv(RUN/'metrics/P_metrics.csv');bm=csv(RUN/'metrics/branch_contribution.csv');boundary=csv(RUN/'metrics/boundary_95_96.csv',dtype={'fipsCode':str})
    model=read(RUN/'model_manifest.json');pred=normalize(pd.read_parquet(RUN/'control_predictions.parquet'))
    frame(OUT/'summary/main_comparisons.csv',comp);frame(OUT/'summary/auxiliary_groups.csv',group)
    focus={'42053','39117','39119','18013','54015','54013','54007','39115','42031','54009'}
    frame(OUT/'summary/focus_counties.csv',counties[counties.fipsCode.isin(focus)])
    round_rows=[]
    for case in NEW:
        for outer in range(5):
            inner=sorted([r for r in model if r['spec']['case']==case and r['spec']['outer_fold']==outer and r['spec']['inner_fold'] is not None],key=lambda r:r['spec']['inner_fold'])
            ref=next(r for r in model if r['spec']['case']==case and r['spec']['outer_fold']==outer and r['spec']['inner_fold'] is None)
            round_rows.append(dict(case=case,outer_fold=outer,inner_best_iterations=','.join(str(r['best_iteration']) for r in inner),refit_rounds=ref['spec']['requested_rounds'],actual_trees=ref['actual_trees'],
                refit_weight_mean=ref['training_weight_mean'],fit_seconds=sum(r['fit_seconds'] for r in inner)+ref['fit_seconds']))
    frame(OUT/'summary/model_rounds.csv',round_rows)
    bound=[]
    for (case,domain),g in boundary.groupby(['case','domain']):
        idx=g.additional_jump.abs().idxmax();row=boundary.loc[idx]
        bound.append(dict(case=case,domain=domain,counties=len(g),mean_abs_additional_jump=g.additional_jump.abs().mean(),max_abs_additional_jump=g.additional_jump.abs().max(),max_county=row.fipsCode))
    frame(OUT/'summary/boundary_summary.csv',bound)
    lineage=dict(reference_tree=dict(identity=TREE_ID,run=str(TREE),case='NB_P24'),reference_a=dict(identity=F2_ID,run=str(F2),case='F2',
        column='a_applied',source_column='source_a_model_id',branch_file=str(F2/'branch_oof_predictions.parquet')),
        feature_file=str(FEATURES/'F2_train.parquet'),feature_count=183,new_branch_file=str(RUN/'branch_predictions.parquet'),cases={})
    for case in CASES:
        lineage['cases'][case]=dict(P1_early='reference' if case=='G0' else 'p71 * reference_a + clipped_new_increment',P1_late='exact_copy_of_NB_P24_P1',
            other_components='exact_copy_of_NB_P24',new_b_column=None if case=='G0' else case+'_raw_b',model_id_column=None if case=='G0' else case+'_model_id',
            new_models=[r['model_id'] for r in model if r['spec']['case']==case and r['spec']['inner_fold'] is None])
    save(OUT/'summary/source_lineage.json',lineage)
    decision=dict(status='seed42_completed_not_adopted',candidate='G2',reference_remains='G0',
        category='meaningful_point_estimate_single_county_dominated',confirmation_candidate=True,automatic_confirmation_started=False,
        reason='1h/6h improve vs G0; gains dominated by Forest, other counties slightly worse, paired CIs cross zero',
        next_scope='If continued, unchanged G2 with G0 on the two frozen CVs; do not expand parameter or county-rule search',new_fits=50,new_outer_models=10,new_inner_models=40,
        test_inference=False,figures_created=0)
    save(OUT/'summary/decision.json',decision)
    score_rows=[]
    for h in [1,6,24,48]:
        s=primary[primary.horizon==h].set_index('case');c=comp[comp.horizon==h].set_index('pair')
        score_rows.append([h,int(s.loc['G0','n']),*[f'{s.loc[g,"rmse"]:.12f}' for g in CASES],f'{c.loc["G1_minus_G0","rmse_change_pct"]:+.3f}%',f'{c.loc["G2_minus_G0","rmse_change_pct"]:+.3f}%'])
    pair_rows=[]
    for r in boots[(boots.group=='all')&boots.horizon.isin([1,6])].itertuples():
        pair_rows.append([r.pair,r.horizon,f'{r.rmse_change_pct:+.3f}%',f'{r.rmse_delta:+.9f}',f'[{r.ci_low:+.9f}, {r.ci_high:+.9f}]'])
    group_names={'all':'全239县','early_73_95':'仅早期目标73–95','late_96_215':'后期目标96–215','test_like':'与测试输入相似的子集',
        'not_test_like':'其余子集','all_except_Forest':'全体除Forest（仅诊断）','test_like_except_Forest':'测试相似子集除Forest（仅诊断）',
        'g_zero':'N71=0','g_positive':'N71>0','g_positive_low_early_damage':'N71>0且早期真实峰值OSI≤.01'}
    aux_rows=[]
    for r in group[(group.pair=='G2_minus_G0')&group.horizon.isin([1,6])].itertuples():
        aux_rows.append([r.horizon,group_names[r.group],int(r.n),f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
    concentration=[];count_rows=[];fold_rows=[]
    for h in [1,6]:
        c=counties[(counties.pair=='G2_minus_G0')&(counties.horizon==h)];f=c[c.fipsCode=='42053'].iloc[0];net=comp[(comp.pair=='G2_minus_G0')&(comp.horizon==h)].iloc[0].sse_reduction
        concentration.append([h,f'{net:+.6f}',f'{f.sse_reduction:+.6f}',f'{net-f.sse_reduction:+.6f}',f'{f.rmse_change_pct:+.3f}%'])
        count_rows.append([h,int((c.sse_reduction>1e-12).sum()),int((c.sse_reduction< -1e-12).sum()),int((c.sse_reduction.abs()<=1e-12).sum())])
        for r in folds[(folds.pair=='G2_minus_G0')&(folds.horizon==h)].itertuples():fold_rows.append([h,r.fold,f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
    p_rows=[]
    for r in pm[(pm.pair=='G2_minus_G0')&(pm.horizon==1)].itertuples():
        p_rows.append([r.space,r.group,int(r.n),f'{r.baseline_rmse:.9f}',f'{r.rmse:.9f}',f'{r.rmse_change_pct:+.3f}%'])
    peaks=[]
    for case in CASES:
        mask=(pred.fipsCode=='42053')&(pred.hour_idx+1<=95);peaks.append(float(pred.loc[mask,f'pred_{case}_v18_rule_osi_target_t01h'].max()))
    best=counties[(counties.pair=='G2_minus_G0')&(counties.horizon==1)].sort_values('sse_reduction')
    case_rows=[[r.countyName+' '+r.stateAbbr,f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'] for r in pd.concat([best.tail(5).iloc[::-1],best.head(5)]).itertuples()]
    # Freeze a simple manifest for convenient downstream candidate references.
    save(OUT/'summary/candidate_manifest.json',dict(protocol='v18_pgrowth_v1',cv_seed=42,identity_hash=ver['identity_hash'],run_directory=str(RUN),cases=list(CASES),
        adopted=False,files={n:dict(path=str(RUN/n),sha256=sha(RUN/n)) for n in ['control_predictions.parquet','component_predictions.parquet','branch_predictions.parquet','verification.json','model_manifest.json']}))
    report=f'''# P1短时增长尺度首轮结果（2026-09-20）

**seed42已完成：G2相对当前G0，完整1h RMSE下降4.948%，6h下降1.798%；G1基本无净收益。但G2的净收益几乎全部来自Forest，排除Forest后的其他县总体略差，主要置信区间跨零。因此暂不替换主线，只保留G2作为有限跨CV确认候选。**

本轮实际50次模型拟合、10个新外层模型；40个内层模型也保存供审计。没有训练另外两套CV、没有测试预测/全量拟合、没有图片。138份旧输入文件保持不变。

## 1. 到底改了什么

G0为最新NB_P24完整树方案：P1 a/b＋F2，P6 a/b，P24邻县，D_G及其他strict来源。本实验不是回到最早的strict基线，也没有把较早F2整张预测覆盖当前方案。

| 方案 | 新训练内容 | 早期增量尺度 | 其他部分 |
|---|---|---|---|
| G0 | 无 | 原b重建 | 当前完整参照 |
| G1 | 只在早期目标训练P1的b | 1−P71 | a、其他来源冻结 |
| G2 | 与G1相同行/183列/树参数训练b | (1−P71)*(1+N71/.01) | 与G1相同 |

新b只用于目标小时73–95（截止后首24小时内的有效目标）。目标96以后显式使用旧P1；通过原C3影响最终1h的73–95和6h的78–95，最终24h/48h完全不变。没有按县或g>0路由；g=0行也可能因重训而变化。

G2检验的是增长尺度的监督归一化、贡献权重和输出重建这一整体修改，不能把效果归因于单独增加一个乘数；现有输入里本来就有N71。

## 2. 完整四输出成绩

范围：seed42全239县、每个horizon的全部有效县小时。n为评分行数；RMSE单位为OSI。相对变化=(候选RMSE/参照RMSE−1)×100%，负数改善；没有平均四个RMSE作为官方总分。

{table(['输出h','n','G0 RMSE','G1 RMSE','G2 RMSE','G1相对G0','G2相对G0'],score_rows)}

G1的1h略差、6h略好；仅缩小b训练时间范围没有明确净收益。G2相对G1的完整1h/6h分别改善4.986%/1.745%，说明新增尺度方案与早期专门训练对照产生了实质差异。

## 3. 区间和分折稳定性

下表ΔRMSE=候选−参照，单位OSI；区间为2000次县整条轨迹配对bootstrap的95%百分位区间。不是测试获胜概率，也不校正此前同事件开发选择偏差。

{table(['比较','输出h','相对RMSE变化','ΔRMSE','ΔRMSE的95%区间'],pair_rows)}

G2相对G0的1h、6h区间均跨零。G2相对G1的1h区间低于零，但它不能替代相对当前G0的主要比较，也不能抵消收益集中这一限制。

以下为G2−G0的每个外折，SSE减少量=参照SSE−候选SSE，正数改善；各折RMSE不能简单平均替代总体成绩。

{table(['输出h','外折编号','相对RMSE变化','SSE减少量'],fold_rows)}

最大收益出现在含Forest的fold3。其余折大多只有小变化，部分恶化；不是每折都获得一致改善。

## 4. 收益集中在哪里

范围仍为完整horizon有效时段。表中的SSE减少量为参照−候选，正数改善；Forest减少量大于全体净减少量，意味着其他县抵消了部分收益，不是计数或百分比错误。

{table(['输出h','全体SSE减少量','Forest SSE减少量','其余238县SSE减少量','Forest RMSE变化'],concentration)}

Forest早期1h预测峰值由G0的{peaks[0]:.6f}升至G2的{peaks[2]:.6f}（G1为{peaks[1]:.6f}）；真实峰值仍为.5447。因此它确实改善了幅度表达，但远未准确预测该峰值。

改善/恶化县数按县SSE差>1e-12/<−1e-12判定，中间为数值不变，分母均239；县数量不衡量误差量级，不能代替总体RMSE。

{table(['输出h','改善县数','恶化县数','数值不变县数'],count_rows)}

下列为1h的SSE改善前5县和恶化前5县，仅作结果诊断，不能据此新增县级规则。

{table(['县','G2相对G0 RMSE变化','SSE减少量'],case_rows)}

## 5. 与测试输入相似的辅助评价

成员在训练前冻结：对于相应horizon和目标日，被至少一个测试县选入context前5近邻的训练县进入test_like。它只利用已知输入，不使用测试未来标签。除“早期/后期”两行外，各行统计该组完整有效时段；n为县小时数。

“除Forest”仅是诊断视角，正式全县成绩保留Forest；“N71>0且早期真实峰值≤.01”使用未来真实值作事后低损失分组，不能成为推理规则。

{table(['输出h','分组','n','G2相对G0 RMSE变化','SSE减少量'],aux_rows)}

test_like在1h/6h分别改善6.093%/2.423%，但它包含Forest。去掉Forest后，test_like的1h略差0.032%、6h仅略好0.006%；因此辅助视角没有提供独立于Forest的广泛收益证据。其余县的轻微恶化也不能据此称为统计上确定的伤害。

已知增长但早期实际损失很轻的县略有恶化，符合需要继续检查误报的预期。Warren与Forest的已知状态相似，只说明该方向与测试输入有联系，不代表本轮已验证Warren的预测收益；本轮没有测试推理。

## 6. P分量是否改善

下表只比较G2−G0：source为原P1来源预测，aligned为C3按同一目标小时对齐后的P；all为完整有效目标，early为73–95。它们都使用官方P真值，RMSE单位为P比例，不是OSI。

{table(['P空间','窗口','n','G0 P RMSE','G2 P RMSE','相对变化'],p_rows)}

早期b增量的P贡献RMSE由{bm.set_index('case').loc['G0','b_increment_rmse']:.9f}降到{bm.set_index('case').loc['G2','b_increment_rmse']:.9f}；a的平方误差项完全不变。a、b自身平方项与交叉项之和重现P误差。

这里没有拿不同尺度u的原始RMSE互相比优劣；评价的是乘回尺度后的P增量、实际P和最终OSI。N/D/R没有被重新训练，其误差抵消/放大会影响最后的OSI收益。

## 7. 边界与严格性

目标95→96处，G2相对G0额外跳变的最大绝对值：P1约0.00809、最终1h/6h OSI约0.00188。完整239县表已保存。该边界是预先冻结的作用范围，本轮没有看完结果后改窗口或平滑；后续如确认，应保留相同规则并继续检查。

- 预检重载原F2的5个a和5个b模型，与保存OOF一致；50个实际训练作用域通过外折/内折及晚期标签和特征扰动隔离检查。
- 正式40次内层探测＋10次外层重训，全部保存模型、作用域、权重归一化、曲线、最佳轮数和收据。
- 独立进程重载40个新内层、10个新外层以及5个参考a模型。新检查点预测最大差0；按原始数据独立重建目标与分量，按县/目标小时重新对齐C3。
- {ver['metric_groups_checked']}条指标分组、48条P指标、24条bootstrap记录及356,349条逐行比较通过独立复算。后期、长期和未改分量逐行精确一致。
- 内层早停只使用b的贡献误差，不调用已见内层验证县标签的外层a模型参与选轮。参考a仅用于其真正外折输出的合成。
- 138个来源文件哈希保持不变；未改旧代码/模型/预测；没有图、测试推理或提交。

## 8. 下一步判断

**保持G0主线，G2只保留为原样跨CV确认候选。** 这次点估计改善足够明显，G1对照也支持增长尺度整体方案产生了作用；但净收益由Forest主导、其余县没有广泛收益、相对G0的区间跨零，尚不足以直接采用。

若继续，只在另外两套已冻结CV复现同一个G2，与各自G0比较，重点看Forest改善是否稳定、测试相似子集除Forest是否出现一致代价、边界和轻灾组是否恶化。本轮不自动启动确认，不增加尺度常数/窗口/县规则搜索；不能因为seed42的4.95%就重新开始围绕Forest无期限调参。

## 文件入口

- [主要比较]({OUT}/summary/main_comparisons.csv)
- [辅助分组]({OUT}/summary/auxiliary_groups.csv)
- [原始逐行最终OSI]({RUN}/control_predictions.parquet)
- [原始分量预测]({RUN}/component_predictions.parquet)
- [新b分支与模型身份]({RUN}/branch_predictions.parquet)
- [模型轮数与归一化]({OUT}/summary/model_rounds.csv)
- [区间]({RUN}/metrics/bootstrap.csv)
- [全县比较]({RUN}/metrics/county_metrics.csv)
- [时间边界]({RUN}/metrics/boundary_95_96.csv)
- [独立校验]({RUN}/verification.json)
- [候选来源清单]({OUT}/summary/candidate_manifest.json)
'''
    safe(OUT/'Results_2026-09-20.md').write_text(report)
    safe(OUT/'README.md').write_text(f'''# P1早期增长尺度：seed42

已完成50次拟合和独立核验，结果见[报告]({OUT}/Results_2026-09-20.md)。G2完整1h/6h改善4.948%/1.798%，收益几乎全部来自Forest；主线未替换，另两CV未运行。

所有写入在本目录，不画图、不测试推理。G0是最新NB_P24整套树方案；仅G1/G2的早期P1 b为新模型。10外层与40内层模型分别位于runs下的models/G*/outer*/。

首次执行顺序：

```bash
cd {ROOT}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/preflight.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/train_growth.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/evaluate_growth.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/verify_growth.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/summarize_results.py
```

训练/评价完成标记存在时拒绝覆盖。未完成训练仅允许匹配身份的--resume；每个已完成拟合有模型和收据，孤立模型不自动复用。验证脚本不做训练。完整性与所有交付哈希见completion.json。
''')
    print('Summary saved:',decision,flush=True)

if __name__=='__main__':main()
