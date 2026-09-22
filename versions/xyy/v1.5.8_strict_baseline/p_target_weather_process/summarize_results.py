"""Report the frozen weather-process candidate, including null/negative results."""
from weather_core import *

def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    ver=read(RUN/'verification.json');assert ver['status']=='PASS'
    primary=csv(RUN/'metrics/primary_scores.csv');comp=csv(RUN/'metrics/comparisons.csv');boots=csv(RUN/'metrics/bootstrap.csv')
    counties=csv(RUN/'metrics/county_metrics.csv',dtype={'fipsCode':str});folds=csv(RUN/'metrics/fold_metrics.csv');days=csv(RUN/'metrics/day_metrics.csv')
    groups=csv(RUN/'metrics/groups.csv');process=csv(RUN/'metrics/process_groups_24h.csv');pm=csv(RUN/'metrics/P_metrics.csv')
    feature=csv(RUN/'metrics/feature_importance.csv');models=read(RUN/'model_manifest.json');stats=csv(OUT/'preflight/new_feature_statistics.csv')
    for name,data in [('main_comparisons',comp),('bootstrap',boots),('auxiliary_groups',groups),('process_groups_24h',process)]:frame(OUT/f'summary/{name}.csv',data)
    focus=['54015','54013','54007','39119','54087','54109','42053','39117','39115','18013','54041']
    frame(OUT/'summary/focus_counties.csv',counties[counties.fipsCode.isin(focus)])
    feature_summary=feature.groupby(['feature','is_new'],as_index=False).agg(total_gain=('gain','sum'),total_splits=('split_count','sum'))
    feature_summary['share_total_gain_pct']=100*feature_summary.total_gain/feature_summary.total_gain.sum()
    frame(OUT/'summary/feature_usage.csv',feature_summary)
    gain_share=float(100*feature.loc[feature.is_new,'gain'].sum()/feature.gain.sum());split_share=float(100*feature.loc[feature.is_new,'split_count'].sum()/feature.split_count.sum())
    round_rows=[]
    for outer in range(5):
        inner=sorted([r for r in models if r['spec']['outer_fold']==outer and r['spec']['inner_fold'] is not None],key=lambda r:r['spec']['inner_fold'])
        ref=next(r for r in models if r['spec']['outer_fold']==outer and r['spec']['inner_fold'] is None)
        old=read(TREE/f'models/outer{outer}/P_direct_t24h.json')
        round_rows.append(dict(outer_fold=outer,inner_best_iterations=','.join(str(r['best_iteration']) for r in inner),W1_refit_rounds=ref['spec']['requested_rounds'],
            W1_actual_trees=ref['actual_trees'],W0_refit_rounds=old['fit']['requested_rounds'],fit_seconds=sum(r['fit_seconds'] for r in inner)+ref['fit_seconds']))
    frame(OUT/'summary/model_rounds.csv',round_rows)
    lineage=dict(reference_case='NB_P24',reference_tree_id=TREE_ID,reference_directory=str(TREE),candidate='W1',new_source='P_t_target_t24h',
        new_feature_columns=NEW_COLUMNS,feature_count=191,other_component_sources='exact_copy_of_NB_P24',G2_applied=False,
        short_pre96='exact_copy',final48='exact_copy',C3='original target-hour alignment and equal mean',
        source_prediction_file=str(RUN/'P24_oof_predictions.parquet'),new_model_ids=[r['model_id'] for r in models if r['spec']['inner_fold'] is None])
    save(OUT/'summary/source_lineage.json',lineage)
    decision=dict(status='seed42_complete_not_adopted',candidate='W1',reference_remains='W0_NB_P24',category='no_observed_increment',
        confirmation_recommended=False,automatic_confirmation_started=False,parameter_search_started=False,
        reason='Full24h, short outputs and frozen test-like subset have no net improvement; fixed 8-column package stops here',
        inference_limit='Does not prove all exposure/recovery representations lack value',new_fits=25,new_outer_models=5,saved_inner_models=20,test_inference=False,plots=0)
    save(OUT/'summary/decision.json',decision)
    save(OUT/'summary/candidate_manifest.json',dict(protocol='v18_pweather_v1',cv_seed=42,identity_hash=ver['identity_hash'],run_directory=str(RUN),cases=['W0','W1'],adopted=False,
        files={n:dict(path=str(RUN/n),sha256=sha(RUN/n)) for n in ['control_predictions.parquet','component_predictions.parquet','P24_oof_predictions.parquet','model_manifest.json','verification.json']}))
    overall=[]
    for h in [1,6,24,48]:
        s=primary[primary.horizon==h].set_index('case');d=comp[comp.horizon==h].iloc[0]
        overall.append([h,int(d.n),f'{s.loc["W0","rmse"]:.12f}',f'{s.loc["W1","rmse"]:.12f}',f'{d.rmse_change_pct:+.3f}%'])
    intervals=[[r.horizon,'全县' if r.group=='all' else '测试相似子集',int(r.n),f'{r.rmse_change_pct:+.3f}%',f'[{r.ci_low:+.9f}, {r.ci_high:+.9f}]'] for r in boots[boots.horizon!=48].itertuples()]
    proc_labels={'no_high':'前48小时没有>30mph阵风','active_high':'目标时刻阵风仍>30mph','post_high_1_6':'最近强风过去1–6小时','post_high_7plus':'最近强风过去7–47小时','multiple_segments':'窗口内至少2段强风（重叠组）'}
    proc_rows=[[proc_labels[r.group],int(r.n),f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'] for r in process.itertuples()]
    focus_rows=[];c24=counties[counties.horizon==24]
    for f in focus:
        r=c24[c24.fipsCode==f].iloc[0];focus_rows.append([r.countyName+' '+r.stateAbbr,f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
    improved=int((c24.sse_reduction>1e-12).sum());worsened=int((c24.sse_reduction< -1e-12).sum())
    fold_rows=[[r.fold,int(r.n),f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'] for r in folds[folds.horizon==24].itertuples()]
    day_rows=[[str((pd.Timestamp('2026-03-11')+pd.Timedelta(days=int(r.target_day))).date()),int(r.n),f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'] for r in days[days.horizon==24].itertuples()]
    p_rows=[[r.horizon,'来源P' if r.space=='source' else 'C3后的P',int(r.n),f'{r.baseline_rmse:.9f}',f'{r.rmse:.9f}',f'{r.rmse_change_pct:+.3f}%'] for r in pm.itertuples()]
    feat_rows=[[r.feature,int(r.unique_values),f'{r.min:.4f}–{r.max:.4f}',r.most_correlated_old_feature,f'{r.max_abs_pearson:.3f}'] for r in stats.itertuples()]
    usage_rows=[[r.feature,int(r.total_splits),f'{r.share_total_gain_pct:.3f}%'] for r in feature_summary[feature_summary.is_new].sort_values('total_gain',ascending=False).itertuples()]
    report=f'''# P24目标前天气过程首轮结果（2026-09-21）

**结论：固定8列天气过程特征W1没有带来seed42净收益。完整24h RMSE上升0.131%，1h/6h分别上升0.016%/0.020%，48h不变。测试相似子集也略差。保留原W0，停止本版本，不追加另外两CV或窗口/阈值搜索。**

这些差异较小、区间跨零，不能说W1已被证明显著更差；但没有支持继续采用/确认的收益证据。结论限于本次固定特征组、模型和分折，不代表所有暴露/恢复信息都无价值。

本轮实际25次拟合（20内层＋5外层），保存全部检查点；独立核验PASS。没有叠加短时G2，没有测试模型推理、全量拟合、提交或图片；{ver['source_files_unchanged']}份旧输入文件保持不变。

## 1. 这轮改动和表头口径

W0是当前完整NB_P24树方案，即前一短时实验保持的G0。W1仅给其中的P24直接LightGBM-Huber增加8列，输入183→191；P1/P6/P48、N/D/R、原C3/OSI组合和后处理保持原定义。

新字段只使用目标s=t+24前的48点已提供gust，包含目标时刻；阈值严格>30mph。内容是6/24小时衰减暴露、最近强风年龄、是否有强风、连续强风段数、段间最长间歇、暴露加权年龄及最近6点占比。无强风时年龄49是编码，不是观测到49小时；段数也不等于真实风暴次数。完整公式、单位和边界见[冻结协议]({OUT}/Experiment_Protocol_2026-09-21.md)。

本报告所有“相对RMSE变化”均为(W1 RMSE/W0 RMSE−1)×100%，负数改善；n为对应范围内的有效县小时数。SSE减少量=W0 SSE−W1 SSE，正数改善。不会把改善县数、相对百分比或四个RMSE的平均当成官方总成绩。

## 2. 完整四输出成绩

范围：seed42全部239县、每个horizon的完整有效目标小时，RMSE单位为OSI。

{table(['输出h','n','W0 RMSE','W1 RMSE','相对RMSE变化'],overall)}

1h/6h发生变化的目标都是96–215，来自同一P24变化经C3传播；两者SSE变化相同，只因各自完整评分范围不同而百分比不同，不能把它们当两份独立确认。目标<96的短时预测与最终48h逐行完全不变。

## 3. 不确定性与测试相关子集

下表95%区间针对ΔRMSE=W1−W0（单位OSI），由2000次县整条轨迹配对bootstrap得到。它不校正此前模型选择，也不是测试获胜概率。

测试相似子集在训练前固定：对应h/目标日被至少一测试县选入旧context前5近邻的训练县。新8列不改变成员；它不是实际测试集评分。

{table(['输出h','范围','n','相对RMSE变化','ΔRMSE 95%区间'],intervals)}

所有可变主要比较的区间均跨零；全县和测试相似子集的点估计都没有改善。没有因看到结果而改子集或挑选局部成绩替代总体。

## 4. 是否更好地区分过程阶段

范围只为24h的28,680有效行。前四组按新输入定义、互不重叠且覆盖全部行；最后一组与前四组重叠，不能加总。分组不使用真实停电结果，也没有在预测时按组选择模型。

{table(['输入定义的过程组','n','相对RMSE变化','SSE减少量'],proc_rows)}

“无强风”组和强风过去较久的组略有改善，但损失规模较小。正在强风、强风后1–6小时、以及多段强风组均略差，没有形成预期的关键阶段净改善。

平均偏差更靠近零也不保证平方误差下降：整体24h bias从约−0.000093变为−0.000067，但RMSE仍上升。本轮不能据此声称模型已准确识别恢复阶段；实际恢复进度并未进入输入。

## 5. 县、外折和目标日

下表为运行前固定关注县的24h完整有效时段。SSE减少量保留误差量级；小基数县的较大相对百分比不一定影响总体很多。

{table(['县','相对RMSE变化','SSE减少量'],focus_rows)}

Clay、Braxton、Muskingum有小幅改善，Calhoun、Roane等有损失。这没有解决原先“同类条件下有的持续低估、有的持续高估”的总体问题。全239县中{improved}县改善、{worsened}县恶化；改善县略多仍不足以降低pooled RMSE，因为各县误差量级不同。

24h外折结果（外折编号为0–4，不是CV seed）：

{table(['外折编号','n','相对RMSE变化','SSE减少量'],fold_rows)}

24h逐目标日结果（所有日期都保留，分母n为该日全县有效小时）：

{table(['目标日期','n','相对RMSE变化','SSE减少量'],day_rows)}

五折中两折改善、三折恶化；日期也有取舍。3月19日相对改善较大，但绝对SSE收益很小，不能用它替代完整时段判断。

## 6. P分量与模型是否实际使用新列

下表RMSE单位为P比例。来源P是各h独立分量来源；C3后的P是同目标小时的平均。最终1/6h用C3，最终24/48h用各自来源；因此C3后的P48变化不等于最终48h发生变化。

{table(['h','P空间','n','W0 P RMSE','W1 P RMSE','相对变化'],p_rows)}

P24来源本身RMSE上升约0.340%，C3后的短时P也略差。这次没有出现P分量改善却仅因其他分量抵消而丢失OSI收益的情况。

以下为新列在训练有效行中的输入检查。“最高旧列相关”是对183个旧字段逐一计算Pearson相关后取绝对值最大者；只是线性冗余描述，不证明独立信息量或因果作用。

{table(['新字段','不同值数量','最小–最大','最高相关旧字段','绝对Pearson相关'],feat_rows)}

8列均非常数、没有与旧列逐行完全相同。模型实际使用了7列新特征；五个外层模型合计新列占分裂gain的{gain_share:.2f}%、分裂次数的{split_share:.2f}%。gain是训练时分裂增益累计，不能解释为验证误差改善贡献。

{table(['新字段','五模型合计分裂次数','占全部字段训练gain'],usage_rows)}

has_high_gust_48h未被用于分裂；其他字段如年龄编码49、段数0也表达部分相同状态。没有据此事后删列再跑。本次结果不是特征未接入或全部被模型忽略造成的。

## 7. 验收与文件

- 训练34,416行、测试9,072行的191列特征已独立重建；新增8列最大差0。有效新特征行分别28,680、7,560；测试只构造特征，没有模型推理。
- 无风、全程强风、窗口首尾强风、两段之间间歇等边界核验通过；目标时刻之后天气扰动不改变此前目标特征；输入函数拒绝传入停电标签列。
- 旧P24五个外层模型重载一致；25个训练作用域的排除标签/特征扰动检查通过。
- 新20内层与5外层模型使用独立重建191列重载，预测最大差0；标签/内层float32指标/曲线/轮数/完整C3分别核验。
- {ver['metric_rows_checked']}条基础指标、5个过程组、1,195个县日、8条区间记录及118,783条逐行比较复算通过；未改分量、最终48h和短时目标<96精确不变。
- 旧文件哈希保持不变；所有新增产物在本目录，0张图。

入口：

- [主要比较]({OUT}/summary/main_comparisons.csv)
- [辅助分组]({OUT}/summary/auxiliary_groups.csv)
- [固定关注县]({OUT}/summary/focus_counties.csv)
- [逐行最终OSI]({RUN}/control_predictions.parquet)
- [逐行分量与来源身份]({RUN}/component_predictions.parquet)
- [P24新模型OOF]({RUN}/P24_oof_predictions.parquet)
- [模型轮数]({OUT}/summary/model_rounds.csv)
- [模型使用新列的情况]({OUT}/summary/feature_usage.csv)
- [独立验收]({RUN}/verification.json)
- [来源清单]({OUT}/summary/candidate_manifest.json)

## 8. 决策

**保留W0，停止当前固定8列W1版本，不做其他CV确认，不扩大窗口、衰减或风速阈值搜索。** 实验有效完成，但没有获得新提升；小幅退化并不构成该机制必然无效的证明。

G2仍是此前冻结、收益集中于Forest的独立候选，本轮没有把它与W1叠加，也没有新增任何替代实验。
'''
    safe(OUT/'Results_2026-09-21.md').write_text(report)
    safe(OUT/'README.md').write_text(f'''# P24目标前天气过程：seed42

已完成25次拟合与独立重载验证；[结果]({OUT}/Results_2026-09-21.md)。W1相对W0完整24h RMSE +0.131%，1h/6h +0.016%/+0.020%，48h不变。当前版本不采用、不追加确认。没有叠加G2、画图或测试推理。

首次执行顺序（已完成训练/评价拒绝覆盖）：

```bash
cd {ROOT}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/preflight.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/train_weather.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/evaluate_weather.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/verify_weather.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/summarize_results.py
```

features/保存新的8列和完整191列训练/测试输入；reference/保存批准设计和旧输入哈希；runs/保存20内层与5外层模型、收据、曲线和逐行指标；summary/保存可读汇总、来源和决策。completion.json为交付哈希。验证脚本不训练。
''')
    print('Summary saved:',decision,flush=True)

if __name__=='__main__':main()
