"""Summarize the completed seed42 comparison and freeze its reviewable delivery."""
from nr_core import *
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    if (OUT/'completion.json').exists():raise RuntimeError('Completed delivery is immutable')
    ver=read(RUN/'verification.json');assert ver['status']=='PASS';manifest=read(RUN/'run_manifest.json');models=read(RUN/'model_manifest.json')
    primary=csv(RUN/'metrics/primary_scores.csv');comparisons=csv(RUN/'metrics/comparisons.csv');boots=csv(RUN/'metrics/bootstrap.csv')
    counties=csv(RUN/'metrics/county_metrics.csv',dtype={'fipsCode':str});groups=csv(RUN/'metrics/group_metrics.csv');components=csv(RUN/'metrics/component_metrics.csv')
    folds=csv(RUN/'metrics/fold_metrics.csv');days=csv(RUN/'metrics/day_metrics.csv');cls=csv(RUN/'metrics/classification.csv',dtype={'fold':str});amps=csv(RUN/'metrics/amplitude_diagnostics.csv')
    comp=norm(pd.read_parquet(RUN/'component_predictions.parquet'));oof=norm(pd.read_parquet(RUN/'NR1_oof_predictions.parquet'))
    for name,d in [('main_comparisons',comparisons),('bootstrap',boots),('auxiliary_groups',groups),('component_metrics',components),('focus_counties',counties[counties.fipsCode.isin(FOCUS)])]:frame(OUT/f'summary/{name}.csv',d)
    rounds=[]
    for c in ['N_t','R_t']:
        for outer in range(5):
            for head in ['q','m']:
                rr=[r for r in models if r['spec']['component']==c and r['spec']['head']==head and r['spec']['outer_fold']==outer]
                probes=sorted([r for r in rr if r['spec']['inner_fold'] is not None],key=lambda r:r['spec']['inner_fold']);ref=next(r for r in rr if r['spec']['inner_fold'] is None)
                rounds.append(dict(component=c,head=head,outer_fold=outer,inner_best_iterations=','.join(str(r['best_iteration']) for r in probes),
                    requested_refit_rounds=ref['spec']['requested_rounds'],saved_trees=ref['actual_trees'],train_rows=ref['spec']['train']['rows'],fit_seconds=sum(r['fit_seconds'] for r in rr)))
    frame(OUT/'summary/model_rounds.csv',rounds)
    concentration=[];top=[]
    for (case,h),g in counties[(counties.reference=='B0')&(counties.horizon<=6)].groupby(['case','horizon']):
        net=g.sse_reduction.sum();gain=g.loc[g.sse_reduction>0,'sse_reduction'].sum();loss=-g.loc[g.sse_reduction<0,'sse_reduction'].sum();forest=g.loc[g.fipsCode=='42053','sse_reduction'].iloc[0]
        best=g.sort_values('sse_reduction',ascending=False).head(5);worst=g.sort_values('sse_reduction').head(5)
        assert abs((gain-loss)-net)<1e-12
        concentration.append(dict(case=case,horizon=h,improved_counties=int((g.sse_reduction>1e-12).sum()),worsened_counties=int((g.sse_reduction< -1e-12).sum()),
            unchanged_counties=int((abs(g.sse_reduction)<=1e-12).sum()),gross_gain= gain,gross_loss=loss,net_sse_reduction=net,
            forest_sse_reduction=forest,forest_share_net=forest/net if net>0 else np.nan,top5_gain=best.sse_reduction.sum(),top5_share_gross_gain=best.sse_reduction.sum()/gain))
        for kind,d in [('largest_gains',best),('largest_losses',worst)]:
            for rank,r in enumerate(d.itertuples(),1):top.append(dict(case=case,horizon=h,kind=kind,rank=rank,fipsCode=r.fipsCode,countyName=r.countyName,stateAbbr=r.stateAbbr,sse_reduction=r.sse_reduction,rmse_change_pct=r.rmse_change_pct))
    frame(OUT/'summary/benefit_concentration.csv',concentration);frame(OUT/'summary/top_county_changes.csv',top)
    tail=[]
    for c in ['N_t','R_t']:
        hit=oof['true_'+c]>.01;truth=oof.loc[hit,'true_'+c];old=comp.loc[hit,f'pred_B0_{c}_target_t01h'];new=oof.loc[hit,c+'_product']
        tail.append(dict(component=c,n=int(hit.sum()),truth_mean=truth.mean(),old_source_mean=old.mean(),new_clipped_m_mean=oof.loc[hit,c+'_clipped_m'].mean(),
            new_product_mean=new.mean(),mean_q=oof.loc[hit,c+'_q'].mean(),old_bias=(old-truth).mean(),new_bias=(new-truth).mean()))
    frame(OUT/'summary/tail_amplitude.csv',tail)
    save(OUT/'summary/source_lineage.json',dict(reference_directory=str(TREE),reference_tree_id=TREE_ID,reference_case='NB_P24',cases=CASES,
        changed_sources=dict(B0=[],N_only=['N1'],R_only=['R1'],NR_both=['N1','R1']),feature_count=163,
        fixed_sources='all P/D plus N6/N24/N48/R6/R24/R48',final_unchanged_horizons=[24,48],G2_applied=False,W1_applied=False,
        C3='original county-target-hour equal component mean',product_formula='q*clip(m,0,1)',outer_model_ids=[r['model_id'] for r in models if r['spec']['inner_fold'] is None]))
    save(OUT/'summary/decision.json',dict(status='seed42_complete_not_adopted',reference_remains='B0_NB_P24',best_point_estimate='N_only',
        recommendation='Stop this fixed two-stage version; preserve direct N/R, do not automatically confirm other CVs',
        reason='N-only final1h gain is 0.04963 percent with CI crossing zero; both source tail errors worsen; R and joint do not justify additional heads',
        inference_limit='Not proof that all two-stage models or N/R information improvements are ineffective',formal_fits=100,new_outer_models=20,saved_inner_models=80,
        confirmation_started=False,parameter_search_started=False,test_inference=False,plots=0))
    save(OUT/'summary/candidate_manifest.json',dict(protocol='v18_nr2stage_v1',cv_seed=42,model_seed=42,identity_hash=ver['identity_hash'],
        run_directory=str(RUN),cases=CASES,adopted=False,files={n:dict(path=str(RUN/n),sha256=sha(RUN/n)) for n in ['control_predictions.parquet','component_predictions.parquet','aligned_components.parquet','NR1_oof_predictions.parquet','model_manifest.json','verification.json']}))
    short=comparisons[(comparisons.reference=='B0')&(comparisons.horizon<=6)]
    overall=[[r.case,r.horizon,int(r.n),f'{r.baseline_rmse:.12f}',f'{r.rmse:.12f}',f'{r.rmse_change_pct:+.5f}%'] for r in short.itertuples()]
    for h in [24,48]:
        r=primary[(primary.case=='B0')&(primary.horizon==h)].iloc[0];overall.append(['全部候选',h,int(r.n),f'{r.rmse:.12f}',f'{r.rmse:.12f}','0（逐行不变）'])
    ctable=[]
    cm=components[(components.reference=='B0')&(components.horizon==1)&(components.group=='all')]
    for c,case in [('N_t','N_only'),('R_t','R_only')]:
        for space in ['source','aligned']:
            r=cm[(cm.component==c)&(cm.case==case)&(cm.space==space)].iloc[0];ctable.append([c[0],'1h来源' if space=='source' else '最终1h采用的C3平均',int(r.n),f'{r.baseline_rmse:.9f}',f'{r.rmse:.9f}',f'{r.rmse_change_pct:+.4f}%'])
    group_labels={'zero':'Z=0','small_positive':'0<Z≤0.01','tail_gt_001':'Z>0.01'};sg=[]
    for c,case in [('N_t','N_only'),('R_t','R_only')]:
        for group in group_labels:
            r=components[(components.reference=='B0')&(components.horizon==1)&(components.component==c)&(components.case==case)&(components.space=='source')&(components.group==group)].iloc[0]
            sg.append([c[0],group_labels[group],int(r.n),f'{100*r.n/34177:.3f}%',f'{100*r.baseline_group_sse_share:.2f}%',f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
    clsrows=[]
    for c in ['N_t','R_t']:
        g=cls[(cls.component==c)&(cls.fold=='all')].set_index('variant');q=g.loc['q_model'];p=g.loc['training_prior']
        clsrows.append([c[0],f'{q.observed_positive_fraction:.4f}',f'{q.mean_q:.4f}',f'{p.logloss:.4f} → {q.logloss:.4f}',f'{p.brier:.4f} → {q.brier:.4f}'])
    tailrows=[[r['component'][0],r['n'],f"{r['truth_mean']:.6f}",f"{r['old_source_mean']:.6f}",f"{r['new_clipped_m_mean']:.6f}",f"{r['mean_q']:.4f}",f"{r['new_product_mean']:.6f}"] for r in tail]
    br=boots[(boots.reference=='B0')&(boots.horizon<=6)]
    intervalrows=[[r.case,r.horizon,'全县' if r.group=='all' else '测试相似子集',int(r.n),f'{r.rmse_change_pct:+.5f}%',f'[{r.ci_low:+.9f}, {r.ci_high:+.9f}]'] for r in br.itertuples()]
    grouprows=[[r.case,r.horizon,'去Forest全县' if r.group=='all_except_Forest' else '测试相似子集去Forest',int(r.n),f'{r.rmse_change_pct:+.5f}%',f'{r.sse_reduction:+.6f}'] for r in groups[(groups.reference=='B0')&(groups.horizon<=6)&groups.group.isin(['all_except_Forest','test_like_except_Forest'])].itertuples()]
    concentration_rows=[[r['case'],r['horizon'],r['improved_counties'],r['worsened_counties'],f"{r['gross_gain']:.6f}",f"{r['gross_loss']:.6f}",f"{r['net_sse_reduction']:+.6f}",f"{r['forest_sse_reduction']:+.6f}"] for r in concentration]
    focus=[]
    for f in FOCUS:
        g=counties[(counties.reference=='B0')&(counties.horizon==1)&(counties.fipsCode==f)].set_index('case');r=g.loc['N_only']
        focus.append([r.countyName+' '+r.stateAbbr,f,f'{r.rmse_change_pct:+.4f}%',f'{g.loc["N_only","sse_reduction"]:+.6f}',f'{g.loc["R_only","sse_reduction"]:+.6f}',f'{g.loc["NR_both","sse_reduction"]:+.6f}'])
    foldrows=[]
    for ff in range(5):
        g=folds[(folds.reference=='B0')&(folds.horizon==1)&(folds.fold==ff)].set_index('case');foldrows.append([ff,int(g.loc['N_only','n'])]+[f'{g.loc[c,"rmse_change_pct"]:+.4f}%' for c in CASES[1:]])
    dayrows=[]
    for day in sorted(days.target_day.unique()):
        g=days[(days.reference=='B0')&(days.horizon==1)&(days.target_day==day)].set_index('case');dayrows.append([g.loc['N_only','target_date'],int(g.loc['N_only','n'])]+[f'{g.loc[c,"rmse_change_pct"]:+.4f}%' for c in CASES[1:]])
    seconds=sum(r['fit_seconds'] for r in models)
    report=f'''# N1/R1两阶段预测首轮结果（2026-09-21）

**结论：本次固定两阶段结构没有足够的保留理由，建议继续使用原N/R。** seed42只换N的完整1h/6h RMSE改善0.04963%/0.02437%；只换R分别退化0.00613%/0.01695%；联合替换没有超过只换N。全部主要短时比较的县bootstrap区间跨零。不是“完全没有变化”，而是当前收益规模和证据不足以支持增加模型复杂度、继续另两CV确认。

实验已有效完成：100次真实拟合、80个内层和20个外层检查点；独立重载验收PASS，预测最大差0。各头拟合累计约{seconds/60:.1f}分钟（不含预检/评价/验收）。最终24/48h逐行不变，旧文件未改，没有模型测试推理、全量拟合、提交或图片。

## 1. 改动范围与指标说明

B0=此前冻结的完整NB_P24树方案：P1 a/b+F2、P6 a/b、P24邻县信息、原P48、D_G及原N/R。未叠加G2/W1。N_only只换N1来源，R_only只换R1来源，NR_both同时替换；其余来源原样保留。

N1/R1均覆盖1h来源模型的全部有效目标小时73–215，而非仅第一小时。每个分量使用相同163列，分别拟合q=Pr(Z>0|X)和正值幅度m，输出q×clip(m,0,1)。无硬分类阈值、无类别重加权、无新增特征；N与R独立且允许同时正值。这里的“发生”只指官方平滑分量Z>0，不等价于客户级故障或修复事件。

县级嵌套CV：每外折的县完全排除；概率内层头按logloss选轮数，幅度内层头仅在正值训练行拟合，但按**全部内层验证行**的q×clip(m)分量RMSE选轮数。q来自同一outer/inner的概率模型，绝无全局OOF删折替代。外层重拟合轮数是四个内层最佳轮数均值向下取整。详细约定见[冻结协议]({OUT}/Experiment_Protocol_2026-09-21.md)。

下表统一口径：n=所述范围内的有效县小时数；相对RMSE变化=(候选/同表参考−1)×100%，负数改善；ΔRMSE=候选−参考，单位OSI或对应分量比例；SSE减少量=参考SSE−候选SSE，正数改善，单位为对应数值平方的县小时累计。bias=预测−真实，负数低估。各h使用完整有效时段，不把百分比平均当成比赛总分。

## 2. 完整四输出成绩

全部239县，参考都是B0。RMSE单位为OSI。

{table(['候选','最终输出h','n','B0 RMSE','候选RMSE','相对RMSE变化'],overall)}

最终1/6h使用同县同目标小时C3分量平均，因此N1/R1变化会传播到最终6h；最终24/48h使用自己的分量来源，精确不变。诊断文件中的C3平均N24/N48可能变化，但它们不用于这两个最终输出，不能误读为长时预测变化。

联合相对只换N，1h/6h分别再退化0.00281%/0.01289%；本轮没有N/R同时换的额外净优势。1h和6h共享不少目标小时及来源变化，不能把它们当成两次独立确认。

## 3. 为什么收益有限：背景更干净，较大值仍被低估

先看1h分量的完整RMSE。来源=该h模型直接输出；C3平均=同县同目标时刻各可用h来源平均，也是最终1h所用的分量。联合候选各分量与相应单独候选相同，因此不重复列。

{table(['分量','预测空间','n','旧RMSE','新RMSE','相对变化'],ctable)}

N来源本身微差、C3平均微好；R来源微好、C3平均反而更差。来源RMSE与组合RMSE不等价，因为各来源误差有相关性；分量RMSE也不能直接替代最终OSI指标。本次不能简化为“R准确了但只因抵消关系变差”，因为最终使用的C3 R本身也变差了。

下面按**真实官方分量**分组，只诊断来源N1/R1，三组互斥且覆盖34177行。行占比的分母为34177；旧分量SSE占比的分母是该分量旧1h来源全部行SSE；分组是事后诊断，不用于预测门控或训练加权。

{table(['分量','真实值组','n','行占比','占旧分量来源SSE','组内RMSE变化','组内SSE减少量'],sg)}

N的零值/小正值两组合计减少约0.012436 SSE，却被较大值组增加约0.012661 SSE抵消；R相应减少约0.006350、增加约0.004782。两阶段确实降低了背景误差，**但没有解决占分量SSE主体的较大幅度误差**。C3后较大值组RMSE也分别退化约0.391%/0.363%。

概率头并非完全无效。下表使用完整1h的34177行；“常数→q”参考为每外折允许训练县的正值比例，在外层县上作常数预测。logloss是二元交叉熵（自然对数），Brier为平均(q−1[Z>0])²，均越小越好；它们衡量正值概率，不能代替OSI RMSE。

{table(['分量','真实正值比例','平均q','logloss：常数→q','Brier：常数→q'],clsrows)}

较大值组的幅度如下，数值均是该组逐行均值，单位分量比例；平均q无量纲。最后一列是逐行q×clip(m)再平均，**不是表中平均q与平均m的乘积**。

{table(['分量','Z>0.01行数','真实均值','旧来源预测均值','新clip(m)均值','平均q','新乘积均值'],tailrows)}

在这些行上，新幅度头单独输出比旧来源均值更高，但仍远低于实际；乘q后再次收缩，新最终来源均值甚至低于旧模型。这里的真实值筛选只用于解释，不能据此在推理时挑选这些行。未尝试取消q、改硬阈值或按本表重新选模型。

这支持的有限结论是：**仅用原163列，把正值概率和正值均值分开，主要带来背景清理；没有获得更可靠的大幅度表达。** 不能由此证明N/R不存在可利用信号，也不能把潜在收益全归因于概率分解，因为本次同时增加了模型容量。

## 4. 不确定性及固定测试相似子集

95%区间是2000次县整条轨迹配对bootstrap的ΔRMSE百分位区间，单位OSI；不是单小时独立抽样，不是测试获胜概率，也没有校正此前的模型选择。

测试相似子集沿用旧context度量：按输出h和目标日，被至少一测试县选入前5输入近邻的训练县并集。成员在训练前冻结，未用测试标签、未重选、未调整训练权重；此列不是实际测试评分。

{table(['候选','输出h','评分范围','n','相对RMSE变化','ΔRMSE的95%区间'],intervalrows)}

全部上述区间跨零。N_only虽为本轮最佳点估计，其1h RMSE绝对减少仅约0.00000538，不能据此宣布稳定泛化提升。

去Forest只作稳健性诊断，以下仍以同组B0为参考。

{table(['候选','输出h','评分范围','n','相对RMSE变化','SSE减少量'],grouprows)}

N_only的1h收益不依赖Forest：去掉Forest后反而改善0.07346%。但它的6h收益更依赖Forest；测试相似子集去Forest只改善约0.00037%。联合6h去Forest则退化。这与此前G2“几乎只改善Forest”不同，仍不足以使本轮微小收益成为必留项。

## 5. 县、折和日期的取舍

下表按完整县轨迹汇总最终OSI。改善/恶化县数以SSE减少量大于/小于±1e−12判定；收益总额=所有改善县SSE减少量之和，损失总额=所有恶化县SSE增加量之和（正数）；净收益=前者−后者。它们不是县RMSE的平均。

{table(['候选','h','改善县数','恶化县数','收益总额','损失总额','净SSE减少量','Forest的SSE减少量'],concentration_rows)}

N_only 1h改善143县、恶化96县；收益最大的县是Venango PA、Cameron PA、Monroe OH等，且被其他县损失抵消一部分。收益前5县名单属于事后描述，见top_county_changes.csv，未据此改变固定关注县或模型。benefit_concentration.csv另存top5_share_gross_gain=五个最大改善县收益/全部改善县收益；forest_share_net=Forest收益/全县净收益，仅全县净收益为正时给值，可能小于0或大于1，不能当概率。

以下是**运行前固定关注县**的完整1h表现。“N相对变化”是N_only最终OSI RMSE相对B0变化；后三列均为各候选最终OSI的SSE减少量。

{table(['县','FIPS','N相对RMSE变化','N_only ΔSSE','R_only ΔSSE','NR_both ΔSSE'],focus)}

外折编号0–4，与CV seed不同。下表全部为完整1h相对B0 RMSE变化，分母n为该外折县小时数。

{table(['外折','n','N_only','R_only','NR_both'],foldrows)}

逐目标日结果如下，全部239县，完整1h；首日从目标小时73开始，因此有23小时，其余每天24小时。所有日期保留，不按结果挑日。

{table(['目标日期','n','N_only RMSE变化','R_only RMSE变化','NR_both RMSE变化'],dayrows)}

N_only第一日改善0.12286%，后续目标96–215合计退化0.02516%；没有因这个结果新增“只在首日用两阶段”的后验候选。其他h、所有县、MAE/bias/SSE和联合对单独的完整比较均保存在metrics文件中。

## 6. 验收与产物

- 预检：旧N1/R1共10模型重载一致；100个训练作用域的排除行标签/特征扰动检查通过；所有概率训练作用域有两类、幅度训练作用域有正值。
- 正式训练100次，保存80内层、20外层；40条内层q→m依赖逐一核验，同outer/inner、逐行验证q、模型哈希和标签作用域均匹配。
- 独立读取原163列和原始DM_Train官方标签，重载全部新模型；预测最大差0。完整验证曲线最佳轮数、外层均值轮数、正值训练/全部行验证和float64乘积指标通过。
- 独立按县/目标小时重建C3，核验全部控制模式；复算{ver['metric_rows_checked']}条指标、{ver['bootstrap_rows_checked']}条bootstrap记录、{ver['row_comparisons_checked']}条逐行比较；分类、幅度、误差变化分解通过。
- P/D及N/R其他h来源精确不变；最终24/48h精确不变；固定测试相似成员一致；{ver['source_files_unchanged']}份引用旧文件哈希未变。

主要入口：

- [主要比较]({OUT}/summary/main_comparisons.csv)
- [分量分组指标]({OUT}/summary/component_metrics.csv)
- [收益集中度]({OUT}/summary/benefit_concentration.csv)
- [固定关注县]({OUT}/summary/focus_counties.csv)
- [新N1/R1逐行q、原始m、截断m、乘积及身份]({RUN}/NR1_oof_predictions.parquet)
- [四候选逐行最终OSI和其他控制模式]({RUN}/control_predictions.parquet)
- [分量来源预测及身份]({RUN}/component_predictions.parquet)
- [独立验收]({RUN}/verification.json)
- [来源清单]({OUT}/summary/candidate_manifest.json)
- [字段说明]({OUT}/Metric_Dictionary.md)

## 7. 决策与下一步边界

**保留B0的原N/R；本固定两阶段版本不纳入主线，不自动追加另两CV或阈值/尾部权重搜索。** N_only作为已完成的微收益对照留档，不将seed42的一次微小点估计写成稳健提升。若将来有新的N/R信息或明确的新机制，可另立协议；本结果不要求为了四分量外观对称而给N/R增加结构。

本轮为“为什么P采用结构化模型、N/R仍采用直接树”提供了实验证据：改进策略由目标误差结构及最终RMSE验证决定，模型头不必形式一致。它也提示后续若重开N/R，问题应落在较大幅度信号与可泛化表达上，而不是只把零/非零分类做得更漂亮。
'''
    atomic(OUT/'Results_2026-09-21.md',lambda p:p.write_text(report))
    dictionary='''# 字段口径

所有CSV为无图表的数值结果。`case`是候选，`reference`是本行参考（不一定B0）；`pair`=case_minus_reference。`horizon`为最终输出或分量来源h（小时），具体以文件及space为准。`fold`为seed42外折0–4。`fipsCode`应按五位字符串读取。CSV复算建议`float_precision="round_trip"`。

- `n`：本行分组内真实标签有限的县小时数。有效目标截止215，四h全县n分别34177/32982/28680/22944。
- `rmse/sse/mae/bias`：候选在对应范围的指标；bias=预测−真实。`baseline_rmse`是同组reference的RMSE。`rmse_delta`=候选−参考；`rmse_change_pct`=100×(候选RMSE/参考RMSE−1)，负数改善。`sse_reduction`=参考SSE−候选SSE，正数改善。空组指标为NaN、SSE=0。
- `primary_scores/comparisons/group_metrics/county_metrics/fold_metrics/day_metrics/row_changes`均评价最终v18_rule OSI；`all_controls`另列C0/C1/C2/C3。1/6h最终用C3，24/48h最终用C1。
- `component_metrics`：单位为N或R比例。`space=source`指各h独立来源，`aligned`指按同县同目标小时C3平均。group=`zero/small_positive/tail_gt_001`分别Z=0、0<Z≤.01、Z>.01；互斥。`positive`是后两组并集，与它们不能相加。`baseline_group_sse`是本组参考SSE；`baseline_group_sse_share`分母为同component/space/horizon/reference的全部有效行参考SSE。
- `test_like`：冻结旧context下，按h/目标日被至少一测试县选为前5近邻的训练县并集；分组是输入定义，不用测试标签。`test_votes`是选中该训练县的不同测试县数，不作训练权重。
- `bootstrap`：ci_low/ci_high为候选−参考RMSE（绝对量纲）的95%区间，2000次县轨迹配对重采样，seed20260910。完全不变的输出区间为[0,0]、replicates=0。未校正历史模型选择，不是测试胜率。
- `classification`：binary真值=1[官方Z>0]；q_model是概率头，training_prior是允许四训练折发生率常数。logloss使用自然对数，数值裁剪[1e−15,1−1e−15]仅防log(0)；Brier=mean((q−真值)²)。两者越低越好。`fold=all`是全部外层OOF汇总。
- `calibration`：固定概率箱[0,.1),...,[.9,1]，n为箱内行数；mean_q与observed_positive_fraction比较校准。均不用于事后校准或阈值选择。
- `amplitude_diagnostics`：在group指定行上求true/q/raw_m/clipped_m/product均值。`conditional_m_rmse`为clip(m)直接对真实Z的组内RMSE，只有正值组才可解释为条件幅度诊断；all/zero只是反事实“不给q缩放”诊断，不能当部署模型成绩。raw_m_below_zero/above_one是该组原始幅度越界行数。模型输出始终q×clip(m)。
- `OSI_change_decomposition`：只针对最终1/6h，令旧未后处理有符号组合误差e=.4P+.35N+.25D−.1R−官方OSI，δ=.35ΔN−.1ΔR。old_error_cross_reduction=−2Σeδ；change_squared_reduction=−Σδ²；两者和是latent_sse_reduction。N_R_change_interaction=−2Σ(.35ΔN)(−.1ΔR)，已经包含在−Σδ²内，不能再加一次。postprocess_remainder=实际最终OSI SSE减少量−latent减少量，含非负/上界/.001后处理的影响；不是额外模型贡献。
- `benefit_concentration`：gross_gain=所有改善县ΔSSE之和，gross_loss=所有恶化县−ΔSSE之和，net=两者差。top5_share_gross_gain分母为gross_gain；forest_share_net仅net>0时计算，可负或超过1。县改进判定容差±1e−12仅用于计数，SSE汇总保留原始浮点数。`top_county_changes`的排名由本次结果产生，仅用于描述。
- `NR1_oof_predictions`：*_q为概率，*_m为原始幅度，*_clipped_m为[0,1]幅度，*_product为逐行乘积；*_prior_q为该外折训练四折的发生率；两个model_id及product_id提供来源。无效尾行均NaN/空身份。`target_hour=hour_idx+1`。
- `source_*`列中的新product_id由q、m两个模型身份及公式生成；旧分量身份从NB_P24逐行原样继承。它不是单个模型文件名。
'''
    atomic(OUT/'Metric_Dictionary.md',lambda p:p.write_text(dictionary))
    readme=f'''# N1/R1两阶段对照（seed42）

已完成100次拟合与独立验收；结论见[Results_2026-09-21.md]({OUT}/Results_2026-09-21.md)。不纳入主线、不自动做其他CV。

训练前协议、脚本、预检、源哈希、固定分组、运行产物及报告均位于本目录；旧文件未改。没有图、测试推理、全量拟合或提交。

执行顺序（项目根目录，已完成的训练/评价/汇总禁止覆盖；中断训练可--resume）：

```bash
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/preflight.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/train_nr.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/evaluate_nr.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/verify_nr.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/summarize_results.py
```

`reference/`冻结来源哈希及测试相似成员；`preflight/`保存10旧模型重载和100作用域检查；`runs/v18_nr2stage_v1_split42_model42/models/{{N_t,R_t}}/outer{{0..4}}/`含80内层+20外层模型、receipt、完整验证预测。`summary/`含主表、决策、来源清单。字段定义见[Metric_Dictionary.md]({OUT}/Metric_Dictionary.md)。

每阶段完成标记核对文件哈希。根completion.json冻结最终交付；不要重写已验收脚本/模型/报告后仍声称同一身份。复核可只读检查completion.json所有哈希并重新运行验证器；验证输出内容确定不变。
'''
    atomic(OUT/'README.md',lambda p:p.write_text(readme))
    # Check quoted headline conclusions against accepted data before freezing.
    index=comparisons.set_index(['case','reference','horizon'])
    assert index.loc[('N_only','B0',1),'rmse_change_pct']<0 and abs(index.loc[('N_only','B0',1),'rmse_change_pct']+.049630)<1e-6
    assert ((br.ci_low<0)&(br.ci_high>0)).all()
    assert ver['maximum_prediction_difference']==0 and ver['final24_48_exact']
    for p,h in read(OUT/'reference/source_hashes.json').items():assert sha(p)==h,p
    save(RUN/'VERIFIED_COMPLETE',dict(identity_hash=ver['identity_hash'],verification_sha256=sha(RUN/'verification.json'),
        verification_code_sha256=sha(OUT/'verify_nr.py'),model_stage_sha256=sha(RUN/'MODEL_CV_COMPLETE'),evaluation_stage_sha256=sha(RUN/'EVALUATION_COMPLETE')))
    save(OUT/'completion.json',dict(status='COMPLETE',identity_hash=ver['identity_hash'],decision='keep_B0_stop_fixed_two_stage',formal_fits=100,
        all_new_files_confined_to=str(OUT),files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='completion.json' and p.suffix!='.partial'}))
    print('Report and immutable delivery complete',flush=True)
if __name__=='__main__':main()
