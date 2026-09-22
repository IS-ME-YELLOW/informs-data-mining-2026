"""Tables and text only; no plots or model operations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
H=['last_P_t','last_D_t','last_N_t','last_R_t','osi_max_72h','osi_mean_72h','osi_trend_last6h']
W=['gust_max','gust_mean','gust_hours_gt30','tp_sum','soil_mean','prior24_gust_max','prior24_tp_sum']
S=['customers_at_cutoff','pop_density','pct_forest','n_utilities']
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def write(name,d):
    p=OUT/name;assert p.resolve().is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False)
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
    m=pd.read_parquet(OUT/'data/support_metrics.parquet');x=pd.read_parquet(OUT/'data/allowed_inputs.parquet')
    names=csv(OUT/'data/county_names.csv',dtype={'fipsCode':str}).set_index('fipsCode')
    tests=csv(OUT/'summary/test_counties.csv',dtype={'fipsCode':str});allcounty=csv(OUT/'summary/all_county_support.csv',dtype={'fipsCode':str})
    changes=csv(OUT/'summary/train_addition.csv',dtype={'fipsCode':str});geo=csv(OUT/'data/geographic_support.csv',dtype={'fipsCode':str,'nearest_fips':str})
    feature_ranges=pd.read_parquet(OUT/'data/original_scale_ranges.parquet')
    context=tests[tests.view=='context'];gtest=geo[geo.scope=='test_to_train'].set_index('fipsCode')
    overview=[]
    for view,g in tests.groupby('view'):
        overview.append(dict(view=view,total_test_counties=len(g),any_d1_counties=int((g.flagged_d1_windows>0).sum()),frequent_d1_counties=int((g.flagged_d1_windows>=11).sum()),
           any_d5_counties=int((g.flagged_d5_windows>0).sum()),frequent_d5_counties=int((g.flagged_d5_windows>=11).sum()),
           any_either_counties=int(((g.flagged_d1_windows>0)|(g.flagged_d5_windows>0)).sum())))
    overview=pd.DataFrame(overview);write('summary/view_counts.csv',overview)
    selected=context[(context.flagged_d1_windows>0)|(context.flagged_d5_windows>0)].copy()
    selected=selected.merge(gtest[['geo_d1_km']],left_on='fipsCode',right_index=True,validate='one_to_one')
    write('summary/main_test_watchlist.csv',selected)
    # Explicitly describe differences at each test county's largest distance/reference-line ratio.
    explanations=[];examples=[]
    for f in selected.fipsCode:
        r=m[(m.scope=='test_to_train')&(m.view=='context')&(m.fipsCode==f)].sort_values(['d1_over_q95','horizon','target_day'],ascending=[False,True,True]).iloc[0]
        full=x[(x.horizon==r.horizon)&(x.target_day==r.target_day)];ref=full[full.dataset=='train'];q=full[full.fipsCode==f].iloc[0];b=full[full.fipsCode==r.nearest_fips].iloc[0]
        for group,columns in [('history',H),('weather',W),('static',S)]:
            for c in columns:
                vals=ref[c].to_numpy();a0,b0=q[c],b[c];a,bv=a0,b0
                if c in H:vals=np.arcsinh(vals/.01);a,bv=np.arcsinh(a/.01),np.arcsinh(bv/.01)
                elif c in ['customers_at_cutoff','pop_density','n_utilities','tp_sum','prior24_tp_sum']:vals=np.log1p(vals);a,bv=np.log1p(a),np.log1p(bv)
                sd=vals.std();sd=sd if sd>1e-12 else 1;contrib=((a-bv)/sd)**2/(3*len(columns))
                explanations.append(dict(fipsCode=f,countyName=names.loc[f,'countyName'],horizon=r.horizon,target_day=r.target_day,nearest_fips=r.nearest_fips,
                    nearest_county=names.loc[r.nearest_fips,'countyName'],group=group,feature=c,query_value=a0,nearest_value=b0,
                    squared_distance_contribution=contrib,share_d1_squared_pct=100*contrib/r.d1**2,train_min=ref[c].min(),train_max=ref[c].max()))
        examples.append(dict(fipsCode=f,horizon=r.horizon,target_day=r.target_day,d1=r.d1,q95=r.reference_d1_q95,ratio=r.d1_over_q95,nearest_fips=r.nearest_fips,
            nearest_county=names.loc[r.nearest_fips,'countyName']))
    explanations=pd.DataFrame(explanations);write('summary/test_distance_explanations.csv',explanations);write('summary/test_display_windows.csv',pd.DataFrame(examples))
    for f,g in explanations.groupby('fipsCode'):
        assert abs(g.share_d1_squared_pct.sum()-100)<1e-10
    # Relevant initial-state comparisons exclude the query itself.
    hist=x[(x.horizon==1)&(x.target_day==3)].copy();range_rows=[]
    for f in ['42053','39117','18013']:
        row=hist[hist.fipsCode==f].iloc[0];tr=hist[(hist.dataset=='train')&(hist.fipsCode!=f)];test=hist[hist.dataset=='test'];joint=hist[hist.fipsCode!=f]
        for col in H:
            range_rows.append(dict(fipsCode=f,countyName=names.loc[f,'countyName'],feature=col,value=row[col],other_train_min=tr[col].min(),other_train_max=tr[col].max(),
                test_min=test[col].min(),test_max=test[col].max(),all_other_min=joint[col].min(),all_other_max=joint[col].max(),
                remains_outside=bool(row[col]<joint[col].min() or row[col]>joint[col].max())))
    write('summary/focus_ranges_excluding_query.csv',pd.DataFrame(range_rows))
    hist=hist.merge(names[['countyName','stateAbbr']],left_on='fipsCode',right_index=True,validate='one_to_one')
    write('summary/known_state_all_counties.csv',hist[['fipsCode','countyName','stateAbbr','dataset']+H+S])
    train=allcounty[(allcounty.scope=='train_loo')&(allcounty.view=='context')].sort_values(['flagged_d1_windows','max_d1_ratio'],ascending=[False,False])
    write('summary/training_context_rank.csv',train)
    mainadd=changes[(changes.comparison=='train_loo_vs_train_plus_test')&(changes.view=='context')]
    focus=mainadd[mainadd.fipsCode.isin(['42053','39117','18013','18111','39149','42063','39091'])]
    write('summary/main_train_addition.csv',focus)
    testrows=[[r.countyName+' '+r.stateAbbr,r.fipsCode,f'{r.flagged_d1_windows}/21',f'{r.flagged_d5_windows}/21',f'{r.max_d1_ratio:.3f}',f'{r.geo_d1_km:.1f}'] for r in selected.itertuples()]
    additionrows=[]
    for f in ['42053','39117','18013','18111','39149','42063','39091']:
        r=focus[focus.fipsCode==f].iloc[0]
        added=m[(m.scope=='train_plus_test')&(m.view=='context')&(m.fipsCode==f)];new=added[added.nearest_split=='test'].nearest_fips.value_counts()
        nearest=(names.loc[new.index[0],'countyName']+f'（{int(new.iloc[0])}/21窗）') if len(new) else '最近一县未改为测试县'
        additionrows.append([r.countyName+' '+r.stateAbbr,f'{r.before_flagged}→{r.after_flagged}',f'{r.before_flagged_d5}→{r.after_flagged_d5}',f'{r.mean_distance_reduction_pct:.1f}%',nearest])
    oofrows=[]
    for f in ['42053','39117','18013']:
        row=[names.loc[f,'countyName']]
        for s in [42,20260917,20260918]:
            r=changes[(changes.comparison=='oof_before_vs_oof_plus_test')&(changes.cv_seed==s)&(changes.view=='context')&(changes.fipsCode==f)].iloc[0]
            row.append(f'{r.before_flagged}→{r.after_flagged}')
        oofrows.append(row)
    state_rows=[]
    for f in ['42053','42123','39117','39019','39021']:
        r=hist[hist.fipsCode==f].iloc[0];state_rows.append([r.countyName+' '+r.stateAbbr,r.dataset,f'{r.last_P_t:.4f}',f'{r.last_N_t:.4f}',f'{r.last_D_t:.4f}'])
    # Geographic distance is verified again using a separate great-circle expression.
    coords=csv(OUT.parent/'p_information_increment/features/v1/coordinates.csv',dtype={'fipsCode':str}).set_index('fipsCode')
    for r in geo.itertuples():
        a,b=coords.loc[r.fipsCode],coords.loc[r.nearest_fips]
        lat1,lat2=np.deg2rad(a.latitude),np.deg2rad(b.latitude);dl=np.deg2rad(a.longitude-b.longitude)
        distance=6371.0088*np.arccos(np.clip(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(dl),-1,1))
        assert abs(distance-r.geo_d1_km)<1e-7
    tgeo=geo[geo.scope=='test_to_train'];common_test=m[(m.scope=='test_to_train')&(m.view=='context')]
    d1_count=int(common_test.flag_d1.sum());d5_count=int(common_test.flag_d5.sum())
    report=f'''# 训练/测试县特征支持分析结果（2026-09-20）

**结论：测试集整体没有普遍缺少训练参照，但存在少数需要关注的条件组合。主指标中Mifflin、Allegheny、Doddridge的最近邻距离在部分或全部窗口超线；Warren能找到Forest这个近邻，但周围同类训练案例仍很少。加入测试县后Forest获得明显更近的无标签参照，Morrow只部分改善，Brown基本不变。**

本轮只使用已知历史、许可天气和静态特征；没有测试未来停电标签、模型训练、预测或图片。旧文件保持不变。两个方向的实验依据、公式、候选、作用范围和训练预算另见[具体实验设计]({OUT}/Growth_Exposure_Experiment_Design_2026-09-20.md)。

## 1. 先说明表头与“离群”的含义

主分析沿用此前context距离：历史7列、天气7列、静态4列，三组等权。地理距离单独计算，不混进特征距离。这里的离群是相对于所选特征、权重和训练参照的稀疏程度，不能直接推断真实RMSE。

所有h×目标日共21个窗口：1h/6h各6日、24h5日、48h4日。它们重叠且共用截止历史，**不是21次独立试验**。

| 本文表头 | 明确口径 |
|---|---|
| 最近1县距离超线窗口数/21 | 21窗口里，目标县d1严格大于原训练最近邻距离95%分位线的个数 |
| 第5近县距离超线窗口数/21 | 同上，但改用到第五近县的距离及其训练95%分位线，检查是否只有孤立近邻 |
| 最大d1/参考线 | 21窗口中d1除以对应q95参考线的最大值；1.23表示该窗口距离高23%，不是RMSE高23% |
| 距最近训练县/km | 县代表坐标之间的球面地理距离；和特征最近的县可以不同 |
| 前→后超线窗口数 | 添加测试县前后，按同一个固定训练尺度和阈值计数；不是重算百分位后比较 |
| 平均最近距离缩短% | 每窗口(前d1−后d1)/前d1的百分比，再对21窗口等权平均；不是预测收益 |

这些95%参考线只由训练输入决定，**与Stella的OSI预测门控q95无关**。详细公式、分母、scope及机器字段见[指标字典]({OUT}/Metric_Definitions.md)。

## 2. 测试县是否离群

参照为全239训练县；变换也只由训练县决定。下表列出context下至少一个窗口在d1或d5超线的全部5县，其余58县在这两个指标下均未超线。

{table(['测试县','FIPS','最近1县超线窗口数/21','第5近县超线窗口数/21','最大d1/参考线','距最近训练县/km'],testrows)}

按d1有3/63县在至少一窗标记、2/63县在至少半数窗口标记；按d5有4/63县在至少一窗标记，其中3县为21/21。合并两个检查共5县，不应只用最近一县的3县概括全部稀疏情况。

从县×窗口统计，d1是{d1_count}/1323（{100*d1_count/1323:.2f}%）超线，d5是{d5_count}/1323（{100*d5_count/1323:.2f}%）超线。这与“测试集到处都是离群县”不符；这些比例也不能当作独立异常率估计，因为窗口重复。

### 具体是什么不相似

- **Mifflin PA**：主指标21/21窗的d1/d5均超线。最明显窗口为6h的3月14日，最近参照Huntingdon；差异中R71（约.0427 vs .0091）、此前降水与末期OSI趋势贡献较大。各变量未必超出训练单列范围，但“低P＋较高R/D＋当前历史形态”的联合状态较少见。距最近训练县代表点仅约20km，明显不是因为地理上远离研究区域。
- **Allegheny PA**：650,536个监测客户、人口密度约1,713，与最近特征参照Westmoreland在规模/密度、历史趋势和部分天气上存在差异。d1仅部分窗口超线，d5全部超线；不能简单称为所有条件完全不可比。
- **Doddridge WV**：主d1仅9/21、d5为0/21，最明显距离只比参考线高约7.5%，属于较边缘、较弱的标记。与所选近邻比较时，N71、土壤湿度、末期趋势有差异。不能把它和Forest式极端状态等量齐观。
- **Warren PA**：主d1为0/21，因为训练集里有Forest；但d5为21/21。它的已知增长状态确有先例，却仍依赖很少的类似县，是需要保留的测试关注对象。
- **Ashtabula OH**：主d1不超线、d5仅1/21超线，属于单窗口轻度敏感，不是持续缺少参照。

每个病例采用其d1/参考线最大的窗口解释距离，明确属于输入诊断展示，不使用未来结果。各列对平方距离的贡献保存在[距离差异解释]({OUT}/summary/test_distance_explanations.csv)，不能将这些贡献解释为模型特征重要性。

## 3. 加入测试县后，训练离群县是否不再离群

本节固定全239训练县拟合的距离尺度。原参照为查询县以外的238训练县，加入后为这238县＋63测试县；阈值仍用原训练参考分布。最近距离下降有候选数量增加的机械因素，不能直接视为效果提升。

下表为原context中所有21/21窗d1超线的7个训练县，两个计数列的分母均为21。

{table(['训练县','最近1县超线数：前→后','第5近县超线数：前→后','平均最近距离缩短','成为最近邻的主要测试县'],additionrows)}

**Forest的答案是“最近一个参照明显改善，但局部案例密度仍不足”。** 它全部21窗最近邻都变为测试Warren，平均d1降低约60%，从21/21超线变为0/21；第五近邻仍是21/21超线。因此不能概括成“训练支持问题完全消失”。

**Morrow只部分改善。** 最近邻变为Champaign，但d1仍有13/21窗口超线。它的P71=.8242、D71=.9047仍高于其他训练县和全部测试县；测试最大P71来自Carroll约.5959，仍未覆盖Morrow的绝对初始幅度。

**Brown没有获得明显更近的第一参照。** d1标记仍21/21，最近距离基本不变；d5只有少量变化。新增县没有普遍补齐所有罕见条件组合。

{table(['县','数据集','已知P71','已知N71','已知D71'],state_rows)}

即使Forest与Warren接近，Forest的N71仍高于其他301县的最大值(.0558)。这没有矛盾：“超出某一列最大值”与“在整体距离上找得到足够近的一个县”是不同概念。Morrow、Brown等的逐列前后范围另存于[排除查询自身的范围表]({OUT}/summary/focus_ranges_excluding_query.csv)。

## 4. 是否仅仅因为解除了CV折排除

不是。本节重新复现原三套CV：原外折仍完全排除，只把63测试县添加到原允许训练池，变换与阈值保持该训练池原值。

表中仍是context的d1超线窗口数，分母21；每个单元格为添加测试县前→后。**本表和上一节的阈值不同，不能跨表直接相减。**

{table(['查询训练县','seed42','seed20260917','seed20260918'],oofrows)}

Forest在三套都是21→0，但d5三套都仍21→21；Morrow只部分缓解，Brown大体不变。因此Forest新增近邻不是把原来同外折的训练县加回来造成的。原30,114个OOF匹配查询的d1/d5和百分位已逐项复现。

还要区分两个方向：预测测试Warren时，最终模型可用Forest的训练标签；验证Forest时，Warren没有未来标签，不能因此自动增加有监督样本。这个不对称说明Forest的OOF困难不能原样套到Warren测试误差上，但仍不足以保证Warren预测准确。

## 5. 为什么同一地区也会有特征离群

地理上测试县到最近训练县代表点的距离范围为{tgeo.geo_d1_km.min():.1f}–{tgeo.geo_d1_km.max():.1f}km，训练最近地理距离的95%参考线约{tgeo.geo_d1_q95_km.iloc[0]:.1f}km。地理距离超线的{int(tgeo.geo_flag_d1.sum())}个测试县并不等于特征主指标超线的那3县。例如Mifflin空间上很近，却缺少相似的历史状态组合。

原因包括：城市/农村的规模与密度不同、每县采集客户分母不同、风暴经过的时间和强度不同、截止时一个县正在增长而另一个正在恢复。239县×许多小时也不等于有同样多独立县域背景；多次窗口复用相同静态和历史条件。

所以你的直觉有一部分得到支持：**总体没有大量测试县与训练完全脱节。** 但区域相连不能保证每种“历史状态×天气×背景”的组合都有很多带标签先例。距离指标也有选择性：等权匹配把某变量看得重要，不代表模型一定依赖它；不能根据距离直接删县或抬高预测。

## 6. 对距离定义是否敏感

下表分母都是63测试县。“任一”表示21窗口至少一窗；“频繁”只表示至少11窗。不同视图使用各自训练参考线，不能直接比较原始距离大小。

{table(['视图','任一窗d1超线县数/63','频繁d1超线县数/63','任一窗d5超线县数/63','频繁d5超线县数/63'],[[r.view,f'{r.any_d1_counties}/63',f'{r.frequent_d1_counties}/63',f'{r.any_d5_counties}/63',f'{r.frequent_d5_counties}/63'] for r in overview.itertuples()])}

history_only更突出Mifflin、Benton、Doddridge、Allegheny；static_only突出监测客户规模较小的Ohio County IN；天气单组的标记多为少数日期。Carroll虽是测试P71/D71最大县，在上述各视图均未出现d1/d5超线，不可把“数值大”简单等同于“没有训练支持”。

source_cache_rank突出Montour，但该县历史OSI全零，hours_since_peak按原设计缺失，本次敏感性把“无峰值”编码为−1；这个编码会影响距离。主context没有该字段，也没有将Montour标记为缺少近邻。因此不能把这一敏感性结果解释成数据异常或未来高误差。其他敏感性病例见[所有测试县、所有视图]({OUT}/summary/test_counties.csv)。

## 7. 对两个实验方向的实际影响

1. **短时增长尺度仍值得先测，但其理由是已知状态表达和稀少样本，而不是某个县码。** Warren提供了与Forest接近的测试输入例子；最终全训练模型有Forest标签可用，但只有少数增长先例，仍需严格CV检验统一规则的收益与副作用。具体建议是保留a分支、比较早期b对照与按已知增长尺度归一化/重建的b，晚期明确复用原预测。
2. **长期目标前暴露顺序仍是信息表达假设。** 同区域并不等于同阶段；现有storm_phase按日期统一赋值、累计量截至原点t，和按目标s刻画最近暴露/间歇/再增强不同。先只给P24加固定8列，保留模型和其余分量，检验完整OSI。

本次支持分析没有证明这两个候选会提分，没有生成测试预测，也不建议把测试县未知结果补成伪标签。若首轮受控实验没有实际增益，应停下该候选，而不是继续用距离解释来替代预测验证。

## 8. 验收与文件

- 6,342个训练/测试县×h×日输入独立重建；其中训练5,019、测试1,323，训练部分与旧诊断逐值一致。
- 117,033条支持指标和414,750个近邻对独立复算，训练专属变换、自排除/整折排除和添加前后固定阈值通过。地理距离另用独立球面公式核对。
- 所有源文件哈希保持不变；0次模型拟合、0次模型重载、0张图。输出仅在本目录。
- [完整指标字典]({OUT}/Metric_Definitions.md)
- [所有测试县汇总]({OUT}/summary/test_counties.csv)
- [按horizon分别统计]({OUT}/summary/by_horizon.csv)
- [训练县添加前后结果]({OUT}/summary/train_addition.csv)
- [全量最近五县]({OUT}/data/nearest_five.parquet)
- [地理距离]({OUT}/data/geographic_support.csv)
- [具体实验设计]({OUT}/Growth_Exposure_Experiment_Design_2026-09-20.md)
- [独立校验]({OUT}/verification.json)
'''
    (OUT/'Results_2026-09-20.md').write_text(report)
    (OUT/'README.md').write_text(f'''# 训练/测试县特征支持

结果见[报告]({OUT}/Results_2026-09-20.md)，新指标见[字典]({OUT}/Metric_Definitions.md)，模型方向见[具体设计]({OUT}/Growth_Exposure_Experiment_Design_2026-09-20.md)。本轮没有训练或画图。

首次执行命令（已完成的主分析拒绝覆盖）：

```bash
cd {ROOT}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/train_test_feature_support/analyze_support.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/train_test_feature_support/verify_support.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/train_test_feature_support/summarize_support.py
```

data保存只含允许信息的输入和全量支持/近邻表；summary保存可读汇总和距离解释；reference保存输入与设计来源哈希。completion.json为最终交付身份。统计变换仅由训练参照决定，测试未来结果未提供且未使用。
''')
    print('Report generated; test watchlist',selected.fipsCode.tolist(),flush=True)

if __name__=='__main__':main()
