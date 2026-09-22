"""Generate transparent case evidence and a bounded research recommendation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
def csv(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def write(name,d):
    p=OUT/name;assert p.resolve().is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False)
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'input_verification.json').read_text())['status']=='PASS'
    d=pd.read_parquet(OUT/'data/route_rows.parquet');case=pd.read_parquet(OUT/'data/case_trajectories.parquet')
    overall=csv(OUT/'summary/overall_metrics.csv');county=csv(OUT/'summary/county_metrics.csv',dtype={'fipsCode':str})
    days=csv(OUT/'summary/county_day_metrics.csv',dtype={'fipsCode':str});selection=csv(OUT/'summary/case_selection.csv',dtype={'fipsCode':str})
    sc=csv(OUT/'summary/stella_county_diagnostics.csv',dtype={'fipsCode':str})
    support=csv(OUT/'matching/support.csv',dtype={'fipsCode':str});cs=csv(OUT/'matching/control_summary.csv',dtype={'fipsCode':str})
    controls=pd.read_parquet(OUT/'matching/ranked_controls_with_outcomes.parquet')
    names=d[['fipsCode','countyName','stateAbbr']].drop_duplicates().set_index('fipsCode')
    concentrations=[];short=[]
    for r in overall.itertuples():
        g=county[(county.cv_seed==r.cv_seed)&(county.horizon==r.horizon)]
        if r.horizon>=24:
            f=selection[(selection.horizon==r.horizon)&(selection.reason=='common_top5')].fipsCode
            sources=sc[(sc.cv_seed==r.cv_seed)&(sc.horizon==r.horizon)]
            concentrations.append(dict(cv_seed=r.cv_seed,horizon=r.horizon,T_rmse=r.T_rmse,S_rmse=r.S_rmse,
                common_score=r.common_score,common_over_T_sse_pct=100*r.common_score/r.T_sse,
                both_under_share_common_pct=100*r.under_common_score/r.common_score,
                top5_T_sse_share_pct=100*g[g.fipsCode.isin(f)].T_sse.sum()/r.T_sse,
                all6_under_share_common_pct=100*sources.common_score_all6_under.sum()/r.common_score,
                gate_share_common_pct=100*sources.common_score_gate.sum()/r.common_score))
        else:
            short.append(dict(cv_seed=r.cv_seed,horizon=r.horizon,T_rmse=r.T_rmse,
                forest_sse_share_pct=100*g.loc[g.fipsCode=='42053','T_sse'].iloc[0]/r.T_sse,
                forest_clay_calhoun_share_pct=100*g[g.fipsCode.isin(['42053','54015','54013'])].T_sse.sum()/r.T_sse))
    con=pd.DataFrame(concentrations);short=pd.DataFrame(short);write('summary/concentration.csv',con);write('summary/short_focus.csv',short)
    # Each case uses its largest-SSE fixed calendar day from seed42, explicitly post-hoc.
    cases=[('42053',1),('54015',24),('54013',24),('39115',24),('18013',24),('39119',24),('54007',48),('54041',48),('39117',1),('42131',6)]
    cards=[];selected_windows=[]
    for f,h in cases:
        win=days[(days.cv_seed==42)&(days.fipsCode==f)&(days.horizon==h)].sort_values(['T_sse','target_day'],ascending=[False,True]).iloc[0]
        day=int(win.target_day);selected_windows.append(dict(fipsCode=f,horizon=h,target_day=day,selection='seed42_largest_T_SSE_fixed_day_for_display_only'))
        for s in [42,20260917,20260918]:
            g=d[(d.cv_seed==s)&(d.horizon==h)&(d.fipsCode==f)&(d.target_day==day)]
            for variant in ['context','source_cache_rank']:
                su=support[(support.cv_seed==s)&(support.horizon==h)&(support.fipsCode==f)&(support.target_day==day)&(support.variant==variant)].iloc[0]
                co=cs[(cs.cv_seed==s)&(cs.horizon==h)&(cs.fipsCode==f)&(cs.target_day==day)&(cs.variant==variant)].iloc[0]
                cards.append(dict(cv_seed=s,fipsCode=f,countyName=names.loc[f,'countyName'],stateAbbr=names.loc[f,'stateAbbr'],horizon=h,target_day=day,
                    date=str((pd.Timestamp('2026-03-11')+pd.Timedelta(days=day)).date()),variant=variant,
                    true_peak=g.truth.max(),T_peak=g['T'].max(),S_peak=g.S.max(),T_mean=g['T'].mean(),S_mean=g.S.mean(),
                    true_mean=g.truth.mean(),P_true_mean=g.P_true.mean(),P_pred_mean=g.P_pred.mean(),
                    nearest_percentile=su.nearest_percentile,low_loss_controls=co.low_loss_controls,high_loss_controls=co.high_loss_controls,
                    control_max_peak=co.control_max_osi_peak,control_max_P_mean=co.control_max_P_mean))
    cards=pd.DataFrame(cards);write('summary/case_evidence_cards.csv',cards);write('summary/display_windows.csv',pd.DataFrame(selected_windows))
    display=pd.DataFrame(selected_windows)[['fipsCode','horizon','target_day']]
    focus_controls=controls.merge(display,on=['fipsCode','horizon','target_day'],validate='many_to_one')
    focus_controls['control_countyName']=focus_controls.matched_fips.map(names.countyName)
    focus_controls['control_stateAbbr']=focus_controls.matched_fips.map(names.stateAbbr)
    write('summary/focus_ranked_controls.csv',focus_controls)
    write('figures/trajectory_data.csv',case)
    write('figures/plot_cases.csv',pd.DataFrame(cases,columns=['fipsCode','horizon']))
    # Simple data-source consistency, without correcting official outcomes.
    raw=csv(ROOT/'data/DM_Train.csv',dtype={'fipsCode':str});raw['target_hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int)
    quality=[]
    for f in sorted(set(selection.fipsCode)):
        g=raw[raw.fipsCode==f]
        quality.append(dict(fipsCode=f,countyName=names.loc[f,'countyName'],rows=len(g),customers_min=g.customersTracked.min(),customers_max=g.customersTracked.max(),
                            count_exceeds_customers_rows=(g.outageCount>g.customersTracked).sum(),
                            max_P_minus_unclipped_count_ratio_abs=(g.P_t-g.outageCount/g.customersTracked).abs().max(),
                            max_P_minus_clipped_count_ratio_abs=(g.P_t-np.clip(g.outageCount/g.customersTracked,0,1)).abs().max()))
    write('summary/raw_count_consistency.csv',pd.DataFrame(quality))
    hist=raw[raw.target_hour<=71].copy();write('summary/focus_known_history.csv',hist[hist.fipsCode.isin(set(selection.fipsCode))][['fipsCode','target_hour','P_t','N_t','D_t','R_t','osi','gust','outageCount','customersTracked']])
    # A cross-horizon matrix shows whether common long-term failures also burden short outputs.
    overlap=d.merge(display[['fipsCode','target_day']],on=['fipsCode','target_day'],validate='many_to_one')
    cross=overlap.groupby(['cv_seed','fipsCode','target_day','horizon']).agg(n=('truth','size'),T_sse=('T_sse','sum'),T_bias=('eT','mean')).reset_index()
    write('summary/focus_cross_horizon.csv',cross)
    headings=['CV','h','共同分数/T SSE','共同分数中低估','冻结前5县/T SSE','六路均低估/共同分数']
    conrows=[[r.cv_seed,r.horizon]+[f'{getattr(r,c):.1f}%' for c in ['common_over_T_sse_pct','both_under_share_common_pct','top5_T_sse_share_pct','all6_under_share_common_pct']] for r in con.itertuples()]
    countyrows=[]
    for h in [24,48]:
        fs=selection[(selection.horizon==h)&(selection.reason=='common_top5')]
        for r in fs.itertuples():
            ranks=[int(county[(county.cv_seed==s)&(county.horizon==h)&(county.fipsCode==r.fipsCode)]['rank'].iloc[0]) for s in [42,20260917,20260918]]
            g=sc[(sc.cv_seed==42)&(sc.horizon==h)&(sc.fipsCode==r.fipsCode)].iloc[0]
            countyrows.append([h,r.countyName+' '+r.stateAbbr,r.fipsCode,' / '.join(map(str,ranks)),f'{100*g.common_score_all6_under/g.common_score:.1f}%',f'{100*g.common_score_gate/g.common_score:.1f}%'])
    cardrows=[]
    for r in cards[(cards.cv_seed==42)&(cards.variant=='context')].itertuples():
        cardrows.append([r.countyName+' '+r.stateAbbr,r.horizon,r.date,f'{r.true_peak:.4f}',f'{r.T_peak:.4f}',f'{r.S_peak:.4f}' if np.isfinite(r.S_peak) else '—',f'{r.nearest_percentile:.1f}',f'{r.low_loss_controls} / {r.high_loss_controls}',f'{r.control_max_peak:.4f}'])
    report=f'''# 共同误差调研结果（2026-09-20）

**结论：目前主要瓶颈是两条路线共同漏判的增长与持续过程，继续调整24h融合很难解决。短时优先研究已知增长状态的幅度表达；长期优先研究目标时刻之前的多轮天气暴露与恢复阶段表示。Morgan式孤立突增的具体原因仍未证实，不应通过统一抬高背景预测来修补。**

本轮新增训练0次、模型重载0次。50份既有输入保持不变；所有新文件仅在本目录。下述为开发数据诊断和假设优先级，不是新模型提升、因果结论或独立隐藏测试证据。

## 1. 与上次极端县分析相比新增了什么

- 使用当前已完成P结构、邻县和D修复后的最新T，并与严格Stella L2逐行比较，不沿用旧v1.12误差排名。
- 检查六路Stella来源，确定是否已有某一路捕捉到事件、只是组合丢掉了它。
- 所有239县、三CV、四h均保存统计；重新按seed42选长期前5县和短时前3县，再追踪确认分折。Morrow等历史县只作历史对照。
- 对全部县和固定目标日做两种不使用未来结果的匹配，再连接未来受灾结果；没有按已知答案挑“相似县”。
- 检查增长、持续与恢复形态，纳入共同高估的Muskingum，避免只研究漏报而提出普遍抬高预测。

## 2. 共同错误占据主导

共同分数为同方向时两者平方误差的较小值。它用于衡量重叠和排序，**不是不可约误差，也不是新模型可实现收益上限**。表中的“共同分数/T SSE”是两个总量的比，不能解读为正式的唯一误差归因。

{table(headings,conrows)}

24h共同分数约为T SSE的90%–93%，48h约94%–96%。其中约77%–90%属于共同低估；Stella六路全部低估的部分几乎覆盖了这些共同低估分数。不能期望通过改平均权重，自动补出所有来源均未预测到的峰值。

六路来源共享大部分输入，T与Stella也复用部分基础来源，并非七个相互独立的证据。共同失败能定位共享瓶颈，仍不能单独区分“缺少信息”和“共享建模方式没有充分利用信息”。

seed42冻结的前5县，在三套里贡献24h T SSE约36%–45%、48h约56%–61%。没有删除这些县或只在有利CV报告。

{table(['h','县','FIPS','共同排名：42 / 17 / 18','seed42六路均低估占共同分数','seed42门控占共同分数'],countyrows)}

Clay/Calhoun很多错误发生在原门控已经触发的区域：尾部回退能保护已有较高预测，却不能补齐L0自身的严重低估。Muskingum则主要共同高估，且门控大量覆盖；它是防止“所有高值都加大”方案的必要反例。

## 3. 短时优先级：Forest仍是最大单县瓶颈

{table(['CV','h','Forest占T SSE','Forest+Clay+Calhoun占T SSE'],[[r.cv_seed,r.horizon,f'{r.forest_sse_share_pct:.1f}%',f'{r.forest_clay_calhoun_share_pct:.1f}%'] for r in short.itertuples()])}

三套中Forest单县仍占1h SSE的31%–35%，6h约20%–23%；它的大部分损失集中于3月14日早期。注意“1h”是输出horizon，观测截止后真实停电并没有持续更新。

Forest在截止前已有显著变化：P从小时69的约0.0083跳到小时70的0.1430，小时71约0.1432；官方允许使用的N71约0.0654。N71在三套都超出外折训练县最大值（约0.0217、0.0312、0.0312）。未来OSI峰值0.5447，seed42最新T在首日的最大预测仅约0.1020。

主匹配最近距离为候选池留一最近距离的100百分位，完整输入摘要匹配仍约96%–97%。有Venango、Clarion等受灾对照，但幅度远小。这支持“已知强增长状态缺少训练支持、输出幅度没有充分传递”，不是完全没有事前线索。仍不能由这一个案例证明按N71外推的模型一定有效。

Morrow首日OSI峰值0.5087，最新T峰值约0.4865，已不在当前短时前三。当前难点分布已经发生变化，后续不应照搬最早那批县的优先级。

## 4. 重点县与相似对照

下表每县展示seed42在固定日分组中T SSE最大的那一天；这只是事后展示窗口，不用于调参。完整各日及其他两CV均另存。低损失对照定义为该日峰值OSI≤.01，高损失为≥.05，中间组不计入这两类。距离百分位不是校准的概率。

{table(['县','h','展示日','真实OSI峰值','T日峰值','S日峰值','最近距离百分位','前5低/高损失县','对照最大峰值'],cardrows)}

### Clay、Calhoun、Braxton：有受灾先例，持续量明显不足

Clay/Calhoun在3月16–17日表现为多次增长与延续，不只是一个峰值错位。seed42的3月17日：Clay真实平均P约0.1608、预测0.0451；Calhoun真实约0.1692、预测0.0548。D同时低估，形成正的P×D误差交叉项。

Clay的相似县包括Roane、Braxton，也包括同类条件下损失较小的Wyoming WV。Calhoun前5对照中有Braxton、Roane、Ritchie、Tyler等受灾先例。距离支持并不差；但相近的广义风雨/林地条件对应不同幅度和持续量。三套有时存在更严重训练参照，模型仍低估，说明不能只归因于“标签值超过所有训练样本”。

这支持优先研究暴露过程/持续量表示和训练中的尾部泛化，同时保留局地损伤、网络脆弱性和恢复条件缺失的可能。县级均值天气与摘要匹配尚不足以分辨这些解释。

### Muskingum：共同高估的恢复反例

3月15日真实平均P约0.0360，T预测约0.1137；真实P>.05仅3小时，预测24小时都>.05。两条OSI曲线都明显偏高。该县初始P71约0.3206，匹配到Coshocton、Tuscarawas、Holmes等持续受灾对照，但其自身随后恢复更快。

上述大幅高估出现在seed42和seed20260918；seed20260917的T平均P约0.0363，已经接近真实0.0360，共同误差排名也从第5降至第171。它是明确的分折敏感案例，不能概括成三套都无法识别恢复，也不能据此认定必须补充外部信息。任何下一步改动都要检查是否破坏已经预测较准的那套。

这是“模型把持续程度估得过高”的曲线证据，不是已证实修复队伍效率更高。后续必须同时检查此类县，不能只根据Clay/Calhoun决定增加持续性。

### Morgan：事件区分信息不足，原因未明

3月18日小时178→179，官方outageCount从95升到5293，客户数均为8636；P从约.0110升到.6129，之后几小时迅速回落。跳变不是分母突变造成的；计数与P一致，但这不验证原始上报系统的事件真实性或具体原因。目标日最大阵风仅16.8mph。

三套、两种匹配的前5对照全部为低损失县。主匹配对照峰值最多.0035，而本县OSI峰值.3277；Stella六路在该峰值都为0。现有证据不支持通过更换树算法或微调融合修复它。

尚未取得可靠的县级原始原因记录，不能认定是计划停电、设备故障、切换操作或报送变化，也不能说它绝对不可预测。暂不把这个孤立案例作为全体县抬高预测的依据。

### Brown、Lewis：证据混合，保留次级位置

Brown的条件组合在两种匹配下均较稀疏；seed42/20260918前5缺少高损失先例，seed20260917出现先例，不能认定只需某一固定县属性就能解释。Lewis的受灾先例随匹配表示变化，存在峰值/过程漏判；需要更具体的信号，暂不单独开发县级方案。

## 5. 分量解释的范围

T的1/6h由四来源按目标小时重建C3，24/48h用相应来源。独立从原分量预测重新对齐，重现最终T。分量平方项、全部交叉项及后处理/标签精度余项与最终SSE闭合。

共同低估案例中P、D经常同向低估，N在突增阶段额外贡献误差，R的误差有时反向抵消。因此“某个分量RMSE改善”不自动等于整体改善。没有给Stella最终OSI强行分配唯一P/N/D/R贡献。

原始计数核查中，Morrow观测期有4行outageCount大于customersTracked、对应P封顶为1。保留官方值；按截断后的计数比核对，重点县P差异均小于5e-8，符合保存精度。Morgan峰值的客户分母稳定。上述检查只排查机械计算/分母问题，不证明原始采集记录或具体故障原因。

## 6. 外部资料核查带来了什么

直接读取了[美国国家气象局3月13日风灾回顾](https://www.weather.gov/cle/event_20260313_Wind)和[PowerOutage.com的同事件研究](https://poweroutage.us/research/march-13-midwest-windstorm)。它们确认广泛强风背景，后者还指出后续风暴使恢复复杂化，支持研究风强度、持续时间、空间覆盖与多轮过程。

这些是区域/事后证据，尚未证明重点县的具体原因。FirstEnergy正文访问失败；Clay/Calhoun搜索没有获得对应日期的可靠县级原因记录；Morgan搜索受到验证限制，替代搜索未返回相关证据。详细查询与排除理由见[外部核查记录]({OUT}/external_research/Source_Notes_2026-09-20.md)。

尤其不能把同事件事后停电图、恢复进度或拟合损伤曲线作为新的独立外部预测信息。暂无足够证据推荐立即大规模采购/下载新数据；先利用已许可信息检验具体过程表达更可控。

## 7. 下一步最多两个实验假设

### 优先一：短时增长状态的显式幅度表达

面向Forest这类已有强增长信号、训练覆盖稀疏的情况，以当前P1 a/b＋F2为对照，检验观测末期的增长尺度是否应直接进入输出重建，而不仅作为树的输入列。先做统一的P1来源结构候选，完整评估1/6h及对照县；不能为Forest指定特殊规则，也不能把趋势无衰减延伸到整个预测期。

已有特征含N71、末6小时趋势等，因此简单再添加同一列不构成新实验。需另立计划明确增长尺度、衰减/作用范围、边界、与现有a/b的差别及反例约束。优先级来自短时误差集中度和可见信号，未承诺收益。

### 优先二：以目标小时为基准的暴露顺序与持续过程

面向Clay/Calhoun/Braxton与Muskingum的相反错误，用允许的天气和截止历史构造“最近一轮暴露、距峰值时间、暴露后间隔/再增强、随时间衰减的累计暴露”等有限候选，检验损伤积累和恢复阶段是否更可区分。

现有163列已含天气最大/均值、阈值小时数、累计量和部分变化率，不能把这些重复包装成新信息。新的假设应明确编码顺序、距目标时刻的年龄或多轮间隔，先核对与现有特征的重合。使用跨县统一定义，不按本次结果设置县名单、日期开关或损伤标签阈值。正式验证以完整OSI为准，既看漏报也看过报。

这两项是待设计实验，不在本轮训练。现阶段不推荐继续扩大融合权重搜索，也不优先围绕Morgan单县调参。若上述具体候选没有跨CV、跨县的有意义增量，应停止该方向，再依据未解决的问题决定是否引入新的局地暴露或基础设施数据。

## 8. 验收、限制和文件

- 356,349条四horizon逐行记录、154,872条Stella长期来源记录、18,000个指标分组通过独立核验。
- 30,114个匹配查询、150,570个近邻对，逐一验证整折排除、排名、距离与支持百分位。排名文件在连接未来结果前冻结哈希。
- 另一个程序从原始数据/冻结缓存重建全部5,019个县×h×目标日输入，检查历史截止、天气聚合、实际输入摘要和结果连接。
- 三CV是同一事件的重复开发划分，案例由误差选出；匹配不构成因果识别。县日均值摘要也不能证明完整条件组合已充分覆盖。
- 全部旧输入保持不变；不改真值、不删县、不调预测、不进行训练/提交。

文件入口：

- [全县共同误差与排名]({OUT}/summary/county_metrics.csv)
- [固定案例与三CV证据卡]({OUT}/summary/case_evidence_cards.csv)
- [案例的具体相似县和结果]({OUT}/summary/focus_ranked_controls.csv)
- [完整匹配支持表]({OUT}/matching/support.csv)
- [输入超出训练范围的逐列证据]({OUT}/matching/focus_feature_ranges.csv)
- [分量及交叉项闭合]({OUT}/summary/component_error_decomposition.csv)
- [六路来源与门控]({OUT}/summary/stella_county_diagnostics.csv)
- [原始计数一致性]({OUT}/summary/raw_count_consistency.csv)
- [独立校验]({OUT}/verification.json)；[输入重建校验]({OUT}/input_verification.json)

## 9. 代表轨迹（seed42；另两套同目录提供）

![Forest截止前已有增长信号]({OUT}/figures/Forest_known_history.png)

![Forest短时增长]({OUT}/figures/split42_42053_h01.png)

![Clay持续损失低估]({OUT}/figures/split42_54015_h24.png)

![Muskingum持续程度高估]({OUT}/figures/split42_39119_h24.png)

![Morgan孤立突增]({OUT}/figures/split42_39115_h24.png)
'''
    (OUT/'Results_2026-09-20.md').write_text(report)
    (OUT/'README.md').write_text(f'''# 共同误差调研

研究已完成，详见 [结果]({OUT}/Results_2026-09-20.md)。0次训练、0个模型重载；新文件全部在本目录。

首次执行（已完成的主分析拒绝覆盖）：

```bash
cd {ROOT}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/diagnosis.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/verify_diagnosis.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/verify_inputs.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/summarize_report.py
/home/jacklo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/plot_cases.py
```

`data/`为全量逐行与案例轨迹；`summary/`为分组、来源、分量和证据卡；`matching/`保留不接触结果的输入、排名、支持及随后连接的结果；`external_research/`保存公开资料与访问限制；`figures/`为可分享SVG/PNG。`completion.json`给出交付文件哈希。

三套CV不构成新事件测试。共同分数不是不可约误差，事后报道不作为可部署特征。未来模型建议尚未训练。
''')
    print('Report generated',len(cards),'case evidence cards')

if __name__=='__main__':main()
