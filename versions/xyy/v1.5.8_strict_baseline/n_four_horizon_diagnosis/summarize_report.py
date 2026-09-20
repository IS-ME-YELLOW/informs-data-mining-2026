"""Render the diagnostic findings from verified tables; no fitting or candidate selection."""
from diagnostic_core import *


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def rng(v,nd=2,sign=False):
    lo,hi=float(min(v)),float(max(v));spec=f'+.{nd}f' if sign else f'.{nd}f'
    return f'{format(lo,spec)}～{format(hi,spec)}'


def link(label,path):return f'[{label}]({path})'


def main():
    for seed in SEEDS:assert read_json(OUT/f'runs/split{seed}/verification.json')['status']=='PASS'
    s=csv(OUT/'summary/four_horizon_summary.csv');o=csv(OUT/'summary/all_oracle_metrics.csv')
    boot=csv(OUT/'summary/all_bootstrap_intervals.csv');county=csv(OUT/'summary/county_priority_all_splits.csv',dtype={'fipsCode':str})
    nbin=pd.concat([csv(OUT/f'runs/split{seed}/tables/distribution_and_bins.csv') for seed in SEEDS],ignore_index=True)
    alignm=pd.concat([csv(OUT/f'runs/split{seed}/tables/alignment_effects.csv') for seed in SEEDS],ignore_index=True)
    report=['# N 四个 horizon 诊断结果（2026-09-19）','',
        '**结论：N 值得做一轮受控的尾部预测实验，四个 horizon 都有诊断空间；主要问题是少数突增被严重低估，整体时间对齐并非当前首要问题。P 仍是更大的整体瓶颈，不建议围绕 N 开展无期限调参。**','',
        '三套冻结CV（42、20260917、20260918）全部完成，独立复算PASS；使用当前完整F2（D_G＋P a/b＋邻县摘要），没有训练或修改模型。下表RMSE绝对值取seed42；百分比范围覆盖三套CV，负数代表相对当前组合改善。真实N替换属于使用标签的诊断，不能部署，也不是可实现收益保证。','']
    head=[];priorities=[]
    for h in HS:
        g=s[s.horizon==h];r=g[g.split_seed==42].iloc[0]
        decision='优先做受控对照' if h==1 else '同一方法一并测试'
        head.append([f'{h}h',f'{r.current_OSI_rmse:.6f}',
            f'{r.N_model_raw_rmse:.6f} / {r.N_component_clipped_rmse:.6f} / {r.N_aligned_rmse:.6f}',
            rng(g.O_N_all_change_pct,sign=True)+'%',rng(g.A_N_zero_change_pct,sign=True)+'%',
            rng(g.A_N71_change_pct,sign=True)+'%','N真值替换3/3改善',decision])
        priorities.append(dict(horizon=h,recommendation=decision,method='frozen features; direct N L2 versus current Huber',
            rationale='tail amplitude compression; positive oracle sensitivity; no global time-shift evidence',
            oracle_change_pct_min=g.O_N_all_change_pct.min(),oracle_change_pct_max=g.O_N_all_change_pct.max(),
            training_authorized=False))
    report += [table(['h','当前OSI RMSE','N RMSE：raw / clip / C3','真实N替换 ΔOSI','固定N=0 ΔOSI','固定N=N71 ΔOSI','三CV一致性','下一步建议'],head),'',
        'raw是树未截断输出，clip是进入组合前的[0,1]分量，C3是同目标小时平均。当前1/6h用C3，24/48h用clip对应的C1；24/48h的C3数字仅供诊断。','',
        '## 1. N的主要误差来自少数高值，幅度低估比全局错时更突出','']
    tail=[]
    for h in HS:
        space='aligned' if h<=6 else 'component_clipped'
        g=nbin[(nbin.horizon==h)&(nbin.space==space)&(nbin.group_type=='truth_N')&(nbin.group=='N>1e-2')]
        full=nbin[(nbin.horizon==h)&(nbin.space==space)&(nbin.group_type=='all')]
        r=g[g.split_seed==42].iloc[0];n=int(full.loc[full.split_seed==42,'n'].iloc[0])
        tail.append([f'{h}h',f'{int(r.n)}/{n} ({100*r.n/n:.2f}%)',
            rng(100*g.sse_fraction_of_full)+'%',f'{r.truth_mean:.6f}',rng(g.prediction_mean,6)])
    report += [table(['h','N>0.01行数与比例','其占N SSE比例，三CV','该组真实均值','该组预测均值，三CV'],tail),'',
        '以1h为例，只有426行的N>0.01（1.25%），却贡献约91%的N误差。该组真实均值约0.0272，预测均值只有约0.0036；大量零值和小值行并不是N分量SSE的主要来源。这里的91%是**N分量SSE**，不是整体OSI误差占比。','',
        '在计划固定的−3至+3小时位移中，使用共同小时子集，三套CV×四个主规则N空间的最佳总体RMSE均出现在**位移0**。这不排除个别县峰值错位，但不支持通过全局移动N预测修复当前模型。','',
        '中心三小时N/R公式再次通过核验，完整窗口最大差小于1e-6；官方末端标签保留。没有发现需要把中心rolling改为后向rolling、截掉N71或整体平移天气的证据。','',
        '## 2. 改善N有整体价值，但目前不是最大的瓶颈','']
    oracle_table=[]
    for h in HS:
        g=s[s.horizon==h]
        oracle_table.append([f'{h}h',rng(g['O_N_partial_0.25_change_pct'],sign=True)+'%',
            rng(g['O_N_partial_0.5_change_pct'],sign=True)+'%',rng(g.O_N_all_change_pct,sign=True)+'%',
            rng(g.O_P_all_change_pct,sign=True)+'%',rng(g.O_D_all_change_pct,sign=True)+'%',rng(g.O_R_all_change_pct,sign=True)+'%'])
    report += [table(['h','N向真值靠近25%','N向真值靠近50%','完全真实N','完全真实P','完全真实D','完全真实R'],oracle_table),'',
        '25%/50%表示把每一行N预测向该行真值插值，不代表已经训练出这样的模型。三套CV中，完整N oracle的四项ΔRMSE的县轨迹bootstrap区间均低于0，部分插值的点估计也同向。这说明改善N对当前组合有价值；实际方法能获得多少仍待训练验证。','',
        '固定N=0只让整体RMSE恶化约0.18%–0.39%，反映当前N模型对最终OSI的净增益仍有限。固定N71在1h有两套小幅改善、seed42略差，6/24/48h均恶化，因此没有稳定替代规则，不能将保持历史N当作已验证改进。','',
        'P、D的oracle敏感性明显大于N。它们来自真实值替换，不能当成容易拿到的收益、不能加总，但提醒我们：N值得做一次明确假设的实验，P方法向6/24/48h的迁移仍应保留较高优先级。','',
        '当前R换成真值反而使整体RMSE上升约1.15%–1.52%，与N形成区别。不能仅凭分量RMSE决定整体方案，也不能因为误差抵消就认定错误预测值得永久保留。','',
        '## 3. 误差交叉项与C3：不宜直接删掉当前组合','']
    decomp=csv(OUT/'runs/split42/tables/osi_decomposition.csv')
    nd=decomp[(decomp.horizon==1)&(decomp['mode']=='main')&((decomp.term_a=='N')|(decomp.term_b=='N'))]
    report += ['seed42、1h的误差分解如下（平方误差单位）：','',
        table(['项','SSE项'],[[f'{r.term_a} × {r.term_b}' + ('（两倍交叉项）' if r.term_type=='twice_cross' else '（自身平方项）'),f'{r.sse_term:.6f}'] for r in nd.itertuples()]),'',
        'N自身平方项之外，P×N、N×D的交叉项明显为正；这些分量会在部分高损失行同向出错。N×R交叉项为负，存在局部抵消。完整后处理与官方标签精度项b已保留并闭合验证，未把0.35的公式权重误当成35%的误差份额。','']
    ar=[]
    for h in HS:
        g=alignm[(alignm.horizon==h)&(alignm.domain=='N')&(alignm.scope=='full')]
        ar.append([f'{h}h',rng(g.rmse_change_pct,sign=True)+'%'])
    report += [table(['来源horizon','同一有效窗口下，C3相对clip的N RMSE变化'],ar),'',
        '四个h、三套CV中C3均小幅改善N。共同目标区间120–215的对照另存于alignment_effects.csv，用于排除评分窗口不同造成的混淆。没有根据本轮结果删来源、调权重或修改主规则。','',
        '仅替换一个来源N时，seed42的最终OSI相对变化：','']
    matrix=csv(OUT/'runs/split42/tables/source_to_output_effects.csv')
    report += [table(['替换的来源N','输出1h','输出6h','输出24h','输出48h'],[
        [f'N{a}h']+[f'{matrix[(matrix.source_horizon==a)&(matrix.horizon==b)].rmse_change_pct.iloc[0]:+.2f}%' for b in HS] for a in HS]),'',
        '这张表说明N24/N48也会通过C3影响短时输出。四个来源oracle的收益不能相加，因组合与平方损失存在交互；不能把来源h的作用范围简化成同名输出h。','',
        '## 4. 高误差县：外推不足与缺少事件线索同时存在','',
        '案例规则在seed42冻结，得到7县：历史6县加Wyoming County, PA。每个案例在每个有效h/小时都按固定8列匹配，排除查询县完整外折；先保存前5名及其哈希，之后才连接未来结果。三套共10,437个查询、52,185个匹配对。','',
        '下表是seed42各案例在对应窗口真实N峰值小时的情况；真实峰值用于事后定位，不是可部署选择规则。距离是折内标准化8维欧氏距离，只用于该设定下的支持判断。','']
    q=csv(OUT/'runs/split42/tables/query_training_support.csv',dtype={'fipsCode':str})
    t=csv(OUT/'runs/split42/tables/case_trajectories.csv',dtype={'fipsCode':str})
    match=csv(OUT/'runs/split42/tables/matched_support.csv',dtype={'fipsCode':str,'donor_fipsCode':str})
    case_rows=[]
    for f,h in [('42053',1),('39115',48),('54015',24),('54013',24),('42131',6),('18013',24),('39117',1)]:
        g=t[(t.fipsCode==f)&(t.horizon==h)];peak=g.loc[g.N_true.idxmax()];hour=int(peak.target_hour)
        a=q[(q.fipsCode==f)&(q.horizon==h)&(q.target_hour==hour)].iloc[0]
        mm=match[(match.fipsCode==f)&(match.horizon==h)&(match.target_hour==hour)]
        outside=a.outside_columns if pd.notna(a.outside_columns) else '无'
        case_rows.append([f'{peak.countyName}, {peak.stateAbbr} ({f})',f'{h}h / s={hour}',f'{peak.N_true:.6f}',
            f'{a.query_pred_N:.6f}',f'{a.donor_all_training_N_max:.6f}',f'{mm.donor_true_N.max():.6f}',outside])
    report += [table(['县','h/目标小时','真实N','当前N','外层训练N最大值','前5匹配县同小时N最大值','超出匹配特征支持的列'],case_rows),'',
        '- **Forest**：三套划分都显示N71超出外层训练县范围。真实峰值0.2431，而1h当前预测在三套中仅约0.0089–0.0163；匹配距离也很大。这是已知强增长状态与训练覆盖不足的直接证据，但不能仅凭它推断准确未来幅度。','- **Morgan**：严重突增时刻的允许输入未超出这8列的单列范围，五个匹配县同小时最大N约0.0010，而本县约0.2043；三套均如此。支持缺少能区分该事件的信息/条件组合这一解释，不能证明它发生了某一种具体外部事件。其峰值还高于对应外层训练N最大值，单靠改损失未必足够。','- **Clay/Calhoun**：同小时参照县存在明显N增长，但幅度低于本县；森林覆盖支持及分折会改变外推程度。三套预测虽不同，峰值仍被大幅压低。','- **Wyoming**：新进入N重点清单。目标s=80真实N约0.1018，五个匹配县同小时最大值约0.0031，模型预测也很低；说明困难不只属于先前人工关注的县。','- **Brown/Morrow**：Brown同类参照的突增较弱；Morrow的P71及历史N摘要在三套都超出对应支持，且匹配距离大。Morrow此前主要是P/D问题，本次不因历史名气把它误列为N误差首位。','',
        'seed42的N加权SSE前三县占比约为1h 50.1%、6h 40.6%、24h 41.4%、48h 61.9%。这些是诊断聚集性，没有删除任何县，也没有添加按FIPS选择预测方法的规则。','',
        '特征支持范围不足不等于已证明模型绝对无法外推；范围内也不等于全部条件组合都有训练覆盖。本轮只能定位与区分证据，不能在损失函数、不可见因素和事件偶然性之间给出唯一因果答案。','',
        '## 5. 下一步建议：一轮统一N对照，随后回到P跨horizon迁移','',
        '1. **先做一个冻结输入的N-L2对照**：以现有N-Huber为对照，同样163列、官方N标签、严格县级内层早停，seed42同时训练四个horizon。每h 25次拟合、5个外层模型，合计100次拟合、20个外层模型。只改变损失，避免同时加邻县信息而无法解释收益来源。','2. **使用同一批新OOF做单来源替换和四来源联合替换**：分别观察N1/N6/N24/N48的独立影响及组合结果，不必为了每个评估组合重新训练。以最终四个OSI RMSE及高N尾部误差共同判断；不能只选好看的分量RMSE。','3. **有实质方向一致的收益，再做其余两套CV确认**。如果主要仍是Morgan/Wyoming一类缺少事件线索的失败，下一步应是明确的信息增量假设，而非继续堆损失或按县修补。','4. **P方法迁移保持下一优先级**。本轮P oracle显示P仍占较大的敏感性空间；N首轮若没有实质增益，就转向P6/P24/P48的a/b＋合法邻县特征迁移。当前没有启动上述任何训练。','',
        'Huber对尾部的处理是合理的待检验假设，不是本轮已证实的原因；L2也不能自动补齐稀缺事件的信息。当前证据不要求为四个h发明四套方法，也不支持立即移除C3、用N71替代模型或重写官方N定义。','',
        '## 6. 复现、完整性与产物','',
        '- 三套当前F2的预测独立复算最大绝对差为2.78e-17，远低于1e-12验收阈值；N的20个来源模型/每套及内层训练、早停范围检查通过。','- 另一个进程以pandas县/目标小时键重建组合，未调用生产诊断的对齐/评分函数。每套验证1,544,179条oracle/锚点行、3,479个匹配查询、40个bootstrap区间，以及全部N分组指标和OSI误差分解。','- 1,375个冻结输入文件在执行前后哈希一致。所有新代码、配置、逐行表、静态SVG/PNG和报告只在本诊断目录；无新训练、无旧产物覆盖、无提交。','- 图使用已有捆绑ReportLab绘图及Sharp渲染；项目环境未安装新包。1/6/24/48h正式有效行仍为34,177 / 32,982 / 28,680 / 22,944；公式无法完整复算的末端仍参与官方标签评分。','- 本轮三CV是已参与开发的同事件、同239县重复划分，bootstrap未消除空间依赖与模型选择偏差；oracle区间也不构成隐藏测试集可实现收益证据。','',
        '关键入口：','',
        '- '+link('完整四h指标表',OUT/'summary/four_horizon_summary.csv'),
        '- '+link('全部真实值替换/锚点指标',OUT/'summary/all_oracle_metrics.csv'),
        '- '+link('配对区间',OUT/'summary/all_bootstrap_intervals.csv'),
        '- '+link('县级三分折一致性',OUT/'summary/split_consistency.csv'),
        '- '+link('独立验收与输入完整性',OUT/'final_integrity_check.json'),
        '- '+link('执行说明',OUT/'README.md'),'',
        '## 7. 图示','',
        f'![N敏感性：oracle不是可部署模型]({OUT}/summary/figures/N_sensitivity.png)','',
        f'![Forest：历史强增长与幅度低估]({OUT}/runs/split42/figures/county_42053_h01.png)','',
        f'![Morgan：后期突增缺少匹配事件]({OUT}/runs/split42/figures/county_39115_h48.png)','',
        '三套CV的Forest、Morgan、Morrow图分别保存在各自runs/split*/figures；SVG是可缩放原图，PNG用于查看。']
    safe(OUT/'Results_2026-09-19.md').write_text('\n'.join(report)+'\n')
    frame(OUT/'summary/priority_recommendation.csv',priorities)
    readme=f'''# N 四horizon诊断

**状态：三套CV诊断及独立核验完成，未训练模型。** 结果见 [Results_2026-09-19.md]({OUT}/Results_2026-09-19.md)。

主参照是完整F2（D_G＋P a/b＋邻县信息），N仍为原strict模型。结论支持做一次统一的N尾部预测对照，但没有采用oracle、锚点或改变组合。

- `reference/approved_plan.md`：用户批准时的计划原文；根目录计划保留编写时状态，当前执行状态以本README、结果和completion为准。
- `diagnostic_config.json`：冻结参数；`reference/initial_input_hashes.json`：1,375个输入文件字节身份。
- `runs/split*/rows/`：完整逐行诊断；oracle含未来真值，显式标记不可部署，不能用作训练特征或提交。
- `runs/split*/tables/`：所有县/时段、尾部分组、峰值与位移、交叉项、匹配支持、bootstrap。
- `summary/`：三套汇总与建议，未合并三套OOF生成模型候选。
- `verification.json`、`final_integrity_check.json`、`completion.json`：独立核验及完成身份。

诊断使用项目 `.venv/bin/python -B`。`run_diagnostics.py`完成后拒绝覆盖；重验已有结果可运行：

```bash
cd {ROOT}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_four_horizon_diagnosis/verify_diagnostics.py --split-seeds 42 20260917 20260918
```

首次执行命令已按批准计划运行，`--stage preflight`和`--stage analyze`都限定三套固定seed。相似支持采用8个合法输入、完整外折排除、前5名先冻结后接结果；因查询县/小时由事后误差聚焦，不作因果解释。

图使用本机捆绑Python的ReportLab，不改变项目环境：

```bash
/home/jacklo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B {OUT}/plot_diagnostics.py
```

`summarize_report.py`只读已验证表生成说明；`plot_diagnostics.py`只读本目录表生成图。无LightGBM拟合入口、无部署/提交入口。独立验证器不复用生产诊断的数值函数；浮点求和顺序差异使用事先冻结的容差，不修改预测或输入。
'''
    safe(OUT/'README.md').write_text(readme)
    print('Report, README and four-horizon recommendation saved')


if __name__=='__main__':main()
