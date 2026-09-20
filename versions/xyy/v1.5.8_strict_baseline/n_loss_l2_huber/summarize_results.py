"""Report the completed frozen-feature loss comparison and exact-equivalence finding."""
from loss_protocol import *


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    v=read_json(RUN/'logs/independent_verification.json');assert v['status']=='PASS'
    audit=read_json(OUT/'summary/loss_equivalence_audit.json');assert audit['status']=='PASS'
    primary=pd.read_csv(RUN/'metrics/primary_scores.csv',float_precision='round_trip')
    ns=pd.read_csv(RUN/'metrics/N_component_metrics.csv',float_precision='round_trip')
    tail=pd.read_csv(RUN/'metrics/N_bins.csv',float_precision='round_trip')
    allctrl=normalize(pd.read_parquet(RUN/'control_predictions.parquet'))
    checks=[]
    for case in CASES[1:]:
        for h in HORIZONS:
            for mode in ('C0_direct_osi','C1_component_osi','C2_equal_blend','C3_aligned_component','v18_rule'):
                a=allctrl[f'pred_{case}_{mode}_{h}'].to_numpy();b=allctrl[f'pred_H0_Huber_{mode}_{h}'].to_numpy()
                assert np.array_equal(a,b,equal_nan=True)
                checks.append(dict(case=case,horizon=h,mode=mode,changed_rows=0,max_abs_difference=0.))
    write_json(OUT/'summary/prediction_equality.json',dict(status='PASS',exact=True,checks=checks))
    for name in ('primary_scores','N_component_metrics','N_bins','bootstrap_intervals','fold_metrics','window_metrics','county_metrics'):
        p=RUN/f'metrics/{name}.csv'
        write_frame(OUT/f'summary/{name}.csv',pd.read_csv(p,float_precision='round_trip',dtype={'fipsCode':str} if name=='county_metrics' else None))
    cases=pd.read_csv(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv',dtype={'fipsCode':str})
    county=pd.read_csv(RUN/'metrics/county_metrics.csv',dtype={'fipsCode':str},float_precision='round_trip')
    focus=county[county.fipsCode.isin(cases.fipsCode.unique())].merge(cases[['fipsCode','countyName','stateAbbr']].drop_duplicates(),on='fipsCode',validate='many_to_one')
    write_frame(OUT/'summary/frozen_focus_counties.csv',focus)
    models=read_json(RUN/'model_manifest.json')['models']
    rows=[]
    for r in models:
        target=r['spec']['target'];outer=r['spec']['outer_fold']
        old=read_json(HUBER/f'models/outer{outer}/{target}.json')
        rows.append(dict(target=target,outer_fold=outer,Huber_rounds=old['fit']['requested_rounds'],
            L2_rounds=r['fit']['requested_rounds'],best_iterations=','.join(map(str,r['fit']['best_iterations'])),fit_seconds=r['fit']['seconds']))
    write_frame(OUT/'summary/round_selection.csv',rows)
    joint=primary[primary['case']=='L2_all']
    osi_rows=[[f'{HORIZON_HOURS[r.horizon]}h',int(r.n),f'{r.baseline_rmse:.12f}',f'{r.rmse:.12f}','0.0000%'] for r in joint.itertuples()]
    nrows=[];trows=[]
    for h in HORIZONS:
        g=ns[(ns['case']=='L2_all')&(ns.horizon==h)&(ns['group']=='full')].set_index('space')
        nrows.append([f'{HORIZON_HOURS[h]}h']+[f'{g.loc[space,"rmse"]:.9f}' for space in ('model_raw','component_clipped','aligned')]+['全部为0'])
        space='aligned' if HORIZON_HOURS[h]<=6 else 'component_clipped'
        r=tail[(tail['case']=='L2_all')&(tail.horizon==h)&(tail.space==space)&(tail['group']=='N>1e-2')].iloc[0]
        trows.append([f'{HORIZON_HOURS[h]}h',int(r.n),f'{r.true_mean:.6f}',f'{r.baseline_mean:.6f}',f'{r.prediction_mean:.6f}','0.0000%'])
    sources=read_json(OUT/'reference/initial_input_hashes.json')
    report=['# 冻结特征的 N-L2 / Huber 对照结果（2026-09-19）','',
        '**结论：seed42四个horizon全部完成，L2与原Huber得到完全相同的20个树模型结构和OOF数值；N分量、整体OSI及高N尾部均为零变化。单纯切换这两个objective不能修复当前高值低估。**','',
        '本轮实际新增100次拟合、20个外层L2模型，不是直接复制旧预测。只改变N损失；同163列特征、同官方标签、同严格内层早停和外层四折重训规则。主参照仍为完整F2（D_G＋P a/b＋邻县信息）。未训练另外两套CV、全量模型或提交。','',
        '## 1. 最终OSI：四个horizon逐行完全相同','',
        table(['h','有效行','当前Huber组合RMSE','四来源N均换L2后RMSE','相对变化'],osi_rows),'',
        '四个单来源候选L2_N01/L2_N06/L2_N24/L2_N48同样全部零变化。六个组合的C0/C1/C2/C3和主规则都已保存；所有候选对H0的预测最大绝对差为0，不是四舍五入后看起来相同。县/折/时段指标均未变化。','',
        '## 2. N分量与高值组也没有改变','',
        table(['h','N raw RMSE','N clip RMSE','N aligned RMSE','相对Huber变化'],nrows),'',
        '以下使用主规则对应的N空间：1/6h为aligned，24/48h为clip。','',
        table(['h','N>0.01行数','真实N均值','Huber预测均值','L2预测均值','该组N RMSE变化'],trows),'',
        'Forest、Morgan、Clay、Calhoun、Brown、Morrow、Wyoming七个已冻结关注县的预测全部不变。高值低估仍存在，本次没有靠降低背景误差掩盖尾部恶化，也没有尾部提升。','',
        '## 3. 为什么两个损失没有产生不同结果','',
        '已保存的原Huber模型实际参数为 **alpha=0.9**。在LightGBM Huber中，这个数是残差阈值；当绝对残差不超过它时，使用二次损失区间，梯度与当前L2一致。它不是“取残差的90%分位数”，也不会自动缩放到N的数值范围。','',
        '本次N标签最大值约0.2431。进一步逐棵检查原Huber的叶值：对每个已保存外层模型的全部树前缀，将各树最小/最大叶值累加，得到覆盖任意输入路径的保守预测界。结合该horizon的N标签范围，20个模型中最大的绝对残差上界仅为 **'+f'{audit["maximum_saved_prefix_residual_bound"]:.6f}'+'**，明显小于0.9。','',
        '同时得到三个直接证据：','',
        '- 20对模型的完整 `tree_info`（分裂、阈值、叶值等）精确一致；模型文件因为objective声明等元数据不同，字节哈希可以不同。','- 80个内层best_iteration及20个外层refit轮数全部一致。','- 20个新模型重新预测外折后，原始N OOF与Huber逐值完全相同；clip、C3和最终OSI自然也一致。','',
        '**因此，此前把“Huber可能压低N尾部”列作待检验假设，本轮应明确排除当前参数下的这一解释。** 现有N-Huber在这批已核实的训练轨迹上没有进入需要截断大残差的区域，实际已表现为L2。','',
        '界限说明：残差保守界覆盖已保存的外层refit树前缀；没有保存的内层probe在best_iteration之后的树不在这项界限证明中。内层轮数相同和外层树/OOF精确相同是另外的实测证据。不能把本轮结果泛化成任意数据、任意alpha或目标缩放下Huber与L2都等价。','',
        '## 4. 对后续路线的影响','',
        '1. **保留当前组合，停止单纯Huber→L2切换方向。** 没有新收益，不需要优先为这个零变化候选追加两套CV确认。','2. **按原计划转向P结构向6/24/48h迁移。** 这是已验证方法的迁移，而不是继续期待名称不同的损失函数提供增益；仍需另立训练范围和目标天气对齐方案。','3. **N的优化问题仍然成立，但需要实质改变。** 可在后续考虑有依据的信息增量、预测结构或目标尺度与正则化配合；这次实验没有区分这些原因，也没有证明树模型绝对无法改善。','4. 不为了让Huber与L2产生差异而直接降低alpha。那将引入新的损失强度，且可能更早削弱高残差样本的作用，需要独立假设与验证。','',
        '本轮仅seed42；没有对另外两套CV作实际L2训练结论。上述优先级是基于本轮精确零变化及已完成N诊断的建议。','',
        '## 5. 严格性和验收','',
        f'- 输入/代码作用域：{len(sources)}个来源文件哈希核验通过，原163列列序及数据环境一致；唯一训练参数差异为objective。外折标签扰动不改变对应训练或早停子集。',
        '- 真实训练：80次内层早停＋20次四折refit，共100次；20个新L2模型。拟合函数累计耗时约'+f'{sum(r["fit"]["seconds"] for r in models):.1f}'+'秒（不含预检、产物写入和验收总耗时）。',
        '- 独立进程重载20个L2和20个Huber模型，检查80个内层probe的标签/指标/最佳轮数；独立pandas同目标小时对齐，复算120组控制指标和32组县轨迹bootstrap区间。',
        '- 32组区间都是[0,0]，原因是比较预测逐行相同，不是统计功效不足或“暂时不显著”。',
        '- P/D/R、直接OSI及未替换来源保持原值；全体239县与四horizon正式评分行不减少；输出没有进入正式提交。',
        '- 所有新代码、配置、模型、逐行产物、审计与文档均位于本目录；旧模型、旧报告和原分折未修改。','',
        '## 6. 文件入口','',
        f'- [整体指标]({OUT}/summary/primary_scores.csv)',
        f'- [N分量指标]({OUT}/summary/N_component_metrics.csv)',
        f'- [高值与其他分组]({OUT}/summary/N_bins.csv)',
        f'- [模型等价性与残差界]({OUT}/summary/loss_equivalence.csv)',
        f'- [逐树前缀界]({OUT}/summary/saved_tree_prefix_bounds.csv)',
        f'- [预测精确相等检查]({OUT}/summary/prediction_equality.json)',
        f'- [独立重载验收]({RUN}/logs/independent_verification.json)',
        f'- [执行说明]({OUT}/README.md)','']
    safe(OUT/'Results_2026-09-19.md').write_text('\n'.join(report))
    safe(OUT/'README.md').write_text(f'''# N 冻结特征 L2 / Huber 对照

**状态：seed42四horizon训练、评估及独立验收完成。100次新拟合、20个L2模型；与Huber的树结构、轮数、OOF及全部指标精确相同。**

[结果与解释]({OUT}/Results_2026-09-19.md)。原Huber alpha=0.9，已保存外层树前缀的残差保守界最高约0.264，仍处于二次损失区域。没有采用新模型，也没有启动另外两套CV。

输入参照是完整F2。只将N的objective从huber改为regression；163列特征、标签、CV、种子和内层早停规则冻结。六个预定组合均有完整逐行产物。

首次执行顺序如下（完成stage拒绝覆盖；只可恢复身份相同的未完成训练）：

```bash
cd {ROOT}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/train_n_loss.py --stage preflight
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/train_n_loss.py --stage train
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/evaluate_n_loss.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/verify_n_loss.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/audit_loss_equivalence.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_loss_l2_huber/summarize_results.py
```

`--resume`仅适用于同身份未完成train；不复用不同配置或不同分折模型。重验可单独运行verify_n_loss.py，不拟合任何模型。

- `reference/`：输入SHA-256与20个目标/外折的预检。
- `runs/v18_nloss_l2_v1_split42_model42/models/`：20个新模型及允许折、特征/标签哈希、轮数与来源receipt。
- 同run的`logs/probes/`：80个内层probe逐行真值、float32指标标签与最佳轮次预测；完整指标曲线在模型receipt中。
- 同run的`n_l2_oof_predictions.parquet`、`component_predictions.parquet`、`control_predictions.parquet`：全县逐行预测及来源；`metrics/`：完整六组合、县/折/窗口、N分组及区间。
- `summary/`：可直接阅读的汇总、固定7县表、损失等价性和保守残差界。
- `MODEL_CV_COMPLETE / EVALUATION_COMPLETE / VERIFIED_COMPLETE`：训练、评估、验收身份；顶层completion.json为本轮完成清单。

没有复制旧Huber模型到本目录；验收从原路径只读重载。训练或核验脚本不写旧日志、旧报告或缓存。
''')
    print('Report and summaries saved; exact zero changes documented')


if __name__=='__main__':main()
