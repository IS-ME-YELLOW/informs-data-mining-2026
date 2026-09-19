"""Produce the diagnostic report from verified saved-result analyses only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;PKG=OUT.parent
RUN=PKG/'outputs/runs/dem_v158_p2_component_seed42_cuda_log'
H1='osi_target_t01h'


def csv(name):return pd.read_csv(OUT/'tables'/name,dtype={'fipsCode':str,'scope':str},float_precision='round_trip')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def table(frame,columns,formats=None):
    formats=formats or {};lines=['| '+' | '.join(columns.values())+' |','|'+'|'.join('---' for _ in columns)+'|']
    for _,r in frame.iterrows():
        cells=[]
        for key in columns:
            v=r[key]
            if pd.isna(v):s='—'
            elif key in formats:s=formats[key].format(v)
            elif isinstance(v,(float,np.floating)) and v.is_integer():s=str(int(v))
            else:s=str(v)
            cells.append(s)
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='PASS'
    summary=csv('overall_metrics.csv');fold=csv('fold_effects.csv');county=csv('county_effects.csv')
    inner=csv('inner_alpha_recomputed.csv');transfer=csv('inner_outer_transfer.csv');final=csv('final_alpha_and_test_replay.csv')
    sensitivity=csv('diagnostic_exclusions_and_rollbacks.csv');features=csv('Newton_feature_support_by_scope.csv')
    contexts=csv('Newton_across_stack_contexts.csv');history=csv('Newton_history_windows.csv');scale=csv('outer2_residual_scale_reconstruction.csv')
    source=json.loads((OUT/'source_audit.json').read_text())
    all1=summary.loc[summary.horizon==H1].iloc[0];n=county.loc[(county.horizon==H1)&(county.fipsCode=='18111')].iloc[0]
    others=sensitivity.loc[sensitivity.scenario=='exclude_Newton_hindsight'].iloc[0]
    z=features.loc[(features.scope=='0,1,3,4')&(features.feature=='pre_event_mean_osi')].iloc[0]
    selected=inner.loc[(inner.horizon==H1)&(inner.outer_fold==2)]
    diagnosis=(f'旧 GAT 的 1h RMSE 从 {all1.base_rmse:.12f} 升至 {all1.gat_rmse:.12f}，恶化 {all1.rmse_change_pct:.2f}%。'
        f'核心损失来自 fold 2 的 Newton County, Indiana（18111）：它仅占 {100*n.n/all1.n:.3f}% 的有效行，'
        f'却贡献 {100*n.fraction_of_net_loss_increase:.2f}% 的净 SSE 增量。'
        '已确认的现象是：历史输入严重超出监督训练县范围，同一县的 GAT 校正在不同训练上下文中剧烈变化；'
        '内层正确选择的系数未能防住外层的极端正校正。')
    report=f'''# 旧 GAT 结果诊断（2026-09-19）

**{diagnosis}**

当前最有依据的解释是“域外历史输入下的神经网络外推不稳定，叠加无界残差输出和内外层模型迁移”。这还不是某个特征或某条网络路径的因果证明：本地旧运行缺少模型和检查点，不能进行固定权重的输入扰动或完整模型重载。

本次仅写入 `diagnostics/`，没有训练模型、修改预测、改变 alpha、改动原代码/报告或重新生成提交。独立诊断复算见 `verification.json`；它不等于原模型包的完整验收。

## 1. 诊断对象和验收边界

- 来源：`{RUN}`。
- 协议为 `dem_v158_nested_v2`，`component_v158` 基线、205 维图输入、MSE、CUDA、220 epoch 上限。运行名中的 `_log` 不能当作 log 目标变换证据；保存代码采用残差标准差缩放，没有 log/expm1 变换。
- `CV_COMPLETE` 及其指向的二进制 OOF、内层预测、alpha 和基模型拟合清单均可核对。源代码与特征 Parquet/JSON 的记录哈希匹配。
- 原运行来自 Windows。本地部分运行 JSON、CSV 和地形 CSV 的换行变成 LF；只在内存中恢复 CRLF 后可匹配记录哈希。源文件没有被写回。逐文件结果见 `tables/source_hash_audit.csv`。
- **原 `verification.json` 明确写着 `independent_reload: not_run_in_this_process`；`FINAL_READY` 也注明独立验证待完成，且没有最终 `COMPLETE`。** 当前目录没有 `.pt`、基模型 `.txt` 或 `.complete.json`；清单要求 416 个基模型和 64 个 GAT 检查点及对应完成记录。项目环境也没有 PyTorch。

因此，本次可以独立验证保存数值、来源和声明的作用域；不能验证缺失权重能否重现这些预测、实际保存的 scaler/epoch/权重状态是否一致。这里不安装依赖、不用重训代替找回原模型。

## 2. 总体恶化来自哪里

重新按官方四个 horizon 计算 pooled RMSE，基线和 GAT 均使用原固定后处理：

{table(summary,{'horizon':'horizon','n':'有效行','base_rmse':'旧树基线 RMSE','gat_rmse':'旧 GAT RMSE','rmse_change_pct':'相对变化','delta_sse':'SSE 增量'},
    {'base_rmse':'{:.12f}','gat_rmse':'{:.12f}','rmse_change_pct':'{:+.4f}%','delta_sse':'{:+.9f}'})}

1h 分折：

{table(fold.loc[fold.horizon==H1],{'outer_fold':'外折','alpha':'内层所选 α','base_rmse':'基线 RMSE','gat_rmse':'GAT RMSE','delta_sse':'SSE 增量'},
    {'alpha':'{:.2f}','base_rmse':'{:.9f}','gat_rmse':'{:.9f}','delta_sse':'{:+.9f}'})}

fold 0/1/4 净改善，fold 3 因 α=0 保持基线；fold 2 独自出现大幅恶化。

1h 最大损失县：

{table(county.loc[county.horizon==H1].sort_values('delta_sse',ascending=False).head(8),
    {'fipsCode':'FIPS','countyName':'县','outer_fold':'折','base_rmse':'基线 RMSE','gat_rmse':'GAT RMSE','delta_sse':'SSE 增量'},
    {'base_rmse':'{:.9f}','gat_rmse':'{:.9f}','delta_sse':'{:+.9f}'})}

Newton 增加 SSE **{n.delta_sse:.9f}**，大于整体增加的 **{all1.delta_sse:.9f}**，因为其余县合计减少 SSE 约 **{n.delta_sse-all1.delta_sse:.9f}**。

作为事后定位，排除 Newton 后，同一批剩余行的基线 RMSE={others.base_rmse:.12f}，GAT={others.gat_rmse:.12f}，改善 **{-others.rmse_change_pct:.4f}%**。这说明该次失败不等于所有县普遍退化，但删除 Newton 或为它关闭模型都不是合法的新泛化结果。

## 3. Newton 的真实轨迹与错误校正

该县已知历史表现为：观测早期停电很重，随后明显恢复，预测期总体停电很低。

{table(history,{'start_hour':'起始小时','end_hour':'结束小时','mean_P':'平均 P','max_P':'最大 P','mean_OSI':'平均 OSI','max_OSI':'最大 OSI'},
    {'mean_P':'{:.8f}','max_P':'{:.8f}','mean_OSI':'{:.8f}','max_OSI':'{:.8f}'})}

小时 71 的 P 约为 0.03148。有效 1h 预测期真实 OSI 平均约 0.00204、最大 0.0172；树基线平均约 0.00383，量级合理。GAT 原始校正却在 **全部 143 个有效行**持续为很大的正值：最小 0.95262、平均 1.03173、最大 1.12995。α=0.2 后平均增加约 0.20635，最终平均预测约 0.21018。

最大错误行对应预测起点小时 119、目标小时 120：真实 OSI=0.0066，基线约 0.01800，原始校正约 1.12995，最终预测约 0.24399。最终值没有超过 0.65，所以上界截断不能阻止这种“范围合法、幅度完全错误”的预测。

这不是单个峰值偏移，而是几乎整个预测期持续高估。完整时间行保存在 `Newton_outer_rows.csv`；原始历史和未来真实轨迹仅作为诊断参考存于 `Newton_official_history_and_future.csv`，未进入任何新模型。

原始数据早期存在 outageCount 大于 customersTracked 的记录，官方 P 已截断为 1。缓存与这些已公开历史数值一致；本次没有把它直接判作数据错误，也没有据此改写标签。是否涉及更早事故或报送口径，需另查数据来源，现有证据只支持“已有严重停电后恢复”的描述。

## 4. 输入支持范围是最明显的风险点

按保存代码的 float32 图输入、监督县/有效时间掩码、mean/std 规则重建标准化。下表是外层 fold 2 使用的 S=0,1,3,4，**不包含 Newton 的监督标签**：

{table(features.loc[features.scope=='0,1,3,4'].sort_values('Newton_max_abs_z',ascending=False).head(8),
    {'feature':'图输入','training_mean':'训练均值','training_std':'训练标准差','training_max':'训练最大值','Newton_min':'Newton 最小值','Newton_max':'Newton 最大值','Newton_max_abs_z':'Newton 最大 |z|'},
    {'training_mean':'{:.7f}','training_std':'{:.7f}','training_max':'{:.7f}','Newton_min':'{:.7f}','Newton_max':'{:.7f}','Newton_max_abs_z':'{:.2f}'})}

最突出的是 `pre_event_mean_osi`：Newton 的值约 **{z.Newton_max:.7f}**，与原始小时 0–47 的平均 OSI 对应；训练县最大仅约 **{z.training_max:.5f}**。训练均值约 {z.training_mean:.7f}、标准差约 {z.training_std:.7f}，对应 **{z.Newton_max_abs_z:.2f} 个标准差**。该值是合法已知历史，不是未来泄漏。

代码只在标准差小于 1e-6 时改成 1；这里标准差约 0.002511，正常通过，所以这个保护不会约束 153σ 的输入。源网络的线性投影及 `skip = Linear(in_dim, hidden)` 路径、最终回归 head 都没有相应的输入范围或校正幅度约束。它们具备把超出监督支持范围的输入放大的能力。

外层 GAT 的训练残差尺度可从该折内层保存的基线预测和标签重建，约为 **{scale.iloc[0].residual_scale_reconstructed:.7f}**。Newton 的校正相当于该尺度的约 **{scale.iloc[0].Newton_normalized_correction_mean:.1f} 倍**；其训练真实残差最大约 {scale.iloc[0].train_true_residual_max:.5f}。这支持“域外外推过大”的解释。源码先用残差除以尺度训练，再乘回尺度；没有发现简单漏乘或逆变换方向错误。

这里的 scaler 是按代码重建的统计，尚未与缺失 checkpoint 内的保存值逐项核对。204 个非 base 图输入可从冻结数据和图重建；外层模型的测试县 base 通道未完整提供，本次没有拿占位 base 做模型推理。

**证据支持风险机制，不能断言就是某一个权重或 skip 分支造成了全部错误。** 要确认具体通道，必须找回原 checkpoint 做固定权重、同输入的局部扰动；仅看 z 分数不能替代该实验。

## 5. 为什么内层 α 没有挡住

全部 160 个内层候选分数已重新计算，选择规则确实是 pooled RMSE 最小、平局取较小 α；记录层面的内/外折依赖集合和基模型选轮也自洽，未发现使用外折未来标签的声明性违例。

fold 2 的内层选择曲线：

{table(selected,{'alpha':'α','rmse':'内层 pooled RMSE','selected':'是否选择'},{'alpha':'{:.2f}','rmse':'{:.12f}'})}

α=0.2 在内层约改善 2.43%，因而按规则被选中。但该池只评价 folds 0/1/3/4，**Newton 所在 fold 2 的未来标签不参与这次选择**，这是正确的外层隔离。模型仍可看到它的公开历史/天气作为图节点输入，不能把“不参与监督评分”误说成“整个节点不存在”。

内层预测来自三折训练的图模型，外层使用四折重训的图模型。即使 α 相同，基模型、标准化、监督规模和按 scope 派生的随机 seed 都变化了；内层稳定不保证新的外层网络在域外点上稳定。

同一个 Newton，在不同、均未以它为监督节点的 1h 图上下文中，保存校正如下：

{table(contexts.loc[contexts.horizon==H1],{'source':'来源','outer_fold':'用于哪个外折','stack_id':'图模型 scope','correction_min':'校正最小','correction_mean':'校正均值','correction_max':'校正最大'},
    {'correction_min':'{:+.6f}','correction_mean':'{:+.6f}','correction_max':'{:+.6f}'})}

校正从负值变成较大正值，四折外层进一步到约 +1.03，说明这里的外推很不稳定。但这些模型同时改变了训练县和派生 seed，本次不能把差异单独归因于随机初始化。

所有外折的内外层增量：

{table(transfer.loc[transfer.horizon==H1],{'outer_fold':'外折','selected_alpha':'所选 α','inner_change_pct':'内层 RMSE 变化','rmse_change_pct':'外层 RMSE 变化','inner_Newton_rows':'内层 Newton 行数','inner_correction_abs_max':'内层最大 |校正|','outer_correction_abs_max':'外层最大 |校正|'},
    {'selected_alpha':'{:.2f}','inner_change_pct':'{:+.3f}%','rmse_change_pct':'{:+.3f}%','inner_correction_abs_max':'{:.6f}','outer_correction_abs_max':'{:.6f}'})}

α 网格包含零，只表示允许选择回退；它不能保证每个外折一定不退化。`outer_alpha_hindsight_diagnostic.csv` 给出了事后外折曲线：fold 2 最好会选零，fold 3 则可能选非零，但不能用这些外折标签重新挑系数后宣称得到了无泄漏成绩。

此外，GAT 的 best checkpoint 使用每个 optimizer step 后、eval 模式下的**训练监督 MSE**选择，没有独立的 GAT 验证损失参与该次 checkpoint 选择。这本身不是标签泄漏，却无法直接约束域外泛化；内层 alpha CV 承担了主要外部选择作用。具体各模型在哪一 epoch 停止，需 checkpoint 的 fit_info 才能核实。

## 6. CV 退化与已有提交要分开

最终模型使用全训练集内部 CV 重新选择 α，不取五个外折 α 的平均或众数。全部 32 个最终候选分数也已从保存的外层基线/校正数值重放：

{table(final,{'horizon':'horizon','final_alpha':'最终 α','test_valid_rows':'测试有效行','test_changed_vs_base':'相对基线改变行数'},{'final_alpha':'{:.2f}'})}

**最终 1h 的 α=0，保存的 9,009 个有效测试预测与后处理后的旧树基线逐行一致。** 因此，0.01787464 是这次嵌套外层 CV 的 GAT 分数，不是“已保存提交一定启用了这套 1h 坏校正”。这一点已从 test_predictions 重放确认，但没有进行缺失权重的提交独立重载。

## 7. 目前结论与下一阶段建议

| 层次 | 结论 |
|---|---|
| 已证实 | 1h 净损失由 Newton 主导；其他县合计改善；保存预测和 α 运算可复算；最终 1h α=0 |
| 由数据/源码强烈支持 | Newton 的多个历史输入超出监督训练支持范围；无界网络/校正与内外模型迁移存在稳定性风险 |
| 尚未证实 | 具体是哪一特征权重、skip 或注意力路径造成放大；实际 checkpoint scaler/epoch 与重载结果；完整真实模型验收 |

因此，下一阶段应先处理神经网络输入/校正的稳定性，并设计同信息 MLP 与 GAT 对照。建议顺序：

1. **找回原模型用于定点核查。** 不重训来冒充复现旧模型。最小 Newton 核查所需文件见 `Checkpoint_Followup_2026-09-19.md`。
2. **在新实验前定义统一的稳健预处理和诊断。** 基于允许训练范围记录每县最大 |z|、范围外字段、校正/残差尺度比；候选的输入缩尾、稳健变换或有界校正幅度须在内层选择，训练与推理一致，不按 Newton 名单关闭模型。
3. **加入真正用于泛化的 checkpoint 稳定性检验。** 在允许训练县内部决定 epoch/超参数，并检查不同 scope/seed 的残差输出；不能只以训练 MSE 很低判断模型可用。不得因此放松外折隔离。
4. **用当前 F2 作新基准，MLP/GAT 共享输入与选择规则。** 旧 `component_v158` 在此代码中是同 horizon 分量直接重组，并不包含 main 当前的 a/b、D_G 和短时 C3 主路由。旧 GAT 的 205 列也包含 DEM、坐标和另一套近邻摘要；不能直接拿它与当前 F2 的成绩比较并归因于图结构。
5. **保留极端真灾县的收益检查。** 简单把所有校正压得很小可能同时损害 Forest 等真正需要大幅校正的县。应在严格内层和全部外折上权衡，不把本次事后回退表当作新候选。

当前证据既不能说明“图模型普遍无用”，也不能证明“去掉 Newton 就能赢”。它明确指出：旧版本在少数域外状态上存在足以吞掉其他县收益的尾部风险，这应当先于扩大 GAT 配置搜索处理。

## 8. 文件与复现

- `diagnose_old_gat.py`：只读来源与 OOF/alpha/输入支持诊断。
- `verify_diagnosis.py`：独立检查核心数值、alpha 选择、Newton 历史和最终 1h 测试回退，并核对未改动其他文件。
- `tables/overall_metrics.csv`、`fold_effects.csv`、`county_effects.csv`：全部损益。
- `tables/Newton_outer_rows.csv`、`Newton_inner_rows.csv`、`Newton_history_windows.csv`：Newton 的轨迹和上下文。
- `tables/Newton_feature_support_by_scope.csv`、`outer2_all_nodes_feature_support.csv`：域外输入诊断。
- `tables/inner_alpha_recomputed.csv`、`inner_outer_transfer.csv`、`final_alpha_and_test_replay.csv`：选择规则及迁移。
- `tables/outer_alpha_hindsight_diagnostic.csv`、`diagnostic_exclusions_and_rollbacks.csv`：明确标识的事后诊断，不是新模型成绩。
- `source_audit.json`、`verification.json`、`reference/before_diagnosis.json`：证据边界和文件保护。

运行命令见 README。所有输出路径强制位于 diagnostics，Python 使用项目 `.venv/bin/python -B`；CSV 用 `float_precision="round_trip"` 复读。
'''
    report_path=OUT/'Old_GAT_Diagnosis_2026-09-19.md';report_path.write_text(report)
    missing=csv('missing_model_artifacts.csv')
    followup=f'''# 原检查点后续核查（2026-09-19）

当前诊断已完成保存结果层面的核验；完整模型重载没有完成，不能用 CV_COMPLETE 代替。

若仅核查 Newton 的外层 fold2、1h，需要优先找回原运行的：

- `models/gat/gat_component_v158_t01h_S0-1-3-4.pt` 及同名 `.complete.json`。
- 该 checkpoint 依赖的 P_t/N_t/D_t/R_t 基模型及完成记录：训练 scope 为 `0-1-3-4`、`1-3-4`、`0-3-4`、`0-1-4`、`0-1-3`，均为 t01h，共 20 个基模型。
- 原 checkpoint 内 feature_mean/std、residual_scale、fit_info、node/time 顺序、base_dependencies 与图身份；若原始训练 stdout 另存，也应保留作 epoch 诊断。

完整原运行的验收要求 416 个基模型、64 个 GAT checkpoint 和对应完成记录。本地缺失清单位于 `tables/missing_model_artifacts.csv`，共 {len(missing)} 项。原运行 Windows 绝对路径需要显式映射到同内容本地副本，不能改写原 manifest 来掩盖变化。

拿到原文件后，先在 diagnostics 下建立只读来源副本/映射记录，并将所有新输出限定在 diagnostics；不要直接调用会重写原运行 verification/COMPLETE 的入口。

核查顺序：

1. 核对 checkpoint/receipt/hash/scope 与原运行身份；不缺省重训。
2. 用保存 scaler 和原基模型重建 outer2 的全节点图输入，预测应重现保存 Newton 校正。
3. 在固定权重上逐组扰动历史重尾字段、skip 输入或消息通道，区分直接路径与图传播的敏感性。此为机制诊断，不是经过重新训练验证的新模型。
4. 新预处理或有界校正若要用于比赛，再按完整内层隔离协议重新训练、独立评估。不能用事后剪掉 Newton 的 OOF 分数宣布改进。

本次没有安装 PyTorch、没有向原目录写入文件，也没有执行上述缺失权重步骤。
'''
    (OUT/'Checkpoint_Followup_2026-09-19.md').write_text(followup)
    manifest={'status':'DIAGNOSIS_COMPLETE','source_run':str(RUN),'source_identity':source['run_identity'],
        'numerical_diagnosis_verification':'PASS','model_training':False,'model_reload':False,
        'original_package_full_acceptance':'not established; checkpoint files and original independent reload missing',
        'source_files_unchanged':verification['source_files_unchanged'],
        'report_sha256':sha(report_path),'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'artifacts':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='diagnostic_manifest.json'}}
    (OUT/'diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(diagnosis)


if __name__=='__main__':main()
