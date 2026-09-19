# v1.8 严格树基线训练计划（2026-09-17）

本计划对应已实现的 `v18_tree_nested_v2`。本轮只修改代码并验证：真实输入预检、旧控制规则复算和小型合成数据测试；没有启动比赛数据的正式训练。正式模型尚无新成绩。

目标顺序固定为：**原seed42建立可信基线 → 一次只测试一个有依据的改动 → 少数候选使用两套已冻结分折确认 → 最后训练全量提交模型。** 当前不叠加D约束、DEM、降水shift或GAT。

## 1. 正式训练前

所有命令从main根目录执行，使用已安装的 `.venv`：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8/train_v18.py --stage preflight --cv-file cv/cv_assignments_balanced_v1_seed42.csv --model-seed 42
```

预检通过应看到239训练县、63测试县、163列；四项训练有效行34177/32982/28680/22944，原五折县数48/47/48/48/48。这个命令只读，不训练、不建run、不改标签。

固定当前代码版本、requirements和输入哈希。历史 `models/`/OOF/提交属于旧协议，不能作为新CV的warm start；旧文件哈希失配记录保留，不在本次正式训练中修写历史清单。

## 2. 第一轮：只跑原seed42的严格基线CV

```bash
.venv/bin/python -B versions/xyy/v1.5.8/train_v18.py \
  --stage cv \
  --cv-file cv/cv_assignments_balanced_v1_seed42.csv \
  --model-seed 42 \
  --run-id v18_tree_nested_v2_split42_model42
```

该轮保留原163列和模型参数，仅改变验证隔离。每个外折内部选择树数后，用四折重训，再评估第五折。20个标量目标产生100个正式外层模型，总计500次LightGBM拟合；CPU单线程固定用于重现，不需要GPU。

此阶段不训练最终全量模型，不生成新提交。不要把 `--stage cv` 改为 `all` 提前消耗最终训练预算。不要传 `--diagnostic` 或降低轮数作为正式结果。

如果中断，原命令加 `--resume`，其余参数和输入完全相同。已经保存完整receipt的模型会校验后复用，中断中的未完成拟合从头重跑。代码或输入已变化时，应使用新run-id及新基线身份，不能绕过拒绝。

完成后执行独立重载验证：

```bash
.venv/bin/python -B versions/xyy/v1.5.8/verify_artifacts.py \
  versions/xyy/v1.5.8/runs/v18_tree_nested_v2_split42_model42
```

CV_COMPLETE表示该run外层产物已完整验证并冻结。独立verifier仍应再次执行，以确认新进程可重现。

## 3. 第一轮结果要看什么

先确认身份、覆盖、模型重载和C3来源正确，再看结果：

| 内容 | 使用方式 |
|---|---|
| `summary_metrics.csv` 的 `v18_rule` | 新严格基线主结果：短期C3，长期C1；各horizon pooled RMSE分别报告 |
| 同表C1 | 之后D/P分量改动的直接锚点，不把它误称为完整v1.8规则 |
| `control_fold_metrics.csv` | 检查fold2是否仍主导后期误差，但不据此修改分折 |
| `county_metrics.csv`、`target_day_metrics.csv` | 确认Forest/Morrow早期与Clay/Calhoun后期的误差结构 |
| `round_selection_cv.csv`、模型receipt | 检查树数和允许县集合；外折标签不参与早停 |
| `oof_component_predictions.parquet` | 为后续D已知历史贡献诊断提供逐行P/N/D/R预测 |
| `paired_county_bootstrap.csv` | 同一严格CV内固定控制的县级配对诊断，不按其结果重新选horizon路由 |

新严格分数可能比历史v1.8更差，因为旧早停看过计分折；这不是应回退到旧评价方法的理由。不能把两种协议的差值直接解释为新特征或模型的收益。

## 4. 第一项改进：D已知历史贡献

严格基线生成后，先做确定性诊断：只在最初四个有效t+1目标小时，对组件预测应用已知D下限，再重新组合C1、C3及固定v18_rule。

这一投影不需要重新训练树，可以复用**同一个严格协议、同一个CV**的分量OOF。它需要独立的分析/候选产物，保留原基线OOF；不能修改原表冒充模型原始输出。本轮尚未实现该投影脚本。

检查D分量误差、最终OSI、全体县和重点县的变化，同时保留全部有效行。若最终OSI无稳定收益，就保留诊断，不把“D物理约束正确”误当成“比赛分数一定改善”。

之后才考虑预测未知贡献再加回已知贡献，以及P/D树上的DEM、邻县信息或早期状态表示。每次只增加一个固定因素；这些属于后续独立实验，不改变当前已冻结基线代码。

## 5. 确认阶段：两套固定分折都使用

只有在seed42上选出少数有依据的候选后，再运行以下两套严格基线，为各自候选提供同分折参照：

```bash
.venv/bin/python -B versions/xyy/v1.5.8/train_v18.py \
  --stage cv \
  --cv-file cv/cv_assignments_balanced_v1_seed20260917.csv \
  --model-seed 42 \
  --run-id v18_tree_nested_v2_split20260917_model42

.venv/bin/python -B versions/xyy/v1.5.8/train_v18.py \
  --stage cv \
  --cv-file cv/cv_assignments_balanced_v1_seed20260918.csv \
  --model-seed 42 \
  --run-id v18_tree_nested_v2_split20260918_model42
```

每套仍为500次拟合。split seed只决定县分配，模型seed始终42。两套都报告，不能因seed20260917仍将Clay/Calhoun放在同一折而换seed。

候选若仅为固定D投影，可以分别复用这两套的组件OOF，不必为该投影重复训练同一树基线。候选若改变特征或监督结构，则必须在相同CV下独立训练、独立命名，并遵循相同隔离方法；不能把seed42模型用于新CV。

确认关注：候选-基线差在两套重复划分上的方向、改善是否只靠少数县、是否同时产生更多高估。县级bootstrap有空间相关限制，多次划分也不是新增独立风暴。没有预定的机械百分比门槛，尤其不能只报最好的划分。

## 6. 最终提交模型放在确认之后

如果最终仍选择当前固定树基线，使用已验证的seed42 run：

```bash
.venv/bin/python -B versions/xyy/v1.5.8/train_v18.py \
  --stage final --resume \
  --cv-file cv/cv_assignments_balanced_v1_seed42.csv \
  --model-seed 42 \
  --run-id v18_tree_nested_v2_split42_model42

.venv/bin/python -B versions/xyy/v1.5.8/verify_artifacts.py \
  versions/xyy/v1.5.8/runs/v18_tree_nested_v2_split42_model42
```

最终阶段在239县内部重新做五折树数选择，再以固定轮数重训全部239县，共120次拟合、20个最终模型。它不会覆盖CV证据。最终主提交为该run的 `submission_v18_rule.csv`；其他控制提交用于诊断，不重新从它们中逐项挑最低OOF。

如果选择了D修正或其他候选，最终提交必须由对应的已确认候选流程生成并独立核验，不能把上述未修正基线提交当作候选提交。C0/C1/C2/C3以及候选之间的选择需要保留完整记录。

## 7. 执行边界与停止条件

- 当前只完成代码与测试，没有比赛数据正式训练结果；不声称新协议已改善成绩。
- 数据/代码/环境身份不符、有效预测缺失、验证失败时停止，不降低校验要求继续训练。
- CV_COMPLETE之后不改写该run的OOF；改代码或实验规则应保存独立实现版本与run身份，避免失去旧基线可恢复的代码。
- 不为每个候选都提前训练最终模型；先比较CV与两套确认结果。
- 不因重复分折分数更低就替换原分折，也不平均不同划分OOF来伪造更多独立样本。
- GAT、降水对齐和Morgan事件调查沿此前优先级分别推进；它们不是本轮严格树基线的组成部分。
