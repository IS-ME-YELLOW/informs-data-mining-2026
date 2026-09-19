# v1.8 第一轮严格基线

本目录统一保存第一轮 `seed42` 严格县级CV的代码快照、标签快照、训练日志、模型、验证和结果分析。协议为 `v18_tree_nested_v2`；163列特征、组件公式、后处理和固定短期C3/长期C1规则保持不变。

本轮仅运行CV：20个目标×5个外折，保存100个外层模型；每个外折模型由4次内层早停探测选轮数，再在该外层训练集合上重训，共500次LightGBM拟合。没有最终全量模型、提交文件、新特征、D投影或GAT。

## 文件位置

- `Results_2026-09-17.md`：完整训练结束后的结果解读。
- `runs/v18_tree_nested_v2_split42_model42/`：不可变CV产物、100个模型及来源记录。
- `analysis/`：新旧协议比较、严格协议内控制比较、逐折/县误差、重点时段、树数及D下限诊断。
- `logs/train_cv.log`、`execution.json`：训练日志、实际命令、起止时间和资源记录。
- `logs/independent_verification.json`：训练完成后新进程独立重载验证的结果。
- `source_snapshot_manifest.json`：从已验证v1.8代码复制到本目录的文件及哈希；复制时没有改训练代码。
- `reference/`：训练计划、实现验收快照、原文件保护清单，以及本轮历史比较所用OOF和清单副本。

`reference/Training_Plan.md`保留的是原计划快照，其中的旧目录命令不作为本目录的复现入口；本次实际命令以以下说明和 `execution.json` 为准。

公共原始CSV、v1.5.6特征及原seed42县分折从项目原位置只读加载；实际路径和内容哈希均记在run的 `input_manifest.json`。

## 执行与恢复

从项目根目录使用指定环境：

```bash
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/run_first_cv.py
```

如果在CV_COMPLETE之前中断，可运行同命令并加 `--resume`。它只复用身份完全匹配且保存完整的外层模型，未完成拟合从固定seed重跑。已完成CV禁止重新覆盖，不要修改快照代码后试图resume。

训练结束后独立验证：

```bash
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/verify_artifacts.py \
  versions/xyy/v1.5.8_strict_baseline/runs/v18_tree_nested_v2_split42_model42
```

结果分析与一致性检查：

```bash
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/analyze_cv_results.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/verify_analysis.py
```

分析脚本要求CV_COMPLETE存在且所有冻结文件哈希正确。它不会训练模型或修正预测；D部分只统计违反已知下限的情况，不评估投影后的候选成绩。

## 解读原则

同一严格CV内部的C0/C1/C2/C3及固定v18_rule可以配对比较。历史v1.8采用外折早停，执行设置也不同；新旧分数差只作口径变化的描述，不能直接当成新特征收益或旧选择偏差的无偏估计。

固定v18_rule仍为1/6h使用C3、24/48h使用C1。即使本次其他控制某项更低，也不在这一轮事后改路由。其他两套CV和最终全量训练不属于本轮。
