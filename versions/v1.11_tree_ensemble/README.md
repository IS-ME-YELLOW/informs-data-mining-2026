# v1.11：直接/分量树模型严格嵌套融合

> 后续更新：本版本的尾部诊断促成了 `versions/v1.12_tail_protected_ensemble/`。v1.12 在保持 t+1/t+6 不变的同时，使 t+24/t+48 RMSE 分别改善 1.957%/0.567%，现为当前内部 OOF 最佳候选。本页“不晋级、保留 v1.8”仅指 v1.11 的全局静态融合本身。

本实验复用六路既有树模型 OOF。所有 34,416 行在县、时间戳、小时索引、州、四个 horizon 的真实值与缺失掩码上完全对齐；五个来源元数据均为 v1.5.6、`balanced_v1`、`seed=42`。

## 主要结果

下表为严格外层 OOF。括号是相对 v1.8 当前规则的变化，负数为改善。

| Horizon | v1.8 RMSE | 简单平均 | 静态凸融合 | 安全凸融合 | 安全融合改善折数 |
|---|---:|---:|---:|---:|---:|
| t+1 | 0.011677 | 0.011763 (+0.74%) | 0.011821 (+1.23%) | 0.011813 (+1.17%) | 3/5 |
| t+6 | 0.010321 | 0.010424 (+1.00%) | 0.010393 (+0.70%) | 0.010391 (+0.68%) | 2/5 |
| t+24 | 0.008373 | 0.008425 (+0.62%) | 0.008444 (+0.85%) | 0.008438 (+0.77%) | 1/5 |
| t+48 | 0.007593 | 0.007700 (+1.41%) | 0.007687 (+1.23%) | 0.007674 (+1.06%) | 3/5 |

安全融合的 MAE 分别改善 2.14%、1.01%、3.40%、3.16%，但 RMSE 四项全部退化。县级 paired bootstrap 的 RMSE 差 95% 区间分别为 `[-0.000061, 0.000458]`、`[-0.000087, 0.000256]`、`[-0.000079, 0.000199]`、`[-0.000081, 0.000230]`，全部跨 0 且点估计为正。

## 诊断

- 六路残差相关很高：t+1/t+6/t+24/t+48 的范围分别为 0.936–0.992、0.918–0.992、0.943–0.994、0.969–0.995。
- 外折权重平均仍给 v1.8 规则 0.67–0.72 权重；安全系数均值为 0.94、0.94、0.91、0.82，但这种回退仍不足以抵消县间分布变化。
- 高 OSI（各 horizon 真实值前 5%）安全融合 RMSE 均退化；t+48 从 0.031746 升至 0.032297。t+48 退化最大的县为 54013、54015、42045。
- convex-hull oracle 仅作诊断，覆盖 60.3%–62.8% 样本，RMSE 可降至 0.010283/0.008974/0.007451/0.007032。它说明逐样本选择存在理论空间，但静态权重无法利用，不能当作可报告模型成绩。
- 简单平均显著改善 MAE，却损害高值尾部 RMSE；当前差异更像损失函数/县域条件差异，而不是可由固定全局权重稳定获取的互补性。

## 决策

不晋级，继续以 v1.8 当前规则为正式 RMSE 基线。现阶段不扩大 ExtraTrees/Random Forest 搜索；若以后尝试 gating，只应使用很浅、预先限定的门控，并完整嵌套在县级外折内，重点约束高 OSI 与 t+48。

## 复现与验证

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe versions/v1.11_tree_ensemble/run_ensemble.py
C:\Users\Stella\.conda\envs\best-route\python.exe versions/v1.11_tree_ensemble/verify_artifacts.py
```

独立验证重载 20 个外折模型和 4 个最终模型；OOF/测试/提交最大差异 `9.9963e-17`，输入及产物哈希全部通过。

主要文件：`artifacts/oof_predictions.parquet`、`artifacts/test_predictions.csv`、`results/summary_metrics.csv`、`results/fold_metrics.csv`、`results/county_metrics.csv`、`results/paired_county_bootstrap.csv`、`results/high_osi_metrics.csv`、`results/convex_hull_oracle.csv`、`results/weights.csv`、`artifacts/run_metadata.json` 和 `artifacts/verification.json`。
