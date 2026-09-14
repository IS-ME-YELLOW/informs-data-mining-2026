# v2.6：P/N/D/R 树模型严格嵌套融合

本实验对 4 个 horizon × 4 个分量分别融合 LightGBM、XGBoost 和 CatBoost，再按官方公式重组 OSI。三个来源的 34,416 条 OOF 在键、真实分量和缺失掩码上完全一致，元数据均为 v1.5.6、`balanced_v1`、`seed=42`。

## 重组 OSI 结果

| Horizon | v1.8 RMSE | 简单平均 | 静态凸融合 | 安全凸融合 | 安全融合改善折数 |
|---|---:|---:|---:|---:|---:|
| t+1 | 0.011677 | 0.011775 (+0.84%) | 0.011920 (+2.09%) | 0.011920 (+2.09%) | 1/5 |
| t+6 | 0.010321 | 0.010501 (+1.75%) | 0.010548 (+2.20%) | 0.010548 (+2.20%) | 1/5 |
| t+24 | 0.008373 | 0.008386 (+0.16%) | 0.008433 (+0.72%) | 0.008433 (+0.72%) | 1/5 |
| t+48 | 0.007593 | 0.007658 (+0.85%) | 0.007659 (+0.86%) | 0.007659 (+0.86%) | 4/5 |

安全融合 MAE 分别改善 2.62%、1.41%、4.07%、6.67%，但所有 RMSE 点估计均退化；paired bootstrap RMSE 区间全部跨 0。t+48 虽有 4/5 折方向改善，少数高损失县仍把池化 RMSE 推高。

## 分量诊断

- 简单平均在 P 的 t+1/t+6/t+24、D 的 t+1/t+6/t+24，以及 N/R 多数 horizon 上带来小幅 RMSE 收益；D t+48 则明显退化。
- 严格嵌套静态权重在县折间不稳定，导致 P/D 多个 horizon 的外折成绩不如简单平均，说明同一组全局分量权重不能稳定跨县迁移。
- N/R 的真实零值占约 50.6%–62.2%。安全融合显著降低零值子集误差和整体 MAE，但 N 的所有正值前 5% 尾部 RMSE 都变差；R 的尾部也大多轻微变差。这正是“零值更好、尾部更差”的典型取舍。
- 分量残差相关仍高：P 为 0.922–0.981，D 为 0.914–0.972，N/R 约 0.978–0.995，可稳定利用的融合空间有限。
- 高 OSI 子集四个 horizon 的 RMSE 全部不如 v1.8，t+48 为 0.032333 对 0.031746。

## 决策

不晋级，不替换 v1.8。逐分量独立最小化 RMSE 不能保证经 `-0.10*R` 和后处理重组后的 OSI RMSE 最优。当前不建议继续大量树族组合；如果再做门控，应以 OSI 联合目标、高值尾部约束和完整嵌套 CV 为前提，而不是继续逐分量独立选权重。

## 复现与验证

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe versions/v2/v2.6_component_tree_ensemble/run_component_ensemble.py
C:\Users\Stella\.conda\envs\best-route\python.exe versions/v2/v2.6_component_tree_ensemble/verify_artifacts.py
```

独立验证重载 80 个外折模型和 16 个最终模型；OOF/测试/提交最大差异 `9.9991e-17`，输入及产物哈希全部通过。

主要文件：`artifacts/oof_component_and_osi_predictions.parquet`、`results/component_summary_metrics.csv`、`results/component_zero_tail_metrics.csv`、`results/osi_summary_metrics.csv`、`results/osi_fold_metrics.csv`、`results/paired_county_bootstrap.csv`、`results/high_osi_metrics.csv`、`results/weights.csv`、`artifacts/run_metadata.json` 和 `artifacts/verification.json`。
