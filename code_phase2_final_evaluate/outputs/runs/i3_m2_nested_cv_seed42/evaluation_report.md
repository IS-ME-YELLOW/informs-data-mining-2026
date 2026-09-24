# I3 baseline vs. m2_robust_input GAT 嵌套外层 CV 结果

运行目录：`i3_m2_nested_cv_seed42`

## 评价协议

- 模型：冻结 I3 scope-aware baseline + final `m2_robust_input` residual GAT；
- CV：`code_phase2_compare` seed-42 县级五折，CV SHA-256 为
  `2ee47b74590051cec9d0d1ed0ac7d7a640250c8370433b1dfdac9b281ffe4021`；
- 每个外层 fold 只使用其余四个 fold 训练 GAT；alpha 只在外层训练数据内部 OOF 选择；
- 评价行数：`outer_oof.parquet` 共 137,664 行；实际 scoreable 行按 horizon 为 34,177 / 32,982 / 28,680 / 22,944；
- 指标在同一批外层验证行上计算，GAT 和 baseline 使用相同后处理；
- `rmse_improvement_vs_baseline` 和 `mae_improvement_vs_baseline` 定义为 baseline 减 GAT，正值表示 GAT 更好。

## Pooled 外层 OOF 指标

| Horizon | N | Baseline RMSE | GAT RMSE | RMSE 改善 | Baseline MAE | GAT MAE | MAE 改善 | Baseline R² | GAT R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t01h | 34,177 | 0.01084069 | 0.01061279 | 0.00022790 (2.10%) | 0.00322729 | 0.00317294 | 0.00005435 (1.68%) | 0.773414 | 0.782841 |
| t06h | 32,982 | 0.00990906 | 0.00985497 | 0.00005410 (0.55%) | 0.00310047 | 0.00308711 | 0.00001336 (0.43%) | 0.733750 | 0.736649 |
| t24h | 28,680 | 0.00841429 | 0.00821388 | 0.00020042 (2.38%) | 0.00268345 | 0.00259282 | 0.00009063 (3.38%) | 0.480975 | 0.505405 |
| t48h | 22,944 | 0.00766153 | 0.00749934 | 0.00016218 (2.12%) | 0.00210679 | 0.00207955 | 0.00002724 (1.29%) | 0.365774 | 0.392341 |

## 外层 fold 稳定性

GAT 相对 baseline 的 RMSE 改善 fold 数为：

| Horizon | RMSE 改善 fold 数 | RMSE 变差 fold 数 | MAE 改善 fold 数 |
|---|---:|---:|---:|
| t01h | 3/5 | 2/5 | 3/5 |
| t06h | 3/5 | 2/5 | 3/5 |
| t24h | 4/5 | 1/5 | 5/5 |
| t48h | 3/5 | 2/5 | 4/5 |

内部选择的 alpha：

```text
t01h: 0.35, 0.35, 0.20, 0.35, 0.50
t06h: 0.35, 0.50, 0.20, 0.20, 0.35
t24h: 0.35, 0.35, 0.50, 0.35, 0.35
t48h: 0.50, 0.35, 0.50, 0.35, 0.50
```

## 县级 paired bootstrap

Bootstrap 按县重采样 2,000 次；`delta_rmse_gat_minus_base` 为 GAT RMSE 减 baseline RMSE，负值表示 GAT 更好。

| Horizon | Delta RMSE | 95% CI | GAT 更好概率 |
|---|---:|---:|---:|
| t01h | -0.00022790 | [-0.00061017, 0.00018867] | 0.8250 |
| t06h | -0.00005410 | [-0.00030757, 0.00019787] | 0.6665 |
| t24h | -0.00020042 | [-0.00069249, 0.00024254] | 0.7760 |
| t48h | -0.00016218 | [-0.00094232, 0.00039452] | 0.6235 |

四个 horizon 的 95% 区间均包含 0。因此结果支持 GAT 有小幅改善趋势，尤其是 t01h、t24h 和 t48h，但按照本次县级 bootstrap，不能宣称相对于 I3 baseline 已达到明确的 95% 统计显著改善。

## 结论

在严格的嵌套外层 CV 下，`m2_robust_input` GAT 的 pooled RMSE 和 MAE 在四个 horizon 均优于冻结 I3 baseline；最大 pooled RMSE 相对改善为 t24h 的 2.38%，最小为 t06h 的 0.55%。改善并非所有外层 fold 都稳定，且县级 bootstrap 区间均跨 0，应将结论写为“预测性能有小幅、方向一致的改善，但证据不足以支持 95% 显著性结论”。

详细原始证据：`outer_oof.parquet`、`fold_metrics.csv`、`cv_summary.csv`、`county_metrics.csv`、`county_bootstrap.csv` 和 `run_manifest.json`。
