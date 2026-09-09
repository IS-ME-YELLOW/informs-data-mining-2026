# v2.1：LightGBM 分量预测

使用 v1.5.2 冻结特征和 `balanced_v1` 固定县级五折，分别预测每个 horizon 的 `P_t/N_t/D_t/R_t`，再按官方权重合成 OSI。共保存 16 个 LightGBM 模型。

| Horizon | 分量合成 RMSE | 直接 LightGBM RMSE | 变化 | 分量合成 MAE | 直接 MAE | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| t+1h | **0.011958** | 0.012232 | **-2.24%** | **0.003591** | 0.003654 | **-1.73%** |
| t+6h | **0.010597** | 0.010677 | **-0.75%** | **0.003412** | 0.003538 | **-3.56%** |
| t+24h | **0.008451** | 0.008637 | **-2.16%** | **0.002918** | 0.002934 | **-0.55%** |
| t+48h | **0.007641** | 0.007687 | **-0.60%** | **0.002291** | 0.002293 | **-0.09%** |

v2.1 是本轮 RMSE 最稳定的分量路线，四个 horizon 全部改善；它也给出全部六套模型中最好的 t+6h、t+24h、t+48h RMSE。

复现：

```powershell
python -m pip install -r versions/v2/v2.1/requirements.txt
python versions/v2/v2.1/run_lightgbm.py
```

主要数据文件：`osi_summary_metrics.csv`、`component_summary_metrics.csv`、`oof_osi_predictions.csv`、`oof_component_predictions.csv`、`test_component_predictions.csv`、`feature_importance.csv`、`run_metadata.json` 和 `submission_v2.1_balanced_v1.csv`。
