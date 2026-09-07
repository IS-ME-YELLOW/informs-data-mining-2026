# v2.2：XGBoost 分量预测

使用 v1.5.2 冻结特征和 `balanced_v1` 固定县级五折，分别预测每个 horizon 的 `P_t/N_t/D_t/R_t`，再按官方权重合成 OSI。共保存 16 个 XGBoost 模型。

| Horizon | 分量合成 RMSE | 直接 XGBoost RMSE | 变化 | 分量合成 MAE | 直接 MAE | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| t+1h | 0.011943 | **0.011900** | +0.36% | **0.003404** | 0.003488 | **-2.41%** |
| t+6h | 0.010685 | **0.010655** | +0.28% | **0.003210** | 0.003298 | **-2.67%** |
| t+24h | **0.008666** | 0.008732 | **-0.75%** | **0.002648** | 0.002734 | **-3.12%** |
| t+48h | **0.007852** | 0.007942 | **-1.14%** | **0.002065** | 0.002104 | **-1.83%** |

分量路线改善了 XGBoost 的所有 MAE 和两个长 horizon 的 RMSE，但短 horizon RMSE 基本持平略退。多个 P/D 折的最佳轮数接近 2,000 上限，后续可把更高轮数上限作为单独调参实验。

复现：

```powershell
python -m pip install -r versions/v2/v2.2/requirements.txt
python versions/v2/v2.2/run_xgboost.py
```

主要数据文件：`osi_summary_metrics.csv`、`component_summary_metrics.csv`、`oof_osi_predictions.csv`、`oof_component_predictions.csv`、`test_component_predictions.csv`、`feature_importance.csv`、`run_metadata.json` 和 `submission_v2.2_balanced_v1.csv`。
