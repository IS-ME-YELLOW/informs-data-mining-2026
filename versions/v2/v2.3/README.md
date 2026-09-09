# v2.3：CatBoost 分量预测

使用 v1.5.2 冻结特征和 `balanced_v1` 固定县级五折，分别预测每个 horizon 的 `P_t/N_t/D_t/R_t`，再按官方权重合成 OSI。共保存 16 个 CatBoost 模型。

| Horizon | 分量合成 RMSE | 直接 CatBoost RMSE | 变化 | 分量合成 MAE | 直接 MAE | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| t+1h | 0.012206 | **0.012179** | +0.22% | **0.003415** | 0.003534 | **-3.34%** |
| t+6h | **0.010913** | 0.010937 | **-0.22%** | **0.003214** | 0.003267 | **-1.62%** |
| t+24h | **0.008529** | 0.008669 | **-1.61%** | **0.002585** | 0.002649 | **-2.45%** |
| t+48h | **0.007647** | 0.007787 | **-1.80%** | **0.002047** | 0.002098 | **-2.43%** |

除 t+1h RMSE 轻微回退外，分量路线改善了 CatBoost 的其余指标。v2.3 给出全部六套模型中最好的 t+24h 和 t+48h MAE。

复现：

```powershell
python -m pip install -r versions/v2/v2.3/requirements.txt
python versions/v2/v2.3/run_catboost.py
```

主要数据文件：`osi_summary_metrics.csv`、`component_summary_metrics.csv`、`oof_osi_predictions.csv`、`oof_component_predictions.csv`、`test_component_predictions.csv`、`feature_importance.csv`、`run_metadata.json` 和 `submission_v2.3_balanced_v1.csv`。
