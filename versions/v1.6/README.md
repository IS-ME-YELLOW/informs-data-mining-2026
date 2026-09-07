# v1.6：XGBoost 模型族对照

日期：2026-09-07

## 实验定义

- 直接读取冻结的 `cache/features_*_v1.5.2.parquet`（141 列），不重新构建特征。
- 使用 `cv/cv_assignments_balanced_v1_seed42.csv` 中固定的县级五折，与 v1.5.4/v1.5.2 LightGBM 完全相同。
- 四个预测期独立训练；最大 2,000 轮，验证 RMSE 100 轮不改善则早停。
- `reg:pseudohubererror`，`huber_slope=0.01`；学习率 0.05，深度 6，行/列采样均为 0.8。
- OOF 与提交都裁剪到 `[0, 0.65]`，再将小于 `0.001` 的预测置零。
- 最终全量模型的轮数为五折最佳轮数的四舍五入均值。

## 结果

主指标为池化 OOF。括号是相对固定 LightGBM 基线的变化，负值表示更好。

| 预测期 | XGBoost RMSE | 相对 LightGBM | XGBoost MAE | 相对 LightGBM |
|---|---:|---:|---:|---:|
| t+1h | **0.011900** | **-2.71%** | **0.003488** | **-4.54%** |
| t+6h | **0.010655** | **-0.20%** | **0.003298** | **-6.78%** |
| t+24h | 0.008732 | +1.10% | **0.002734** | **-6.83%** |
| t+48h | 0.007942 | +3.32% | **0.002104** | **-8.25%** |

XGBoost 在短预测期 RMSE 上有价值，且四个 MAE 都优于 LightGBM；但 t+24h/t+48h RMSE 回退，说明尾部大误差控制较弱。它适合进入按 horizon 的 OOF 融合候选，不应整体替换 LightGBM。

## 复现

```powershell
python -m pip install -r versions/v1.6/requirements.txt
python versions/v1.6/run_xgboost.py
```

本次实际运行环境为 Python 3.11.13、XGBoost 3.2.0。若依赖装在仓库根目录的 `.python_packages/`，脚本会自动加载。

## 产物

- `run_xgboost.py`：模型适配器与命令行入口；通用实验协议在 `versions/gbdt_experiment.py`。
- `versions/verify_gbdt_artifacts.py`：复算 OOF 指标并重新加载模型核对提交值。
- `summary_metrics.csv`、`fold_metrics.csv`：池化与逐折结果。
- `oof_predictions.csv`：每个训练行的真实值和严格 OOF 预测，可直接用于融合/残差分析。
- `feature_importance.csv`：四个模型的完整重要性排序。
- `run_metadata.json`：参数、版本、输入文件 SHA-256、基线和预测分布。
- `xgboost_*.json`：四个全量训练模型。
- `submission_v1.6_balanced_v1.csv`：已验证的 9,072 行提交文件。

注意：部分折的最佳轮数接近 2,000 轮上限，因此未来可把“提高轮数上限”作为单独调参实验；本版保持 2,000 轮是为了与 v1.5.4 的训练预算一致。
