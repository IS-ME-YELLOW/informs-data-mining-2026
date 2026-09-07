# v1.7：CatBoost 模型族对照

日期：2026-09-07

## 实验定义

- 直接读取冻结的 `cache/features_*_v1.5.2.parquet`（141 列），不重新构建特征。
- 使用 `cv/cv_assignments_balanced_v1_seed42.csv` 中固定的县级五折，与 v1.5.4/v1.5.2 LightGBM 完全相同。
- 四个预测期独立训练；最大 2,000 轮，验证 RMSE 100 轮不改善则早停。
- `Huber:delta=0.01`；学习率 0.05，深度 6，`rsm=0.8`，L2 叶正则为 3。
- OOF 与提交都裁剪到 `[0, 0.65]`，再将小于 `0.001` 的预测置零。
- 最终全量模型的轮数为五折最佳轮数的四舍五入均值。

v1.5.2 全部输入都是数值特征，因而本实验没有 CatBoost 原生类别特征；它检验的是 CatBoost 的对称树与正则化机制，而不是类别编码优势。

## 结果

主指标为池化 OOF。括号是相对固定 LightGBM 基线的变化，负值表示更好。

| 预测期 | CatBoost RMSE | 相对 LightGBM | CatBoost MAE | 相对 LightGBM |
|---|---:|---:|---:|---:|
| t+1h | **0.012179** | **-0.43%** | **0.003534** | **-3.30%** |
| t+6h | 0.010937 | +2.43% | **0.003267** | **-7.66%** |
| t+24h | 0.008669 | +0.37% | **0.002649** | **-9.70%** |
| t+48h | 0.007787 | +1.30% | **0.002098** | **-8.49%** |

CatBoost 仅在 t+1h RMSE 上小幅优于 LightGBM，但四个 MAE 均改善；其中 t+6h、t+24h、t+48h MAE 是本轮三种模型中最低。它适合作为融合候选和低绝对误差方案，不应按全部 horizon 直接替换 LightGBM。

## 复现

```powershell
python -m pip install -r versions/v1.7/requirements.txt
python versions/v1.7/run_catboost.py
```

本次实际运行环境为 Python 3.11.13、CatBoost 1.2.10。若依赖装在仓库根目录的 `.python_packages/`，脚本会自动加载。

## 产物

- `run_catboost.py`：模型适配器与命令行入口；通用实验协议在 `versions/gbdt_experiment.py`。
- `versions/verify_gbdt_artifacts.py`：复算 OOF 指标并重新加载模型核对提交值。
- `summary_metrics.csv`、`fold_metrics.csv`：池化与逐折结果。
- `oof_predictions.csv`：每个训练行的真实值和严格 OOF 预测，可直接用于融合/残差分析。
- `feature_importance.csv`：四个模型的完整重要性排序。
- `run_metadata.json`：参数、版本、输入文件 SHA-256、基线和预测分布。
- `catboost_*.cbm`：四个全量训练模型。
- `submission_v1.7_balanced_v1.csv`：已验证的 9,072 行提交文件。

注意：CatBoost 的特征重要性定义与 LightGBM/XGBoost 不同，数值不可跨模型族直接比较；应比较同一模型内的排序，或另行计算统一的 permutation/SHAP 指标。
