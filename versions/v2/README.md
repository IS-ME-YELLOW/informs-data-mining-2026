# v2：先预测 P/N/D/R，再计算 OSI

日期：2026-09-07

## 路线定义

v1.5.4/v1.6/v1.7 都直接回归四个 `osi_target_*`。v2 改为对每个 horizon 分别预测目标时刻的四个 OSI 分量：

```text
X -> P_hat(t+h), N_hat(t+h), D_hat(t+h), R_hat(t+h)
  -> max(0, 0.40*P_hat + 0.35*N_hat + 0.25*D_hat - 0.10*R_hat)
  -> clip [0, 0.65] -> 小于 0.001 置零
```

三个子版本为：

| 版本 | 模型族 | 入口 |
|---|---|---|
| v2.1 | LightGBM | `v2.1/run_lightgbm.py` |
| v2.2 | XGBoost | `v2.2/run_xgboost.py` |
| v2.3 | CatBoost | `v2.3/run_catboost.py` |

每个版本训练 4 个 horizon × 4 个分量，即 16 个独立最终模型；交叉验证阶段对应 80 个折模型。全部使用冻结的 v1.5.2 141 列特征、`balanced_v1` 固定县级五折和 seed=42。

## 分量目标数据

`build_component_targets.py` 从 `DM_Train.csv` 在县内按 horizon 移位得到 16 个分量目标，保存为 `component_targets_v2.parquet`。它不使用测试预测窗口中隐藏的分量，也不会把验证折的真实分量用于合成预测。

四分量重算值与官方四位小数 `osi_target_*` 的最大绝对差约为 `0.00005001`，RMSE 约为 `0.000024`。该微小差异来自官方 OSI 目标的四位小数舍入；行数、NaN 掩码和目标时刻均已验证一致。详细证据见 `component_target_validation.json`。

## 正式结果

下表为合成 OSI 的池化 OOF 指标。

| Horizon | v2.1 LightGBM RMSE / MAE | v2.2 XGBoost RMSE / MAE | v2.3 CatBoost RMSE / MAE |
|---|---:|---:|---:|
| t+1h | 0.011958 / 0.003591 | **0.011943 / 0.003404** | 0.012206 / 0.003415 |
| t+6h | **0.010597** / 0.003412 | 0.010685 / **0.003210** | 0.010913 / 0.003214 |
| t+24h | **0.008451** / 0.002918 | 0.008666 / 0.002648 | 0.008529 / **0.002585** |
| t+48h | **0.007641** / 0.002291 | 0.007852 / 0.002065 | 0.007647 / **0.002047** |

相对各自的直接 OSI 版本：

- v2.1 LightGBM：四个 RMSE 均改善 `0.60%–2.24%`。
- v2.2 XGBoost：t+1h/t+6h RMSE 轻微回退 `0.36%/0.28%`，t+24h/t+48h 改善 `0.75%/1.14%`；四个 MAE 均改善。
- v2.3 CatBoost：t+1h RMSE 回退 `0.22%`，其余改善 `0.22%–1.80%`；四个 MAE 均改善。

在全部直接与分量模型中，t+1h RMSE 仍由直接 XGBoost v1.6 最优；t+6h/t+24h/t+48h RMSE 均由 v2.1 最优。MAE 则由 v2.2 赢得 t+1h/t+6h，v2.3 赢得 t+24h/t+48h。

## 重要限制

四个分量是独立训练的，因此预测结果不强制满足以下动态恒等关系：

- `N_t/R_t` 应与连续两小时 `P_t` 或 outageCount 的变化一致；
- `D_t` 应是 `P_t` 的六小时滚动均值；
- 四分量联合误差最小不等于每个分量单独 RMSE 最小。

因此 v2 是有效的受控对照和候选路线，但后续可继续测试“联合损失/结构约束”或直接模型与分量模型的 OOF 融合。

## 复现和校验

```powershell
python versions/v2/build_component_targets.py
python versions/v2/v2.1/run_lightgbm.py
python versions/v2/v2.2/run_xgboost.py
python versions/v2/v2.3/run_catboost.py
python versions/v2/verify_v2_artifacts.py
```

共用协议位于 `component_experiment.py`，模型适配器位于 `adapters.py`。每个子版本都包含 16 个模型、分量/OSI 折指标、完整 OOF、测试分量预测、特征重要性、提交及运行元数据。
