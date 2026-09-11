# v1.9：分量专用 LightGBM 与县域响应画像

v1.9 在 v1.8 分量建模框架上完成了两组实验：

1. 为 `P/N/D/R` 分别尝试适合其分布的 LightGBM 目标；
2. 从预测时点之前的 72 小时观测构建 27 个县域响应画像特征。

本轮完成 M1–M4，模型、OOF、测试预测、诊断提交和独立校验均已保存。四个 horizon 的 OOF 最优组合相对当前 v1.8 规则分别改善 `0.54% / 0.68% / 0.61% / 0.03%`，但县级配对 bootstrap 的 95% 区间全部跨过 0。按方案中的晋级门槛，v1.9 暂不替换 v1.8，也不进入 M5 的正式候选阶段。

## 主要结论

- 响应画像确实被模型使用，但只在 `t+6` 的整体 Huber 对照中转化为约 `0.49%` 的 OSI RMSE 改善；其余 horizon 退化。
- `P/D` 的 L2 候选对最终 OSI 有一些帮助，尾部加权 L2 同时损伤 RMSE 和 MAE，没有达到预期。
- `N/R` 的 Tweedie 与 hurdle 显著降低 MAE，却通常提高 RMSE，说明零值/小值学习改善了，但少数大值误差被放大。
- 最有希望的是 `t+6` 的画像分量组合，但相对当前规则的改善概率为 `90.6%`，95% 区间仍跨 0，只能作为后续定向实验的候选。

详细分析见 [Report_v1.9.md](./Report_v1.9.md)，原定方案见 [Plan_v1.9.md](./Plan_v1.9.md)。

## 复现顺序

在项目根目录依次运行：

```powershell
python versions/v1.9/build_response_profiles.py
python versions/v1.9/train_v19.py
python versions/v1.9/evaluate_v19.py
python versions/v1.9/verify_artifacts.py
```

`train_v19.py --resume` 会复用已保存且配置一致的折模型；全量模型的轮数采用五折最佳轮数的均值，与 v1.8 保持一致。

## 关键资产

- 特征数据：`features_train_v1.9.parquet`、`features_test_v1.9.parquet`
- 响应画像：`response_profiles_train_v1.9.parquet`、`response_profiles_test_v1.9.parquet`
- 模型：`models/`，共 576 个文件
- OOF 与测试预测：`oof_predictions.parquet`、`test_predictions.parquet`
- 指标：`summary_metrics.csv`、`component_metrics.csv`、`paired_county_bootstrap.csv`
- 诊断提交：`diagnostic_submission_v1.9_*.csv`
- 独立校验：`verification.json`

诊断提交仅用于保存完整实验资产，不代表已选定正式提交。
