# v1.8：v1.5.6 特征上的分量 LightGBM

本版固定使用 v1.5.6 的 163 列特征和 `balanced_v1` 县级五折，比较直接预测 OSI、预测 P/N/D/R 后重组、两者等权平均，以及同目标小时分量对齐。没有增加 v1.5.7 F1 或其他新未来天气特征。

## 主要结果

| Horizon | C0 直接 OSI | C1 分量重组 | C2 等权融合 | C3 同目标小时分量对齐 | 最低 RMSE |
|---|---:|---:|---:|---:|---|
| t+1h | 0.011979 | 0.011865 | 0.011901 | **0.011677** | C3 |
| t+6h | 0.010677 | 0.010513 | 0.010567 | **0.010321** | C3 |
| t+24h | 0.008464 | 0.008373 | 0.008403 | **0.008330** | C3 |
| t+48h | 0.007644 | **0.007593** | 0.007608 | 0.007667 | C1 |

C1 四个 horizon 的 RMSE 都优于 C0，说明分量监督在 v1.5.6 特征上仍然成立。C2 的点估计没有超过 C1，但其四个 horizon 相对 C0 的县级 bootstrap 95% 区间均低于 0，是最稳健的单一组合。C3 在 t+1 和 t+6 上的额外提升最可靠；t+48 的 RMSE 回退 0.295%，不应统一使用 C3。

## 复现

标准环境：

```powershell
python -m pip install -r versions/v1.8/requirements.txt
python versions/v1.8/train_v18.py
python versions/v1.8/verify_artifacts.py
```

如果训练完成后只需从已保存模型继续生成评估产物：

```powershell
python versions/v1.8/train_v18.py --resume
```

当前工作站已有 LightGBM 和 PyArrow，但缺少 SciPy。为完成本次稠密 DataFrame 训练，`_compat/scipy` 只提供 LightGBM 导入所需的稀疏类型占位；它不实现稀疏矩阵操作。安装完整 requirements 后不需要使用该兼容层。

## 主要文件

- `Plan_v1.8.md`：实验方案和决策规则。
- `Report_v1.8.md`：完整结果、bootstrap、解释和结论。
- `train_v18.py`、`protocol.py`、`config.py`：训练与统一协议。
- `verify_artifacts.py`、`verification.json`：模型重载与产物核验。
- `models/cv/`：100 个五折模型；`models/final/`：20 个全量模型。
- `summary_metrics.csv`、`paired_county_bootstrap.csv`：主指标与不确定性。
- `oof_predictions.parquet`、`oof_component_predictions.parquet`：完整 OOF。
- `submission_v1.8_*.csv`：C0/C1/C2/C3 四套合规提交。
- `run_metadata.json`、`model_manifest.csv`：配置、输入和模型哈希。
