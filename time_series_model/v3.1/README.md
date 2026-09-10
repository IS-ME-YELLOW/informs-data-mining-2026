# v3.1 时序残差模型

## 当前状态

截至 2026-09-09，里程碑 M1、M2、M3 已完成。

- M1：将原始小时数据整理为县级定长序列缓存；
- M2：实现双路 GRU 残差模型、折内预处理、掩码损失和检查点保存；
- M3：生成诊断级 LightGBM OOF、GRU OOF、测试预测、指标及提交文件，并完成独立重载复算。

所有 M3 结果均标记为 `diagnostic_only`。它们用于确认工程闭环和观察模型行为，不能写入根目录 `Results.md` 作为正式模型收益。正式比较需要完成计划中的 M4 嵌套交叉验证。

完整设计见 [Plan_v3.1.md](Plan_v3.1.md)，本次实现与结果见 [M3_Implementation_Report_2026-09-09.md](M3_Implementation_Report_2026-09-09.md)。

## 运行顺序

以下命令均在项目根目录执行：

```powershell
python time_series_model\v3.1\build_sequences.py --overwrite
python time_series_model\v3.1\baseline_crossfit.py --overwrite
python time_series_model\v3.1\train_diagnostic.py --max-epochs 30 --overwrite
python time_series_model\v3.1\verify_artifacts.py
```

仅检查已有序列缓存时：

```powershell
python time_series_model\v3.1\build_sequences.py --verify
```

## 主要产物

`artifacts/` 中保存：

- `sequences_train_v3.1.npz`、`sequences_test_v3.1.npz`：县级定长序列；
- `sequence_manifest.json`：字段、形状、输入与输出哈希；
- `baseline_predictions_v3.1.npz`：v1.5.6 LightGBM 原始 OOF 和测试预测；
- `checkpoints/baseline_*.txt`：20 个 LightGBM 折模型；
- `checkpoints/gru_residual_fold*.pt`：5 个 GRU 折模型及各自预处理参数；
- `diagnostic_predictions.npz`：基线、残差与组合预测矩阵；
- `oof_predictions.parquet`、`test_predictions.parquet`：可逐行审计的预测；
- `fold_metrics.csv`、`summary_metrics.csv`：逐折与汇总指标；
- `run_metadata.json`：运行配置、训练历史和文件哈希；
- `verification.json`：独立复算结果。

提交格式文件为 `submission_v3.1_balanced_v1.csv`。

## 依赖说明

版本记录在 `requirements.txt`。CPU 版 PyTorch 可从官方 CPU wheel 源安装：

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
```

当前机器无法通过证书校验下载 SciPy 和 SymPy，因此本版本对受影响部分采用两个局部兼容措施：LightGBM 的密集 DataFrame 路径使用最小 `scipy.sparse` 类型占位；训练使用标准公式的本地 AdamW 实现。这两个措施只服务于当前精简环境，算法配置仍为 LightGBM 与 AdamW。在依赖完整的环境中可以换回库自带实现，并在 M4 做一次数值一致性检查。

