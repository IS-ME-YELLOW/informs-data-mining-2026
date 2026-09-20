# P a/b结构三来源独立迁移

**状态：seed42完成，150次拟合、30个新模型，独立核验PASS。** [结果报告](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/Results_2026-09-19.md)。

候选仅B0、AB_P06、AB_P24、AB_P48；没有联合替换、直接P-L2对照或邻县特征增量。B0是完整F2，P1h已有的F2仍保留，新训练P6/P24/P48只用原163列。

P6迁移使整体1h/6h RMSE下降约1.26%/1.55%，但C3后P6分量RMSE略升；P24/P48的同名整体RMSE分别上升约2.73%/3.95%。建议只优先确认P6，尚未替换当前模型。

首次执行命令如下；完成stage拒绝覆盖：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/train_transfer.py --stage preflight
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/train_transfer.py --stage train
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/evaluate_transfer.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/verify_transfer.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/summarize_results.py
```

`--resume`只适用于同代码/配置/来源身份的未完成train。单独运行verify_transfer.py可重新验收，不拟合模型。

- `reference/`：源文件哈希、P71/端点/外折隔离和原E2一致性预检。
- `runs/v18_ptransfer_ab_v1_split42_model42/models/`：各外折、各horizon的a/b模型和receipt。
- 同run的`logs/probes/`：内层验证逐行分支标签、权重及最佳轮次预测；完整曲线保存在模型receipt。
- 同run的`branch_oof_predictions.parquet`：原始/截断a/b、贡献、支持、P重建与模型来源；`reconstruction_registry.json`关联两个分支与P71来源。
- 同run的`component_predictions.parquet`、`control_predictions.parquet`和`aligned_unique_components.parquet`：全体239县的完整单来源候选结果。
- `summary/`：所有指标、固定7县及误差交叉项；`Results_2026-09-19.md`为结论。
- 训练、评估、验收完成标记与顶层`completion.json`记录本轮身份；没有全量提交模型。

统计与逐行预测均使用项目venv。没有外部下载或新增依赖。全部新增写入限定本目录，旧产物只读。
