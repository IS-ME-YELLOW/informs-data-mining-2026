# P6 a/b迁移确认

**状态：seed20260917、seed20260918的P6确认完成，独立核验PASS。** [三套结果](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/Results_2026-09-19.md)。

支持将P6 a/b纳入后续实验的优先组合。首轮seed42产物保持只读；本轮每套50次拟合、10个外层模型，共100次拟合、20个模型。只对P6使用原163列a/b，模型seed固定42，没有联合候选或新特征。

`confirmation_runtime.py`要求显式`--split-seed`，每个CV输出到自己的split目录；代码共用，模型和配置不共用。适配审计记录在reference，数值函数保持原样。

首次执行顺序：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/prepare_confirmation.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/train_transfer.py --stage preflight --split-seed 20260917
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/train_transfer.py --stage preflight --split-seed 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/run_confirmations.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/summarize_confirmation.py
```

已完成训练/评估stage拒绝覆盖。重新核验某套可单独运行`verify_transfer.py --split-seed 20260917`（或20260918）；不拟合新模型。

- `reference/`：父产物哈希和AST适配审计。
- `split*/reference/`：各CV来源哈希、预检；`split*/experiment_config.json`：完整配置。
- `split*/runs/v18_ptransfer_ab_v1_split*_model42/`：模型、内层逐行预测和曲线、全OOF、所有控制、指标与独立验证。
- `split*/summary/`：本CV指标及误差交叉项；`summary/`：三CV按split汇总。
- `final_integrity_check.json`、`completion.json`：本轮完整性与完成清单。

没有改写父目录README、结果或完成标记；确认后的状态以本目录入口为准。没有全量模型或提交文件。
