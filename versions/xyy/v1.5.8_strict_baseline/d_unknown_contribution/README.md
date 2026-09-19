# D 未知贡献监督实验

**seed42 已完成并通过独立验证；seed20260917、seed20260918 尚未执行。** 本轮遵循用户“先做 seed42”的范围。结果见 [Results_2026-09-18.md](Results_2026-09-18.md)，方案见 [Experiment_Plan_2026-09-18.md](Experiment_Plan_2026-09-18.md)。

相对下界投影候选，固定主规则 1h RMSE 从 0.011755140271 降至 0.011631455064，改善 1.052180%；6h 仅改善 0.015270%，24/48h 精确不变。晚期 1h 和部分重点县仍有退化，结果尚待其他分折确认。

## 固定实现

- `unknown_d_protocol.py`：源身份、路径、先按允许县取子集的 U 监督、D 重建和预检。
- `known_history.py`：与此前下界实验逐字节相同的历史和 K 模块。
- `train_unknown_d.py`：严格内层选轮数、外层重训及恢复；当前 CLI 只接受 seed42。
- `unknown_d_evaluation.py`、`metric_helpers.py`：A/B/C 全行产物与预定比较。
- `verify_unknown_d.py`、`independent_numerics.py`：模型重载、独立求和/对齐/指标和边界验证。
- `analyze_unknown_d.py`：已验证 seed42 产物的分析。
- `experiment_config.json`：冻结公式、设置、路径、来源与本轮执行范围。

只训练 U1h：`U_raw=官方D−K`，保留容差内小负标签；推理 `D=min(1,K+max(0,U_hat))`。原 163 列特征不变，不新增 K 或未知小时数。共 20 次内层探测、5 次重训，复用同分折其余 95 个模型对应预测。

## 文件位置

```text
d_unknown_contribution/
├── Experiment_Plan_2026-09-18.md
├── Results_2026-09-18.md
├── experiment_config.json
├── *.py
├── reference/
│   ├── approved_plan.md                 # 批准时计划原文
│   ├── initial_workspace_snapshot.json
│   ├── source_manifest.json             # 旧模型/OOF/标签只读引用
│   ├── implementation_manifest.json
│   └── code_snapshot/
├── preflight/seed42.json
├── preflight/implementation_tests.json
├── logs/train_seed42.log
├── runs/v18_ud01_nested_v1_split42_model42/
│   ├── models/outer0..4/                 # 各一个 U 模型与 receipt
│   ├── logs/independent_verification.json
│   ├── run_manifest.json
│   ├── base_source_manifest.json
│   ├── new_model_manifest.json
│   ├── execution.json
│   ├── round_selection.csv
│   ├── target_audit.parquet
│   ├── u_oof_predictions.parquet
│   ├── component_predictions.parquet
│   ├── control_predictions.parquet
│   ├── aligned_unique_components.parquet
│   ├── primary_row_effects.parquet
│   ├── metrics/
│   ├── verification.json
│   └── U_CV_COMPLETE
├── summary/seed42/
├── final_integrity_check.json
└── completion.json                      # 仅 seed42 阶段完成
```

父目录原基线、已完成的下界投影、共享特征/标签均只读引用。新 U 模型不写入父目录 `runs/`。

## 命令与恢复

以下为本轮使用的入口，从 main 根目录执行。当前 run 已冻结，应使用只读验证命令复核，不再次启动同名 CV 或覆盖既有验证日志。

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026

# 无训练预检。
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/train_unknown_d.py --stage preflight --split-seed 42

# 首次运行命令。当前已完成，不再重复执行。
.venv/bin/python -B -u versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/train_unknown_d.py --stage cv --split-seed 42

# 已完成产物的新进程只读验证，结果输出到终端。
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/verify_unknown_d.py
```

仅对中断、尚无 U_CV_COMPLETE 的同身份 run，可给 CV 命令添加 `--resume`。代码/配置/数据/环境或来源变化会拒绝恢复；已冻结 CV 不覆盖。不能修改原目标文件后冒用旧模型身份。源码副本用于追溯，不应在已冻结实现上直接调参覆盖本轮。

## 产物语义

A 为原基线，B 为原基线＋D 下界，C 为本次未知贡献模型。主要比较是 C−B 的固定主规则 1h RMSE，同时报告 C−A、B−A 和 6h 副作用。

`target_audit.parquet` 含未来监督标签，仅用于审计。`u_oof_predictions.parquet` 保存原始 U、非负处理、K、重建 D、截断标记和新 U 模型 ID；预测函数不接收审计标签。

`component_predictions.parquet` 的 `pred_A/B/C_*` 是各版本分量，`source_*` 保留来源模型。B 的 D 来源 ID 指向投影前旧树；C 的 D1h 来源 ID 指向 U 树，数值经加回 K 重建，不能视为 booster 直接输出 D。其他 95 个来源保留原 ID。

`control_predictions.parquet` 保存全部 34,416 元数据行，按各 horizon 有效掩码评分，尾部预测为 NaN；包含 A/B/C 五个控制、实际 OSI、原始直接预测和变化标记。`aligned_unique_components.parquet` 保存 C 的对齐结果，独立验证检查候选数及来源 horizon。

`metrics/*.csv` 完整保留所有行、县、折的正负结果。CSV 用 `float_precision="round_trip"` 复读；`rmse_delta=后者−前者`，负数改善；`sse_reduction=前者SSE−后者SSE`，正数改善。
