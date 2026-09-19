# 仅在 K>0 时切换到 U 重建

本目录复用三套已冻结的严格 OOF，评估固定切换候选 G。没有训练或拟合新参数。既有原基线、下界投影和整段 U 模型的代码、模型及结果全部只读引用。

**已完成并通过三套新进程独立验证。** 相对下界候选，1h RMSE 分别改善 1.074645%、1.225771%、1.061724%，6/24/48h 逐行不变。结果、D 分量区间及恶化县见 [Results_2026-09-18.md](Results_2026-09-18.md)。

规则是在**分量重组和 C3 对齐之前**，仅修改 D1h：

```text
K > 0：D_G = min(1, K + max(0, U_raw_prediction))
K = 0：D_G = 原 D 模型的 [0,1] 截断预测
```

K 只使用小时 0–71 的 P，严格判断 `K>0`，没有容差阈值、训练出的开关或县名单。A 为原基线，B 为 A＋下界修复，C 为整段 U 版本；主要比较 G−B，并完整保存 G−A、G−C。

**解释边界：** 这条规则源自已经看过的 U 实验结果。本次是三套既有分折的开发回放，不是新独立保留集的验证。代码路径不读取未来标签来决定切换，仍须将方案选择的不确定性与预测依赖隔离分开看待。

## 目录

```text
k_positive_switch/
├── experiment_config.json              # 评估前固定的规则、来源和统计口径
├── switch_protocol.py                  # 纯切换函数、来源验证和边界检查
├── known_history.py                    # 既有 K 实现的原样副本
├── gated_evaluation.py / metric_helpers.py
├── evaluate_switch.py                  # 不调用训练 API
├── verify_switch.py / independent_numerics.py
├── summarize_switch.py
├── reference/initial_workspace_snapshot.json
├── logs/evaluate_*.log
├── runs/
│   ├── split42/
│   ├── split20260917/
│   └── split20260918/
│       ├── manifest.json               # 候选身份、输入/代码哈希
│       ├── preflight.json
│       ├── gate_decisions.parquet      # K、使用的分支、原/U/所选模型 ID
│       ├── component_predictions.parquet
│       ├── control_predictions.parquet # A/B/C/G，全部行与全部控制
│       ├── aligned_unique_components.parquet
│       ├── eligible_rows.csv           # 全部 784 个符合切换条件的行
│       ├── metrics/                    # pooled/fold/county/window/bootstrap/D
│       ├── routing_stats.json
│       ├── verification.json
│       ├── logs/independent_verification.json
│       └── G_COMPLETE
├── summary/                            # 三套并列比较，绝不合并 OOF 打分
├── Results_2026-09-18.md
├── final_integrity_check.json
└── completion.json
```

`component_predictions.parquet` 保存 G 的分量及所选来源。符合 K>0 的 D1h 来源是 U 模型，其输出经重建；其余来源保留原树 ID。`gate_decisions.parquet` 同时保留原 D 模型与 U 模型 ID，不把重建 D 冒充原始 booster 输出。

## 使用与复核

从 main 根目录使用现有 `.venv`。下面以 seed42 为例，另两套只替换 `--split-seed`。

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026

# 首次回放，不训练；G_COMPLETE 已存在时拒绝覆盖。
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/k_positive_switch/evaluate_switch.py --split-seed 42

# 已完成产物的新进程只读复核，输出到终端。
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/k_positive_switch/verify_switch.py --split-seed 42
```

三套均独立验证后，由 `summarize_switch.py` 生成本轮汇总。已封存产物不覆盖；不同代码或来源身份拒绝混写已有候选目录。

`metrics/pooled.csv` 为所有固定控制、horizon 和三组对照；`D_component.csv` 是原始分量 D 指标，`D_spaces.csv` 区分直接 D1h 与 C3 后 D1h/D6h，`D_bootstrap.csv` 提供对应配对区间。县级 bootstrap 沿用 2,000 次、seed20260910，排除 Morrow 仅作诊断。

CSV 用 `float_precision="round_trip"` 复读；`rmse_delta<0`、`sse_reduction>0` 表示改善。门槛满足行数、D 实际改变行数、最终 OSI 改变行数分别记录，不混为一谈。
