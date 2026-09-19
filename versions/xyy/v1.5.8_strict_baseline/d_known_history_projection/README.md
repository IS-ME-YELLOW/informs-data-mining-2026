# D 已知历史贡献投影候选

本目录独立保存固定 D 下界投影的实现、seed42 评估与验证，不改写父目录的严格基线。结果见 [Results_2026-09-17.md](Results_2026-09-17.md)。1h 主 RMSE 改善 0.241728%，99.6099% 的净 SSE 收益来自 Morrow；尚待其他固定分折确认。

## 实现与执行

- `project_d.py`：只依赖观测历史和原预测的纯投影模块，无标签输入或训练 API。
- `experiment_plan.json`：评估前固定的公式、主规则、范围与 bootstrap 配置。
- `run_experiment.py`：复用冻结 OOF，保存独立候选、完整评估和输入/输出哈希。
- `verify_experiment.py`：独立公式、重组、指标、bootstrap 和未来 P 扰动验证。
- `baseline_snapshot.json`：执行前 264 个已有严格基线文件的 SHA256。
- `final_integrity_check.json`：收尾时输入/输出校验及验证后非输入报告变动记录。
- `results_seed42/`：本次数据产物；完整清单由其中的 `manifest.json` 记录。

在原始快照一致的工作区中，从 main 根目录重现本次确定性评估：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_known_history_projection/run_experiment.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_known_history_projection/verify_experiment.py
```

第一条命令不会训练。它默认读取父目录 `runs/v18_tree_nested_v2_split42_model42`，并写入本目录 `results_seed42`。同身份重跑会重新生成该候选产物；代码、规则或输入身份不同则拒绝复用已有候选目录。改实验时保留本目录代码和证据，使用独立版本。第二条命令为当前 seed42 产物验证器，包含本轮 92 行 / 52 县的已知结果断言；不直接用这些固定计数检查其他 CV 候选。

**当前工作区注记：** 本次独立验证通过后，父目录 `analysis/Quantitative_Summary.md` 在 19:46:29 被更新，原快照保留。该文件不是模型或本实验输入，全部数值输入/输出仍校验通过，但上述命令的全目录快照保护现在会拒绝重跑。详见结果报告第五节与 `final_integrity_check.json`；不要覆盖这份后来更新的报告，也不要为继续运行而绕过身份校验。新一轮应保存独立执行版本和当时的工作区快照。

## 后续推理如何使用

模块接受与基线相同的、已通过预检的官方小时元数据和完整观测历史。`meta` 含 `fipsCode`、`timestamp_et`、整数 `hour_idx`，预测起点为 72–215；`components` 各 horizon 含 P_t/N_t/D_t/R_t 的已截断 [0,1] 数组，评分尾部按基线规则为 NaN。原始 P 只需提供 0–71 小时，函数会过滤掉未来 P 行。

```python
from project_d import observed_history, known_bounds, project_components

# raw_observed 只需要 fipsCode、timestamp_et、P_t；不需要任何目标标签。
history = observed_history(raw_observed)
bounds = known_bounds(meta, history)
projected = project_components(components, bounds)

# 在父基线已有的分量重组/C3 对齐前传入 projected。
# direct 必须是原始直接 OSI 预测；函数会沿原规则处理 C0/C2。
candidate_controls, candidate_aux = build_controls(meta, direct, projected)
```

这说明接入顺序，不代表本次已经训练最终模型或生成测试提交。`known_bounds_test.parquet` 仅是供未来推理使用的下界，不是预测文件。

## 数据文件

| 文件（位于 results_seed42） | 内容 |
|---|---|
| `candidate_oof_predictions.parquet` | 34,416 元数据行，四个 horizon 的真实值、基线和候选各控制预测、掩码与身份 |
| `projected_component_predictions.parquet` | 原始/投影分量、K、投影标记与原模型 ID |
| `known_bounds_train.parquet` / `known_bounds_test.parquet` | 239 / 63 县的四个 horizon 下界；评分尾部 NaN |
| `candidate_aligned_unique_components.parquet` | C3 唯一县/目标小时/分量预测、候选数及来源 horizon |
| `metrics_comparison.csv` | 所有 horizon、固定控制的 pooled 指标 |
| `fold_metrics.csv` / `county_metrics.csv` | 每折 / 每县的完整前后指标 |
| `D_component_metrics.csv` | D 分量误差与投影行数 |
| `changed_rows.csv` | 所有被投影的 D 行、分量真值与各控制的最终变化 |
| `window_metrics.csv` | 初始四小时、其余时段、最终预测改变行的诊断 |
| `primary_county_influence.csv` | 主模型 1h 县级净 SSE 贡献及占比 |
| `paired_county_bootstrap.csv` | 2,000 次县级配对重抽样，含排除 Morrow 的敏感性诊断 |
| `manifest.json` | 基线与候选身份、输入/输出哈希、执行范围 |
| `verification.json` | 新进程独立验证结果与验证器哈希 |

CSV 使用 `float_precision="round_trip"` 复读。`rmse_delta = candidate − baseline`，负数改善；`sse_reduction = baseline SSE − candidate SSE`，正数改善。不能用去掉 Morrow 的诊断替代全部县的主比较，也不能把这次 C1/C3 的相同局部变化当作独立复制。
