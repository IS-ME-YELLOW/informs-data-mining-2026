# P 初始状态结构实验

**状态：seed42 已完成，真实模型独立重载验证 PASS。** 本目录统一管理 P 的预测结构实验；另两套分折尚未执行。

结果：[Results_2026-09-18.md](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/Results_2026-09-18.md)。E2 相对 E0 的主 1h RMSE 改善 **4.5346%**、6h 改善 **2.6919%**，24/48h 逐行不变。原始 P1h RMSE 改善 4.0235%，C3 后改善 5.8976%。E1 的收益很小，支持继续确认结构候选。

详细方案：[Experiment_Plan_2026-09-18.md](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/Experiment_Plan_2026-09-18.md)。

固定已知初始停电水平 `p=P71`，学习两个有界分支，重建：

$$\widehat P=p\widehat a+(1-p)\widehat b.$$

a/b 分别表示真实 P 在初始水平以内及超过初始水平的归一化部分，是数学分解，不是官方 N/R，也不代表已识别的客户群。

| 对照 | P1h 来源 | D 来源 |
|---|---|---|
| E0 | 冻结严格基线 P-Huber | 冻结 D_G |
| E1 | 新直接 P-L2 | 同 E0 |
| E2 | 新 a/b-L2 分支，按 p 重建 | 同 E0 |

只改 P1h；原 163 列特征、其他模型、C3 和主规则固定。E2 分支使用贡献权重 `p²`、`(1-p)²`，仅在每次训练支持集内归一化；详细端点、早停和隔离规则见方案。

已完成 seed42：60 次内层探测＋15 次外层重训，共 75 次正式拟合，约 7.29 分钟（含本次评价和独立验证）。所有 239 县均评分；150 县的主 1h SSE 改善、89 县恶化，Morrow 贡献净改善约 54%。三方比较、分支交叉项、时段与县级损益、D_B 敏感性和 bootstrap 均已保存。

运行目录：

`/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/runs/v18_pstate_nested_v1_split42_model42`

关键文件：

- `models/outer0..4/`：每折直接 P、a、b 三个模型及 receipt。
- `logs/probes/`：60 次内层探测在最优轮数的逐行验证预测；完整早停曲线在模型 receipt 中。
- `p_branch_oof_predictions.parquet`、`target_audit.parquet`：原始分支输出、支持标记、权重、P71、重建值及官方标签。
- `component_predictions.parquet`、`control_predictions.parquet`：E0/E1/E2 完整分量与所有控制 OOF，保留来源 ID。
- `D_B_control_predictions.parquet`：预定 D_B 交互对照。
- `metrics/`：全体、分折、全部县、时间窗、初始状态组、bootstrap、分支及组合误差。
- `logs/independent_verification.json`、`P_CV_COMPLETE`：新进程验证结果和冻结标记。
- 本目录 `summary/`：阅读用汇总及逐小时曲线数据；`reference/approved_plan.md` 是批准稿。

使用项目环境复核已完成模型（不训练、不改冻结产物）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/verify_p_structure.py --split-seed 42
```

重新生成本实验的汇总报告（只更新 summary、结果报告和完成信息）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/summarize_seed42.py
```

训练入口为 `train_p_structure.py`，默认 `--stage preflight` 不拟合；明确 `--stage cv --split-seed 42` 才训练。当前完成 run 拒绝覆盖；`--resume` 仅用于同身份的未完成 run。没有 final 阶段，未生成测试提交。当前配置与 CLI 只允许 seed42，确认后续分折时需明确扩展授权范围和各自来源身份。

现有数据、CV、模型和 D_G 产物只读引用；原历史 1,165 个文件的完整性核对 PASS。数值代码、环境、配置、数据、来源模型和分折均纳入运行身份；汇总脚本与报告另有 summary manifest。CSV 用 `float_precision="round_trip"` 复读。

当前证据来自已经用于开发的 seed42；建议按批准方案，在 seed20260917、seed20260918 上同时确认 E1/E2，再决定是否采用。未自动启动后续训练。
