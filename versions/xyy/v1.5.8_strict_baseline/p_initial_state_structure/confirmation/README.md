# P 初始状态结构：确认实验

**状态：seed20260917、seed20260918 均已完成，独立模型重载验证 PASS。** 本轮新增 150 次正式拟合、30 个外层模型；三套合计 225 次拟合、45 个外层模型。

[三分折汇总与分析](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/confirmation/Results_2026-09-18.md)。E2 相对同分折 E0 的主 RMSE：

| 分折 | 1h 变化 | 6h 变化 |
|---|---:|---:|
| 42 | −4.5346% | −2.6919% |
| 20260917 | −7.4402% | −4.0088% |
| 20260918 | −4.9212% | −2.5569% |

24/48h 主预测逐行不变。三套 E2 均优于直接 L2 对照 E1，排除 Morrow 后仍有净收益；部分县、低初始停电组和晚期窗口仍有退化。建议保留固定 E2 为当前优先 P 候选，尚未训练最终模型或生成测试提交。

[执行协议](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/confirmation/Confirmation_Protocol_2026-09-18.md)说明来源隔离、数值代码复用和三套汇总口径。

本目录保存共用确认入口、配置、两套预检和最终汇总；新模型统一放在父实验的 `runs/v18_pstate_nested_v1_split20260917_model42/` 与 `runs/v18_pstate_nested_v1_split20260918_model42/`。原 seed42 文件保持冻结。

训练和验证使用项目环境。下面以 seed20260917 为例；另一套将参数换成 20260918：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/confirmation/train_p_structure.py --stage preflight --split-seed 20260917
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/confirmation/train_p_structure.py --stage cv --split-seed 20260917
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/confirmation/verify_p_structure.py --split-seed 20260917
```

已完成 run 拒绝覆盖。`--resume` 仅用于同身份的未完成 run；验证命令只读。每套 75 次正式拟合、15 个新外层模型，模型 seed 均为 42。

重新生成三套汇总：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/confirmation/summarize_confirmation.py
```

`summary/` 保存完整比较、P 精度、bootstrap、县级一致性、固定窗口和逐时数据；`split20260917/`、`split20260918/` 保存各套预检和简报。`completion.json` 与 `final_integrity_check.json` 记录本轮完成和 1,272 个历史文件未改动的核对结果。

父目录原 README、实验计划和 seed42 报告保留原阶段记录；本页是两套确认后的结果入口。数值代码没有随结果调整。
