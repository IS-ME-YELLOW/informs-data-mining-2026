# F2 近邻信息确认实验

**状态：seed20260917、seed20260918 的 F2 均已完成，独立特征重建、真实模型重载及指标复算 PASS。** 本轮新增 100 次拟合、20 个外层模型；单计 F2，三套累计 150 次拟合、30 个模型。

[三分折结果汇总](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_information_increment/confirmation/Results_2026-09-19.md)。相对各自完整 E2 的整体 RMSE：

| 分折 | 1h 变化 | 6h 变化 |
|---|---:|---:|
| 42 | −0.9908% | −0.6611% |
| 20260917 | −1.1035% | −1.0214% |
| 20260918 | −0.6103% | −0.5984% |

24/48h 主预测逐行不变。三套四个固定时段的 1h 点估计均改善；第三套区间跨零，支持小幅同向增益，不能称为三套都显著。建议保留 F2 为当前优先输入方案，继续保留完整 E2 对照；未训练 F1、组合特征或最终提交模型。

[执行协议](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_information_increment/confirmation/Execution_Protocol_2026-09-19.md)。两套共 100 次新拟合、20 个新外层模型；同分折完整 E2 为 F0，固定 183 列输入和原有全部数值规则。

所有模型位于父目录 `runs/v18_pinfo_v1_split{seed}_model42/`；本目录管理两套的共用入口、配置、预检、日志和三套汇总。原 seed42 文件及特征包只读。

每个进程必须显式指定 CV seed。例如：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_information_increment/confirmation/train_information.py --stage preflight --split-seed 20260917
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_information_increment/confirmation/train_information.py --stage cv --split-seed 20260917
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_information_increment/confirmation/verify_information_exact.py --split-seed 20260917
```

另一套改为 seed20260918。当前入口拒绝 seed42；完成 run 拒绝覆盖，`--resume` 仅接受同身份未完成 run。无需重新构建特征，验证过程直接采用已修正的排名顺序求和。

重新生成三套汇总：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_information_increment/confirmation/summarize_confirmation.py
```

`summary/` 保存 F2 的整体/P 分量/区间、全部县、时段和方向一致性；`split*/Results_2026-09-19.md` 为单套简报。各 run 的 `INFO_CV_COMPLETE`、`logs/independent_verification.json` 固定了独立验证结果；本目录 `completion.json` 和 `final_integrity_check.json` 记录本轮完成及 1,633 个历史文件未改动。

父目录保留 seed42 阶段记录，本页为确认后的结果入口。原完整特征包仍含已存在的 F1 矩阵，只做来源包检查；本轮实际拟合与汇总候选仅为 F2。
