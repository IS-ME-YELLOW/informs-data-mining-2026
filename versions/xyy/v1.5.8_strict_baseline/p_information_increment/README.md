# P 信息增量实验

**状态：seed42 首轮已完成，100 次拟合、20 个外层模型，独立特征重建/模型重载/指标复算 PASS。** 本目录统一管理当前 E2 之后的 DEM 与近邻信息增量实验。

[首轮结果报告](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_information_increment/Results_2026-09-18.md)。相对 F0，F1 的整体 1h/6h RMSE 改善 0.6779%/0.2224%；F2 改善 0.9908%/0.6611%。24/48h 主预测逐行不变。

[详细实验计划](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_information_increment/Experiment_Plan_2026-09-18.md)。

| 对照 | 输入 | 本轮变更 |
|---|---|---|
| F0 | 163 列 | 复用已冻结的完整 E2，含 D_G |
| F1 | 170 列 | 给 P1h 的 a/b 模型增加 7 项 DEM |
| F2 | 183 列 | 给 P1h 的 a/b 模型增加 20 项近邻历史/阵风摘要 |

F1/F2 分别与 F0 比较，原其他模型和路由固定。近邻为全部 302 县中按 Haversine 距离选择的八个最近县，不含自身；只聚合已公开历史和允许天气，不使用任何预测期真实停电、标签、残差或 OOF 预测。精确字段、时间窗口、缺失规则、跨折边界和验收标准见计划。

两项均达到预定点估计晋级条件。F2 的 1h/6h bootstrap 区间位于零以下、四个固定时段均改善，建议优先确认；F1 的区间跨零，约 97% 的净 SSE 收益来自 Forest，稳定性需要后两套验证。没有自动运行其他分折、叠加两组特征或生成提交。

固定运行目录：

`/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_information_increment/runs/v18_pinfo_v1_split42_model42`

- `features/v1/`：冻结增强特征、白名单输入、近邻表、逐行输入依赖及身份；包含测试输入，不包含测试预测。
- `models/F1/outer0..4/`、`models/F2/outer0..4/`：各十个外层模型和 receipt。
- run 内 `logs/probes/`：80 次内层探测在最优轮数的逐行预测；完整曲线在 receipt 中。
- run 内 `branch_oof_predictions.parquet`、`component_predictions.parquet`、`control_predictions.parquet`：完整 F0/F1/F2 的分支、分量和控制 OOF。
- `summary/`：主要指标、bootstrap、全部县、窗口、初始状态组、逐时数据及特征使用情况。
- `reference/approved_plan.md`、`experiment_config.json`、`preflight/split42/checks.json`：批准稿、实际配置和无训练预检。
- run 内 `logs/independent_verification.json`、`audit_finalization.json`、`INFO_CV_COMPLETE`：当前独立验收、零额外拟合的审计修正和冻结标记。

只读复验已完成模型的当前入口：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_information_increment/verify_information_exact.py --split-seed 42
```

重新生成汇总（不训练、不改变 run）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_information_increment/summarize_information.py
```

首个独立聚合实现的求和顺序与训练特征存在浮点尾差，导致少数树分裂阈值不一致。已用 `verify_information_exact.py` 的独立逐邻居求和纠正，要求增强特征逐位一致后再运行所有核心检查；预测容差没有放宽，原模型、输入和 OOF 的哈希均未改变，额外拟合为零。原失败日志、核心审计代码和原训练身份保留，新审计身份单独冻结。直接运行旧 `verify_information.py` 会重现旧复算问题，复验请使用上述当前入口。

`train_information.py` 是本次冻结训练记录对应入口；完成 run 拒绝覆盖。后续扩展分折时须沿用修正的审计入口并建立新的完整运行身份，不直接修改本次已冻结代码。`finish_information.py` 记录了本次无重训的最终验收流程，完成后也拒绝重复冻结。

原 E2、D_G、严格基线、main 原数据和 Wendy 地理源均只读引用，1,461 个历史文件完整性核对 PASS。DEM 是冻结成品表，本轮未重算原始栅格；不要把信息增量结果当作逐栅格验收。`preflight/development_feature_build/` 为未训练的预检开发包，正式模型只使用 `features/v1/`。
