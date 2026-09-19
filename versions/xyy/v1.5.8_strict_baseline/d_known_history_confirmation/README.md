# D 历史下界：两套冻结分折确认实验

本目录负责 seed20260917、seed20260918 的严格树基线执行记录、D 投影和三分折对照分析。模型 seed 始终为 42；163 列输入、嵌套树数选择、模型参数和原 D 投影公式均冻结。seed42 是开发参照，两套新分折是确认分折。

**已全部完成并验证通过。** 两套 1h 主 RMSE 分别改善 0.226417%、0.177510%；收益仍主要来自 Morrow。完整分析见 [Results_2026-09-17.md](Results_2026-09-17.md)，完成记录与路径总索引见 `completion.json`。

## 文件位置

树模型使用父目录原有的统一 `runs/` 约定；本目录按分折存储小型执行记录与候选产物。`confirmation_plan.json` 是各路径和身份的唯一配置来源。

```text
v1.5.8_strict_baseline/
├── runs/
│   ├── v18_tree_nested_v2_split42_model42/          # 原开发基线，保持原样
│   ├── v18_tree_nested_v2_split20260917_model42/    # 确认基线一：模型、OOF、CV_COMPLETE
│   └── v18_tree_nested_v2_split20260918_model42/    # 确认基线二：模型、OOF、CV_COMPLETE
├── d_known_history_projection/                   # 原 seed42 投影，保持原样
└── d_known_history_confirmation/
    ├── confirmation_plan.json                    # 分折、路径、规则、基线代码哈希
    ├── experiment_plan.json                      # 不变的投影与评价约定
    ├── project_d.py                              # 与 seed42 逐字节相同
    ├── run_baseline.py / complete_split.py        # CV 启动与完成后的验证入口
    ├── run_projection.py / verify_projection.py  # 确认候选及独立重算
    ├── verify_implementation.py                  # 与 seed42 代码/数值的一致性检查
    ├── summarize_confirmation.py                 # 三套分折的完整比较
    ├── status.py / progress.json                 # 训练进度
    ├── reference/                               # 相对 seed42 的执行器差异
    ├── split20260917/
    │   ├── execution.json                        # 命令、时间、退出状态、资源用量
    │   ├── logs/                                 # 预检、训练和独立验证日志
    │   ├── projection/                           # 候选 OOF、指标、manifest、verification
    │   └── completion.json                       # 该分折全部步骤完成后生成
    ├── split20260918/                            # 与上面结构相同
    └── summary/                                  # 各分折并列指标、县变化与来源哈希
```

共新增两套基线，每套 400 次内层早停探测 + 100 次外层重训，保存 100 个外层模型。两个进程各保持原定 LightGBM 单线程并独立写目录，不训练最终全县模型或生成测试提交。

## 执行命令

所有命令从 main 根目录执行，使用项目现有 `.venv`。

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026

# 当前两项已完成。不要另行启动重复训练；先查看 status。
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_known_history_confirmation/status.py

# 新工作区首次执行，或中断后用完全相同输入加 --resume。
.venv/bin/python -B -u versions/xyy/v1.5.8_strict_baseline/d_known_history_confirmation/run_baseline.py --split-seed 20260917
.venv/bin/python -B -u versions/xyy/v1.5.8_strict_baseline/d_known_history_confirmation/run_baseline.py --split-seed 20260918

# 对应 CV_COMPLETE 存在后：独立重载 100 个模型 → 投影 → 独立投影验证。
.venv/bin/python -B -u versions/xyy/v1.5.8_strict_baseline/d_known_history_confirmation/complete_split.py --split-seed 20260917
.venv/bin/python -B -u versions/xyy/v1.5.8_strict_baseline/d_known_history_confirmation/complete_split.py --split-seed 20260918

# 两套 completion.json 都存在后汇总；seed42 只读引用，不重新训练。
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_known_history_confirmation/summarize_confirmation.py
```

树训练 `--resume` 只复用已完整保存且身份一致的 receipt；CV 已完成则拒绝再训。完成后的候选重跑也先验证身份；不覆盖其他分折或 seed42 的产物。若日志显示失败，先查看对应 `split*/logs/`；不要降低轮数、跳过校验或另换 seed。

## 与上一轮的关系

`implementation_verification.json` 验证了：D 投影文件逐字节不变；指标、bootstrap、独立重算辅助函数的 AST 相同；所有 seed42 控制预测通过新模块重放后精确一致。执行器修改只涉及路径/入口、分折身份校验、数值保护清单，以及把独立 verifier 的 seed42 固定计数 92/52 改成核对当次产物实际计数。原文件保持原样，差异存放在 `reference/*.diff`。

新保护分两层：

- `numerical_reference_snapshot.json` 固定原模型、OOF、代码及 seed42 投影数据；各新 run 的 `CV_COMPLETE` 固定对应新基线。数值输入、代码或输出不一致时拒绝继续。
- `initial_workspace_snapshot.json` 记录本次开始时已有文件，用于收尾确认有无外部变动。非输入的说明文档不进入新候选数值身份；发现变动照实记录，不更新旧快照或还原别人的编辑。

上一轮 `analysis/Quantitative_Summary.md` 的报告改动已存在于本次初始快照，不需要触碰上一轮旧快照来运行本确认实验。

## 结果解释约定

主比较始终是每套分折内部的固定 v18_rule：1/6h 用 C3，24/48h 用 C1。完整报告四个 horizon、全部 239 县和所有有效行；不按确认结果重新选择县、分折、模型 seed 或 horizon 路由。

跨分折 OOF 来自同一批县和同一场事件，只并列报告，不混合预测后当成更多独立样本。两套确认差值的均值/范围仅作描述；原 seed42 结果单独标为开发参照。县级 bootstrap 仍为 2,000 次、seed20260910，空间相关与单事件限制依然存在。排除 Morrow 的表格仅用于依赖性诊断，不替代主成绩。

CSV 复读使用 `float_precision="round_trip"`。`rmse_delta = 候选 − 基线`，负数改善；`sse_reduction = 基线 SSE − 候选 SSE`，正数改善。
