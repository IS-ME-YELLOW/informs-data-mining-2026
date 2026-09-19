# D 未知贡献监督：冻结分折确认

本目录已完成 seed20260917、seed20260918 的 U1h 确认实验，并只读引用 seed42 结果进行汇总。两套训练和新进程独立验证均 PASS，详见 [Results_2026-09-18.md](Results_2026-09-18.md)。相对下界候选，1h 分别改善 0.762597%、0.862038%，6h 分别恶化 0.507794%、0.189254%；当前候选存在明确的早晚时段取舍。

方法与 seed42 相同：`U_raw=官方D−K`，保留容差内微负标签；预测 `D=min(1,K+max(0,U_hat))`。固定 163 列特征、模型 seed42、Huber 目标、原参数与 C3/C1 路由。每套新增 20 次内层探测和 5 次外层重训，复用同分折其余 95 个模型对应预测；本次合计 50 次新拟合、10 个新模型。

## 路径与来源

```text
d_unknown_contribution/
├── runs/v18_ud01_nested_v1_split42_model42/       # 原 seed42，保留原样
├── summary/seed42/                              # 原开发参照
└── confirmation/
    ├── confirmation_plan.json                  # 固定分折、来源和目录
    ├── verify_adaptation.py                    # 检查与 seed42 的科学逻辑一致
    ├── finish_split.py                         # 新进程验证和单套分析
    ├── summarize_all_splits.py                 # 三套并列比较
    ├── reference/
    │   ├── initial_workspace_snapshot.json
    │   └── adaptation_verification.json
    ├── split20260917/
    │   ├── experiment_config.json
    │   ├── *.py                                # 固定执行器与验证器
    │   ├── reference/seed42_adaptation.diff
    │   ├── preflight/seed20260917.json
    │   ├── logs/train_cv.log
    │   ├── runs/v18_ud01_nested_v1_split20260917_model42/
    │   │   ├── models/outer0..4/
    │   │   ├── logs/independent_verification.json
    │   │   ├── *.parquet                        # U、分量、A/B/C OOF、逐行误差
    │   │   ├── metrics/                         # 全部县/折/窗口/bootstrap
    │   │   ├── execution.json
    │   │   ├── verification.json
    │   │   └── U_CV_COMPLETE
    │   ├── summary/split20260917/
    │   └── completion.json
    ├── split20260918/                           # 同样结构，独立写入
    ├── summary/                                # 三分折汇总和哈希
    ├── Results_2026-09-18.md
    ├── final_integrity_check.json
    └── completion.json
```

旧 seed42 代码、配置、报告和完成清单不改写，其文档记录的是当时仅完成 seed42 的状态。每套确认自带配置和固定代码，避免将新分折配置覆盖到旧运行身份。四个数值模块与 seed42 逐字节相同；U 目标、重建、训练作用域和完整拟合循环经 AST 比较一致。验证器仅调整分折元数据，未削弱检查。差异文件位于各 `reference/seed42_adaptation.diff`。

原 A 基线引用 `v1.5.8_strict_baseline/runs/v18_tree_nested_v2_split{seed}_model42/`；B 引用 `d_known_history_confirmation/split{seed}/projection/`。新 U 模型仅存放在本目录各 split 下，不重训已有完整树基线。

## 执行与复核

从 main 根目录执行，使用现有 `.venv`。下面以 seed20260917 为例，另一套将路径和参数中的 seed 同时替换为 20260918。每个分折入口只接受自己配置的 seed。

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026

.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/confirmation/split20260917/train_unknown_d.py --stage preflight --split-seed 20260917

# 首次正式运行；已完成时不要重复启动。
.venv/bin/python -B -u versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/confirmation/split20260917/train_unknown_d.py --stage cv --split-seed 20260917

# U_CV_COMPLETE 存在后：新进程验证，并生成该套分析。
.venv/bin/python -B -u versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/confirmation/finish_split.py --split-seed 20260917

# 两套完成后汇总；不合并或平均 OOF 预测作为新评分。
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/confirmation/summarize_all_splits.py
```

中断且尚未生成 U_CV_COMPLETE 的同身份运行可以给 CV 命令加 `--resume`；代码/数据/参数/环境变化会被拒绝。已冻结的运行只读验证可直接调用各分折的 `verify_unknown_d.py`，输出到终端，不覆盖原验证日志。当前完成状态可查看各 split 的 `completion.json`。

## 结果口径

A 为严格原基线，B 为 A＋D 历史下界，C 为未知贡献监督模型。主要比较为同分折固定主规则的 **1h C−B**，同时报告 C−A、6h 副作用、晚期 1h、所有恶化县和排除 Morrow 的诊断。

县级 bootstrap 沿用 2,000 次、seed20260910；两套固定确认都报告。三套使用同一批县和同一场事件，不构成三套独立样本；确认差值均值仅是描述，不是合并 OOF 的新成绩。CSV 用 `float_precision="round_trip"` 复读。`rmse_delta<0`、`sse_reduction>0` 表示改善。
