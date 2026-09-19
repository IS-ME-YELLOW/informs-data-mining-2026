# P 初始状态结构：两套冻结分折确认

用户已授权完整执行 seed20260917、seed20260918。每套均训练 E1 直接 P-L2 和 E2 的 a/b 两条分支，模型 seed 始终为 42。每套 60 次内层探测、15 次外层重训；本轮新增 150 次正式拟合、30 个外层模型。

实验定义严格沿用 [批准方案](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure/reference/approved_plan.md)：只替换 P1h；固定 163 列输入、分支支持条件、贡献权重、训练支持集均值归一化、内层早停指标及轮数规则。E0 使用同分折严格 P-Huber＋D_G；E1/E2 共用相同 D_G。C3、1/6h 用 C3、24/48h 用 C1 的路由保持固定。

两套都将完整执行；正常负收益不会成为跳过另一套的理由。未授权全县最终模型、测试提交或新增门控、阈值和调参。

## 冻结来源与代码复用

seed42 的模型、代码、配置、报告和完成标记均保留原文件。其数值代码已纳入原运行身份，修改这些文件会使原验证失效。因此本轮采用一个共用的确认入口，供两套新分折使用：

- `weighted_scoped_training.py`、`evaluate_p_structure.py`、`preflight_checks.py`、`metric_helpers.py`、`independent_numerics.py` 直接从父实验目录只读导入，没有复制或修改。
- 确认入口的 protocol、runner、verifier 由 `prepare_confirmation.py` 对原文件进行限定文本替换生成，只改变 CV 选择、来源身份、配置/代码哈希位置和输出路径。
- `reference/runner_adaptation.diff` 保存完整差异；`reference/adaptation_verification.json` 核对原代码哈希、替换范围和核心函数 AST。a/b 标签、损失、权重、重建和模型参数不变。
- 每套运行身份包含自己的 CV、基线、D_G、共享数值代码及确认入口。不会将 seed42 的其他分量 OOF 混入新分折。

实际配置见本目录 `experiment_config.json`。它独立于原 seed42 配置。每个进程必须显式传入自己的 `--split-seed`，两个运行的元数据和模型路径分别隔离。

## 输出布局

实验根目录：

`/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_initial_state_structure`

```text
p_initial_state_structure/
├── runs/
│   ├── v18_pstate_nested_v1_split42_model42/           # 已冻结，只读
│   ├── v18_pstate_nested_v1_split20260917_model42/     # 本轮新增
│   └── v18_pstate_nested_v1_split20260918_model42/     # 本轮新增
└── confirmation/
    ├── experiment_config.json
    ├── _runtime.py
    ├── prepare_confirmation.py
    ├── p_structure_protocol.py
    ├── train_p_structure.py
    ├── verify_p_structure.py
    ├── summarize_confirmation.py
    ├── reference/                                   # 来源快照与适配审计
    ├── split20260917/                               # 本套预检、来源及报告
    ├── split20260918/
    ├── logs/
    ├── summary/                                    # 三套分折并列汇总
    ├── Results_2026-09-18.md
    └── completion.json
```

每个新 run 继续保存全部分支/分量/控制逐行 OOF、60 次早停探测预测、15 个模型、完整 receipt、分县/分折/窗口/初始状态指标、D_B 交互对照和县 bootstrap。新进程独立重载验证通过后才写入各自 `P_CV_COMPLETE`。

## 汇总与判断

三套逐一比较 E2−E1、E2−E0、E1−E0 的主 1h/6h RMSE、P1h 的 C3 前后精度、区间及县/时段收益。24/48h 主预测须逐行不变。报告所有恶化县，固定检查排除 Morrow 后的净效果和早期/晚期差异。

不平均不同分折的 OOF 预测形成新成绩，不挑选最佳 split，不在看到新结果后修改公式。三套使用同一批县和同一场事件，且已用于此前开发分析；结论是重复分折的开发稳健性证据，不是三个独立外部验证。

本轮结束时核对开工前全部历史文件的哈希，并输出两套新运行的完成标记和三套汇总来源清单。
