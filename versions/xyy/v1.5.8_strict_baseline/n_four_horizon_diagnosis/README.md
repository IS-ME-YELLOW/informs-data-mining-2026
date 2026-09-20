# N 四horizon诊断

**状态：三套CV诊断及独立核验完成，未训练模型。** 结果见 [Results_2026-09-19.md](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/n_four_horizon_diagnosis/Results_2026-09-19.md)。

主参照是完整F2（D_G＋P a/b＋邻县信息），N仍为原strict模型。结论支持做一次统一的N尾部预测对照，但没有采用oracle、锚点或改变组合。

- `reference/approved_plan.md`：用户批准时的计划原文；根目录计划保留编写时状态，当前执行状态以本README、结果和completion为准。
- `diagnostic_config.json`：冻结参数；`reference/initial_input_hashes.json`：1,375个输入文件字节身份。
- `runs/split*/rows/`：完整逐行诊断；oracle含未来真值，显式标记不可部署，不能用作训练特征或提交。
- `runs/split*/tables/`：所有县/时段、尾部分组、峰值与位移、交叉项、匹配支持、bootstrap。
- `summary/`：三套汇总与建议，未合并三套OOF生成模型候选。
- `verification.json`、`final_integrity_check.json`、`completion.json`：独立核验及完成身份。

诊断使用项目 `.venv/bin/python -B`。`run_diagnostics.py`完成后拒绝覆盖；重验已有结果可运行：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_four_horizon_diagnosis/verify_diagnostics.py --split-seeds 42 20260917 20260918
```

首次执行命令已按批准计划运行，`--stage preflight`和`--stage analyze`都限定三套固定seed。相似支持采用8个合法输入、完整外折排除、前5名先冻结后接结果；因查询县/小时由事后误差聚焦，不作因果解释。

图使用本机捆绑Python的ReportLab，不改变项目环境：

```bash
/home/jacklo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B /home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/n_four_horizon_diagnosis/plot_diagnostics.py
```

`summarize_report.py`只读已验证表生成说明；`plot_diagnostics.py`只读本目录表生成图。无LightGBM拟合入口、无部署/提交入口。独立验证器不复用生产诊断的数值函数；浮点求和顺序差异使用事先冻结的容差，不修改预测或输入。
