# 旧 GAT 结果诊断

只读诊断来源：`outputs/runs/dem_v158_p2_component_seed42_cuda_log`。所有新文件都位于本目录，原代码、数据、预测、alpha 和报告保持原值；未训练模型。

[完整诊断报告](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_dem_eval_v158/diagnostics/Old_GAT_Diagnosis_2026-09-19.md)。

主要发现：1h 的净 SSE 恶化由 Newton County（18111、fold2）主导。其已知历史输入严重超出监督训练范围，GAT 持续给出约 +1 的原始校正；内层所选 α=0.2 在外层失效。最终提交阶段的 1h α 实际为 0，保存测试预测与旧树基线一致。

保存预测、alpha 选择、来源哈希和输入支持范围已经复算。当前原运行没有模型/检查点及完成记录，原 independent_reload 也标为未运行；本诊断不等于完整模型验收。所需原模型文件见 [后续核查说明](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_dem_eval_v158/diagnostics/Checkpoint_Followup_2026-09-19.md)。

复现命令（不训练、不改原运行；只更新 diagnostics）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu
/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -B code_phase2_dem_eval_v158/diagnostics/diagnose_old_gat.py
/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -B code_phase2_dem_eval_v158/diagnostics/verify_diagnosis.py
/home/jacklo/XYY/INFORMS/informs-data-mining-2026/.venv/bin/python -B code_phase2_dem_eval_v158/diagnostics/write_report.py
```

`tables/` 包含完整外层行损益、全部县/折/时段、Newton 轨迹、内外 alpha 曲线及域外输入统计。标为 hindsight 的表只用于解释失败，不能当作无泄漏的新候选成绩。

`verification.json` 核实诊断数值和其他文件未改动；`diagnostic_manifest.json` 固定本次结果来源与产物哈希。Windows 转来的文本换行差异只在内存中核对，没有写回源文件。
