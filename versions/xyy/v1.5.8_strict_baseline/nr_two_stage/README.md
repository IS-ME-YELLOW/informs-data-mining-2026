# N1/R1两阶段对照（seed42）

已完成100次拟合与独立验收；结论见[Results_2026-09-21.md](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/nr_two_stage/Results_2026-09-21.md)。不纳入主线、不自动做其他CV。

训练前协议、脚本、预检、源哈希、固定分组、运行产物及报告均位于本目录；旧文件未改。没有图、测试推理、全量拟合或提交。

执行顺序（项目根目录，已完成的训练/评价/汇总禁止覆盖；中断训练可--resume）：

```bash
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/preflight.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/train_nr.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/evaluate_nr.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/verify_nr.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/nr_two_stage/summarize_results.py
```

`reference/`冻结来源哈希及测试相似成员；`preflight/`保存10旧模型重载和100作用域检查；`runs/v18_nr2stage_v1_split42_model42/models/{N_t,R_t}/outer{0..4}/`含80内层+20外层模型、receipt、完整验证预测。`summary/`含主表、决策、来源清单。字段定义见[Metric_Dictionary.md](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/nr_two_stage/Metric_Dictionary.md)。

每阶段完成标记核对文件哈希。根completion.json冻结最终交付；不要重写已验收脚本/模型/报告后仍声称同一身份。复核可只读检查completion.json所有哈希并重新运行验证器；验证输出内容确定不变。
