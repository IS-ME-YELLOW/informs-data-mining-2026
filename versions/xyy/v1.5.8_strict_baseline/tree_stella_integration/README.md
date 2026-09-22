# 最新树与Stella路线整合

状态：三套CV的B0/I1/I2/I3全部完成，独立核验PASS。新增训练0次、模型0个；[结果](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/Results_2026-09-20.md)。

1h/6h全部沿用最新树T。I1为T24＋Stella48；I2为Stella24/48；I3为固定1:1平均24h（再按原规则后处理）＋Stella48。B0保留完整T。

I3相对树24h三套改善，相对Stella24h为两套改善/一套恶化。建议I1作为下一阶段参照，I3作为待压力测试候选，保留I2；没有最终赢家或自动部署声明。

首次执行顺序：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/integrate_routes.py --stage preflight --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/integrate_routes.py --stage evaluate --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/verify_integration.py --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/summarize_integration.py
```

已完成评价不覆盖；单独运行verify_integration.py可重验，不训练、不重载源模型。

- reference/approved_plan.md：批准原文；根计划保留编写时状态，本README和completion为当前状态。
- reference/source_manifest.json：实际读取文件路径、哈希匹配方式与parquet schema；没有复制大模型。
- runs/split*/source_predictions.parquet：T四输出、S长期及原门控；S旧短时不进入组合。
- runs/split*/integrated_predictions.parquet：完整34,416行、四输出、四候选及真值/掩码。
- runs/split*/row_changes.parquet：全有效行的五个固定比较；blend_postprocessing_audit.parquet：平均、阈值和误差恒等式。
- runs/split*/metrics/、verification.json：全县/折/日期/严重度/尾部、互补、bootstrap及独立验收。
- summary/：三套分别汇总、判断和候选来源；completion.json：本轮完成身份。

本轮只组合最终OSI，不重算C3、不改Stella q95、不更换其内部L0，不学习权重。原始平均只用于诊断；无第五候选。地理压力测试、随机邻居负对照、重新训练或提交均不在此次执行范围。
