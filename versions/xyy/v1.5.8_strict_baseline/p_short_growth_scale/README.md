# P1早期增长尺度：seed42

已完成50次拟合和独立核验，结果见[报告](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/Results_2026-09-20.md)。G2完整1h/6h改善4.948%/1.798%，收益几乎全部来自Forest；主线未替换，另两CV未运行。

所有写入在本目录，不画图、不测试推理。G0是最新NB_P24整套树方案；仅G1/G2的早期P1 b为新模型。10外层与40内层模型分别位于runs下的models/G*/outer*/。

首次执行顺序：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/preflight.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/train_growth.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/evaluate_growth.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/verify_growth.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_short_growth_scale/summarize_results.py
```

训练/评价完成标记存在时拒绝覆盖。未完成训练仅允许匹配身份的--resume；每个已完成拟合有模型和收据，孤立模型不自动复用。验证脚本不做训练。完整性与所有交付哈希见completion.json。
