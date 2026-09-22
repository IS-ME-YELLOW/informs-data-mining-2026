# P24目标前天气过程：seed42

已完成25次拟合与独立重载验证；[结果](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/Results_2026-09-21.md)。W1相对W0完整24h RMSE +0.131%，1h/6h +0.016%/+0.020%，48h不变。当前版本不采用、不追加确认。没有叠加G2、画图或测试推理。

首次执行顺序（已完成训练/评价拒绝覆盖）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/preflight.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/train_weather.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/evaluate_weather.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/verify_weather.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_target_weather_process/summarize_results.py
```

features/保存新的8列和完整191列训练/测试输入；reference/保存批准设计和旧输入哈希；runs/保存20内层与5外层模型、收据、曲线和逐行指标；summary/保存可读汇总、来源和决策。completion.json为交付哈希。验证脚本不训练。
