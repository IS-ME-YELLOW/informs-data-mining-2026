# 训练/测试县特征支持

结果见[报告](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/train_test_feature_support/Results_2026-09-20.md)，新指标见[字典](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/train_test_feature_support/Metric_Definitions.md)，模型方向见[具体设计](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/train_test_feature_support/Growth_Exposure_Experiment_Design_2026-09-20.md)。本轮没有训练或画图。

首次执行命令（已完成的主分析拒绝覆盖）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/train_test_feature_support/analyze_support.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/train_test_feature_support/verify_support.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/train_test_feature_support/summarize_support.py
```

data保存只含允许信息的输入和全量支持/近邻表；summary保存可读汇总和距离解释；reference保存输入与设计来源哈希。completion.json为最终交付身份。统计变换仅由训练参照决定，测试未来结果未提供且未使用。
