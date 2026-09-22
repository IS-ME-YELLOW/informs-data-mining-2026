# 共同误差调研

研究已完成，详见 [结果](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/Results_2026-09-20.md)。0次训练、0个模型重载；新文件全部在本目录。

首次执行（已完成的主分析拒绝覆盖）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/diagnosis.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/verify_diagnosis.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/verify_inputs.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/summarize_report.py
/home/jacklo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B versions/xyy/v1.5.8_strict_baseline/common_error_diagnosis/plot_cases.py
```

`data/`为全量逐行与案例轨迹；`summary/`为分组、来源、分量和证据卡；`matching/`保留不接触结果的输入、排名、支持及随后连接的结果；`external_research/`保存公开资料与访问限制；`figures/`为可分享SVG/PNG。`completion.json`给出交付文件哈希。

三套CV不构成新事件测试。共同分数不是不可约误差，事后报道不作为可部署特征。未来模型建议尚未训练。
