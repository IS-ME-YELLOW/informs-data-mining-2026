# P24邻县信息确认

状态：seed20260917、seed20260918完成，独立验收PASS。新增50次拟合、10个外层模型；[三套结果](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/confirmation/Results_2026-09-20.md)。

24h与短时方向均复现，支持将P24直接树＋邻县信息纳入后续优先组合。每套使用自己的P6增强完整B0，只重训直接P24-Huber，原冻结183列输入。首轮特征包只读复用。

首次执行：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/confirmation/prepare_confirmation.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/confirmation/run_confirmations.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/confirmation/summarize_confirmation.py
```

所有单套入口要求显式`--split-seed 20260917`或`20260918`。训练入口内部先完成预检再拟合；已完成stage拒绝覆盖，--resume仅限同身份未完成run。可单独运行verify_neighbors.py --split-seed相应值重新核验，不训练模型。

- reference：父文件哈希及AST适配核验。
- split*/reference和experiment_config.json：对应CV来源身份与预检。
- split*/runs：5个P24模型、20个probe的逐行输出/曲线、全OOF与控制、指标和独立验证。
- split*/summary：单套指标、误差分解、Clay逐行轨迹。
- summary：三套分开报告的指标、Clay和全县稳定性；没有平均OOF。
- completion.json、final_integrity_check.json：完成清单和原来源/特征包不变证明。

父目录README、报告和模型不回写。没有确认P6/P48邻县候选、没有联合候选、没有全量模型或提交。
