# v1.13：目标时刻多提前量加权对齐

本实验把 v1.8 C3 的提前量 1/6/24/48 等权平均替换为严格外折学习的静态非负权重。最终全 OOF 权重为 `0.10 / 0.15 / 0.35 / 0.40`，但该结果没有在外折泛化。

| Horizon | v1.12 RMSE | 候选 RMSE | 变化 | 改善折 | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| t+1 | 0.011677 | 0.011679 | +0.024% | 2/5 | [-0.000069, 0.000077] |
| t+6 | 0.010321 | 0.010324 | +0.032% | 2/5 | [-0.000079, 0.000089] |
| t+24 | 0.008209 | 0.008209 | 不变 | 不变 | [0, 0] |
| t+48 | 0.007550 | 0.007550 | 不变 | 不变 | [0, 0] |

结论：C3 的价值更像等权稳健化，固定可学习权重不稳定，继续保留 v1.12。独立验证重载 6 个模型，最大复算差 `9.9747e-17`，哈希全部通过。

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions/v1.13_weighted_target_alignment/run_weighted_alignment.py
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions/v1.13_weighted_target_alignment/verify_artifacts.py
```
