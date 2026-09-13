# v1.14：复用 v1.9 尾部/零膨胀专家

本实验没有再训练一批近似树模型，而是复用 v1.9 中已经完成的原始候选。为了不继承 v1.9 `best-subset` 在完整 OOF 上筛选的偏差，只预先固定两套未筛选专家：base/profile 的 P、D 使用加权 L2，N、R 保持 Huber。专家名称和安全系数均在外层验证县之外学习。

| Horizon | v1.12 RMSE | 候选 RMSE | 变化 | 改善折 | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| t+1 | 0.011677 | 0.011764 | +0.746% | 0/5 | [-0.000013, 0.000218] |
| t+6 | 0.010321 | 0.010444 | +1.193% | 0/5 | [-0.000067, 0.000360] |
| t+24 | 0.008209 | 0.008209 | 不变 | 安全回退 | [0, 0] |
| t+48 | 0.007550 | 0.007550 | 不变 | 安全回退 | [0, 0] |

短提前量都是 0/5 折改善；长提前量没有满足内层安全条件，完整回退 v1.12。因此不晋级，也没有理由再搜索更多相近损失组合。独立验证重载 24 个模型，最大复算差 `9.7145e-17`，哈希全部通过。

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions/v1.14_tail_specialist_reuse/run_tail_specialist.py
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions/v1.14_tail_specialist_reuse/verify_artifacts.py
```
