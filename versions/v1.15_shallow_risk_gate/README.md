# v1.15：三叶浅层风险门控

这是在 v1.12 之后唯一追加的浅层 gating 尝试。门控只看可部署的 v1.8 预测风险，以内层 q90/q95 切成三叶；高风险叶强制使用 v1.8，其他叶只允许在简单平均方向做安全收缩。

| Horizon | v1.12 RMSE | 候选 RMSE | RMSE 变化 | MAE 变化 | 改善折 |
|---|---:|---:|---:|---:|---:|
| t+1 | 0.011677 | 0.011677 | 不变 | 不变 | 不变 |
| t+6 | 0.010321 | 0.010321 | 不变 | 不变 | 不变 |
| t+24 | 0.008209 | 0.008211 | +0.021% | -0.036% | 0/5 |
| t+48 | 0.007550 | 0.007552 | +0.024% | -0.034% | 0/5 |

分区收缩再次出现“MAE 略好、RMSE 略坏”的模式，没有超过 v1.12 的硬 q95 规则。最终模型在 t+24 回到 `[1,1,0]`，t+48 低风险叶为 0.873、其余为 `[1,0]`；外折仍未泛化。因此停止继续扩大 gating 搜索。独立验证重载 24 个模型，最大复算差 `9.9963e-17`，哈希全部通过。

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions/v1.15_shallow_risk_gate/run_shallow_gate.py
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions/v1.15_shallow_risk_gate/verify_artifacts.py
```
