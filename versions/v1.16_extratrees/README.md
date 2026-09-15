# v1.16 ExtraTrees direct OSI

这是与 boosting 差异最大的直接 OSI bagging 基线。运行：

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions\v1.16_extratrees\run_extratrees.py
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions\v1.16_extratrees\verify_artifacts.py
```

## 结果

| Horizon | ET-A RMSE | ET-B RMSE | 相对 v1.8 |
|---|---:|---:|---:|
| t+24 | 0.008906 | 0.008946 | +6.37% / +6.84% |
| t+48 | 0.008252 | 0.008235 | +8.68% / +8.45% |

两配置均不晋级，也不扩展短 horizon。ET-A 与 v1.12 的残差相关为 0.948/0.943；实际高 OSI 前 5% RMSE 仍比 v1.12 差。24 个 Pipeline 全部独立重载，最大预测差 `6.94e-17`，输入/产物哈希通过。运行约 332 秒，压缩模型合计约 1.15 GB。

`models/` 保存包含逐折中位数填充器的完整 sklearn Pipeline；`artifacts/` 保存 OOF、test、submission、元数据和哈希；`results/` 保存汇总、逐折、逐县、高 OSI、bootstrap、相关性与运行时间。
