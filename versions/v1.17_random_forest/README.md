# v1.17 Random Forest direct OSI

该单配置用来区分 bagging 树平均收益与 ExtraTrees 随机阈值收益。运行：

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions\v1.17_random_forest\run_random_forest.py
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions\v1.17_random_forest\verify_artifacts.py
```

## 结果

RF 的 t+24/t+48 RMSE 为 0.008873/0.007937，相对 v1.8 退化 5.97%/4.53%；县级 paired bootstrap 95% 区间均完全高于 0。与 v1.12 的残差相关为 0.962/0.963。高 OSI t+48 的 MAE 较 v1.12 改善，但 RMSE 仍差，因此不晋级、不扩展短 horizon。

12 个完整 Pipeline 全部独立重载，最大预测差 `4.16e-17`，哈希通过。运行约 768 秒，压缩模型合计约 0.47 GB。目录结构与 v1.16 相同。
