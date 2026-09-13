# v1.19 weather-mechanism tree experts

本版本用深度 2/3 的浅分类树依据无标签天气机制选择 v1.12、ET-A、ET-B 或 RF，再用训练折内安全系数收缩到 v1.12，并保留高风险 v1.8 回退。完整门控 Pipeline 和参数包以 joblib 保存。

```powershell
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions\v1.19_weather_tree_experts\run_weather_tree_experts.py
C:\Users\Stella\.conda\envs\best-route\python.exe -B versions\v1.19_weather_tree_experts\verify_artifacts.py
```

## 结果

深度 2 的所有外折安全系数均为 0，预测完全回退 v1.12。深度 3 仅 t+24 fold 1 的安全系数为 0.0966、路由 144 行到 RF；池化 t+24 RMSE 相对 v1.12 退化 0.0051%，t+48 完全回退。说明可见的局部 ET/RF 优势无法由这些天气机制在县外折间稳定迁移，`promote=false`，不继续更深门控。

24 个门控 Pipeline 全部独立重载，最大预测差为 0，输入/产物哈希通过。
