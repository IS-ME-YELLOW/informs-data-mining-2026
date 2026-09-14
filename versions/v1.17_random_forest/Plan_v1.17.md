# v1.17 Random Forest 直接 OSI 基线计划

固定单配置：600 棵树、squared error、`max_features=0.5`、`min_samples_leaf=5`、bootstrap、`max_samples=0.8`、seed 42、`n_jobs=-1`。使用 v1.5.6 的 163 列特征和 balanced_v1 县级五折，先训练 t+24/t+48。

中位数填充器严格逐训练折拟合并和森林共同保存在 Pipeline 中。与 v1.8/v1.12 比较 RMSE、MAE、逐折、逐县、高 OSI、残差相关性与 paired county bootstrap，不做额外参数搜索。

## 执行结论

t+24/t+48 RMSE 为 0.008873/0.007937，相对 v1.8 退化 5.97%/4.53%，bootstrap 区间完全高于 0；不扩展短 horizon。12 个 Pipeline 重载与哈希验证通过。
