# v1.16 ExtraTrees 直接 OSI 基线计划

固定使用 v1.5.6 的 163 列特征与 balanced_v1 县级五折。第一轮只训练 t+24/t+48：ET-A 为 600 棵树、`max_features=0.5`、`min_samples_leaf=5`；ET-B 为 600 棵树、`max_features=0.7`、`min_samples_leaf=20`。二者均使用 squared error、seed 42。

每个外折都以仅在其余四折拟合的中位数填充器和 ExtraTrees 构成完整 Pipeline。保存五个外折 Pipeline 与全训练 Pipeline，独立重载复算 OOF/test。是否扩展短 horizon 由长 horizon 的 RMSE、多样性和尾部诊断决定，不追加超参数搜索。

## 执行结论

已完成两个长 horizon。ET-A/ET-B 的 t+24 RMSE 为 0.008906/0.008946，t+48 为 0.008252/0.008235，均明显差于 v1.8 与 v1.12；不扩展短 horizon。24 个 Pipeline 重载与哈希验证通过。
