# v1.18 不同树族严格嵌套融合计划

候选固定为 v1.12、v1.12+ExtraTrees、v1.12+Random Forest、v1.12+ExtraTrees+Random Forest，不扩大组合搜索。对每个外折，ET-A/ET-B 的选择、非负和为 1 的权重、v1.8 正预测 q95 风险回退阈值都只使用其余四折；当前外折只评分。短 horizon 保持 v1.12。

输出残差相关、预测分歧、实际高 OSI 前 5%、t+48 极端县、convex-hull oracle、逐折/逐县与 paired county bootstrap。全 OOF 模型只用于 test 推理，不能作为 OOF 成绩。

## 执行结论

三个含新树族的候选在 t+24/t+48 均未超过 v1.12；最接近的 v1.12+RF 仍退化 0.006%/0.121%。ET 的局部外折与高 OSI 收益不足以改善池化 RMSE，`promote=false`。48 个融合模型重载与哈希验证通过。
