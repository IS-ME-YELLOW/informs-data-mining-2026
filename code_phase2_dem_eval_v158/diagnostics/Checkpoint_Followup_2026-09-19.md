# 原检查点后续核查（2026-09-19）

当前诊断已完成保存结果层面的核验；完整模型重载没有完成，不能用 CV_COMPLETE 代替。

若仅核查 Newton 的外层 fold2、1h，需要优先找回原运行的：

- `models/gat/gat_component_v158_t01h_S0-1-3-4.pt` 及同名 `.complete.json`。
- 该 checkpoint 依赖的 P_t/N_t/D_t/R_t 基模型及完成记录：训练 scope 为 `0-1-3-4`、`1-3-4`、`0-3-4`、`0-1-4`、`0-1-3`，均为 t01h，共 20 个基模型。
- 原 checkpoint 内 feature_mean/std、residual_scale、fit_info、node/time 顺序、base_dependencies 与图身份；若原始训练 stdout 另存，也应保留作 epoch 诊断。

完整原运行的验收要求 416 个基模型、64 个 GAT checkpoint 和对应完成记录。本地缺失清单位于 `tables/missing_model_artifacts.csv`，共 960 项。原运行 Windows 绝对路径需要显式映射到同内容本地副本，不能改写原 manifest 来掩盖变化。

拿到原文件后，先在 diagnostics 下建立只读来源副本/映射记录，并将所有新输出限定在 diagnostics；不要直接调用会重写原运行 verification/COMPLETE 的入口。

核查顺序：

1. 核对 checkpoint/receipt/hash/scope 与原运行身份；不缺省重训。
2. 用保存 scaler 和原基模型重建 outer2 的全节点图输入，预测应重现保存 Newton 校正。
3. 在固定权重上逐组扰动历史重尾字段、skip 输入或消息通道，区分直接路径与图传播的敏感性。此为机制诊断，不是经过重新训练验证的新模型。
4. 新预处理或有界校正若要用于比赛，再按完整内层隔离协议重新训练、独立评估。不能用事后剪掉 Newton 的 OOF 分数宣布改进。

本次没有安装 PyTorch、没有向原目录写入文件，也没有执行上述缺失权重步骤。
