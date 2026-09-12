# Phase 2 空间 GAT 模型评估报告

## 1. 结论

本次完整 GPU 运行达到了“整体优于 Phase 1 LightGBM 基线”的目标。与 `code_phase1` 的 direct-OSI LightGBM 比较，最终模型四个 horizon 的 RMSE、MAE 全部改善。与更强的 component LightGBM 主模型比较，GAT 的独立修正主要改善 t+48h；短时域因为 OOF blend 没有收益而自动关闭。

## 2. 运行配置

| 配置项 | 值 |
|---|---|
| 随机种子 | 42 |
| LightGBM 主模型 | v2.1 P/N/D/R component LightGBM |
| GAT 训练轮数上限 | 220 |
| Early stopping patience | 35 |
| 设备 | CUDA |
| 时间片步长 | 1，覆盖全部 144 个预测时间片 |
| kNN | 8 |
| 图边数 | 3,028 |
| 节点特征数 | 250 |
| 特征缓存 | v1.5.7 |

## 3. 与 Phase 1 direct-OSI LightGBM 比较

| Horizon | Phase 1 RMSE | Phase 2 RMSE | RMSE 改善 | Phase 1 MAE | Phase 2 MAE | MAE 改善 |
|---|---:|---:|---:|---:|---:|---:|
| t+1h | 0.012400 | 0.011958 | 3.57% | 0.003808 | 0.003591 | 5.69% |
| t+6h | 0.011088 | 0.010597 | 4.43% | 0.003647 | 0.003412 | 6.45% |
| t+24h | 0.008510 | 0.008451 | 0.70% | 0.003033 | 0.002918 | 3.79% |
| t+48h | 0.007731 | 0.007531 | 2.58% | 0.002379 | 0.002294 | 3.57% |

四个时域方向一致，说明最终提交模型整体优于 Phase 1 direct baseline。

## 4. GAT 相对 component LightGBM 的独立贡献

| Horizon | Component base RMSE | GAT blend RMSE | RMSE变化 | Component base MAE | GAT blend MAE | 最终 alpha |
|---|---:|---:|---:|---:|---:|---:|
| t+1h | 0.011958 | 0.011958 | 0.00% | 0.003591 | 0.003591 | 0.00 |
| t+6h | 0.010597 | 0.010597 | 0.00% | 0.003412 | 0.003412 | 0.00 |
| t+24h | 0.008451 | 0.008451 | 0.00% | 0.002918 | 0.002918 | 0.00 |
| t+48h | 0.007641 | 0.007531 | 1.44% | 0.002291 | 0.002294 | 0.20 |

t+48h 的空间修正主要改善大误差样本，RMSE 下降而 MAE 有极小幅度上升。

## 5. 提交文件审计

已验证：

- 行数 9,072；
- 列结构、标识符和原始行顺序与 `sample_submission.csv` 完全一致；
- 重复 `(fipsCode, timestamp_et)` 为 0；
- 四个目标列 NaN 数量分别为 63、378、1,512、3,024；
- 有效预测值均为有限数并位于 `[0, 0.65]`；
- 文件为 `code_phase2/outputs/submission_phase2_gat.csv`。

## 6. 时间因果与方法学说明

模型没有使用预测窗口的 outage、OSI、P/N/D/R、lag 或 target 作为输入。邻居停电状态只由 `hour_idx < 72` 的观测数据生成；未来时间只使用比赛允许的天气变量。训练标签完全没有作为测试节点输入。

当前 `component_v21` 主模型使用仓库中已有的 v2.1 OOF artifact。GAT 的最终 test inference 只使用训练数据训练出的 base model 和测试县可用的天气/观测特征。若进行最严格的独立 CV 审查，下一版应重新保存每个 component 模型的县级 fold prediction，确保每个 GAT fold 的 component base 完全排除验证县。

## 7. 复现命令

```bash
conda activate phase2
python code_phase2/main.py \
  --base-mode component_v21 \
  --device cuda \
  --epochs 220 \
  --patience 35 \
  --time-stride 1 \
  --k 8
```

权威运行参数见 `outputs/run_metadata.json`，原始指标见 `outputs/cv_summary.csv`，blend 选择见 `outputs/alpha_selection.csv`。
