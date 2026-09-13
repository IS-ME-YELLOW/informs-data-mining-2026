# v1.11 树模型融合实验计划

日期：2026-09-13

## 目标与固定协议

- 复用 v1.8 LightGBM、v1.9_xgboost、v1.10 CatBoost、v2.4 XGBoost 分量和 v2.5 CatBoost 分量的既有 OOF/测试预测，不重训基础模型。
- 固定 v1.5.6 的 163 列特征、`balanced_v1` 县级五折、`seed=42`、OSI 截断 `[0, 0.65]` 与 `<0.001` 置零。
- 正式对照为 v1.8 当前规则：t+1/t+6 使用 C3，t+24/t+48 使用 C1。
- 历史目录只读；新产物仅写入 `versions/v1.11_tree_ensemble/`。

## 候选与方法

每个 horizon 使用六路候选：LightGBM 直接、LightGBM 分量当前规则、XGBoost 直接、CatBoost 直接、XGBoost 分量、CatBoost 分量。

比较三种预先确定的方法：

1. 六路简单平均；
2. 非负、和为 1 的静态凸融合；
3. `baseline + alpha * (convex - baseline)` 安全回退融合，`alpha` 限制在 `[0,1]`。

## 严格嵌套评估

对每个外层县级折：只在其余四折 OOF 行上求凸权重和安全系数，在当前外折评分。全 OOF 拟合的最终权重只用于测试集推理，不用于报告 OOF 成绩。小维度凸优化枚举全部非空 active set，确定性求解带和约束的最小二乘，不做组合搜索。

## 诊断与验收

- 验证全部 OOF 的行顺序、county、timestamp、horizon、缺失掩码和真实标签，并从元数据验证特征/CV/seed。
- 保存残差相关、预测分歧、逐折/逐县、高 OSI、convex-hull oracle、县级 paired bootstrap。
- 建议晋级：RMSE 至少改善约 0.5%，至少 4/5 折改善，bootstrap 95% 区间低于 0，且无 horizon/高 OSI/t+48 明显退化。
- 保存外折模型、最终模型、OOF、测试预测、提交、元数据与 SHA-256；独立重载复算。

## 执行状态

已完成。三个融合方案均未通过 RMSE 晋级条件，保留 v1.8 为正式基线；详见 `README.md` 和 `results/`。
