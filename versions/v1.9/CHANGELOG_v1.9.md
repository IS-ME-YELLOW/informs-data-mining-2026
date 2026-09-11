# v1.9 改动记录

日期：2026-09-11

## 1. 新增实现

### `config.py`

- 集中定义 v1.5.6 输入、v1.8 基线资产、v1.9 输出路径。
- 固定 horizon、P/N/D/R 分量、五折版本、随机种子和 LightGBM 公共参数。
- 定义 P/D 的 Huber、L2、尾部加权 L2，以及 N/R 的 Huber、Tweedie、hurdle 候选。

### `build_response_profiles.py`

- 读取 v1.5.6 已冻结的训练/测试特征，不改写原文件。
- 从预测时点前 72 小时的 P/N/D/R 与阵风观测构建 27 个县域响应画像。
- 生成 190 列 v1.9 特征数据、画像明细、特征名称与定义。
- 校验原 163 列基础特征逐列不变、训练测试列一致、行数和县数一致、无未来信息进入特征。

### `protocol.py`

- 复用 v1.8 的县内标签平移、官方 OSI 重组、预测裁剪、县级五折和提交尾部缺失逻辑。
- 新增分量标签对齐与 v1.8 OOF/测试预测读取接口。

### `train_v19.py`

- 训练基础特征和响应画像特征两套模型。
- P/D 支持 Huber、L2、折内 95% 分位数定义的尾部加权 L2。
- N/R 支持 Huber、Tweedie，以及二阶段 hurdle（是否为正 + 正值 Gamma 幅度）。
- 保存折模型、全量模型、OOF、测试预测、折指标、特征重要性和 SHA-256 模型清单。
- 支持 `--resume` 复用配置一致的折模型。

### `evaluate_v19.py`

- 生成 E0/E1/E2/E3、aligned 组合、最佳分量子集及逐 horizon 最优候选。
- 输出总体、分量、折、县、严重度、目标日、分量贡献和 hurdle 诊断指标。
- 对候选与 v1.8 参考做 2,000 次县级配对 bootstrap。
- 生成六套带 `diagnostic_submission_` 前缀的诊断提交，避免误认为正式晋级提交。

### `verify_artifacts.py`

- 独立重载 576 个模型并核对哈希。
- 重新计算 96 套候选预测与 OSI 指标。
- 检查特征文件、预测文件、诊断提交和 horizon 尾部缺失规则。

## 2. 新增数据和结果资产

- 190 列训练/测试特征：`features_train_v1.9.parquet`、`features_test_v1.9.parquet`。
- 27 列画像数据：`response_profiles_train_v1.9.parquet`、`response_profiles_test_v1.9.parquet`。
- 576 个模型：`models/`。
- 分量候选、OOF 和测试预测：`*_component_candidates.parquet`、`oof_predictions.parquet`、`test_predictions.parquet`。
- 总体和分层指标：各类 `*_metrics.csv`、`component_contribution.csv`、`profile_gain_summary.csv`。
- 六套诊断提交：`diagnostic_submission_v1.9_*.csv`。
- 机器可读运行与验证记录：`training_metadata.json`、`run_metadata.json`、`verification.json`。

## 3. 运行中修正

首次训练后发现，全量模型曾使用五折最佳轮数的中位数，而 v1.8 使用均值。为保持测试预测和旧基线协议一致，随后做了以下修正：

1. 折模型和 OOF 保持不变；
2. 将普通回归、hurdle 分类和正值幅度模型的全量迭代轮数全部改为五折最佳轮数的均值；
3. `--resume` 增加全量模型轮数核对，轮数不一致时只重训全量模型；
4. 重新生成测试预测、诊断提交与独立校验结果。

最终 `verification.json` 对应修正后的资产，最大预测复算差异为 0。

## 4. 仓库级记录

- 在 `Results.md` 追加 Phase 1.9 的协议、主要结果与决策。
- 在 `logs/experiments.log` 追加 v1.9 运行记录。
- 该历史日志中原有 24 处 GBK 编码的 `±`/`—` 混入 UTF-8 内容；追加前仅将这 24 个字符转为等价 UTF-8 字节，使文件恢复为有效 UTF-8，未改动原有文字和数值。

## 5. 最终决策

- 完成 M1–M4。
- 四个 horizon 的点估计均有微小改善，但对当前 v1.8 规则的县级配对 bootstrap 95% 区间全部跨过 0。
- 按预注册门槛不进入 M5，不替换 v1.8；t+6 E3 aligned 留作下一轮唯一优先复核候选。
