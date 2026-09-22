# 最新树方案与 Stella L2 的路线整合计划（2026-09-20）

**状态：待用户确认。本轮仅写计划，尚未计算新的组合预测、运行组合评价或训练模型。**

本实验回答一个固定问题：**保留最新短时预测，48h使用Stella L2时，24h采用当前信息增强树、Stella L2，还是两者固定1:1组合，更值得进入后续验证？**

三套既有CV按同一冻结方案一起评价，不在seed42看完结果后调整权重再称其他两套为原方案确认。新增拟合数为 **0**，新增模型数为 **0**；只组合已完成、同分折的最终OSI OOF预测，并独立核验来源和计算。

所有新增代码、配置、逐行产物、核验和报告统一放在：

`/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/tree_stella_integration`

## 1. 两个来源的准确含义

对CV seed s、最终输出h，记：

- **T_h**：当前最新树方案的最终OSI预测，取P24邻县实验的 `NB_P24` 主规则。它已经包含P1 a/b＋F2、P6 a/b（163列）、P24直接Huber＋邻县（183列）、D_G，以及其他原strict来源。
- **S_h**：Stella严格实验已保存的 `L2` 最终OSI预测。L2在大多数行用六路均值L1，在原L0预测大于其严格内层正预测q95时回退L0。本轮只使用S24、S48。

这里的 **Stella L2是候选名称，不是平方损失名称**。T24是分量树C1重组后的完整OSI，不是单独的P24分量。

Stella文件里的1h/6h仍固定为较早的F2，不含后续P6结构和P24邻县增量。它们不进入本轮候选。不能把Stella整张预测表直接覆盖当前四个输出。

已有证据是：T24在seed42/20260917优于S24，S24在seed20260918更好；S48在三套都优于T48。两个24h方案相对原L0的预测修正相关性约0.21–0.23，支持检验互补，但这不保证固定平均提分。

这些来源成绩、县误差和相关性已经被查看。本计划是运行前冻结一个有限比较，**不把它描述为从未接触过开发数据的独立预注册实验**。

## 2. 仅保留三个整合候选和一个完整参照

| 编号 | 1h | 6h | 24h | 48h | 作用 |
|---|---|---|---|---|---|
| **B0：当前树参照** | T1 | T6 | T24 | T48 | 保留当前完整组合 |
| **I1：树24＋Stella48** | T1 | T6 | T24 | S48 | 只整合已有48h候选 |
| **I2：Stella长期** | T1 | T6 | S24 | S48 | 使用Stella两个长期输出 |
| **I3：固定半数融合，主候选** | T1 | T6 | H((T24＋S24)/2) | S48 | 本轮唯一新增的24h融合假设 |

所有CV使用相同候选定义。**不按CV、fold、县、日期、真实严重程度或本轮误差选择I1/I2/I3。**

本轮不增加0.2/0.8等其他权重，不做权重优化、加权回归、堆叠、残差拟合、新的q95或逐行门控，也不引入Stella L1作为额外可选提交方案。L0/L1和原门控标记可用于解释及来源核对。

### 2.1 组合层级与后处理

组合发生在**最终OSI层**。底层P/N/D/R预测、源模型、特征和C3均不修改、不重算：

```text
B0：直接复制 T 的四个最终输出
I1：复制 T1、T6、T24，最终48h替换为 S48
I2：复制 T1、T6，最终24h/48h替换为 S24/S48
I3：复制 T1、T6、S48，最终24h按下式生成

mean24 = (T24 + S24) / 2.0
clipped24 = clip(mean24, 0.0, 0.65)
I3_24 = 0.0 if clipped24 < 0.001 else clipped24
```

- T24、S24均取已保存、已完成各自后处理的最终OSI值。不能改成平均原始树输出、平均分量或将新的长期值送回C3。
- 严格小于0.001才置零，等于0.001保留；不round预测。计算使用float64。
- 无效尾部保持NaN。有效行任何来源缺失/Inf都报错，不用nanmean、单来源回退或删行来补救。
- I1/I2的来源复制不再做额外变换，必须逐行精确相同。
- 本轮不为I2/I3虚构一套P/N/D/R分量；其新产物是最终OSI组合。

### 2.2 q95边界

Stella原L2已经包含其门控，本轮完整复用该结果，不替换其内部L0，也不重新计算阈值。I3只是在它的最终输出之外做固定平均。

**最终平均可能弱化Stella原有的高值保护。** 即使S24在某行回退L0，I3也不再保证等于该L0；因此必须单独检查原门控行的损益，不能把I3称为“仍然完整保留尾部回退性质”。

如果本轮后续想让最新T24取代Stella的L0、修改六路中的LightGBM分量来源或学习融合权重，应另立实验；阈值/权重须在严格内层重新估计，不能直接复用旧q95或在全局OOF上择优。

## 3. 冻结数据来源和身份

根目录：`/home/jacklo/XYY/INFORMS/informs-data-mining-2026`。

T的权威入口是：

[最新树候选来源清单](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json)

读取其中对应seed、case=`NB_P24`的 `control_predictions.parquet`，选列：

```text
pred_NB_P24_v18_rule_osi_target_t01h
pred_NB_P24_v18_rule_osi_target_t06h
pred_NB_P24_v18_rule_osi_target_t24h
pred_NB_P24_v18_rule_osi_target_t48h
```

S的固定run目录为：

`/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/stella_v112_strict/runs/stella_v112_nested_v1_split{seed}_model42`

读取 `candidate_predictions.parquet` 中 `pred_L2_osi_target_t24h`、`pred_L2_osi_target_t48h`；诊断另读 `pred_L0_...`、`pred_L1_...`、`theta_...`、`anchor_gt_theta_...`。只接受下表三个完成run，不能按mtime选择 `sequential_partial` 目录。

| CV seed | T run identity | S run identity |
|---|---|---|
| 42 | `bd4468e5c37fba7e93832153bc9847f807b8ae0054e45586f3c3853e96835bbd` | `0a019474a437758214bb2eb3de8e13fb0a112320f6c69df0fe76b41166788be4` |
| 20260917 | `847bc47ba2970bcebb9711cd554874f4e423d6b3d7a0df9683c93afc89dd6fcb` | `a859aa7b9cbd9edc5121c6755f0a8eb0c32850bda08f31457187988fb7564b1b` |
| 20260918 | `e8cf35263d42073aaae928f05ba58ab0b1f66df89b5d9cfe7a3944345c44dfe1` | `3e67a67e6fa8ed7d435ee58805a8af4584dfee8974a298d710da60ac2abe6699` |

三套CV显式固定为42、20260917、20260918；上游模型seed均为42。每套T与S必须使用对应的 `cv/cv_assignments_balanced_v1_seed{seed}.csv`。

编写本计划时仅核对了文件schema、行数及上述身份，没有计算I3。每个来源文件均有34,416行；Stella的split_id分别为seed42/seed20260917/seed20260918，T表没有split_id列。因此新表的cv_seed由清单和CV文件确定，不能要求不同来源用同一编码，也不能靠改一个fold列伪造来源身份。

## 4. 运行前完整性与对齐

1. 验证T清单的run identity及所用文件SHA-256；检查相应完成标记、独立验证记录。S检查 `run_manifest.json`、`CV_COMPLETE`、`verification.json` 的身份一致及所用预测、阈值、指标文件哈希。
2. 将实际读入文件及其哈希写入本目录source_manifest，执行前后复核。二进制parquet逐字节匹配；文本如出现换行差异，按上游明确的移植规则记录原始/LF规范化哈希，不忽略其他差异，不改写旧文件或回执。
3. FIPS规范成五位字符串。以县＋预测起点时间唯一键一对一对齐，并核对hour_idx、target_hour、outer_fold与本CV清单。不同CV不得混用预测。
4. 保留原有冻结时间戳语义，小时0为2026-03-11 00:00；不做新的UTC转换。每县起点72–215，目标s=t+h。
5. T的 `true_osi_target_...` 与S的 `truth_osi_target_...`、冻结官方目标表逐值一致。有效性只由确定的时间条件与官方标签决定，不能取“两个来源都恰好有预测”的子集。
6. 各来源有效预测必须finite，位于[0,.65]，且为0或≥.001；尾部NaN位置与官方一致。现有S门控行应满足 `L0 > theta`，且L2等于相应L0/L1；这只核对保存算术，不重新选阈值。
7. 从保存OOF复算T、S的四horizon来源指标，并与各自已发布指标核对；Stella短时列只作来源说明，不参与候选。

正式覆盖：239县×144起点＝34,416行；有效评分行如下：

| 输出h | 有效目标小时 | 有效行 |
|---|---|---:|
| 1 | 73–215 | 34,177 |
| 6 | 78–215 | 32,982 |
| 24 | 96–215 | 28,680 |
| 48 | 120–215 | 22,944 |

本轮依赖上游已完成的严格训练/模型重载验收，新增核验负责预测身份、对齐和组合算术。**不把此次OOF指标复算说成重新训练、重新加载或重新审计了全部源模型。** 不调用会向旧目录写入的验收入口。

## 5. 比较、指标和不确定性

### 5.1 预定比较

| 比较 | 主要输出 | 用途 |
|---|---|---|
| **I3−I1** | 24h | 固定平均相对信息增强树的增量，主要比较 |
| **I3−I2** | 24h | 固定平均相对Stella L2的增量，主要比较 |
| I2−I1 | 24h | 两个单独长期来源的直接比较 |
| I1−B0 | 48h | 整合已有Stella48h的效果与算术复现 |
| I3−B0 | 四输出 | 完整主候选相对当前整套树方案的变化 |

所有比较都保留四输出的完整指标向量；确定不变的项直接验证为零。主要的新证据是24h融合，不把各候选共用的48h既有收益重复包装成I3的新发现。

每个CV、候选、horizon报告全县全行pooled RMSE、SSE、MAE、bias和相对变化。折/县RMSE均值不能替代pooled RMSE；不平均三套预测，不合并三套样本成为717个独立县。

官方按各horizon的RMSE名次平均排名，平局优先1h；本地没有其他参赛队分数，不能由自己的四个RMSE推算官方总排名，也不将原始RMSE简单平均称为官方成绩。

### 5.2 固定分组与尾部检查

- 输出所有239县、五个外折、全部目标日期的损益，不删除极端县。
- 固定目标窗口96–119、120–143、144–215，与各h有效范围取交集；24h额外报告共同120–215，不能用它替代完整24h评分。
- OSI真值分组固定为0、(0,.01]、(.01,.05]、>.05，仅用于事后诊断，不用于选择预测。
- 按Stella原24h门控标记分为“回退L0”和“使用L1”两组，报告I3相对T24/S24的SSE、bias及高值低估，检查平均是否破坏原尾部保护。
- 关注县固定为此前7县清单（Forest、Morrow、Brown、Clay、Calhoun、Morgan、Wyoming PA），再加Stella报告中的39157、39149、54101、54105。按FIPS核对名称；全县表仍是主依据。
- 按固定规则列出各比较SSE改善/恶化前10县，明确属于结果诊断；不据此新增县级回退。

### 5.3 互补性与后处理拆解

24h同时报告T/S的误差相关性、相对原严格L0的修正相关性，以及以下恒等式：

```text
mean24 = (T24+S24)/2
SSE(mean24) = 0.5*SSE(T24) + 0.5*SSE(S24) - 0.25*sum((T24-S24)^2)
```

再单列 `SSE(H(mean24))−SSE(mean24)`、置零行数和逐行后处理变化，区分平均互补与阈值影响。`mean24`只作算术诊断，**不是第五个候选**，不能看完结果后改用“不做后处理”的版本。

### 5.4 配对区间

对上表可能变化的输出计算县整条轨迹配对bootstrap：2,000次，seed20260910、95%百分位区间。每次基准/候选抽相同县集合，保留完整轨迹，按抽样SSE/行数计算RMSE。不变项区间固定[0,0]。

各CV分别报告；不把1h/6h或三个CV视作独立重复试验。区间只描述当前固定预测的抽样不确定性，不校正此前反复看过开发集所造成的选择偏差；不将bootstrap负差比例称为测试集获胜概率。

## 6. 判断规则与停止条件

1. **技术有效性优先**：来源、CV、标签/行键、掩码或短时不变量失败，相关比较不能作为正式结果。追查原因并记录，不静默修预测或缩小评分集。
2. 若I3在三套24h上相对I1和I2均不恶化（仅将绝对RMSE差≤1e-12视为数值相等），并有实际改善，则它是较强的整合候选；仍结合区间、县/时段和尾部损益判断证据强度。
3. 若I3稳定改善一个来源、但相对另一个来源有跨CV取舍，则将其保留为待压力测试的候选，**不能称它已统一胜过两条路线**。不根据每个CV分别选最佳方案。
4. 若I3没有带来清晰增量，或收益由脆弱的少数县/后处理边界主导，不强行采用固定平均；保留I1/I2的比较结论，不继续在同一OOF扫描权重。
5. 任一候选若仅存在很小且不稳定的差异，优先维持简洁参照，不能将“完成实验”写成“获得新提升”。不设置事后才决定的最小收益阈值。
6. 本轮只决定哪些整套候选进入下一步。地理分块压力测试、随机邻居负对照、学习权重、更新q95、新数据/新模型、全量训练和提交均不属于本轮；需要时另立计划。

计划不预先要求某个方案必须胜出。即使固定平均失败，明确“整合收益有限/来源存在取舍”也是完整实验结论。

## 7. 文件布局与执行顺序

**本次只创建本计划文件。** 确认后拟生成：

```text
tree_stella_integration/
  Experiment_Plan_2026-09-20.md
  README.md
  integration_config.json
  integrate_routes.py
  verify_integration.py
  summarize_integration.py
  reference/
    approved_plan.md
    source_manifest.json
    initial_input_hashes.json
  runs/
    split42/
      run_manifest.json
      source_predictions.parquet
      integrated_predictions.parquet
      source_lineage.json
      row_changes.parquet
      blend_postprocessing_audit.parquet
      metrics/
        primary_scores.csv
        comparisons.csv
        paired_county_bootstrap.csv
        county_metrics.csv
        fold_metrics.csv
        target_window_metrics.csv
        target_day_metrics.csv
        severity_metrics.csv
        original_gate_metrics.csv
        complementarity.csv
      verification.json
    split20260917/                # 同结构
    split20260918/                # 同结构
  summary/
    three_cv_scores.csv
    comparison_consistency.csv
    county_consistency.csv
    fixed_focus_counties.csv
    decision.json
  Results_<实际完成日期>.md
  final_integrity_check.json
  completion.json
```

每套保存全部34,416个起点及四horizon的候选宽表，包含县码、起点/目标时间、外折、cv_seed、真实值、有效掩码、T/S源值、四候选预测及源run identity。逐行变化另存长表，保留比较名称、SSE差和原门控状态。不能只存总体分数或改善行。

源模型和旧大文件不复制、不修改；source_manifest记录绝对路径、SHA-256、schema、来源run/case和所用列。source_lineage明确每个候选每个输出的来源及固定权重。所有新写入解析后必须位于本目录。

用户确认后的执行顺序：

1. 将本计划原文快照到reference，冻结配置和来源清单。
2. 对三套来源完成只读预检，并先复现T/S指标。
3. 一次性按固定定义生成三套的B0/I1/I2/I3，运行预定评价；不以seed42作为新增调参回合。
4. 新进程使用独立实现核对来源、混合/后处理、指标及区间，再生成统一结果和完整性证明。

预定接口如下（脚本尚未创建，本轮不执行）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/integrate_routes.py --stage preflight --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/integrate_routes.py --stage evaluate --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/verify_integration.py --split-seeds 42 20260917 20260918
```

## 8. 独立验收清单

- 两类源预测与各自完成标记、run identity和本CV清单匹配，239县与全部时间键一一对应；官方真值一致。
- 四候选1h/6h与T逐行精确相同；I1的24h与T精确相同；I2的24h与S精确相同；三个整合候选48h与S精确相同；B0四输出与T精确相同。
- I3只在24h生成新数值；有效行完整、区间合规、尾部NaN严格一致。独立实现用另一种索引/计算路径复算，预测与指标绝对差≤1e-12。
- 针对0、.001边界、来源一零一正、相等来源、有效行缺失和无效尾部做有意义的边界检查。相等来源应复现原值，缺失有效来源应拒绝。
- 验证平均MSE恒等式及后处理差额闭合；分组SSE/行数能加总回对应完整指标；bootstrap重抽样单位为县轨迹。
- 原q95、原门控标记、C3和源模型不改变；不把I3的混合结果送回任何源模型或内层阈值。
- 源文件前后哈希一致；所有新增产物位于本目录。最后才写completion，并明确“训练次数0、无测试集推理、无全量模型、无提交”。

最终报告先给三套×四输出的整体表及I3相对两个24h来源的对照，再解释尾部、县/日期稳定性和下一步选择。**不会把共用的48h改善或源模型旧成绩当作本轮固定融合的新收益。**
