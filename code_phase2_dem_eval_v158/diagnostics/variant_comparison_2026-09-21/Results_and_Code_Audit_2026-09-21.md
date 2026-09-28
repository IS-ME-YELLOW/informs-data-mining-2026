# M0–M3 限制实验：结果分析与代码审查

日期：2026-09-21。使用保存结果与源码进行轻量审查；没有重新训练或重载模型推理，没有修改原代码、模型、预测及验收标记。

**结论：本轮确实修复了旧 GAT 的主要短时失稳现象，但没有一个变体在所有 horizon 上取得明显、稳定的优势。M1 是当前较有希望的 1h 候选，M2 是较均衡的候选；M3 缓解了 Newton 的爆炸，却没有解决它的 1h 高估。训练与内层选择的隔离设计基本成立，保存数值可以复算；正式验收仍未完成，并且验收脚本存在两个实际会阻断通过的问题。**

## 1. 必须先分清的源码和结果版本

四套结果位于 `code_phase2_experiment`，解压后的真正 run root 在多层 `models/gat/runs/<run_id>` 内。不要根据中间重复的 M0 文件夹名判断模型变体；本次以最内层 `run_manifest.json` 为准。确切路径保存在同目录 `audit.json`。

| 对象 | 实际内容 | 与四套结果的关系 |
|---|---|---|
| `code_phase2_compare` | 205 维输入，M0/M1/M2/M3，协议 dem_compare_v1_nested_v1 | 四套 manifest 记录的 13 个源码哈希全部与此目录匹配 |
| 用户指定的 `code_phase2_dem_eval_v158_limit` | 173 维输入，去除 16 个 neighbor_mean 和 16 个 neighbor_delta | 是另一项实验，不是这次 M0–M3 的生成源码 |

以下 M0–M3 的严谨性审查基于哈希匹配的 `code_phase2_compare`；173 维版本另见第 7 节。本次没有把两个版本混为一套实验。

## 2. 整体效果：相对于同一旧树基线

四套实验使用完全相同的数据、seed42 县分折、图和树基线。416 个基模型按 model_id 对应的文件哈希完全一致，外层逐行 base、标签、评分 mask 和分折也一致。比较对象是旧 component_v158 的同 horizon 分量树组合，**不是当前主线 F2**。

表中括号为 RMSE 相对树基线的变化，负值表示改善。

| 模型 | 1h RMSE | 6h RMSE | 24h RMSE | 48h RMSE |
|---|---:|---:|---:|---:|
| 旧树基线 | 0.01197027 | 0.01066853 | 0.00856008 | 0.00783086 |
| M0 完整 GAT | 0.01787464（+49.33%） | 0.01056003（−1.02%） | 0.00850021（−0.70%） | 0.00778967（−0.53%） |
| M1 去 skip | **0.01173229（−1.99%）** | 0.01057229（−0.90%） | 0.00863678（+0.90%） | 0.00813990（+3.95%） |
| M2 输入 z 截断 | 0.01193689（−0.28%） | **0.01049906（−1.59%）** | **0.00823029（−3.85%）** | 0.00780357（−0.35%） |
| M3 有界校正 | 0.01214228（+1.44%） | 0.01054550（−1.15%） | 0.00832734（−2.72%） | **0.00777381（−0.73%）** |

因此，不应把“从失败的 M0 大幅改善”表述为同等幅度的模型增益。比赛上更有意义的是能否超过树基线：M1 的 1h 实际增益约 1.99%，M2 的 1h 约 0.28%。也不应把四个 horizon 随意平均成新的排名指标。

按外折看：

| 候选/目标 | 改善外折数 | 仍需关注的外折 |
|---|---:|---|
| M1 / 1h | 4/5 | fold0 恶化 4.48% |
| M1 / 48h | 2/5 | fold3 恶化 36.44% |
| M2 / 1h | 3/5 | fold1 恶化 4.49%，该折内层选出 α=0.75 |
| M2 / 24h | 4/5 | fold0 恶化 12.83% |
| M2 / 48h | 4/5 | fold0 恶化 17.18% |
| M3 / 24h | 5/5 | 五折方向一致，但整体幅度小于 M2 |

M2 的“四个 horizon 都改善”是 pooled 结果，不意味着每折或每县都稳定。

## 3. Newton 是否修好，以及是否伤害其他县

Newton, IN，FIPS=18111，1h 共 143 个有效预测。真实 OSI 均值约 0.002036。

| 模型 | Newton 1h RMSE | 平均原始 OSI 校正 | 外层 α | 平均最终预测 |
|---|---:|---:|---:|---:|
| 树基线 | 0.003985 | — | — | 约 0.00383 |
| M0 | 0.208337 | +1.031726 | 0.20 | 0.210178 |
| M1 | 0.003490 | −0.011818 | 0.20 | 0.002312 |
| M2 | **0.003193** | −0.018076 | 0.10 | 0.002229 |
| M3 | 0.051897 | +0.249891 | 0.20 | 0.053811 |

**M1、M2 的重新训练结果支持之前的机制诊断：去掉直接 skip 或约束极端标准化输入，可以消除这次灾难性的正外推。** 它们已经不只是上一轮固定权重的探针。

M3 则接近正上限：0.25 × 0.2 ≈ 0.05。虽然数学上有界，对真实 OSI 只有约 0.002 的县仍然过大。Newton 的新增 SSE 从 M0 的 6.204523 降到 M3 的 0.382876，但依然吞掉其他县的合计改善；M3 其余县合计减少 SSE 0.241126，所以总体 1h 仍恶化。该上限控制了最坏幅度，没有让模型学会正确方向或量级。

### 3.1 M1 的 1h 收益并非只靠 Newton

相对于树基线，M1 在 Newton 上只减少约 0.000529 SSE；其他县合计减少约 0.192254 SSE。事后仅为诊断排除 Newton 后，M1 仍改善 1.98%。这不是新评分方案，只用来区分收益来源。

主要受益县包括：

| 县 | 树基线 1h RMSE | M1 1h RMSE | SSE 变化 |
|---|---:|---:|---:|
| Morrow, OH（39117） | 0.052486 | 0.046810 | −0.080591 |
| Clay, WV（54015） | 0.052187 | 0.047059 | −0.072777 |
| Tuscarawas, OH（39157） | 0.024900 | 0.020995 | −0.025627 |

也有代价：Guernsey, OH 的 1h RMSE 从 0.015678 升至 0.022889，Clarion, PA 从 0.012909 升至 0.017989。M1 的 48h 则在 Logan, OH、Wetzel, WV 和 Wood, WV 引入明显损失。**不能把“去 skip”作为所有 horizon 的统一升级。**

### 3.2 M2 更均衡，但 1h 收益相互抵消

M2 在 1h 改善 Forest, PA：RMSE 0.095979 → 0.092115，减少 SSE 0.103920；同时 Paulding, OH 增加 SSE 0.088589、Wood, OH 增加 0.057085，Morrow, OH 增加 0.042592。因此修复 Newton 后，整体短时收益仍只有 0.28%。

M2 的 24h 收益主要来自 Calhoun, WV、Brown, IN、Muskingum, OH 等；Calhoun 的 RMSE 从 0.048073 降至 0.038529。但 Clay, WV 的 24h 仍退化。这说明稳健输入处理有实质价值，同时没有解决全部高误差县的信息或预测结构问题。

全量县级结果见 `tables/county_metrics.csv`；最大损益见 `largest_harm.csv` 和 `largest_benefit.csv`。

## 4. 统计支持到什么程度

本次独立重算了保存的按县配对 bootstrap：每次完整保留抽到县的所有小时，共 2,000 次。与原结果一致。

| 结果 | RMSE 差值（模型−基线） | 95% bootstrap 区间 |
|---|---:|---:|
| M1 / 1h | −0.00023798 | [−0.00053112, +0.00005471] |
| M2 / 1h | −0.00003338 | [−0.00044536, +0.00046543] |
| M2 / 24h | −0.00032979 | [−0.00079909, +0.00007325] |
| M3 / 24h | −0.00023274 | [−0.00051908, +0.00003176] |
| M1 / 48h | +0.00030904 | **[+0.00005124, +0.00074672]** |

所有正向改善对应的区间都跨零，目前支持“值得确认的候选”，还不足以宣称稳定显著提升。M1 的 48h 退化反而有更明确的证据。保存字段 `gat_better_probability` 只是重采样中模型更好的比例，不是“真实泛化提升的概率”。县之间仍存在空间相关性，这也限制区间的解释。

此外，这些限制实验受此前 Newton 诊断启发，使用同一套 seed42 县分折。即使每次模型训练和 alpha 选择都遵守标签隔离，反复观察外层 OOF 再选结构仍会产生实验选择偏差。“配置写着 preregistered”本身不能证明预注册时间；当前材料能证明固定配置入了 run identity，不能证明外层结果从未影响设计。

下一步更换冻结分折/增加初始化可以检查稳定性，但仍不是新的独立数据集。按本表事后选择不同 horizon 的赢家，也只能形成下一轮待确认方案，不能当作已独立验收的组合成绩。

## 5. M0–M3 实现是否合规、隔离是否严谨

### 5.1 实现与声明一致

- **M1**：`ResidualGAT` 不再建立 skip 线性层，在两层图消息输出之后直接接 head。是重新训练结构变体，不是推理时把旧权重的 skip 置零。
- **M2**：每个 scope 只用监督训练节点和有效时刻拟合 mean/std，之后对全部 205 维标准化输入 clip 到 [−5,5]。训练和保存模型推理都走相同 policy；不是用全县拟合 scaler，也不是只在测试时截断。
- **M3**：训练过程中使用 tanh 有界输出。设残差尺度为 s、未约束的标准化输出为 z，则原始 OSI 校正是 **0.25 × tanh(s z / 0.25)**。传入网络的 normalized limit 是 0.25/s，输出后乘回 s，所以 ±0.25 的单位确实是原始 OSI 校正；最终增量上限还要乘 α。这与训练完再硬 clip 不同。

定位：[输入限制和 scope 标准化](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_compare/stacking.py:80)、[训练/推理一致应用](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_compare/stacking.py:310)、[M1/M3 网络结构](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_compare/gat_model.py:83)、[M3 尺度转换](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_compare/stacking.py:455)。

### 5.2 目前通过的隔离与可比性检查

1. 外层 fold k 的 scope 为其余四折；inner fold j 的整套树/GAT scope 再排除 j。内层 alpha 的 base 和 correction 来自配套的三折 GAT 上下文，没有拿外层四折模型预测冒充 inner OOF。
2. 树模型的 early stopping probe 及重拟合轮数均限制在本 scope。GAT 监督节点的 base 采用排除本折的树模型，scope 外节点使用完整本 scope 模型；已核对所有树清单与 GAT 完成记录中的依赖。
3. 残差标签、残差标准差和 mean/std 只从允许范围读取/拟合；完整图使用 302 个节点的公开特征，是全图建模设计，不等于使用验证县未来标签。
4. alpha 只根据相应 inner OOF 选取；最终 alpha 另在全训练内部 CV 选择。独立复算了 **768 个候选分数**（四次实验各 160 个外层选择候选与 32 个最终候选），选择结果与逐行应用一致。
5. 每个 horizon 的 239 县 OOF 行数、官方标签、尾部 NaN、后处理、有限性和唯一覆盖符合约定。模型失败或缺失行没有被静默剔除。
6. 四套冻结 CV manifest/产物哈希、合计 **1,920 个模型完成记录**及对应文件哈希、依赖和来源源码匹配。共同数据包哈希与本地冻结包一致，基线一致。

因此，**本次源码与保存证据中没有发现直接用外层或 inner 留出标签训练、早停或选择 alpha 的违规。** 这是轻量源码/产物审查结论，不等于重新执行了所有标签扰动测试，也不等于四套模型已完成独立推理重载。

另有两点方法限制：GAT best checkpoint 仍按 eval 模式下的训练 MSE 选取，并非独立验证损失；四套只有一套分折/全局 seed。M1 删除层还会改变后续随机初始化的抽样位置，因此单次差异不能解释成“保持其余权重相同的纯路径因果效应”。

## 6. 两个应修正的验收代码问题

### 6.1 M1–M3 的 alpha 标识检查误用 M0

位置：[artifact_checks.py:229](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_compare/artifact_checks.py:229)。`check_alpha_group` 没有接收 variant，而第 234 行调用 `identity("alpha", ...)` 使用默认 `m0_full`。上层 `check_rows` 虽然知道 variant，却未传入这个函数。

本次直接用真实保存的 fold0/1h 结果调用该检查：M0 通过，M1、M2、M3 都报 `AssertionError: alpha selection id: mismatch`。独立重算各变体分数和胜出 alpha 则全部通过。

修正方案：给 `check_alpha_group` 显式增加 variant 参数；构造 selection_id 时传入；外层和 final 两处调用均传入同一实际 variant。至少验证 M0/M1/M2/M3 的外层及最终 selection_id 均能正确匹配。

### 6.2 路径字段做了不一致的比较

位置：[artifact_checks.py:208](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_compare/artifact_checks.py:208)。前面的 model_path 检查会把反斜杠转成正斜杠，但完成记录 details 的逐字段比较又回到原始字符串。

四套结果各有 **416 个**同类差异：Parquet 清单中是 `models\base\...txt`，完成记录中是 `models/base/...txt`。只存在分隔符差异，文件哈希与其他元数据均一致。这个错误在原 Windows 环境中也可能出现，并非只要换回 Windows 就自然消失。

修正方案：对 model_path 字段在内存中按同一相对路径规则规范化后比较，其他字段继续严格匹配。173 维 `_limit` 的验收代码也继承了这段比较，应同步处理。

**这两个问题阻断正式验收，不构成本轮 RMSE 数值错误的证据，也不需要因此重训。** 四套 `verification.json` 均写着 `independent_reload: not_run_in_this_process`，且都没有 `COMPLETE`，所以当前不能称为“已完整验收”。

修改验收脚本还会改变记录在身份中的源码哈希。建议为历史模型做独立审计补丁、记录审计版本及原训练来源，保留旧 manifest/identity；不要直接改旧 manifest 使其迁就新 verifier。后续新实验使用修正代码与新 run_id。

## 7. 单独审查 173 维 `_limit` 版本

[config.py:21](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_dem_eval_v158_limit/config.py:21) 明确为 `GRAPH_INPUT_DIM=173`、`direct_gat_no_neighbor_summary_v1`；[stacking.py:266](/home/jacklo/XYY/INFORMS/informs-data-mining-wendyxu/code_phase2_dem_eval_v158_limit/stacking.py:266) 使用原始 173 列，不追加邻县摘要。schema 版本进入配置身份、产物及 checkpoint，并在加载时检查维度和版本，能区分旧 205 维模型。

源码上它保留原严格 scope 设计，没有因为删除摘要而扩大标签读取范围。删除显式摘要本身是合理的信息/结构消融，但它：

- 仍保留 skip，仍使用无界残差输出；没有实现 M2 的 z 截断或 M3 的有界策略。
- 仍保留本县 `pre_event_mean_osi` 等极端历史字段，因此不能认为删除摘要就解决了已定位的风险。
- 不属于本轮四套结果，不能引用 M1/M2 的改善作为它的训练证据。
- 如果下一轮同时改变 205→173 维、skip、输入处理，需设配套对照才能区分收益来源。

本次在项目 `.venv` 的 CPU 环境执行无训练预检查，结果为 **PREFLIGHT_OK**，日志另存 `limit_preflight.log`。这只说明本地输入/图构造契约能执行，不代表泛化收益或完整验收。

## 8. 下一步的具体判断

1. **短时优先：保留 M1 的 1h 候选。** 它超过树基线、四折改善，也对 Morrow/Clay 等大误差县有收益；先确认跨冻结分折/初始化的稳定性，不把它推广到 24h/48h。
2. **均衡路线：保留 M2，优先确认 6h/24h。** 1h 增益较弱，不必因其“四项都正向”就取代 M1。特别跟踪 Paulding、Wood OH、Morrow 和 fold0 的变化。
3. **M3 暂不作为 1h 升级。** 它在 24h 五折同向、48h 点估计最好，可以留作中长时备选；现有 ±0.25 对 Newton 明显过宽。不要利用 Newton 的外层标签继续搜索最优上限后把成绩当作独立验证。
4. **先补齐轻量验收代码问题和版本对应，再决定候选迁移。** 这些收益都是相对旧树基线；是否能在当前 F2 上叠加，必须重新构造相应 scope 的 F2 base/residual 后比较，不能直接相加，也不能把旧 GAT 权重接到新基线上宣称增益。

本轮比较支持继续研究稳健的神经残差模型，同时仍未单独证明图消息比同信息 MLP 更有价值。M0–M3 都保留图消息路径，M1 也不是无图对照。

## 9. 交付与复算

- `analyze_results.py`：只读原产物，独立计算指标、scope/alpha、来源哈希与 bootstrap；不加载模型权重、不训练。
- `audit.json`：四套结果路径、核查状态、缺失正式验收状态、验收脚本问题复现。
- `source_snapshot.json`：本次审查的 3,981 个原文件哈希。脚本结束时均未改变。
- `tables/overall_metrics.csv`、`fold_metrics.csv`、`county_metrics.csv`：完整结果。
- `tables/Newton.csv`、`largest_harm.csv`、`largest_benefit.csv`：重点县。
- `tables/bootstrap.csv`、`final_alpha.csv`：不确定性及最终提交路由。
- `tables/excluding_Newton_diagnostic_only.csv`：明确标记的事后排除诊断，不作为新模型成绩。
- `tables/source_code_match.csv`、`verifier_findings.csv`：源码对应和验收函数问题。

最终提交的 alpha（依次 1h/6h/24h/48h）：M0=0/0.35/0.20/0.10；M1=0.35/0.20/0.20/**0**；M2=0.35/0.35/0.50/0.35；M3=0.10/0.35/0.35/0.10。例如 M1 的保存测试提交在 48h 回退树基线，不能把其外层 +3.95% 恶化直接当成该测试提交的表现；测试标签未知，不能在本地计算真正的测试 RMSE。
