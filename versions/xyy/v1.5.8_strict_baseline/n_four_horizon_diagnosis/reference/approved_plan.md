# N 分量四个 horizon 诊断计划（2026-09-19）

**状态：待用户确认；本轮仅创建计划，尚未运行诊断或训练。**

目标是在当前优先组合 **strict v158＋D_G＋P 初始状态 a/b＋邻县 F2** 上，判断 N 是否值得优先优化、应处理哪些 horizon，以及问题主要来自幅度、时间定位、信息覆盖还是分量组合。诊断不能预先假定“下一个分量一定是 N”，也不能把真实值替换得到的收益当成新模型成绩。

本次工作目录：

`/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/n_four_horizon_diagnosis`

用户确认本计划后，实施诊断脚本、逐行分析、独立核验和报告；**不训练任何模型，不修改既有代码、模型、分折、特征、预测或报告，不生成提交文件**。后续 N 训练、P 方法向其他 horizon 迁移、Stella/GAT 实验均另立实验。

## 1. 必须回答的问题

1. N 在原始 1/6/24/48h 模型上分别错在哪里？是大量小值的背景误差，还是少数突增的低估、错时或假峰？
2. C3 同目标小时平均对 N 是改善还是稀释峰值？哪一个来源 horizon 对当前 1/6h OSI 的影响较大？
3. N 分量误差下降有多大可能转化为整体 OSI 收益？是否存在 P、D、R 的误差抵消？
4. 困难是否集中在既有高误差县？这些县的合法历史、天气与静态属性，在对应外层训练县中是否有可比支持？
5. 下一步应优先做 N 的损失/预测结构实验、N 的信息增量实验、P 的跨 horizon 迁移，还是暂缓 N？需要给出证据和仍无法回答的部分。

## 2. 当前参照与读取来源

### 2.1 参照定义

- 主参照记为 **B_current**：各 CV 自己的 `p_information_increment` **F2** 完整产物。F2 已包含 D_G、P1h 的 a/b 和邻县信息；不是原始树基线，也不是 F0/E2，更不含 F1 的 DEM。
- 历史对照记为 **B_strict**：同 CV 的原始 strict v158。只用来说明 P/D 改善后 N 的相对瓶颈是否变化，不重新筛选当前基线。
- 当前 N 四个 horizon 均仍为原 strict v158 的 Huber LightGBM，使用原 163 列特征。F2 的新增邻县特征只训练过 P1h，不能写成 N 已使用这些特征。
- 全部分析使用现有县级严格 OOF；模型 seed 均为 42。三套 split seed 是 `42 / 20260917 / 20260918`，同一套内所有分量对同一县必须使用同一外折。
- 三套都按本计划运行，分别报告结果；不挑一套最优分折，也不平均/拼接三套 OOF 生成新候选。它们是同一事件、同一批县的重复开发划分，不是三份独立测试集。

### 2.2 已定位的主要文件

以下绝对目录用符号缩写，仅用于文档；脚本配置必须保存解析后的绝对路径：

- `ROOT` = `/home/jacklo/XYY/INFORMS/informs-data-mining-2026`
- `BASE` = `/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline`
- `OUT` = `/home/jacklo/XYY/INFORMS/informs-data-mining-2026/versions/xyy/v1.5.8_strict_baseline/n_four_horizon_diagnosis`

| 来源 | 目录/文件（相对上面的根） | 用途 |
|---|---|---|
| 当前 F2 | `BASE/p_information_increment/runs/v18_pinfo_v1_split{seed}_model42/` | 主参照；读取下述三个 parquet 与身份文件 |
| 原始严格树 | `BASE/runs/v18_tree_nested_v2_split{seed}_model42/` | 原 N 预测、未截断输出、外折模型身份、历史对照 |
| 分量标签 | `BASE/component_targets_v1.8.parquet` | 对齐后的官方 P/N/D/R 标签 |
| 原始训练数据 | `ROOT/data/DM_Train.csv` | N/R 公式、时间戳、客户数及官方 OSI 核对 |
| 冻结特征 | `ROOT/versions/xyy/v1.5.6/` | `features_train`、`meta_train`、`targets_train` 的 `_v1.5.6.parquet` 与列名清单 |
| 冻结分折 | `ROOT/cv/cv_assignments_balanced_v1_seed{seed}.csv` | 外折身份与训练支持范围 |
| F2 特征来源 | `BASE/p_information_increment/features/v1/feature_manifest.json` 及其引用文件 | 来源核对及已有合法邻县摘要的描述性检查 |

当前 F2 每套读取：

- `component_predictions.parquet`：选择 `pred_F2_{component}_target_t{hh}h` 和对应 `source_F2_...`。seed42 同时含 F0/F1/F2，另外两套含 F0/F2，不能按列位置读取。
- `control_predictions.parquet`：读取 `pred_F2_v18_rule_osi_target_t{hh}h`，并保留 C1/C3 对照与官方 `actual_...`。
- `aligned_unique_components.parquet`：必须先过滤 `case == F2`，再读取 N 等分量的目标小时预测、候选数与来源 horizon。
- `run_manifest.json`、`source_manifest.json`、`INFO_CV_COMPLETE`、`logs/independent_verification.json`：只读验证来源及已有 PASS，不调用会向旧目录写文件的训练/验收入口。

原 strict 每套读取 `base_predictions_cv.parquet` 的 `raw_N_t_target_t{hh}h`、`oof_component_predictions.parquet`、对应控制预测和模型身份记录；执行时按实际 schema 建立字段映射，拒绝猜测或缺列回退。

编写计划时只读取了源码、报告、manifest 和 parquet 元数据；三个 F2 记录的独立验证状态均为 PASS。其当前 run identity 为：

| split seed | F2 identity_hash |
|---|---|
| 42 | `e39b76d75f191891abda1939dc47933815cc4fa15796227b0c200eb1b96bfcac` |
| 20260917 | `64f8263f5356378124c5e981fcb7c1117ad808c97b46c383b64b7cdbe109484d` |
| 20260918 | `a3f16d3463442c01b1e1315df498317eed53298a1f89ef087b016ab8bd0c9532` |

这些身份记录不替代执行阶段的文件哈希核验；若来源发生变化，先记录差异，不能静默接受并沿用旧身份。

## 3. N 的语义、时间与评分边界

### 3.1 保留官方定义

设 O_s 为小时 s 的 `outageCount`，C_s 为同小时 `customersTracked`：

```text
q_s+ = max(O_s - O_(s-1), 0) / C_s
q_s- = max(O_(s-1) - O_s, 0) / C_s
N_s = (q_(s-1)+ + q_s+ + q_(s+1)+) / 3
R_s = (q_(s-1)- + q_s- + q_(s+1)-) / 3
```

这是净停电客户变化的正/负部分再做**中心三小时平均**，不识别具体客户的故障/恢复。因此：

- N/R 同时为正是合法的；不能强制互斥。
- N 不等于 `max(P_s-P_(s-1),0)`。中心窗口、客户数分母及先取正再平均都会造成区别；也不将 P 的 a/b 分支解释为 N/R。
- `N_s-R_s` 可与同一中心窗口的有符号 `ΔO/C` 均值核对；客户数变化时不能直接当作 P 的三小时差。
- 停电观测固定在小时 0–71。官方给出的 N71/R71 可以直接使用；不得读取隐藏的小时72停电重新计算观测末端。
- 中心平滑的 s+1 是**标签定义**，不构成可以读取未来停电特征的许可。本轮不改为后向 rolling，不做气象/降水一小时平移实验。
- 仅在窗口及差分所需原始小时完整时复算公式。s=215 缺少 s+1，保留官方标签、标记为“不可完整复算”，不能补造未来值或删除其正式评分行。

已有依据：[前置审查第2节](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/review/2026-09-17/v18_preconditions/V18_Precondition_Audit_2026-09-17.md:15)，原始定义来自 [OSI_Methodology.pdf](/home/jacklo/XYY/INFORMS/informs-data-mining-2026/reference/OSI_Methodology.pdf)。诊断仍需核对本次实际读取的数据与标签，没有发现新问题前不推翻已验收口径。

### 3.2 horizon 与目标小时分开记录

小时0为 `2026-03-11 00:00`；预测行起点 t∈[72,215]，目标 s=t+h。**h=1 不代表距最后已知停电只有一小时**；真正的历史间隔是 s−71。

| h | 有效目标小时 s | 每县有效行 | 全239县有效行 |
|---|---|---:|---:|
| 1 | 73–215 | 143 | 34,177 |
| 6 | 78–215 | 138 | 32,982 |
| 24 | 96–215 | 120 | 28,680 |
| 48 | 120–215 | 96 | 22,944 |

每套缓存保留34,416行；无效尾部不能进入指标。有效标签有预测 NaN/Inf 必须报错，不能用 truth/prediction 同时 finite 的筛选静默删行。

当前主规则为 **1/6h 用 C3、24/48h 用 C1**。C3 对同县同目标小时的各 horizon 分量先截断到[0,1]、等权平均，再重组 OSI。目标小时对应候选为：

| s | 可用来源 horizon |
|---|---|
| 73–77 | 1 |
| 78–95 | 1、6 |
| 96–119 | 1、6、24 |
| 120–215 | 1、6、24、48 |

因此 N24/N48 的质量也可能影响当前短时 OSI。跨 horizon 比较必须另外报告共同目标区间 **120–215**，不能把完整评分窗口的难度差异解释为模型能力差异。

## 4. 诊断模块与固定方法

### A. 来源与当前组合复现（先通过，再做解释）

1. 校验输入 SHA-256、完成标记、identity、CV字节及行级模型来源；每个县只能属于一个外折。读取模型记录核对外层训练/内层选择未包含该外折县，继承既有严格验收边界，不重新拟合。
2. FIPS 规范为五位字符串；以 `(split_seed, fipsCode, hour_idx, horizon)` 唯一键对齐，并独立核对 `timestamp_et`、目标时间、fold；不依赖不同文件恰好行序一致。
3. 官方 N 标签与原始训练数据按同县 s=t+h 核对；复算完整窗口的 N/R 公式并单列分母变化、范围异常、端点。未来真值只进入此类标签审查和诊断表。
4. 验证当前 F2 的 N 四个 horizon 及模型来源与同 CV 原 strict 完全相同；P/D/R 从完整 F2 产物读取，不能拼入另一个 CV 的预测。
5. 独立重建 C1、C3 和 `v18_rule`，与当前 F2 保存结果及汇总核对。组件clip、OSI `[0,0.65]` clip和 `<0.001` 置零保持原规则，预测不额外 round。
6. 保存“当前组合复现通过”之后才产出诊断结论。该步骤无需运行旧脚本，也无需重新加载/训练全部树；若发现实际产物不一致，单独报告阻断原因。

### B. N 分量本身的误差形态

三个预测空间分别报告，不能统称为 raw：

- `model_raw`：树的未截断输出，来自原 strict `base_predictions_cv.parquet`。
- `component_clipped`：模型输出clip到[0,1]，对应进入 C1/C3 前的 N。
- `aligned`：C3 同目标小时等权平均后的 N；四个输出 horizon 都可报告，但24/48h主规则实际用前一层。

每个 split、horizon、空间报告 N 的 RMSE、MAE、SSE、bias（预测−真实）、真值/预测均值和分位数、预测方差、零值率、负值/超1截断率。按县/外折/目标时段补充覆盖数和误差；全体 pooled RMSE 是主要统计，不能用县RMSE均值替代。

预先冻结以下分组，空组仍列出 n=0，不事后合并：

- 真值 N：`N=0`、`0<N≤1e-4`、`1e-4<N≤1e-3`、`1e-3<N≤1e-2`、`N>1e-2`；报告各组行数、SSE份额、偏差和平均预测。按真值分层属于事后解释，不能作为部署门控。
- 固定目标时段：73–76、77–95、96–143、144–215，与已有 P/D 报告衔接；另外按 C3 候选数1/2/3/4分组。所有分组均与各 h 的有效窗口取交集。
- 可部署历史分层：P71=`0`、`(0,0.01)`、`[0.01,0.1)`、`[0.1,0.5)`、`[0.5,1]`；N71=`0`、`(0,1e-3]`、`>1e-3`。分层不改变预测或评分权重。
- 以 `N>1e-3` 固定定义“明显正值”做辅助混淆表：漏报、虚报、precision/recall；不把它宣称为官方事件阈值，不扫描阈值挑最优。

幅度和时间错位分开看：

1. 每县/每h/每空间，在相同有效目标范围记录真实峰值、预测峰值、最早峰值小时及差值、真实峰值小时的预测/真值比；真实最大N≤1e-3时标记低活动，不计算不稳定比值或解读峰值偏移。并列峰值数量另存。
2. 固定考察 ell∈{-3,-2,-1,0,1,2,3} 的 `pred_N(s+ell)` 对 `true_N(s)`，只在全部位移共同可用的小时集合比较RMSE、相关性，基准ell=0也用同一子集。该项只描述错时，**不挑最优ell改预测，不报告部署提分**。
3. 列出高估与低估对SSE的贡献；中心平滑本来就会展宽峰值，不把平滑真值的宽峰自动归因为模型反应慢。

### C. C3 的影响与来源 horizon

- 在同一 h 的同一行比较 `component_clipped` 与 `aligned` 的 N误差和最终 C1/C3 OSI误差；最终OSI变化同时包含其他分量变化，不能全部归给N。
- 在共同目标区间120–215构造四个来源N误差的4×4相关/交叉乘积表；每个 `(县,目标小时)` 只计一次。其余候选数区间分别汇总，不重复计为独立样本。
- 独立验证平均预测误差的平方等于各来源误差乘积的加权和，分开报告减少方差与共同偏差仍然存在的情况。
- 保留每行来源horizon、源起点、fold、模型ID、候选数和权重1/m。未出现的来源不填0参与平均。
- 对齐收益、峰值衰减和来源质量均作描述，不在本轮学权重、删除较差horizon或替换C3规则。

### D. N 对最终 OSI 的影响：精确分解与真实值替换

必须使用当前 F2 的 P/D/R 来计算，不能只复用原 strict 的 N 贡献排名。当前公式为：

```text
z = 0.40*P + 0.35*N + 0.25*D - 0.10*R
OSI_pred = H(z)
H：clip到[0,0.65]，随后小于0.001置零
```

**D1：误差分解。** 对 C1、C3 以及主规则各自使用的分量，定义 `e_j=w_j*(pred_j-true_j)`；R 的权重必须取 −0.10。设 `z_true=sum(w_j*true_j)`，`b=H(z_pred)-z_pred+z_true-OSI_official`，则逐行严格满足：

```text
OSI_error = e_P + e_N + e_D + e_R + b
```

报告各自平方和、两两交叉项及 b，重建总体SSE。另报告 `sum(e_N*OSI_error)` 作为有符号分摊值，允许为负；不能把 `0.35²*SSE_N` 或权重35%称为N占整体误差的比例。官方OSI与分量重组有精度差，后处理也会改变误差，不能删去b强行获得等式。

**D2：真实值替换（oracle，标签辅助诊断）。** 只在本目录的诊断数组中做，所有替换行明确标记 `uses_future_truth=true / deployable=false`：

| 固定场景 | 操作 | 回答的问题 |
|---|---|---|
| O_N_all | 四个来源horizon的N同时替换成对应真实N，重算C1/C3与主规则 | 其他分量不变时，N完全准确会怎样 |
| O_N_source_h | 每次只替换一个来源horizon的N，共1/6/24/48四种；其他来源不变 | 来源h→输出h的4×4影响矩阵，含C3传导 |
| O_N_partial | 所有来源N按 `N_new=(1-alpha)*N_hat+alpha*N_true` 插值，固定alpha=0.25、0.50；alpha=1由O_N_all提供 | 误差部分缩小时，整体方向是否一致 |
| O_P_all / O_D_all / O_R_all | 各自单独将一个分量的全部来源替换成真值 | 与N比较固定参照下的敏感性，辅助安排优先级 |
| O_all_components | 四个分量全部替换成真值 | 验证剩余差异来自官方精度和冻结后处理；不要求RMSE为0 |

`N_hat` 取进入组合的clipped组件值；其他分量、有效掩码和C3权重不变。每个场景报告完整四horizon的OSI RMSE、SSE、bias、相对B_current的ΔRMSE和逐行ΔSSE。另在B_strict上做O_N_all，展示P/D改进前后N的作用是否变化。

Oracle结果不是新模型OOF，不是可实现收益保证，**也不是任意N模型的严格性能上界**：原N错误可能抵消P/D/R错误，完全正确的N仍可能使OSI变差。各单分量oracle收益不可相加；插值alpha不能按成绩挑选，更不能写入推理规则。

**D3：合法的简单锚点。** 全时段 `N=0` 和 `N=N71` 两个固定预测，仅用作退化基准，按同样C3/后处理重算。它们不使用未来标签、不拟合、不调参数，但本轮也不自动替换当前N。若树连这些锚点都不能稳定超过，应先检查模型/目标/输入，而不是直接增加复杂度。

### E. 高误差县与合法信息支持

全239县均输出N误差、OSI误差、N分摊、O_N_all的ΔSSE和峰值诊断；保留原始总体结果，不能删除极端县获得主结论。

案例清单固定生成规则：在seed42上，取各h主规则实际使用的N空间中 `0.35²*SSE_N` 最大的前3县并集（FIPS破同分），再加入历史关注的 Forest、Morrow、Brown、Clay、Calhoun、Morgan。历史县身份从既有报告中按州名与FIPS精确解析，不能仅按县名匹配。保存清单后在另外两套CV上按同一清单对照；另外两套新出现的高误差县仅列探索表。

相似支持检查采用明确的小规模规则，避免看到结果后换匹配方法：

1. 候选仅为该split的训练县，**排除查询县整个外折**，不只排除自身。比较同一目标小时、同一来源horizon的合法特征行。
2. 固定匹配列为 `last_P_t / last_N_t / last_R_t / N_t_mean_72h / N_t_max_72h / gust_at_t{h}h / log_customers / pct_forest`。这些列须存在于已冻结schema；不含未来N、OSI、severity、残差和误差排名。
3. 按该外层训练县在该h/s的均值和总体标准差作标准化，零方差列不参与距离并记录；欧氏距离按 `(distance,FIPS)` 排序，保存前5个邻县及距离。缺列/有效行缺值报错，不临时换变量。
4. 先冻结全部邻县排名再连接未来结果。展示全部5个的N/OSI真值与误差；不得只保留低损失邻居。查询县/小时因事后误差被选择，所以这是探索性对照，不能称为盲测或因果识别。
5. 在对应外层训练样本中检查查询特征是否超出支持范围，并列出训练标签N的范围、分布与当前预测位置。标签只用于诊断训练支持，不能反写成特征。
6. 当前N未使用F2邻县新增列；可描述这些现有合法摘要与失败窗口的关系，但相关性或少数相似县不能证明加入它们必然提分。

支持表覆盖案例县全部有效小时，主报告聚焦误差最大的窗口。绘图只选每h排名第一县与Forest/Morrow的并集（至多6县），展示N真值/原始预测/C3、P/R、官方OSI/当前预测及允许天气；不足以覆盖的现象用表补充，不扩展为全县逐图人工检查。

### F. 三套划分复现与不确定性

- 全部总体、县/时段和oracle指标按split分开报告；方向一致性按同县同h对齐，禁止先混合预测。
- 对O_N_all、四个O_N_source_h、两个O_N_partial及两个合法锚点相对B_current的四horizon ΔRMSE，做县整条轨迹配对bootstrap：2,000次，seed=20260910。基准/场景使用同一次抽样，按抽样SSE/样本数计算pooled RMSE，不平均县RMSE。
- C3对N的aligned−clipped比较同样给出配对区间；常量/空组导致相关性未定义时保留NaN和原因，不能强置零。
- 给出95%百分位区间和逐split点估计。不把同事件重复CV当作独立三倍样本，bootstrap也未消除县间空间相关和反复开发选择偏差。
- 官方按四个horizon的RMSE名次平均排序，平局优先1h；本诊断给出四项指标向量，不自造四RMSE均值当官方总分，也不只凭短时收益忽略长时。

## 5. 诊断之后如何做决策

本轮只产出建议，以下均不是预先授权的训练任务：

| 证据组合 | 优先建议 | 不能直接得出的结论 |
|---|---|---|
| N在OSI中影响较大，训练支持充分，主要低估幅度；部分oracle改善方向稳定 | 先做原N-Huber与直接N-L2等受控损失/目标结构比较，再考虑更复杂结构 | 不能凭Huber名称断定损失函数是原因 |
| 高N集中在现有输入无法区分的县/窗口，允许邻域或天气描述有可复现线索 | 先做N的信息增量实验，并保留原输入对照 | 不能把相关性写成因果解释或保证收益 |
| 同目标来源分歧大，C3均值抹平峰值 | 另立来源模型/组合实验，使用严格内层选择 | 不能在本次OOF上学组合权重后仍称完全独立评分 |
| 完整及部分N oracle收益很小，P/D/R敏感性更突出 | 暂缓N，优先P跨horizon迁移或其他证据更充分的方向 | 不能仅凭N自己的RMSE较小就认定它不重要 |
| N分量更准确但OSI恶化，交叉项显示误差抵消 | 保留分量准确性与OSI双重评价，考虑后续组合/联合结构问题 | 不应据此声称“故意预测错误才正确” |
| 标签窗口、分母、时钟或来源身份异常 | 先复核具体数据/代码问题，单独提出修复计划 | 不在诊断中静默修标签、移动天气或删除异常县 |

最终为每个h给出“优先做 / 暂缓 / 证据不足”和主因，说明来源h与输出h的区别。若证据接近，优先成本低、可复用的方法；不要求为四个h分别发明模型，不以本轮oracle大小机械决定模型架构。

## 6. 文件管理与交付

**现在只创建本计划文件。** 以下均为确认后生成的预定布局，不表示脚本或结果已经存在：

```text
n_four_horizon_diagnosis/
  Diagnosis_Plan_2026-09-19.md
  README.md
  diagnostic_config.json
  run_diagnostics.py
  verify_diagnostics.py
  reference/
    approved_plan.md
    source_manifest.json
    initial_input_hashes.json
  runs/
    split42/
      manifest.json
      rows/
        component_diagnostics.parquet
        alignment_sources.parquet
        osi_error_decomposition.parquet
        oracle_predictions.parquet
        anchor_predictions.parquet
      tables/
        integrity_checks.json
        component_metrics.csv
        distribution_and_bins.csv
        peak_and_lag_diagnostics.csv
        alignment_effects.csv
        source_error_cross_products.csv
        osi_decomposition.csv
        oracle_metrics.csv
        source_to_output_effects.csv
        county_and_window_metrics.csv
        matched_support.csv
        bootstrap_intervals.csv
      figures/
      verification.json
    split20260917/                  # 同结构
    split20260918/                  # 同结构
  summary/
    four_horizon_summary.csv
    split_consistency.csv
    priority_recommendation.csv
    case_selection.csv
    figures/
  Results_<实际完成日期>.md
  final_integrity_check.json
  completion.json
```

逐行表至少保留split/模型seed、县码、fold、起点t、目标s、h、真实N、各预测空间N、scoreable、源模型ID与run identity；误差分解保留各加权误差与b；oracle表保留场景、alpha、被替换来源、基准/场景OSI、逐行ΔSSE及不可部署标记。不可只保存汇总或关注县。

来源manifest记录实际读取文件的路径、SHA-256、schema、输入身份、计划/代码/config哈希、包版本；大模型/旧parquet不复制到新目录。所有新写入路径必须解析后位于OUT，读写路径分开检查。

图为诊断报告的静态附件，统一放本目录下。若项目环境缺绘图库，使用可用的本地绘图运行时或明确报告限制；不为本轮静默安装依赖。历史目录、日志、缓存均只读，不调用带写入副作用的旧verifier。

## 7. 执行顺序及验收标准

用户确认后的顺序：

1. 将批准版本复制到本目录reference，实现纯读取分析和独立验证入口；冻结来源与本计划参数。
2. 三套CV均先做A模块预检和B_current复现；失败的来源不能继续出正式指标。
3. 同一配置完成B–F，不在seed42看完结果后改阈值/分组再称另两套为原方案确认。
4. 独立进程复算关键数值、输出汇总和报告，检查旧输入未变；最后写completion。

实现后的预定命令接口（**本轮不执行，脚本尚未创建**）：

```bash
cd /home/jacklo/XYY/INFORMS/informs-data-mining-2026
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_four_horizon_diagnosis/run_diagnostics.py --stage preflight --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_four_horizon_diagnosis/run_diagnostics.py --stage analyze --split-seeds 42 20260917 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/n_four_horizon_diagnosis/verify_diagnostics.py --split-seeds 42 20260917 20260918
```

验收必须满足：

- **来源正确**：三个已冻结CV与F2身份匹配；读取范围明确；跨horizon没有跨fold/跨split拼接；N模型来源与原strict一致。
- **行数与掩码正确**：239县、完整34,416个起点及四h有效计数全部符合第3节；不因预测失败、公式端点或筛选县而缩小正式评分集。
- **指标复现**：当前预测逐行、C1/C3/主规则、主要RMSE与保存F2一致，数值绝对误差≤1e-12；NaN位置完全一致。
- **公式审查边界**：完整窗口N/R复算沿用既有绝对容差1e-6；官方标签与缓存逐值对齐；官方OSI与分量重组沿用5.1e-5精度容差，禁止修改官方标签凑齐。
- **分解闭合**：逐行误差等式、SSE平方项/交叉项总和独立核对，float64采用 `atol=1e-12, rtol=1e-10`；不能只重复调用生成函数证明自身。
- **oracle隔离**：标签辅助值仅进入标记清楚的诊断产物；不进入特征、训练、推理或提交。alpha=0应精确复现基准；没有被替换的P/D/R/N来源、C3权重与掩码保持不变。
- **传播正确**：N来源h未覆盖的目标小时不受该source oracle影响；24/48h主规则仅响应自身N来源，1/6h按C3覆盖传导；all-source结果独立核对。
- **对照公平**：所有比较使用相同有效行；lag用共同子集；跨h用共同目标小时；匹配先排名后连结果并排除完整外折，保存足够数据重建排序。
- **文件完整性**：CSV读取使用 `float_precision='round_trip'`，FIPS显式字符串；输入前后哈希一致；全新输出限于OUT；不启动LightGBM/GAT/其他模型拟合。
- **报告边界**：分别写清已证实事实、诊断假设与后续待训练验证事项。即使N优先级很低或oracle恶化，也按预定表格完整汇报。

最终报告以一张四horizon决策表开头，列当前OSI RMSE、N raw/clipped/aligned误差、N oracle及两个合法锚点、三CV一致性和下一步建议；随后给出主要县/时段证据。没有新模型，因此不得使用“已完成N优化”或“测试集可获得X%收益”的措辞。
