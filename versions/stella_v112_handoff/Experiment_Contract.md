# v1.12 长期集成：严格迁移实验规格

状态：2026-09-19 交接提案。先实现和无训练预检；本文件不授予正式训练权限。建议协议名 `stella_v112_nested_v1`。正式运行前冻结本规格、实现代码及环境。

## 1. 问题和控制组

比较同一分折、相同行、相同 163 列下的长期预测。

| 候选 | 1h/6h | 24h/48h |
|---|---|---|
| L0 | 同分折当前 F2，逐行复制 | 严格 LightGBM 分量 C1 |
| L1 | 与 L0 完全相同 | 六路处理后的 OSI 等权平均，再做 OSI 后处理 |
| L2，主候选 | 与 L0 完全相同 | L0 预测大于内层正预测 q95 时用 L0；其余用 L1 |

六路固定顺序：`lgbm_direct, lgbm_component, xgboost_direct, catboost_direct, xgboost_component, catboost_component`。每路权重 1/6；分量一路指四个树模型合成后的一个 OSI 预测，不是把所有分量树当等权成员。

F2 的 P1h 初始状态 a/b 模型、D_G 和近邻输入已得到三分折验证，但其主 24h/48h 一直保持原严格 C1。本次长期学习器继续使用 163 列，不把 F2 的 183 列用于全部学习器。不要以当前 F2 的长期 C3 列替换 C1。

不增加候选/特征，不改损失，不搜权重、q、县名单或 horizon 路由；不重新分折；不训练最终全量模型或生成提交。保留六路单模型成绩作为诊断，不借此新增第七个自动选择策略。

## 2. 数据和评分不变量

- 三套冻结分折为 `cv/cv_assignments_balanced_v1_seed{42,20260917,20260918}.csv`；首轮只用 42。折编号为 0–4；模型随机种子始终 42。
- 239 个训练县，每县 origin hour 72–215，34,416 行。63 个测试县，9,072 行，本轮不推理测试集。
- 键为五位字符串 `fipsCode`＋`timestamp_et`，`hour_idx` 与键一致；分折在县级固定，禁止随机拆行。同一个县整条轨迹只能处于一个外折。
- 以 `2026-03-11 00:00:00` 为 hour 0，使用冻结时间戳语义，不另做 UTC 转换。目标为 `origin+h`，评分有效条件 `72 <= origin` 且 `origin+h <= 215`。每个 horizon 的有效训练行数分别为 34,177、32,982、28,680、22,944；24/48h 分别每县 120/96 行。
- 先按县作用域和确定性的时间条件选行，再取对应标签。有效行缺标签/缺预测应报错，不得用“实际有预测的交集”悄悄减少评分行。尾部保留 NaN，真实零值参与训练和评分。
- 输入冻结为 `versions/xyy/v1.5.6/` 的特征、列序、metadata、OSI 标签及严格基线目录的分量标签。不要回退到旧 `cache/` 或重新生成一套看似同名的特征。
- 已提供小时 0–71 的停电历史及比赛允许天气可用；预测期真实停电不可作输入。N/R 的平滑与官方已提供最后观测值保持现状；不在此轮重新解释 1h 延迟/滚动窗口，不重算带未来真实值的特征。
- FIPS、fold、真实 severity、未来目标、残差、其他模型 OOF 均不能擅自追加为特征。权威是冻结的 163 列名单和输入哈希。
- 六路、L0–L2 必须按键对齐后比较，不能依赖旧 CSV 的偶然行序。Parquet 为主产物；CSV 读数使用 `float_precision='round_trip'`。

## 3. 外折训练：把早停也包含在隔离中

设所有折为 F={0,1,2,3,4}。评估外折 q 时，允许监督集合 S=F\{q}。

对每个 family/target 的外层模型：

```text
for validation_fold v in sorted(S):
    probe.fit(训练折=S\{v}, early_stop折={v})
    保存 best_iteration，转换为正整数轮数
rounds = max(1, floor(mean(全部 |S| 个 best_rounds)))
model.fit(完整 S, 固定 rounds, 无 eval_set/早停)
predict(外折 q 的有效行)
```

特征预处理若涉及学习也只能在相应 fit 的训练集合拟合；本轮没有新增这类转换。数值类型转换是固定转换。外折 q 标签只能进入最终评分，不能进入选择轮数、fit 参数、重加权、阈值或学习器回退分支。

复用的 LightGBM 模型必须已经满足以上要求，且 CV、数据/特征、目标、参数、模型种子、作用域和回执完整匹配。不能拿旧 Stella 模型套一个新的 strict 文件名。

## 4. q95 额外需要一层内层预测

对外折 q 的 S，依次取 r∈S；令 T=S\{r}，长度为 3。先在 T 内做“三次两折训练/一折早停”，floor 平均轮数后在完整 T 重训，预测 r。对 P/N/D/R 分别执行，再按 C1 合成 r 的 anchor。拼接四个 r 得到只用于外折 q 阈值的 inner OOF。

例：外折 q=2、内层被预测折 r=0 时，T={1,3,4}。所有梯度、早停和重训只访问这些折的标签；q=2 与 r=0 的标签均不可进入生成 r 预测的模型。

```text
外折 q 的最终预测
├─ 六路外层模型：四折 S 内选轮 → 在 S 重训 → 预测 q
└─ theta[q,h]
   └─ 拼接四个 r 的 C1 anchor 预测
      └─ 每个 T=S\{r}：T 内选轮 → 在 T 重训 → 预测 r
```

`theta[q,h] = np.quantile(inner_anchor[inner_anchor > 0], 0.95, method='linear')`，仅纳入该 h 的有效行。有效 inner 预测必须全为有限值，不用过滤非有限预测掩盖错误；无正预测时报错。

阈值按行的正预测分位数计算，不先求各县分位数再平均。不得用训练内拟合值、q 的预测分布、q 的真实标签、全局旧 OOF 或全五折训练模型的预测来算 theta。

**即使其他四折的旧 OOF 没有使用各自行的标签，生成它们的模型仍可能用过 q 的标签。** 因此从全局 OOF 中删掉 q 行，不等于得到可用于 q 的 inner OOF。

因为没有学集成权重，新增三折内层预测只需要 LightGBM 分量 anchor，不必给另外五路都再做一轮内层模型。相同三折 T 的模型可复用，但应以排序后的完整 scope、target、split/data/code/config/env 身份索引，不能只用 `outer{q}` 命名。每个复用来源必须有依赖记录。

## 5. 预测算术：顺序不能互换

定义 `post(x)`：先截断 [0,0.65]，再将严格小于 0.001 的值置 0（等于 0.001 保留）。

1. 每个 direct 模型：`source_direct = post(raw_direct)`。
2. 每个 component family：分别把 raw P/N/D/R 截断到 [0,1]，计算 `max(0, 0.40P + 0.35N + 0.25D - 0.10R)`，再 `post`，得到一个 `source_component`。
3. `L0 = lgbm_component`，`L1 = post(mean(six processed sources))`。
4. `L2 = where(L0 > theta[q,h], L0, L1)`。等于 theta 时使用 L1。门控依据预测值，不是真实 top5% 标签。
5. 24/48h 不做 C3；1/6h 直接复制 F2 已冻结主预测。**禁止把新长期分量送回短期 C3 重算**，否则“短期不变”的实验定义被改变。

`protocol_examples.json` 给出截断、分量加权、q95、等于阈值、尾部 mask 和轮数取整案例。新实现应调用自己的生产函数复算这些案例；不能只自测复制在测试里的另一套公式。

## 6. 学习器规格

下面保留旧 Stella 的模型家族/结构/损失，统一迁移隔离与轮数选择。**不是所有模型都改成 L2**，也不复用旧 runner 的自动五折最终训练流程。

| 学习器 | 输入 | 损失/指标 | 最大探测轮数 | patience |
|---|---|---|---|---|
| LightGBM direct / components | 原163列缓存数值类型 | huber / rmse | 2000 | 100 |
| XGBoost direct | 原163列，固定转 float32 | reg:pseudohubererror，huber_slope=0.01 / rmse | 4000 | 100 |
| XGBoost components | 同上 | 同上 | 8000 | 100 |
| CatBoost direct / components | 原163列，固定转 float32；无 categorical 新编码 | Huber:delta=0.01 / RMSE | 4000 | 100 |

LightGBM：learning_rate=.05，num_leaves=31，max_depth=-1，min_child_samples=50，feature_fraction=.8，bagging_fraction=.8，bagging_freq=5，lambda_l1=.1，lambda_l2=1，num_threads=1，force_col_wise=True，deterministic=True；seed、feature_fraction_seed、bagging_seed、drop_seed、data_random_seed 均42。依据严格 `config.make_lgbm_params`。

XGBoost：learning_rate=.05，max_depth=6，min_child_weight=50，subsample=.8，colsample_bytree=.8，reg_alpha=.1，reg_lambda=1，tree_method=hist，random_state=42，n_jobs=1，verbosity=0；CPU。CatBoost：learning_rate=.05，depth=6，l2_leaf_reg=3，random_strength=1，rsm=.8，random_seed=42，thread_count=1，allow_writing_files=False，verbose=False；CPU。

未覆盖的库默认值必须在环境及有效参数中记录，首轮开始前固定版本；不要因机器上存在另一个默认版本而混跑。XGB/Cat 的早停 best_iteration 是从 0 开始的索引，转轮数要 +1；Cat 无 best iteration 时记录并使用实际 tree_count。LightGBM best_iteration 已是轮数。最终重训不传外部验证集，不保留早停开关，不用外折挑最佳树数。

迁移明确改变旧代码的 `n_jobs/thread_count=-1`、使用评分外折早停和“全部外折轮数 round 平均”的行为。线程改1是复现设置；选轮改为**当前允许作用域内部**的 floor 平均。记录实际保存轮数；LightGBM 允许 1≤实际树数≤requested（树无可分裂时可能提前停止），不伪造树数或强制补树。

## 7. 运行阶段和预算

A：阅读、对齐、实现、输入校验、无训练依赖 dry-run。输出 `Protocol_Alignment.md`、`resolved_config.json`、`dependency_plan.json`、预检报告和待执行命令，等待训练授权。

B：授权后仅 seed42 正式 CV；完成独立重载与评分验收后汇报，先不自动跑确认折。

C：负责人审阅 seed42 后，若确认固定候选值得继续，再运行20260917和20260918。不能看两套确认结果后重选参数并把它们继续称为独立确认。三套都是同一批县的重复分折，不是三份独立测试集。

seed42 预算（不含隔离诊断的小模型测试）：

| 项目 | 拟合数 | 新保存模型数 |
|---|---:|---:|
| 已有严格 LGB 的24/48h direct及四分量，5外折 | 0 | 0（引用50个） |
| XGB/Cat的20个标量目标×5外折×(4探测+1重训) | 500 | 100 |
| LGB anchor：10种三折scope×2horizon×4分量×(3探测+1重训) | 320 | 80 |
| 合计 | 820 | 180 |

阈值10个 JSON 不是新树模型。不保存探测模型也要保存各次fit记录；重训模型和回执必须保存。若不能复用50个 LGB 外层模型，需要额外250次拟合和50个模型，并作为新的环境运行记录，先报告原因；禁止静默切换或套用原身份。

## 8. 分析及决策

优先报告四个 horizon 的 pooled RMSE，不平均五个折 RMSE，也不擅自把四个 horizon 压成一个加权总分。1/6h 必须逐行不变。24/48h 比较 L1−L0、L2−L0、L2−L1，保留完整预测。

给出逐折/逐县/目标日指标、最大改善与恶化县、六路误差相关性、门控覆盖、q95及正预测数、真实 top5% 与预测尾部两种诊断（真实 top5% 仅作事后描述，不能进入门控）。与旧 v1.12 分数的差异只能描述，不归因为纯集成收益。

按县整条轨迹配对 bootstrap：2000次，seed20260910，固定县排序，候选与 L0 使用同一次抽样；报告 ΔRMSE 及95%百分位区间。复算程序必须冻结抽样实现。重复分折逐套报告，不把三份同县轨迹视为717个独立县。

首轮是否继续确认的建议：L2 在24/48h都有负的ΔRMSE点估计，并检查改善是否集中于单一县、单一折及是否引入严重退化；区间跨零应如实称证据有限，不能自动宣称无效或显著有效。无论结果如何先汇报，由负责人决定确认范围。最终采用看跨分折复现、尾部损益和竞赛优先级，不因模型数量多就自动晋级。
