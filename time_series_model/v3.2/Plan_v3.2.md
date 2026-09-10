# v3.2：统一目标小时的动态融合时序模型实施方案

日期：2026-09-10  
状态：方案已确定，尚未实现  
LightGBM 基线：v1.5.6 直接 OSI 模型  
验证划分：`balanced_v1`，按县五折

## 1. 本版结论先行

v3.2 将预测对象从 `(county, origin_hour, horizon)` 改为：

```text
(county, target_hour) -> 一个 OSI 预测
```

随后再把这个唯一值映射回提交表中的 t+1、t+6、t+24、t+48 四列。只要两个提交位置指向同一个县、同一个真实目标小时，它们在后处理前后都必须完全相等。

v3.2 的主模型不再自由预测 LightGBM 残差。主模型采用**小型因果 GRU 动态融合器**：它读取前 72 小时观测、已知未来天气、县级静态属性，以及指向同一目标小时的最多四个 v1.5.6 LightGBM 原始预测；GRU 只输出这四个预测的融合权重。权重必须非负、和为 1，最终结果始终位于已有 LightGBM 预测的范围内。

选择这一结构的原因是：

1. v3.1 中真正稳定的收益来自同一目标小时的等权平均；
2. v3.1 GRU 残差在中长 horizon 上放大了极端县误差；
3. v3.1 已证明未来气象在 t+24h 存在增量信息；
4. 训练集只有 239 个真正独立的县级序列，首先应限制神经网络的自由度；
5. 动态融合器可以从等权平均出发学习“何时更信任哪个 horizon”，而不会生成脱离四个基线预测的新极端值。

v3.2 暂不引入 P/N/D/R 分量 LightGBM 或分量目标。前 72 小时真实 P/N/D/R 仍作为合法的历史观测输入保留；这里的“不考虑分量”专指不接入分量预测模型。

## 2. v3.1 给 v3.2 的约束

v3.1 M4 的严格 OOF 结果表明：

- 等权目标时刻平均 B1 在 t+1h、t+6h、t+24h 的 RMSE 分别改善 0.97%、2.13%、1.54%；
- B1 在 t+48h 的 RMSE 退化 0.94%；
- 完整 GRU 残差 B3 在 t+24h、t+48h 的 RMSE 分别退化 2.23%、5.44%；
- B3 优于非时序 MLP，且在 t+24h 优于不读未来天气的 GRU，说明时序顺序和未来天气并非完全无效；
- B3 几乎没有降低同一目标小时的跨 horizon 离散程度。

因此，v3.2 需要同时满足两点：

1. 从结构上消除同一目标小时的多套预测；
2. 让时序模型在等权平均之上做受约束的选择，而不是再次自由修正 OSI。

这里有一个直接后果：v3.1 报告中“t+1/t+6/t+24 用 B1、t+48 用 B0”的候选策略不能用于 v3.2，因为同一个目标小时会再次因 horizon 不同而得到两个值。v3.2 必须在统一预测下重新衡量四个 horizon 的总体取舍。

## 3. 时间索引和唯一目标

统一定义：

| 符号 | 含义 |
|---|---|
| `c` | 县 |
| `o=71` | 最后一个有真实停电观测的小时 |
| `t` | 提交表中的预测原点，范围 72～215 |
| `h` | horizon，取 1、6、24、48 |
| `s=t+h` | 真实目标小时 |
| `y(c,s)` | 县 `c` 在目标小时 `s` 的唯一真实 OSI |
| `b_h(c,s)` | horizon 为 `h`、指向目标小时 `s` 的 v1.5.6 原始预测 |
| `p(c,s)` | v3.2 对目标小时 `s` 的唯一预测 |

原有预测可以改写为：

```text
b_h(c,s) = b(c, t=s-h, h)
```

只有 `72 <= s-h <= 215` 且 `s <= 215` 时，该候选预测可用。统一目标轴为 hour 73～215，共 143 个目标小时；hour 72 只用于未来天气编码器预热，没有监督目标。

| 目标小时 `s` | 可用 LightGBM 候选 | 候选数 |
|---|---|---:|
| 73～77 | t+1 | 1 |
| 78～95 | t+1、t+6 | 2 |
| 96～119 | t+1、t+6、t+24 | 3 |
| 120～215 | t+1、t+6、t+24、t+48 | 4 |

每县原提交目标有 `143+138+120+96=497` 个有效单元格，去重后只有 143 个目标小时。训练集由 118,783 个有效提交单元格变为 34,177 个唯一 `(county,target_hour)` 目标；测试集最终只需先生成 `63×143=9,009` 个唯一预测，再映射为 31,311 个有效提交单元格。

映射公式固定为：

```text
submission_prediction(c,t,h) = p(c,t+h)
```

不能在四个 horizon 映射后分别做校准或使用不同阈值，否则会破坏唯一性。正确顺序是先得到唯一的 `p_raw(c,s)`，统一后处理一次，再广播到所有对应提交位置。

## 4. 数据输入与缓存契约

### 4.1 冻结输入

继续使用 v1.5.6：

```text
cache/features_train_v1.5.6.parquet
cache/features_test_v1.5.6.parquet
cache/targets_train_v1.5.6.parquet
cache/meta_train_v1.5.6.parquet
cache/meta_test_v1.5.6.parquet
cache/feature_names_v1.5.6.json
cv/cv_assignments_balanced_v1_seed42.csv
data/DM_Train.csv
data/DM_Test.csv
```

v3.2 不改动 v1.5.6 的特征工程和参数，也不接入 v1.5.7 或 v2.1 分量预测。

v3.1 的县级序列缓存可以只读复用，但必须先核对源文件和缓存 SHA-256、字段顺序、数组形状及时间网格。v3.2 的所有新缓存、模型和报告只能写入 `time_series_model/v3.2/`，不能覆盖 v3.1 产物。

### 4.2 唯一标签构造

对每个 `(c,s)` 收集所有有效 horizon 对应的目标值，并执行：

1. 所有非空值必须在固定浮点容差内相等；
2. 相同目标不能出现一部分缺失、一部分有效的异常结构；
3. 通过检查后只保存一个 `y(c,s)`；
4. 同时保存它映射回原始行和 horizon 的索引。

若同一目标小时的官方目标不一致，构建过程直接失败并输出县、目标小时、来源 horizon 和原值，不能取平均掩盖问题。

### 4.3 对齐后的 LightGBM 候选

每个目标小时保存：

```text
candidate_raw[4]       # t+1/t+6/t+24/t+48 原始预测
candidate_mask[4]      # 该候选是否存在
candidate_equal_mean   # 只对有效候选求原始值均值
candidate_min
candidate_max
candidate_std
candidate_count
```

`candidate_raw` 必须是裁剪和置零之前的预测。缺失候选可在模型输入中填 0，但必须同时提供 mask；计算均值、标准差和最终融合时不得把填充值当成候选。

正式训练中的候选必须与折分相匹配：

- 融合器训练县使用固定轮数、按县 cross-fit 产生的 LightGBM OOF 原始预测；
- 外层验证县使用只在外层训练县上拟合的四个 LightGBM；
- 最终测试使用在全部训练县上拟合的四个 LightGBM；
- 任何县自己的标签都不能参与生成该县供融合器训练或验证的基线预测。

### 4.4 时序和静态输入

沿用 v3.1 已核验的输入：

- hour 0～71：P/N/D/R 和同期天气；
- hour 72～215：已知未来天气；
- 静态连续变量：`pop_density`、`n_utilities`、`pct_forest`、`pct_wetland`、`tree_canopy_pct`、`tree_canopy_std`；
- 静态类别变量：`rural_urban_code`；
- 最后观测状态：`last_P_t`、`last_N_t`、`last_D_t`、`last_R_t`、`last_osi`。

不输入 `fipsCode`、州或县 embedding。第一版不输入绝对日期；相对序列位置由 GRU 自身和候选可用 mask 表达，减少模型记忆单次训练风暴日程的机会。

所有归一化、填补和类别字典只在相应训练县上拟合。原始候选值同时保留一份不缩放副本用于最终加权求和，缩放值只作为门控网络输入。

## 5. 主模型：统一轨迹 GRU 动态融合器

### 5.1 总体结构

```text
hour 0~71: P/N/D/R + weather
                 |
                 v
          Observed GRU
                 |
                 v
        county initial state
                 |
hour 72~215: future weather
              + aligned LGBM candidates/masks/statistics
              + county static projection
                 |
                 v
        causal Future GRU
                 |
                 v
     4 candidate logits at target hour s
                 |
          masked softmax
                 |
                 v
        nonnegative weights, sum=1
                 |
                 v
  one weighted prediction p_raw(c,s)
```

未来天气编码器保持单向因果结构。预测 `s` 时只使用 hour 72～`s` 的天气和候选轨迹。虽然比赛中 `s` 之后的天气也已知，第一版不使用双向编码，避免扩大参数量，也避免模型通过整场风暴形状过度记忆训练事件。若 v3.2 主模型通过严格验证，再单独评估非因果已知天气编码。

### 5.2 受约束输出

在目标小时 `s`，网络输出四个 logit。不可用候选在 softmax 前被屏蔽：

```text
w_dynamic(c,s) = masked_softmax(logits(c,s))
w_equal(c,s)   = candidate_mask / candidate_count
```

增加一个安全收缩系数：

```text
w_final = (1-alpha) * w_equal + alpha * w_dynamic
p_raw   = sum_h(w_final_h * b_h_raw)
```

其中 `alpha` 只允许从以下集合选择：

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

`alpha=0` 完整退回 v3.1 的等权目标时刻平均；任意 alpha 下权重仍非负且和为 1。第一版只选择一个全局 alpha，不按 horizon 选择，也不按目标阶段分别选择，避免重新引入多套目标定义和过多调参。

门控输出层的权重和偏置初始化为 0，因此训练开始时 `w_dynamic=w_equal`。hour 73～77 只有一个候选，模型输出必然等于 t+1 LightGBM；这一段不会被神经网络任意修改。

第一版不加入额外残差头、不允许负权重、不允许权重和偏离 1。只有受约束融合在严格 OOF 中超过等权平均后，才考虑一个有明确幅度上限的校准残差对照。

### 5.3 初始规模

| 模块 | 初始设置 |
|---|---|
| Observed GRU | 1 层，hidden size 32，单向 |
| Future GRU | 1 层，hidden size 32，单向 |
| 静态投影 | Linear → 16 维 → SiLU |
| 门控头 | Linear → 32 → SiLU → Dropout → 4 logits |
| dropout | 0.10 |
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| county batch size | 16 |
| max epochs | 50 |
| early stopping patience | 10 |
| gradient clipping | 1.0 |
| seed | 42 |

第一轮不同时搜索 hidden size、层数和 dropout。模型若没有超过非时序门控，不通过扩大网络寻找偶然收益。

## 6. 损失、选择指标与后处理

### 6.1 预测损失

同一个 `p(c,s)` 会映射到多个 horizon。训练损失应反映原提交结构，而不能简单把 143 个目标小时无权平均。默认使用 horizon 等权的 MSE：

```text
L = (1/4) * sum_h mean_{c, s in S_h}[(p(c,s) - y(c,s))^2]
```

其中：

```text
S_1  = 73~215
S_6  = 78~215
S_24 = 96~215
S_48 = 120~215
```

这等价于在唯一目标轴上给后期目标更高权重，因为它们会出现在更多 horizon 的评分中，同时保证四个 horizon 在损失中等权。训练使用原始连续预测，不在损失内执行 `<0.001` 置零。

### 6.2 模型选择指标

内层选择 epoch 和 alpha 时，使用映射回四个 horizon、完成固定后处理后的：

```text
primary_score = mean(RMSE_t1, RMSE_t6, RMSE_t24, RMSE_t48)
```

同时保存四个 RMSE、四个 MAE、唯一 143 小时轨迹 RMSE，以及原提交有效单元格的 pooled RMSE。官方评分尚未明确前，不把单一 pooled 指标当作唯一结论。

### 6.3 后处理

沿用 v1.5.6/v3.1 的固定规则：

```text
clip to [0, 0.65]
values < 0.001 -> 0
```

报告中必须同时给出 raw、clipped、postprocessed 三套结果。阈值不能在外层验证折上重选。

## 7. 必做对照与诊断上界

所有正式对照使用完全相同的外层县折和 LightGBM 候选：

| 编号 | 模型 | 是否满足唯一目标 | 作用 |
|---|---|---:|---|
| B0 | 原始 v1.5.6 四 horizon LightGBM | 否 | 保留项目基线，量化统一预测的净变化 |
| B1 | 对齐后等权平均 | 是 | v3.2 的直接结构基线 |
| B2 | 每种候选可用组合下的静态凸权重 | 是 | 检查固定权重能否解决 B1 的 t+48 问题 |
| B3 | 逐目标小时 MLP 动态凸融合 | 是 | 检查收益是否只来自当前特征和非线性门控 |
| B4 | GRU 动态凸融合，不读未来天气 | 是 | 未来天气消融 |
| B5 | 完整 GRU 动态凸融合 | 是 | v3.2 主模型 |

B2 只能根据训练县学习非负、和为 1 的权重，并使用强收缩回等权。B3 使用与 B5 相同的候选、mask、静态属性、最后状态和目标时刻天气，但不跨小时传递状态。B4 保留观测历史和候选预测序列，只把 hour 72～215 的天气通道置零。由此：

- B5 对 B3 衡量逐小时时序状态的增量；
- B5 对 B4 衡量未来天气的增量；
- B5 对 B1 衡量整个时序动态融合方案的净收益。

另计算一个只用于诊断、禁止生成测试预测的 oracle 凸包下界：对每个训练目标，把真实值投影到该时点候选预测的 `[min,max]` 区间。它回答“仅靠凸融合最多还有多少理论空间”。oracle 使用真实标签，不能参与模型选择、不能写成可提交模型。

## 8. 严格验证协议

外层继续使用 `balanced_v1` 五折，每个县的完整轨迹只能出现在一个外层验证折。

对每个外层折执行：

1. 仅在外层训练县中完成 v1.5.6 LightGBM 的内层早停；
2. 每个 horizon 取内层最佳树数中位数作为固定轮数；
3. 用固定轮数在外层训练县内按县 cross-fit，生成融合器训练所需的四个 OOF 候选；
4. 用全部外层训练县拟合四个固定轮数 LightGBM，生成外层验证县候选；
5. 在外层训练县内部做四折融合器验证，汇总四个内层验证折后选择统一 epoch 和 alpha；
6. 使用全部外层训练县的 OOF 候选、折内预处理和已选 epoch 训练最终融合器；
7. 对外层验证县产生 143 个唯一预测，再映射回四个 horizon；
8. 外层验证标签只用于本折最终评分，不参与任何参数、轮数、epoch、alpha 或后处理选择。

相较 v3.1 M4 只用一个固定内层折选择神经网络 epoch，v3.2 使用四个内层折的汇总结果，降低选择结果对单组县的依赖。所有批次、折分和 bootstrap 的最小单位仍为县。

可以复用 v3.1 M4 的 LightGBM 模块和已有产物，但只有在以下内容均可审计时才允许直接复用：外层训练县、内层训练县、固定轮数、raw 预测数组、行索引和输入哈希。缺少任一项就重新生成该层候选，不能根据文件名推断其无泄漏性。

## 9. 评价内容

### 9.1 正式指标

每个模型至少报告：

- t+1、t+6、t+24、t+48 的 pooled OOF RMSE 和 MAE；
- 四 horizon 平均 RMSE、原提交有效单元格 pooled RMSE；
- 143 小时唯一目标轨迹的 RMSE 和 MAE；
- 五个外层折的各项指标及改善折数；
- 每县 RMSE、MAE、SSE 和最差县贡献；
- 四个候选可用阶段的指标；
- 风暴阶段和目标小时指标；
- 高 OSI 区间、峰值附近与接近零区间的误差；
- 预测均值、最大值、零比例和误差一小时自相关。

对 B1～B5，目标时刻一致性必须在 raw、clipped、postprocessed 三个阶段均严格为 0；这里是程序契约，不是统计指标。

### 9.2 权重诊断

保存并汇总：

- 各候选 horizon 的平均权重和分位数；
- 按候选数量、外层折、风暴阶段、天气强度的权重；
- alpha 的内层选择结果；
- 权重熵以及接近单一候选的比例；
- 极端县上动态权重相对等权的变化。

若权重几乎固定，说明 B2 足够；若 B5 只靠长期压低某个 horizon 权重获益，也应优先使用更简单的静态融合。

### 9.3 不确定性

使用至少 2,000 次按县整条轨迹配对重采样的 bootstrap，比较：

```text
B5 vs B1   # 时序动态融合的净收益
B5 vs B2   # 动态权重相对静态权重
B5 vs B3   # 跨小时状态的增量
B5 vs B4   # 未来天气的增量
B1 vs B0   # 强制唯一预测本身的代价与收益
```

不能逐行或逐目标小时 bootstrap，因为这些目标共享同一县和同一场风暴。

## 10. 晋级与停止条件

B5 只有同时满足以下条件，才进入最终训练：

1. 严格外层 OOF 的四 horizon 平均 RMSE 优于 B1；
2. B5 相对 B1 的改善在至少 4/5 个外层折方向一致；
3. t+48h RMSE 优于 B1，并尽量恢复到 B0 水平，不能继续扩大 B1 的长时段退化；
4. 任一 horizon 相对 B1 的 RMSE 退化不超过 0.5%；
5. 收益不由一两个极端县单独贡献；
6. B5 至少稳定优于 B3 或 B4 之一，才能把收益归因于时序顺序或未来天气；
7. 模型重载后的唯一轨迹、融合权重和提交映射逐值一致。

若 B5 不优于 B2/B3，则停止扩大 GRU，保留表现最好的简单统一融合器。若所有学习式融合都不优于 B1，则 v3.2 的结论是“等权平均已经吃掉当前候选的大部分可用收益”，不继续增加自由残差头。

由于训练数据只有一场事件、县级交叉验证只能检验空间泛化，无法完整检验跨风暴泛化。任何依赖特定目标时段的权重模式都必须在报告中明确标注，不能把县级 OOF 的稳定性直接等同于新事件稳定性。

## 11. 里程碑

### M1：唯一目标轴和映射闭环

- 构建 hour 73～215 的唯一标签；
- 构建 `(origin,horizon) <-> target_hour` 双向索引；
- 验证 34,177 个训练唯一目标和 9,009 个测试唯一位置；
- 验证每县 497 个有效提交单元格均可无遗漏映射；
- 保存字段、形状、有效数和哈希清单。

**执行状态（2026-09-10）：已完成。** 唯一目标数、候选阶段、标签一致性和 497 单元格双向映射均通过检查。

### M2：LightGBM 候选与确定性对照

- 生成或审计严格折内的四路 raw LightGBM 候选；
- 实现 B0、B1、B2 和 oracle 诊断；
- 复现 v3.1 B0/B1 指标，差异需低于固定容差；
- 验证 B1/B2 的目标时刻离散度严格为 0。

**执行状态（2026-09-10）：已完成。** 已实现候选对齐、等权平均、静态凸融合和 oracle 诊断，并复现 v3.1 B0/B1。

### M3：最小模型闭环

- 实现 B3、B4、B5；
- 完成一个小规模 county batch 的前向、反向、保存和重载；
- 检查 masked softmax、权重和、不可用候选权重和初始等权行为；
- 生成诊断级 OOF 和测试结构，但标记为 `diagnostic_only`。

**执行状态（2026-09-10）：已完成。** 20 个检查点、诊断 OOF、测试结构和提交已独立重载复核；结果不作为正式收益。

### M4：严格嵌套验证

- 完成五个外层县折和四折内层模型选择；
- 生成 B0～B5 严格 OOF；
- 完成县级 bootstrap、误差结构和权重诊断；
- 独立重载并复算所有核心产物；
- 根据第 10 节规则决定是否晋级。

**执行状态（2026-09-10）：已完成。** B5 没有超过 B1/B2，也未显示时序或未来天气净收益，未通过晋级条件。

### M5：最终训练与提交

只有 M4 通过后执行：

- 冻结模型、epoch、alpha、后处理和随机种子策略；
- 在全部训练县上生成合法 cross-fit 候选并训练统一融合器；
- 在全部训练县上训练四个最终 v1.5.6 LightGBM；
- 对测试县生成 9,009 个唯一目标预测；
- 映射并复核提交的 31,311 个有效单元格；
- 保存提交、检查点、运行元数据和所有输入输出哈希；
- 将正式结果追加到项目 `Results.md`。

**执行状态：不执行。** M4 未通过预设条件，没有生成正式 v3.2 提交，也未修改项目 `Results.md`。

## 12. 计划目录和接口

建议目录：

```text
time_series_model/
└── v3.2/
    ├── Plan_v3.2.md
    ├── README.md
    ├── config.py
    ├── build_target_timeline.py
    ├── dataset.py
    ├── model.py
    ├── baseline_crossfit.py
    ├── train_diagnostic.py
    ├── train_nested_cv.py
    ├── train_final.py
    ├── evaluate.py
    ├── verify_artifacts.py
    ├── artifacts/
    │   ├── target_timeline_manifest.json
    │   ├── target_alignment_v3.2.npz
    │   ├── m4/
    │   │   ├── nested_oof_predictions.npz
    │   │   ├── nested_oof_predictions.parquet
    │   │   ├── fusion_weights.parquet
    │   │   ├── summary_metrics.csv
    │   │   ├── fold_metrics.csv
    │   │   ├── county_metrics.csv
    │   │   ├── stage_metrics.csv
    │   │   ├── paired_bootstrap.csv
    │   │   ├── run_metadata.json
    │   │   ├── verification.json
    │   │   └── checkpoints/
    │   └── final/
    └── submission_v3.2_balanced_v1.csv
```

建议命令接口：

```powershell
# M1：唯一目标轴与映射
python time_series_model/v3.2/build_target_timeline.py --verify

# M3：快速工程闭环，仅作诊断
python time_series_model/v3.2/train_diagnostic.py

# M4：严格嵌套按县验证
python time_series_model/v3.2/train_nested_cv.py

# 独立重载和复算
python time_series_model/v3.2/verify_artifacts.py --milestone m4

# M5：仅在 M4 通过后运行
python time_series_model/v3.2/train_final.py
```

所有入口必须拒绝：输入哈希变化、县内时间不连续、唯一标签冲突、候选对齐错位、训练验证县重叠、不可用候选得到非零权重、权重和偏离 1、同一目标小时映射后预测不一致、提交行顺序或 NaN 掩码变化。

## 13. v3.2 最终要回答的问题

v3.2 不以“使用了 GRU”作为成功标准，而要回答以下四个问题：

1. 强制同一目标小时只有一个预测后，整体误差结构是否更稳定？
2. 学习非等权融合能否保留 B1 的短中期收益，同时修复 t+48h？
3. 动态时序权重是否稳定优于静态权重和逐时点 MLP？
4. 未来天气是否能在统一目标结构下转化为相对等权平均的实际净收益？

只有第 3 或第 4 个问题得到严格 OOF 支持，才说明 v3.2 真正获得了时序模型带来的收益。
