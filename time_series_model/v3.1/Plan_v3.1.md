# v3.1：v1.5.6 LightGBM + GRU 残差模型实施方案

## 1. 版本目标

v3.1 的目标是完成第一版时序模型最小闭环：以 v1.5.6 直接预测 OSI 的 LightGBM 为基线，用小型 GRU 学习其折外残差，并生成可复现的 OOF 结果、测试预测和提交文件。

最终预测定义为：

```text
baseline_raw = v1.5.6 LightGBM 原始预测
delta        = GRU 预测的正负残差
combined_raw = baseline_raw + alpha_h * delta
prediction   = postprocess(combined_raw)
```

其中 `h` 为 1、6、24、48 小时，`alpha_h` 是按 horizon 独立选择的残差收缩系数。

v3.1 首先解决以下问题：

1. 前 72 小时停电轨迹能否解释 LightGBM 未捕捉的状态变化；
2. 从最后停电观测时刻到目标时刻的逐小时天气顺序能否解释残差；
3. GRU 的改进是否超过非时序残差模型和简单的目标时刻对齐平均；
4. 改进能否在按县严格留出的验证中成立。

## 2. 本版范围

### 2.1 本版包含

- 使用冻结的 v1.5.6 特征、目标、元信息和 LightGBM 参数；
- 直接 OSI LightGBM，不在本版改为 P/N/D/R 分量模型；
- 72 小时观测历史编码器；
- 144 小时已知未来天气编码器；
- 共享时序表示和四个 horizon 残差头；
- 按县嵌套交叉验证；
- 开发闭环和正式评估闭环；
- OOF、测试预测、指标、模型和运行元数据的完整保存；
- 与 v1.5.6、目标时刻对齐平均和非时序 MLP 的比较。

### 2.2 本版不包含

- 不加入 v1.5.7 的 48 列 F1 特征；
- 不新增 F2/F3 手工天气特征；
- 不训练 P/N/D/R 的 16 个分量残差模型；
- 不使用 LSTM、TCN、Transformer 或 TFT；
- 不使用真实测试期停电、真实测试期 OSI 或真实残差；
- 不把模型预测写回原始停电字段进行自回归；
- 不使用 `fipsCode`、`stateAbbr`、县 ID embedding 或其他标识符作为模型输入。

这些限制用于保证第一版变化来源单一：与 v1.5.6 相比，只增加合法的时序表示和残差修正。

## 3. 时间和样本定义

统一定义：

| 符号 | 含义 |
|---|---|
| `c` | 县 |
| `o=71` | 最后一个有真实停电观测的小时 |
| `t` | 提交表中的预测原点，范围为 72～215 |
| `h` | horizon，取 1、6、24、48 |
| `s=t+h` | 实际目标小时 |
| `y(c,t,h)` | 县 `c` 在目标小时 `s` 的真实 OSI |
| `b(c,t,h)` | v1.5.6 LightGBM 对该目标的原始预测 |

只有 `s <= 215` 的目标有效。所有损失、指标和提交都使用与官方目标一致的有效掩码。

训练单位按县组织。每个县包含：

- 一条长度 72 的停电和天气观测序列；
- 一条长度 144 的未来天气序列；
- 最多 `144 × 4` 个 `(t,h)` 位置，其中越过 hour 215 的位置被掩码；
- 一组县级静态特征。

虽然表格形式有 34,416 行，模型和验证都不能把这些高度重叠的行当作独立样本。batch、数据划分和 bootstrap 的最小单位均为县。

## 4. 数据输入

### 4.1 冻结基线输入

使用以下 v1.5.6 文件：

```text
cache/features_train_v1.5.6.parquet
cache/features_test_v1.5.6.parquet
cache/targets_train_v1.5.6.parquet
cache/meta_train_v1.5.6.parquet
cache/meta_test_v1.5.6.parquet
cache/feature_names_v1.5.6.json
cv/cv_assignments_balanced_v1_seed42.csv
```

每次运行记录上述输入文件的 SHA-256。v3.1 不覆盖任何 v1.5.6 文件。

### 4.2 原始逐小时序列来源

从以下文件按 `fipsCode` 和 `timestamp_et` 排序重建序列：

```text
data/DM_Train.csv
data/DM_Test.csv
```

`fipsCode` 只用于连接和分组，不进入模型。

### 4.3 观测历史序列

观测历史范围固定为 hour 0～71。

第一版停电通道：

```text
P_t, N_t, D_t, R_t
```

第一版同期天气通道：

```text
gust
wind_speed
wind_direction_sin
wind_direction_cos
t2m
tp
sp
r2
csnow
soil_moist
```

若原始字段命名与上述语义名不同，实施时建立显式映射，并在元数据中保存实际列名。OSI 不作为第五个时序通道，因为它是 P/N/D/R 的确定性线性组合；`last_osi` 可以作为输出头的跳跃输入。

### 4.4 未来天气序列

未来天气范围固定为 hour 72～215，使用与观测期相同的天气通道，不包含停电字段。

未来天气编码器为单向模型。预测目标小时 `s` 时，只使用 hour 72～`s` 的编码状态，不使用 `s` 之后的天气。完整未来天气虽然在比赛中可知，但 v3.1 先采用目标截止的因果表示，以减少事件时间代理和不必要的模型自由度。

### 4.5 县级静态特征

第一版从 v1.5.6 选择少量静态列：

```text
pop_density
pct_forest
pct_wetland
tree_canopy_pct
tree_canopy_std
n_utilities
rural_urban_code
```

`log_customers` 和 `forest_x_customers` 在 v1.5.6 中随预测小时轻微变化，不属于严格县级静态量，因此不进入 GRU 静态分支；它们仍通过 v1.5.6 LightGBM 基线预测影响最终结果。

`rural_urban_code` 使用 one-hot 或明确的有序编码。若某列在 v1.5.6 中不存在，构建阶段直接失败并报告，不静默替换。

### 4.6 输出头的额外输入

每个 `(c,t,h)` 位置向输出头提供：

```text
历史编码状态 z_c
目标小时天气状态 u_c,s
LightGBM 原始预测 b(c,t,h)
last_P_t, last_N_t, last_D_t, last_R_t, last_osi
s_minus_o = s - 71
hours_since_obs = t - 71
horizon 编码
县级静态特征
```

不把全部 163 列 v1.5.6 特征再次输入神经网络。LightGBM 已经吸收这些表格特征；GRU 只保留必要的基线预测、状态跳跃输入、静态属性和时序输入，以形成互补表示。

## 5. 归一化和缺失处理

所有变换参数只在当前训练县上拟合，再应用于验证县和测试县。

| 类型 | 处理方式 |
|---|---|
| `gust/wind/t2m/sp/r2/soil_moist` | 按训练县拟合全局均值和标准差 |
| `tp/csnow` | `log1p` 后标准化 |
| `P/N/D/R` | 保留绝对量级后做全局标准化 |
| 静态连续变量 | 标准化或稳健标准化，方式在配置中固定 |
| 类别变量 | one-hot，未知类别保留单独位置 |
| 缺失值 | 使用训练集统计量填充，并增加缺失标志 |

禁止只做县内 z-score，因为这会删除绝对阵风强度和绝对停电严重程度。所有 scaler、类别字典和通道顺序随模型保存。

## 6. 模型结构

### 6.1 总体结构

```text
hour 0~71: P/N/D/R + weather
              |
              v
       Observed GRU Encoder
              |
              v
        county state z_c
              |
              +-----------------------+
                                      |
hour 72~215: known future weather      |
              |                       |
              v                       |
       Future Weather GRU <------------+
              |
              v
   state u_c,s at each target hour s
              |
              + static + last state + baseline_raw + time
              |
              v
       1h / 6h / 24h / 48h residual heads
              |
              v
     baseline_raw + alpha_h * delta
              |
              v
          clip / zero threshold
```

### 6.2 初始参数

| 模块 | v3.1 初始设置 |
|---|---|
| Observed GRU | 1 层，hidden size 32，单向 |
| Future Weather GRU | 1 层，hidden size 32，单向 |
| 静态特征投影 | Linear → 16 维 → 激活 |
| Horizon 表示 | 4 类 one-hot 或 4 维 embedding |
| 残差头 | 小型两层 MLP，四个 horizon 独立 |
| 激活 | SiLU 或 ReLU，配置中固定一种 |
| 显式 dropout | 0.1 |
| 输出层 | 线性层，允许正负值，权重和偏置零初始化 |

单层 GRU 的框架内置层间 dropout 不生效，因此 dropout 必须显式放在输入投影、GRU 输出或残差头中。

第一版禁止使用两层 hidden 128。只有 hidden 32 的闭环和验证通过后，才将 hidden 64、dropout 0.2 作为有限对照。

### 6.3 输出公式

对每个有效位置：

```text
delta_std = head_h(...)
delta     = delta_std * residual_scale_h
combined  = baseline_raw + alpha_h * delta
```

`residual_scale_h` 只根据当前训练县的 LightGBM OOF 残差计算。`alpha_h` 第一轮固定为 1.0；正式评估可在内层验证中从以下集合选择：

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

`alpha_h=0` 等价于退回 v1.5.6，是必要的安全边界。

## 7. 残差定义和损失

### 7.1 残差标签

```text
residual(c,t,h) = y(c,t,h) - baseline_oof_raw(c,t,h)
```

必须使用后处理前、且对应县未参与基线训练的 LightGBM 折外预测。真实残差只作为训练标签，不作为任何时刻的 GRU 输入。

### 7.2 训练目标

默认对标准化残差使用 Huber loss。四个 horizon 分别在有效掩码内取均值，再对四个 horizon 等权平均：

```text
loss = mean_h(Huber(residual_std_h, delta_std_h))
```

不预设短 horizon 权重更高。MSE 只作为后续有限对照，不与第一轮同时大规模调参。

### 7.3 最终后处理

依次保存并评估三种输出：

1. `combined_raw`：不裁剪、不置零；
2. `combined_clipped`：裁剪到合法物理范围；
3. `combined_postprocessed`：裁剪后将小于既定阈值的值置零。

阈值不得在外层验证折上选择。由于现有阈值通常改善 MAE、轻微损害 RMSE，最终提交策略应在官方评分公式明确后确定。

## 8. 两级运行闭环

### 8.1 A 级：开发闭环

目的只是验证数据、模型、预测和文件链路能够完整运行。

允许复用现有 v1.5.6 OOF 或减少 epoch，但产物必须标记：

```text
evaluation_status = diagnostic_only
```

A 级闭环必须完成：

1. 读取 v1.5.6 和原始逐小时数据；
2. 构建一个县级 batch；
3. 完成前向传播和反向传播；
4. 生成四个 horizon 的残差预测；
5. 与 LightGBM 原始预测合成；
6. 映射回提交结构；
7. 完成保存、重新加载和逐值复核。

A 级结果不能写入 `Results.md` 作为正式模型收益。

### 8.2 B 级：正式嵌套验证闭环

外层使用现有 `balanced_v1` 五折，每次完整留出一组县。

对每个外层折：

1. 只在外层训练县中选择 LightGBM 轮数和预处理参数；
2. 在外层训练县内部按县交叉拟合，生成每个训练县的 LightGBM OOF 原始预测；
3. 由这些预测生成 GRU 残差标签；
4. 只在外层训练县中选择 GRU epoch、`alpha_h` 和其他配置；
5. 使用全部外层训练县重训该折的 LightGBM；
6. 使用全部外层训练县的合法残差标签训练该折的 GRU；
7. 对外层验证县生成 LightGBM 预测和 GRU 修正；
8. 外层验证县标签只用于最终评分。

合并五个外层折后形成完整系统的严格 OOF 预测。

为避免用某个内层验证县自己的标签选择该县基线预测的树数，正式实现优先采用：

1. 先在外层训练县内部确定每个 horizon 的固定轮数；
2. 再以固定轮数重新执行按县交叉拟合，生成残差标签。

如果为了最小实现暂时使用内层早停折本身选择轮数，运行元数据必须明确标记该近似，且该结果不得作为最终定版证据。

## 9. 最终全量训练和测试预测

正式 OOF 协议完成并冻结配置后：

1. 在全部训练县内部交叉拟合 v1.5.6 LightGBM，生成全量合法 OOF 残差标签；
2. 根据外层验证得到的固定配置和 epoch 训练最终 GRU；
3. 在全部训练县上训练最终 v1.5.6 LightGBM；
4. 对测试县生成 LightGBM 原始预测；
5. GRU 读取测试县前 72 小时停电历史和完整天气，预测残差；
6. 按 horizon 合成、裁剪、后处理并写入提交模板；
7. 验证标识列、行顺序、NaN 掩码、数值范围和输出哈希。

第一版最终模型可以使用单个 seed=42。只有 v3.1 在正式 OOF 中显示稳定增益后，再训练 3 个种子并平均预测。

## 10. 训练设置

初始训练配置：

| 参数 | 初始值 |
|---|---:|
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| county batch size | 8 或 16 |
| max epochs | 200 |
| early stopping patience | 20 |
| gradient clipping | 1.0 |
| random seed | 42 |

epoch 内的 loss 先对每个 county/horizon 的有效时间位置求均值，再在 county 间平均，避免有效位置较多的序列获得额外权重。

训练日志至少记录：

- train/validation loss；
- 每个 horizon 的 RMSE/MAE；
- 每个 horizon 的残差均值、标准差和 `alpha_h`；
- 梯度范数；
- 最佳 epoch；
- 学习率；
- 有效县数和目标数；
- 运行耗时和随机种子。

## 11. 对照实验

v3.1 的正式报告至少包含四个同折对照：

| 编号 | 模型 | 作用 |
|---|---|---|
| B0 | v1.5.6 LightGBM | 原始基线，`delta=0` |
| B1 | 目标时刻对齐平均 | 检验跨 horizon 一致性带来的廉价收益 |
| B2 | 非时序 MLP 残差 | 与 GRU 使用相同静态、时间和基线输入，但不读逐小时序列 |
| B3 | GRU 残差 | v3.1 主模型 |

如果 B2 与 B3 改善相近，说明收益主要来自二级校准或附加输入；只有 B3 稳定优于 B2，才能支持“时序顺序提供了额外信息”。

可选负对照是在训练县内打乱天气时间顺序。该实验不属于最小闭环必需项。

## 12. 评价和通过标准

### 12.1 必报指标

每个 horizon 报告：

- 池化 OOF RMSE 和 MAE；
- 相对 v1.5.6 的百分比变化；
- 五个外层折的 RMSE/MAE；
- 每县 RMSE/MAE；
- 最大误差县及其平方误差占比；
- 预测零比例、均值、最大值；
- 合成前后残差的一小时自相关；
- 按目标小时和风暴阶段的误差；
- 同一 `(county, target_hour)` 下不同 horizon 预测之间的离散程度。

不把折 RMSE 的普通标准差误写为整体置信区间。

### 12.2 不确定性

使用按县整条轨迹重采样的 paired bootstrap，对 B3 与 B0/B1/B2 的 RMSE、MAE差值计算区间。禁止逐行 bootstrap。

### 12.3 v3.1 通过条件

满足以下条件，才进入多 seed 或更复杂模型：

1. B3 在严格外层 OOF 上优于 B0，而不是只在开发闭环中改善；
2. 改善方向在多数外层折中一致；
3. 改善不完全由一两个极端县贡献；
4. B3 相对 B2 有可重复优势，或显著改善目标时刻一致性；
5. 没有 horizon 出现不可接受的大幅退化；
6. 保存模型重新加载后的预测与运行时预测逐值一致。

如果只改善个别 horizon，则保留按 horizon 启用 GRU 的方案：无增益的 horizon 设置 `alpha_h=0`。

## 13. 计划目录结构

实现后目录建议为：

```text
time_series_model/
└── v3.1/
    ├── Plan_v3.1.md
    ├── README.md
    ├── requirements.txt
    ├── config.py
    ├── build_sequences.py
    ├── dataset.py
    ├── model.py
    ├── baseline_crossfit.py
    ├── train_diagnostic.py
    ├── train_nested_cv.py
    ├── train_final.py
    ├── evaluate.py
    ├── verify_artifacts.py
    ├── artifacts/
    │   ├── sequence_manifest.json
    │   ├── fold_metrics.csv
    │   ├── summary_metrics.csv
    │   ├── county_metrics.csv
    │   ├── oof_predictions.parquet
    │   ├── test_predictions.parquet
    │   ├── run_metadata.json
    │   └── checkpoints/
    └── submission_v3.1_balanced_v1.csv
```

所有生成文件只写入 `time_series_model/v3.1/`，不得覆盖 v1.5.6、v1.5.7 或 v2 的缓存、模型和结果。

## 14. 计划命令接口

实现后的建议入口：

```powershell
# 构建并验证县级序列缓存
python time_series_model/v3.1/build_sequences.py --verify

# 快速开发闭环，只验证管线，不产生正式结论
python time_series_model/v3.1/train_diagnostic.py

# 正式嵌套按县交叉验证
python time_series_model/v3.1/train_nested_cv.py

# 冻结配置后训练全量模型并生成提交
python time_series_model/v3.1/train_final.py

# 独立重新加载并复核全部产物
python time_series_model/v3.1/verify_artifacts.py
```

脚本必须拒绝以下情况：

- v1.5.6 输入哈希与记录不一致；
- 县内时间不是严格连续的 216 小时；
- 训练和测试通道顺序不同；
- 观测编码器读取 hour 72 以后真实停电；
- 验证县出现在相应模型的训练县中；
- 目标或提交 NaN 掩码错误；
- 提交标识列或行顺序发生变化。

## 15. 实施顺序

### 里程碑 M1：数据闭环

- 定义配置和字段映射；
- 构建县级张量及掩码；
- 验证 239 个训练县、63 个测试县、每县 216 小时时间连续；
- 保存通道、归一化和输入哈希清单。

**执行状态（2026-09-09）：已完成。** 训练和测试序列缓存、字段清单及哈希 manifest
已保存到 `artifacts/`。

### 里程碑 M2：模型闭环

- 实现双段单向 GRU；
- 用少量县和少量 epoch 完成前向、反向和模型重载；
- 验证输出维度、正负残差和有效掩码。

**执行状态（2026-09-09）：已完成。** 前向输出为 144×4，五折训练完成，检查点重载
预测一致，并由 M3 的独立复算再次验证。

### 里程碑 M3：开发预测闭环

- 接入 v1.5.6 原始预测；
- 生成诊断级 OOF、测试预测和提交；
- 独立复算指标和提交值；
- 将结果明确标为 `diagnostic_only`。

**执行状态（2026-09-09）：已完成。** 实际运行、诊断指标、复算结果和文件清单见
`M3_Implementation_Report_2026-09-09.md`。该结果没有写入根目录 `Results.md`。

### 里程碑 M4：正式验证闭环

- 完成外层五折和内层基线交叉拟合；
- 生成严格 OOF；
- 完成 B0/B1/B2/B3 对照；
- 完成按县 paired bootstrap 和误差诊断。

### 里程碑 M5：定版

- 根据正式 OOF 冻结配置；
- 训练最终全量系统；
- 生成并复核 `submission_v3.1_balanced_v1.csv`；
- 将最终结果追加到 `Results.md`，开发闭环分数不写入正式结果。

## 16. 后续版本接口

v3.1 跑通后再决定后续路线：

- v3.2：小型 TCN 与 GRU 的同协议比较；
- v3.3：目标时刻一致性约束或统一未来 OSI 轨迹；
- v3.4：以 v1.5.6 特征重训的 P/N/D/R LightGBM 作为基线；
- v3.5：分量残差或多任务时序模型；
- 后续实验：只在严格消融下重新测试精简 F1/F2/F3 特征。

版本号仅表示实验路线，不表示后续版本必然替换 v3.1。最终选择始终以同一验证协议下的完整系统指标为准。
