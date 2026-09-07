# INFORMS 2026 Data Mining Society Data Challenge — 初版方案

## 一、比赛概要

**INFORMS 2026 Data Mining Society Data Challenge**

**任务**: 基于风暴气象数据,预测县级停电严重程度指数 (OSI)。

**背景**: 2026年3月13日,两轮连续风驱动风暴袭击印第安纳(IN)、俄亥俄(OH)、宾夕法尼亚(PA)、西弗吉尼亚(WV)四州302个县,造成大面积停电。

**核心目标**: 在预测时刻 t,利用3月11–13日(前72小时)的停电观测数据 + 全窗口气象数据,预测3月14–19日(144小时)内4个时间尺度的 OSI:
| 目标 | 含义 |
|---|---|
| `osi_target_t01h` | t+1小时(即时下一小时) |
| `osi_target_t06h` | t+6小时(人员预部署窗口) |
| `osi_target_t24h` | t+24小时(次日互援规划) |
| `osi_target_t48h` | t+48小时(两天应急展望) |

**OSI公式**: `OSI = 0.40·P_t + 0.35·N_t + 0.25·D_t − 0.10·R_t` (clip≥0)
- P_t: 当前停电比例 | N_t: 新增停电增长率 | D_t: 6小时滚动均值 | R_t: 恢复率

**数据结构**:
- **Train**: 239县 × 216小时 = 51,624行, 63列, 全量数据含所有目标
- **Test**: 63县 × 216小时 = 13,608行, 43列; 停电变量仅3月11–13有值, 3月14–19为NaN; 气象全有; 目标完全隐藏
- **提交**: 9,072行 (63县 × 144小时), 填4个目标列

**关键约束 — 时间因果规则**:
- ✅ 允许: 时刻 ≤ t 的停电特征 + 任意时刻的气象特征(含未来)
- ❌ 禁止: 时刻 > t 的停电特征; 从未来行计算OSI/停电衍生量
- 代码将被审查合规性

**关键数据特性**:
- 按县划分(非时间划分), 训练/测试县完全不重叠 → 需学习跨县通用模式
- ~43%目标为零(零膨胀分布)
- 双风暴结构: 两轮独立风事件, OSI轨迹呈"双峰"
- 预事件窗口非干净基线: 236/239县有2月遗留停电
- 目标均值~0.006, 最大~0.65; gust与OSI同时刻相关系数仅0.145 → 需用时滞特征和时序模式
- `severity_tier` 仅用于分层交叉验证, 不可作输入特征
- 可选外部数据: EIA Form 861(线路里程、客户数)、NLCD树冠覆盖、USDA城乡代码

---

## 二、相关文献综述

### A. 风暴停电预测 — 基础与经典方法

| # | 论文 | 年份 | 核心贡献 | 与本赛相关性 |
|---|---|---|---|---|
| 1 | **Guikema et al., "Predicting hurricane power outages to support storm response planning,"** IEEE Access | 2014 | SGHOPM模型: 用风速+持续时间+植被+土壤等特征, 两步法(先分类是否有停电→再回归数量), 被引270次, 该领域奠基性工作 | 核心范式: 两步法、气象+环境特征组合 |
| 2 | **McRoberts, Quiring, Guikema, "Improving hurricane power outage prediction models through the inclusion of local environmental factors,"** Risk Analysis | 2018 | SGHOPM增强版: 加入海拔、土地覆盖、土壤、降水、植被特征, 精度提升~17%, 被引128次 | 特征工程参考: 环境因子的重要性 |
| 3 | **Cerrai et al., "Predicting storm outages through new representations of weather and vegetation,"** IEEE | 2019 | 非飓风风暴停电预测, Random Forest + 集成回归, 植被+气象新表征, 被引162次 | 直接对应风风暴(非飓风)场景 |
| 4 | **Cerrai et al., "Outage prediction models for snow and ice storms,"** Sustainable Energy, Grids and Networks | 2020 | 冬季风暴(雪/冰)停电预测, 统计+ML方法, 被引98次 | 低温+降雪对线路的影响, 与本赛t2m/csnow特征呼应 |

### B. 深度学习与时空方法 — 前沿方向

| # | 论文 | 年份 | 核心贡献 | 与本赛相关性 |
|---|---|---|---|---|
| 5 | **Yang et al., "From Forecast to Action: A Deep Learning Model for Predicting Power Outages During Tropical Cyclones (STOCAST),"** Risk Analysis | 2026 | **GRU+全连接层**的时空深度学习框架, 整合静态环境/基础设施属性与动态气象/停电序列, Leave-One-Storm-Out交叉验证, 支持6h短临+60h长期双模式, 实时数据同化 | 最直接相关: 滚动预测、GRU时序建模、多尺度预报 |
| 6 | **Prieto-Godino & Pelaez-Rodriguez, "Predicting weather-related power outages in large scale distribution grids with deep learning ensembles,"** Elsevier | 2025 | 5种DL架构集成的停电预测算法, 大规模配电网 | DL集成方法参考 |
| 7 | **Alpay et al., "Dynamic modeling of power outages caused by thunderstorms,"** Forecasting (MDPI) | 2020 | **雷暴(非飓风)停电动态建模**, 对比Poisson回归/KNN/Random Forest/**LSTM**, 被引59次 | 与本赛风风暴场景高度吻合, LSTM时序方法 |
| 8 | **Tervo et al., "Short-term prediction of electricity outages caused by convective storms,"** IEEE Trans. | 2019 | **对流风暴短期停电预测**, ML方法, 被引25次 | 短临预测方法, 与t+1h/t+6h目标对应 |
| 9 | **Aljurbua et al., "Early prediction of power outage duration through hierarchical spatiotemporal multiplex networks,"** Springer | 2024 | 层次化时空复用网络, 提前3小时预测停电严重程度, 被引12次 | 时空网络建模思路 |
| 10 | **Ranjan et al., "A PGRF-Augmented Transformer Framework for Short-Term Power Outage Forecasting,"** IISE Annual | 2026 | **Transformer框架**, 83个密歇根县级短期停电预测, Huber loss降低41.8%, 优于LSTM基线 | 县级+Transformer, 与本赛任务几乎同构 |

### C. 综述与方法论

| # | 论文 | 年份 | 核心贡献 | 与本赛相关性 |
|---|---|---|---|---|
| 11 | **Xie & Alvarez-Fernandez, "A review of machine learning applications in power system resilience,"** IEEE PES | 2020 | ML在电力系统韧性中的全面综述, 被引154次 | 方法论全景图 |
| 12 | **Yang et al., "Quantifying uncertainty in machine learning-based power outage prediction model training,"** Sustainability | 2020 | OPM训练不确定性量化, 样本量敏感性, 被引70次 | 小样本场景(239县)的不确定性处理 |
| 13 | **Yang et al., "Enhancing weather-related power outage prediction by event severity classification,"** IEEE Access | 2020 | 按事件严重度分级分类→再预测, 被引76次 | 分级预测思路, 与severity_tier呼应 |
| 14 | **Arora & Ceferino, "Probabilistic and machine learning methods for uncertainty quantification in power outage prediction due to extreme events,"** NHESS | 2023 | 概率+ML方法的不确定性量化, 被引59次 | 概率预测方法 |
| 15 | **Lee et al., "A data-driven approach to predicting power outages during winter storms leveraging nonparametric ML models,"** Springer | 2025 | 冬季风暴停电预测, 非参数ML, 2021德州冬季风暴Uri | 冬季风暴+非参数方法 |

### 文献核心启示

1. **两步法范式**(Guikema 2014, McRoberts 2018): 先分类"是否停电"→再回归"停电严重度" → 适合本赛43%零膨胀
2. **GRU/LSTM时序建模**(STOCAST 2026, Alpay 2020): 将停电历史序列+气象序列输入RNN → 捕捉双峰动态
3. **Transformer在县级停电预测中优于LSTM**(Ranjan 2026): 注意力机制能捕捉跨县/跨时段模式
4. **风特征为核心驱动**(所有文献一致): gust是最强单预测因子, 但需结合时滞和环境上下文
5. **事件严重度分级**(Yang 2020): severity_tier可用于分层CV和分级建模
6. **实时数据同化/滚动预测**(STOCAST 2026): 随时间推进更新预测 → 本赛144小时窗口可分阶段处理

---

## 三、初版构建方案

### 整体架构: 三层渐进式

```
Phase 1: LightGBM 强基线 (快速迭代, 验证特征工程)
    ↓
Phase 2: 序列模型 GRU/Transformer (捕捉时序动态)
    ↓
Phase 3: 集成融合 + 分阶段滚动预测
```

**当前进展（2026-09-07）**：已完成固定 `balanced_v1` 县级五折下的三种直接 OSI 模型，以及 v2.1/v2.2/v2.3“先预测 P/N/D/R、再合成 OSI”的三种分量模型。当前 RMSE 最优组合是直接 XGBoost t+1h，加 v2.1 LightGBM t+6h/t+24h/t+48h；MAE 最优组合是 v2.2 t+1h/t+6h，加 v2.3 t+24h/t+48h。下一步优先基于六套同折 OOF 预测做按 horizon 的受约束融合和重复分组验证，再决定是否投入 GRU/Transformer。

### Phase 1: LightGBM 多目标回归基线

**1.1 训练样本构造**
- 对训练集的3月14–19每个小时 t, 构造一条样本
- 特征 = t时刻可用的停电信息(≤t的lag特征) + 气象特征(任意时刻) + 县级静态特征
- 目标 = 4个horizon的OSI值
- 严格模拟测试条件: 不使用t之后停电数据作为输入

**1.2 特征工程**

| 类别 | 特征 | 说明 |
|---|---|---|
| **停电状态(72h窗口)** | 最近可用OSI、P_t、N_t、D_t、R_t | 观测窗口末尾值 |
| | osi_lag1h/3h/6h/24h/48h | 预计算lag(窗口内有值的部分) |
| | outage_pct_lag系列 | 停电比例lag |
| | 72h窗口统计量: mean/max/std/trend | OSI轨迹摘要 |
| | 预事件基线(Mar 11-12均值) | 区分遗留停电vs新停电 |
| | 第一波信号(Mar 13峰值/趋势) | 首轮风暴影响程度 |
| | 停电是否正在恶化/恢复 | N_t/R_t趋势 |
| **气象(任意时刻)** | t时刻 + 目标时刻的gust/wind_speed | 核心驱动 |
| | gust的滚动max/mean(未来1h/6h/24h/48h) | 预测窗口内的风力峰值 |
| | wind_sin/wind_cos | 风向编码(去除0-360跳变) |
| | t2m, d2m → 降雪/结冰风险标志 | t2m<273.15且高湿度→icing_risk |
| | tp, rain, csnow, sdwe | 降水/降雪/积雪 |
| | soil_moist | 土壤湿度→树木根固力 |
| | 气象变化量(delta) | t→t+horizon的gust增量 |
| | blh, 云量系列 | 对流活动/天气系统 |
| **县级静态** | customersTracked | 县规模(归一化) |
| | 外部数据特征(见下表) | 基础设施/植被/城乡差异 |
| **时间** | hour_of_day, days_since_onset | 日周期/事件进度 |
| | 是否第二风暴窗口(Mar 16-17) | 双峰结构 |

> **关于标识符列的使用限制**: `fipsCode`、`stateAbbr` 等标识符列**不可直接作为模型输入特征**(文档明确标注 "Do not use as model input features")。用 `fipsCode`/`stateAbbr` 作为**连接键(join key)**拼接外部数据,将底层因素的实际数值作为特征进入模型,而标识符本身不进模型。比赛文档已提供可选外部数据源:

| 底层因素 | 外部数据源 | 对应维度 | 用法 |
|---|---|---|---|
| 线路里程、客户数 | EIA Form 861 | 基础设施质量/治理水平 | 按 fipsCode 拼接, 得到县内线路密度等 |
| 树冠覆盖 | NLCD Tree Canopy (USGS) | 植被/树木倒伏风险 | 按 fipsCode 拼接, 得到县内树冠覆盖率 |
| 城乡连续代码 | USDA Rural-Urban Codes | 城市化程度/人口密度 | 按 fipsCode 拼接, 得到城乡等级 |
| 土壤湿度 | 已内置 `soil_moist` | 地形/树木根固力 | 无需外部拼接 |
| 海拔、土地覆盖 | McRoberts(2018)验证有效 | 地形 | 可从公开GIS数据获取 |

**1.3 模型设计**
- 4个独立LightGBM回归器(每个horizon一个), 因各horizon特性不同(t+48h与t+1h相关仅0.056)
- 损失函数: Huber loss(对零膨胀和极端值鲁棒)
- 交叉验证: 5折, 按(state, severity_tier)分层分组, GroupKFold确保同县不跨折
- 零膨胀处理(可选): 两阶段 — 先LightGBM分类器预测OSI>0概率, 再回归器预测非零值, 最终 `pred = P(nonzero) × regression_value`

**1.4 关键实现要点**
- 测试集中lag特征在3月14-19为NaN → 需用72h窗口的末尾值构造"距观测窗口末尾的偏移量"作为替代特征
- 气象未来特征可直接使用(规则允许)
- 所有随机种子固定(seed=42与比赛一致)

### Phase 2: 序列深度学习模型

**2.1 模型架构: GRU + 多头输出**

```
输入序列 (72h × features):
  [outage_seq: osi, P_t, N_t, D_t, R_t, outage_pct]  (6维)
  [weather_seq: gust, wind_speed, wind_sin, wind_cos, t2m, ...]  (22维)
        ↓
  GRU Encoder (2层, hidden=128)
        ↓
  [hidden_state h_T]
        ↓
  拼接: h_T + future_weather_at_horizon + county_static
        ↓
  4个全连接头 → OSI(t+1h, t+6h, t+24h, t+48h)
```

**2.2 设计要点**
- **编码器**: GRU编码72小时停电+气象序列 → 获取初始状态向量
- **解码器**: 将hidden state与目标时刻气象特征拼接 → 全连接预测
- **多horizon共享编码器, 独立解码头**: 参数共享提高泛化, 独立头适应不同尺度
- **损失**: 加权Huber loss, 短horizon权重更高(精度更重要)
- **输入归一化**: 按县内z-score或全局min-max
- **正则化**: Dropout 0.2, weight decay, early stopping

**2.3 可选进阶: Temporal Fusion Transformer (TFT)**
- 论文#10(Ranjan 2026)显示Transformer在县级停电预测优于LSTM
- TFT天然支持: (a)已知未来输入(气象), (b)静态元数据(县级), (c)可解释注意力权重
- 但需更多调参, 建议作为Phase 2的进阶选项

### Phase 3: 集成与滚动预测

**3.1 集成策略**
- LightGBM + GRU 加权平均(权重由CV性能决定)
- 加入persistence baseline: `pred(t+h) = osi(t)` 作为安全网(短期有效)
- 最终: `pred = w1·lgbm + w2·gru + w3·persistence`

**3.2 分阶段滚动预测**
- **阶段A (Mar 14, 前24h)**: 72h窗口lag特征充足 → 依赖停电历史+气象
- **阶段B (Mar 15-16, 中间48h)**: lag逐渐失效 → 转向气象驱动+模型自回归
- **阶段C (Mar 17-19, 末尾72h)**: 几乎无lag → 纯气象+县模式预测
- 模型权重随阶段调整: 早期停电特征权重高, 后期气象权重高

**3.3 评估指标**
- 比赛未明确, 但OSI为连续值, 预计RMSE或MAE
- CV中按horizon分别监控, 重点关注t+24h和t+48h(最难, 价值最高)

### 实施路线图

| 步骤 | 内容 | 预计工时 |
|---|---|---|
| 1 | 数据加载+EDA: 目标分布/零膨胀/双峰/县间差异/特征相关性矩阵 | 0.5天 |
| 2 | 特征工程管线: 构造训练样本+所有上述特征 | 1天 |
| 3 | Phase 1 LightGBM基线: 4模型+分层CV+两阶段零膨胀 | 1天 |
| 4 | 模型分析: 特征重要性/误差分析/per-horizon诊断 | 0.5天 |
| 5 | Phase 2 GRU序列模型: 数据集构造+训练+调参 | 1.5天 |
| 6 | (可选) TFT Transformer | 1天 |
| 7 | Phase 3 集成+滚动策略+提交管线 | 0.5天 |
| 8 | 报告撰写(≤6页) | 0.5天 |

### 风险与注意事项

1. **小样本风险**: 仅239县×1次事件 → 过拟合风险高, 需强正则化+简单模型优先
2. **时间因果合规**: 代码审查会检查 → 所有特征工程必须严格按timestamp ≤ t
3. **lag特征在测试集的NaN处理**: 3月14后lag逐小时变NaN → 用"距观测窗口末尾的hours_since_observed"替代, 或用模型自回归生成
4. **severity_tier不可用作特征**: 仅CV分层用
5. **外部数据(可选)**: EIA线路里程、树冠覆盖、城乡代码可通过 fipsCode/stateAbbr 作连接键拼接,捕获基础设施/植被/城乡等底层县级差异 → 如果基线不足则引入
