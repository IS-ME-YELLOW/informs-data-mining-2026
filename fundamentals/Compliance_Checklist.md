# 比赛硬性约束检查清单 (Compliance Checklist)

> 本文档整理自 `problem_description.pdf`、`osi_methodology.pdf`、`variable_descriptions.pdf`、`weather_variables.pdf` 及数据文件实际结构验证。用于写代码时逐条对照,确保不违规。

---

## 0. 事件时间线速查

```
3月11  ├──────预事件窗口(48h)──────┤  in_event_window=False
       │(非干净基线, 有2月遗留停电) │
3月12  └──────────────────────────┘
3月13  ├──风暴第一波(24h)──────────┤  in_event_window=True
       │  (观测窗口最后24h)        │
3月14  ├──预测窗口(144h) ──────────┤  ← 提交从这里开始
       │                          │
3月16  ├──风暴第二波──┤            │
       │             │            │
3月19  └─────────────┴────────────┘  ← 提交到这里结束
```

| 窗口名 | 时间 | 小时数 | 索引 | 停电数据(Test) | 气象数据 |
|---|---|---|---|---|---|
| 预事件窗口 | 3月11–12 | 48h | 0–47 | ✅ 有值 | ✅ |
| 观测窗口(末24h) | 3月13 | 24h | 48–71 | ✅ 有值 | ✅ |
| **观测窗口合计** | **3月11–13** | **72h** | **0–71** | **✅ 有值** | ✅ |
| 预测窗口 | 3月14–19 | 144h | 72–215 | ❌ NaN | ✅ |
| **全窗口** | **3月11–19** | **216h** | **0–215** | | ✅ |

> 注意: `in_event_window` 仅区分预事件(False=3月11–12) vs 风暴(True=3月13–19), 不区分观测 vs 预测。预测窗口(3月14–19)是 `in_event_window=True` 的子集。

---

## 1. 时间因果规则 (Temporal Causality Rule) — 最核心约束

在预测原点时刻 t,以下规则严格适用:

| 规则 | 内容 |
|---|---|
| ✅ 允许 | 任何停电特征值在 timestamp **≤ t** 的行(即当前及过去时刻) |
| ✅ 允许 | 气象特征在**任意 timestamp**(包括未来时刻 t+1, t+6, t+24, t+48 等) |
| ❌ 禁止 | 任何停电特征值在 timestamp **> t** 的行(即未来时刻的停电数据) |
| ❌ 禁止 | 从未来行计算 OSI 或任何停电衍生量 |

**执行方式**: 代码提交将被人工审查合规性。任何被发现使用了未来停电值的提交将被**自动拒绝**,无论预测精度如何。

**实操含义**:
- 构造训练样本时,输入特征只能用到 timestamp ≤ t 的停电数据
- 测试集中停电变量在 3月14–19 全部为 NaN,因此实际可用的停电数据仅限 3月11–13(72小时观测窗口)
- 气象数据在训练/测试集中均完整覆盖全部 216 小时,可作为"已知未来"输入

---

## 2. OSI 公式与组件定义

### 2.1 OSI 复合公式
```
OSI_t = 0.40 × P_t + 0.35 × N_t + 0.25 × D_t − 0.10 × R_t    (结果 clip ≥ 0)
```
- 典型范围: 0–0.60; 理论最大: 0.90; 训练数据中观测最大: 0.65
- R_t 主动从严重度中减去: 一个正在恢复供电的县,其评分低于相同停电水平但无恢复活动的县

### 2.2 四个组件公式

| 组件 | 类型 | 公式 | 含义 | 取值范围 |
|---|---|---|---|---|
| **P_t** | 状态(State) | `outageCount(t) / customersTracked` | 当前停电比例 | 0–1 |
| **N_t** | 流量(Flow) | `max(0, ΔoutageCount) / customersTracked` | 本小时新增停电增长率(恶化中) | 0–0.379 |
| **D_t** | 状态(State) | `rolling_mean(P_t, 6小时)` | 6小时持续停电趋势 | 0–1 |
| **R_t** | 流量(Flow) | `max(0, −ΔoutageCount) / customersTracked` | 本小时恢复率(改善中) | 0–0.204 |

其中 `ΔoutageCount = outageCount(t) − outageCount(t−1)`

### 2.3 测试集中 OSI 的重建
- 测试文件提供了 3月11–13 观测窗口的全部四个 OSI 组件 (P_t, N_t, D_t, R_t)
- 可用上述公式重建该窗口内的 OSI 及其 lag 特征
- **重建必须遵守时间因果规则**: 在预测原点 t 时, OSI lag 特征只能从同一县内 timestamp ≤ t 的行派生
- 3月14–19 测试行: OSI 组件被**置为 NaN**(withheld),**不可重建、不可填补**,它们是预测目标窗口

---

## 3. 训练/测试集划分

| 项目 | 内容 |
|---|---|
| 划分方式 | 县级划分 (county-level), **非**时间划分 |
| 划分比例 | 80/20, 按 state × severity level 分层 |
| severity level 定义 | 综合"峰值OSI排名"与"平均停电时长排名"(县内), 用于分层 |
| 随机种子 | 42 |
| 县重叠 | **无** — 每个县只出现在训练或测试中,不交叉 |
| 训练县数 | 239 |
| 测试县数 | 63 |
| 总县数 | 302 |

各州分布:
| 州 | 训练 | 测试 | 合计 |
|---|---|---|---|
| IN | 72 | 20 | 92 |
| OH | 70 | 18 | 88 |
| PA | 52 | 15 | 67 |
| WV | 45 | 10 | 55 |
| **合计** | **239** | **63** | **302** |

---

## 4. 列级可用性矩阵 — Train vs Test

### 4.1 Train 有但 Test 没有的列(20列)

| 列名 | 类别 | 说明 |
|---|---|---|
| `severity_tier` | 元数据 | 0–4, 仅用于分层CV, **不可作模型输入特征**, Test中不存在 |
| `osi` | 停电衍生 | Test中3月11–13可从组件重建, 3月14–19为NaN |
| `osi_lag1h` | Lag | Test中不存在(需自行重建3月11–13部分) |
| `osi_lag3h` | Lag | 同上 |
| `osi_lag6h` | Lag | 同上 |
| `osi_lag24h` | Lag | 同上 |
| `osi_lag48h` | Lag | 同上 |
| `peak_pct` | 事件级结果 | 全县恒定值, 事后统计量, **不可用于预测** |
| `peak_customers` | 事件级结果 | 同上 |
| `time_to_restore_h` | 事件级结果 | 同上 |
| `osi_target_t01h` | 预测目标 | 仅Train有, Test完全隐藏 |
| `osi_target_t03h` | 参考目标 | **非评分目标**, 仅为参考 |
| `osi_target_t06h` | 预测目标 | 仅Train有 |
| `osi_target_t24h` | 预测目标 | 仅Train有 |
| `osi_target_t48h` | 预测目标 | 仅Train有 |
| `osi_delta_t01h` | 增量目标 | `= osi_target_t01h − osi`, 非直接评分, 仅为便利 |
| `osi_delta_t03h` | 增量目标 | 非评分目标 |
| `osi_delta_t06h` | 增量目标 | 非直接评分 |
| `osi_delta_t24h` | 增量目标 | 非直接评分 |
| `osi_delta_t48h` | 增量目标 | 非直接评分 |

### 4.2 Test 中停电变量的 NaN 模式

测试集 13,608 行 (63县 × 216小时),停电变量的可用性:

| 时间段 | 索引范围 | 停电变量(outageCount, outage_pct, P_t, N_t, D_t, R_t) | outage_pct_lag系列 |
|---|---|---|---|
| 3月11日 0:00 – 3月13日 23:00 (观测窗口, 72h) | 0–71 | ✅ 有值 | ✅ 有值(开头有少量NaN因lookback超出数据起点) |
| 3月14日 0:00 – 3月19日 23:00 (预测窗口, 144h) | 72–215 | ❌ 全部 NaN | ❌ 全部 NaN |

**关键细节**:
- `customersTracked` 在测试集所有216小时均有值(它是分母,非停电计数)
- 预测窗口中,即使 outage_pct_lag1h 在3月14日0:00理论上可从3月13日23:00的值计算,但**预计算列已全部置为 NaN**,需自行重建
- 气象变量(22列)在训练/测试集均完整覆盖全部216小时

### 4.3 列分类速查

**标识符与元数据 — 不可作模型输入特征**:
- `timestamp_et` — 仅用于时间排序和因果规则执行; 可从中**派生**时间特征(hour_of_day等), 但原始时间戳不直接输入
- `fipsCode` — 可用于分组CV(GroupKFold); 可作**外部数据连接键**; **不可作输入特征**(target encoding亦属违规)
- `countyName` — 不可直接用; **注意**: countyName 非唯一标识, 302县仅有187个唯一名称(跨州有重复), 唯一标识须用 `fipsCode`
- `stateName`, `stateAbbr` — 不可作输入特征(含one-hot); 可作外部数据连接键拼接底层因素数据
- `in_event_window` — 布尔; False=3月11–12预事件窗口, True=3月13–19风暴窗口; **注意**: 它划分的是预事件 vs 风暴, 而非观测 vs 预测; 预测窗口(3月14–19)是 in_event_window=True 的子集
- `split` — TRAIN/TEST标识; **注意文档矛盾**: variable_descriptions.pdf 声称 "split...train file only",但实际测试文件**包含此列**(值为 "TEST"); 以实际数据为准
- `severity_tier` — **严禁作输入特征**, 仅CV分层用
- `event_duration_h` — 恒为168, **无预测信息**, 不可用

**原始停电变量 — 预测窗口(3月14–19)中为NaN**:
- `outageCount`, `outage_pct`, `P_t`, `N_t`, `D_t`, `R_t` — 这6列在测试集3月14–19为NaN

**全程可用的分母变量**:
- `customersTracked` — 在训练/测试集**全部216小时均有值**(是分母,非停电计数); 变量描述明确标注 "Present for all 216 hours in both files"

**Lag特征 — 预计算列在预测窗口为NaN, 需自行重建观测窗口部分**:
- `outage_pct_lag1h`, `outage_pct_lag3h`, `outage_pct_lag6h`, `outage_pct_lag24h`, `outage_pct_lag48h`
- `osi_lag1h`, `osi_lag3h`, `osi_lag6h`, `osi_lag24h`, `osi_lag48h` (Test中不存在, 需从组件重建)

**事件级结果变量 — 事后统计, 不可用于预测**:
- `peak_pct`, `peak_customers`, `time_to_restore_h`

**预测目标(4个评分 + 1个参考)**:
- 评分: `osi_target_t01h`, `osi_target_t06h`, `osi_target_t24h`, `osi_target_t48h`
- 参考(不评分): `osi_target_t03h`

**增量目标(便利列, 非直接评分)**:
- `osi_delta_t01h`, `osi_delta_t03h`, `osi_delta_t06h`, `osi_delta_t24h`, `osi_delta_t48h`
- `osi_delta_tXXh = osi_target_tXXh − osi` (等价于目标, 加回当前OSI即可还原)

**气象变量(22列, 全216小时可用)**:
- URMA(2.5km, 多边形均值): `gust`, `wind_speed_10m`, `wind_dir_10m`, `t2m`, `d2m`
- ERA5(~31km, 县质心): `sp`, `mslma`, `blh`, `tp`, `rain`, `csnow`, `sdwe`, `tcc`, `lcc`, `mcc`, `hcc`, `sdswrf`, `direct_rad`, `diffuse_rad`, `r2`, `vpd`, `et0`, `soil_moist`

---

## 5. NaN 处理要求

### 5.1 目标列 NaN
OSI 目标列在训练集中对每个县最后 k 行为 NaN(因未来窗口超出数据集末尾):

| 目标 | 每县末尾NaN行数 | 总NaN数 |
|---|---|---|
| `osi_target_t01h` | 1 | 239 |
| `osi_target_t06h` | 6 | 1,434 |
| `osi_target_t24h` | 24 | 5,736 |
| `osi_target_t48h` | 48 | 11,472 |

**要求**:
- 训练时: **drop 这些 NaN 行**(不参与训练)
- 提交时: 对应行**保留为 NaN**(不填值), 不被评分

### 5.2 Lag 特征 NaN
- 每县序列开头, lookback 超出数据集起点的行为 NaN (lag1h: 1行, lag3h: 3行, lag6h: 6行, lag24h: 24行, lag48h: 48行)
- 测试集预测窗口(3月14–19)所有预计算 lag 列为 NaN
- **禁止**填补或插值预测窗口的停电/lag NaN → 它们是目标窗口

### 5.3 严格禁止的操作
- ❌ 对测试集3月14–19的停电变量做插值/填补
- ❌ 用未来停电值填充当前行的 lag 特征
- ❌ 使用 `peak_pct`, `peak_customers`, `time_to_restore_h` 作为输入特征(它们是事后统计量)
- ❌ 使用 `severity_tier` 作为输入特征
- ❌ 使用 `event_duration_h` 作为输入特征(恒定值,无信息)

### 5.4 关键目标统计与分布特性

**OSI 目标范围与均值**:

| 目标 | 范围 | 均值 | 每县末尾NaN行数 |
|---|---|---|---|
| `osi_target_t01h` | 0–0.599 | 0.0062 | 1 |
| `osi_target_t06h` | 0–0.599 | 0.0063 | 6 |
| `osi_target_t24h` | — | — | 24 |
| `osi_target_t48h` | 0–0.598 | 0.0071 | 48 |

**关键特性**:
- **零膨胀**: ~43–44% 的目标值为零(osi 本身零占比 44%, outageCount 零占比 43%)
- **t+48h 最独立**: `osi_target_t48h` 与 `osi_target_t01h` 的相关系数仅 **0.056** — 文档明确标注 "Most independent horizon",需单独建模,不可简单外推
- **N_t 零占比 59%**, R_t 零占比 52% — 流量组件大部分时间为零(仅恶化/恢复时非零)

**OSI Delta 范围(增量目标, 非直接评分)**:

| 增量列 | 范围 | 备注 |
|---|---|---|
| `osi_delta_t01h` | −0.369 ~ 0.596 | |
| `osi_delta_t06h` | −0.369 ~ 0.596 | |
| `osi_delta_t48h` | −0.640 ~ 0.593 | 更宽的范围反映48小时不确定性更大 |

**原始停电变量统计(用于数据验证)**:

| 变量 | 范围 | 均值 | 零占比 |
|---|---|---|---|
| `outageCount` | 0–97,457 | 379 | 43% |
| `customersTracked` | 1,232–696,427 | — | — |
| `outage_pct` | 0–100(%) | 0.91% | — |
| `P_t` | 0–1 | 0.0091 | — |
| `N_t` | 0–0.379 | — | 59% |
| `D_t` | 0–1 | 0.0092 | — |
| `R_t` | 0–0.204 | — | 52% |
| `osi` | 0–0.65 | 0.0062 | 44% |

---

## 6. 数据质量注意事项

### 6.1 outage_pct 上限
- `outage_pct` 和 `P_t` 被封顶在 1.0 (100%)
- 原因: 2个县的 `outageCount` 超过 `customersTracked` (因服务区域边界不匹配)
- **无需修正**: 分数使用行级分母,内部一致

### 6.2 customersTracked 变动
- 239个训练县中有 204 个县的 `customersTracked` 在不同小时略有变动
- 原因: 多电力公司报告时间差异
- 变动很小(县内中位数范围: 0.05%)
- **无需修正**: 所有停电分数使用行级分母,内部一致

### 6.3 预事件停电(非干净基线)
- 3月11–12预事件窗口**不是**干净基线
- 239个训练县中有 236 个存在2月先前事件的遗留停电
- 1个县(FIPS 18111)在预事件窗口显示 100% 停电率 — 这是真实的历史状态信号,**非数据错误**
- **禁止**零填充或排除预事件行 — 它们携带有效的 lag 信号(尤其对 `osi_lag24h` 和 `osi_lag48h` 在风暴起始时)

### 6.4 双风暴结构
- 3月13–19窗口内发生两次独立风事件:
  - 第一波: 3月13–14
  - 第二波: 3月16–17
- 许多县的 OSI 轨迹呈"双峰"
- `osi_lag24h` 和 `osi_lag48h` 对区分第一波 vs 第二波动态特别有用

---

## 7. 气象数据处理要求

### 7.1 风向编码(必须)
- `wind_dir_10m` 是**圆形变量** (0–360°)
- 359° 和 1° 在数值上差358,但物理上几乎相同方向
- **必须编码为**:
  ```
  wind_sin = sin(2π × wind_dir_10m / 360)
  wind_cos = cos(2π × wind_dir_10m / 360)
  ```
- 使用 `wind_sin` 和 `wind_cos` 作为特征,**不使用原始角度**

### 7.2 单位与数据源
- URMA 风场(gust, wind_speed_10m)单位为 **mph** (从 m/s 转换, ×2.2369)
- URMA 温度(t2m, d2m)单位为 **Kelvin**
- ERA5 气压(sp, mslma)单位为 **hPa**
- ERA5 边界层高度(blh)单位为 **m**
- ERA5 降水(tp, rain)单位为 **mm**
- ERA5 降雪(csnow)单位为 **cm**, 雪深(sdwe)单位为 **m** (水当量)
- ERA5 云量(tcc, lcc, mcc, hcc)为 **%**
- ERA5 相对湿度(r2)为 **%**
- ERA5 辐射(sdswrf, direct_rad, diffuse_rad)为 **W/m²**
- ERA5 蒸汽压差(vpd)为 **kPa**
- ERA5 参考蒸散(et0)为 **mm/hr**
- ERA5 土壤湿度(soil_moist)为 **m³/m³**

### 7.2.1 数据源与分辨率说明
- **URMA (NOAA, 2.5km)**: 观测校正分析(observation-corrected analysis), 修正了粗分辨率再分析中的风速膨胀伪影; 适合驱动停电的高价值风/温度场; 许可: 公开数据
- **ERA5 via Open-Meteo (~31km)**: URMA 不携带的补充场(云、辐射、降水相态、雪、土壤湿度、气压); 适合缓慢变化的大气上下文变量, **不适合**驱动停电的精细风场; 许可: **CC BY 4.0**
- **NOAA HRRR (备选, 3km)**: 可作为 URMA 的替代风场来源
- 数据源 URL:
  - NOAA URMA: https://registry.opendata.aws/noaa-rtma/
  - Open-Meteo ERA5: https://open-meteo.com/en/docs/historical-weather-api
  - NOAA HRRR: https://registry.opendata.aws/noaa-hrrr-pds/

### 7.3 时间对齐注意
- URMA 时间戳从 UTC 转换, 使用固定 −5h (EST)
- ERA5 通过 Open-Meteo 获取, 使用本地 America/New_York 时区(事件期间为 EDT, −4h, 因夏令时3月8日已开始)
- 两个序列都是密集小时序列, 共享相同的行标签
- **URMA 和 ERA5 在同一标记小时可能描述最多相差1小时的物理状态**
  - 对缓慢变化的 ERA5 上下文场(气压、湿度、土壤湿度)无影响
  - 对 URMA 风变量无影响
  - 可能对 ERA5 降水场(tp, rain)在风暴活跃通过期间略有影响

### 7.4 气象聚合方式
- **URMA**: 多边形均值 — 对每个县边界内所有2.5km网格点取均值(point-in-polygon空间连接), 所有302县至少含1个网格点, 使用真实多边形均值
- **ERA5**: 县质心采样 — 在每个县人口加权质心处取单值

---

## 8. 提交要求

### 8.1 三组件
提交必须包含三个组件,组织在名为 `TeamName_Submission/` 的文件夹中:

| 组件 | 要求 |
|---|---|
| **Component 1: 预测文件** | 以 `sample_submission.csv` 为起点, 填入4个目标列的预测值; **不改动标识符列(fipsCode, countyName, stateAbbr, timestamp_et)和行顺序** |
| **Component 2: 可复现代码** | `code.zip`, 必须在提供的数据文件上端到端复现提交的预测, 无人工步骤; 包含库版本、显式随机种子、相对文件路径 |
| **Component 3: 书面报告** | `report.pdf`, **不超过6页**(不含参考文献); 描述方法论、关键建模决策、结果批判性讨论; 报告是评估的核心部分 |

### 8.2 提交行数
- 9,072 行 = 63县 × 144小时 (3月14–19预测窗口)
- 超出3月19的目标行(即 horizon 超出数据末尾的行)**保留为 NaN**

### 8.3 评分目标(仅4个)
- `osi_target_t01h`, `osi_target_t06h`, `osi_target_t24h`, `osi_target_t48h`
- `osi_target_t03h` 和所有 `osi_delta_*` 列**不评分**

### 8.4 截止日期
- 2026年9月25日 (AOE — Anywhere on Earth)

### 8.5 提交方式
- 上传 `TeamName_Submission/` 文件夹到自己的 Google Drive
- 设置查看权限共享
- 通过 Google Form 提交链接

---

## 9. 可选外部数据

文档提及以下基础设施特征(静态, 可选增强), 但非必须:

| 数据源 | 内容 | URL |
|---|---|---|
| EIA Form 861 | 线路里程、客户数 | https://www.eia.gov/electricity/data/eia861/ |
| NLCD Tree Canopy (USGS) | 树冠覆盖 | https://www.mrlc.gov/data |
| USDA Rural-Urban Codes | 城乡连续代码 | https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/ |
| NOAA HRRR (备选气象) | 3km 分辨率风场, 可替代 URMA | https://registry.opendata.aws/noaa-hrrr-pds/ |

---

## 10. 官方文档中的变量建模提示

> 以下非硬性约束,而是 variable_descriptions.pdf 中官方对各变量的建模建议,供特征工程参考。

**气象变量**:
| 变量 | 官方提示 |
|---|---|
| `gust` | "Strongest single predictor of outage onset" — 最强单预测因子 |
| `t2m` | "Determines rain vs snow/ice" — 决定降水相态 |
| `r2` | "High + near-freezing = icing risk" — 高湿度+近冰点=结冰风险 |
| `soil_moist` | "High values reduce tree root-holding strength, increasing wind-throw risk" — 高土壤湿度→树木倒伏风险 |
| `hcc` | "May precede fronts by 12–24h" — 高云可能提前12–24小时预示锋面 |
| `blh` | "High values indicate convective activity" — 高边界层=对流活动 |

**Lag 特征**:
| 变量 | 官方提示 |
|---|---|
| `outage_pct_lag1h` | "Strongest short-horizon predictor of current outage state" |
| `outage_pct_lag24h` | "Particularly informative for the two-storm structure; reveals county state during first storm wave" |
| `outage_pct_lag48h` | "For storm-onset hours, draws on the pre-event window and reveals baseline state" |
| `osi_lag1h` | "Richer than outage_pct_lag1h because it encodes flow history" |
| `osi_lag24h` | "Key for detecting two-storm interaction; high osi_lag24h means county already hit by first wave" |

**目标**:
| 变量 | 官方提示 |
|---|---|
| `osi_target_t48h` | "Most independent horizon (correlation with t+1h: 0.056)" — 需独立建模 |
| `osi_delta_t48h` | "Wider range reflects greater 48-hour uncertainty" |

---

## 11. 代码合规自检清单

写代码时逐条确认:

- [ ] **CAUSAL-1**: 所有停电特征输入仅来自 timestamp ≤ t 的行
- [ ] **CAUSAL-2**: 气象特征可来自任意 timestamp(含未来)
- [ ] **CAUSAL-3**: 未从未来行计算任何停电衍生量(包括 OSI 本身)
- [ ] **CAUSAL-4**: 未对测试集3月14–19的停电NaN做插值/填补
- [ ] **FEAT-1**: `severity_tier` 未用作输入特征(仅CV分层)
- [ ] **FEAT-2**: `peak_pct`, `peak_customers`, `time_to_restore_h` 未用作输入特征
- [ ] **FEAT-3**: `event_duration_h` 未用作输入特征(恒定无信息)
- [ ] **FEAT-4**: `wind_dir_10m` 已编码为 sin/cos, 未直接使用原始角度
- [ ] **FEAT-5**: 标识符列(timestamp_et, fipsCode, countyName, stateName, stateAbbr, split, in_event_window)未直接作为模型输入特征; `fipsCode`/`stateAbbr` 仅用作外部数据连接键(join key),拼接后的实际数值(线路里程、树冠覆盖率、城乡代码等)才进入模型
- [ ] **FEAT-6**: `customersTracked` 可用作输入特征(全程有值, 非停电计数); 注意它有小幅逐时变动, 使用行级值
- [ ] **FEAT-7**: OSI lag 特征在测试集中需从4个组件(P_t, N_t, D_t, R_t)自行重建(仅3月11–13部分); 重建遵守时间因果规则
- [ ] **TRAIN-1**: 训练样本构造严格模拟测试条件(仅用72h观测窗口停电数据+全窗口气象)
- [ ] **TRAIN-2**: 目标列为 NaN 的行已 drop, 不参与训练
- [ ] **TRAIN-3**: 预事件窗口(3月11–12)的停电行未零填充或排除(携带有效 lag 信号)
- [ ] **MODEL-1**: t+48h 与 t+1h 相关性仅0.056, 不宜简单外推, 需独立建模
- [ ] **SUBMIT-1**: 提交文件以 sample_submission.csv 为起点, 标识列和行顺序未改动
- [ ] **SUBMIT-2**: 超出3月19的目标行保持为 NaN
- [ ] **SUBMIT-3**: 仅填4个评分目标列, 无多余列
- [ ] **REPRO-1**: 代码可端到端复现, 无人工步骤
- [ ] **REPRO-2**: 包含库版本、随机种子、相对路径
- [ ] **REPRO-3**: 随机种子固定(建议 seed=42 与比赛划分一致)
- [ ] **REPORT-1**: 报告 ≤6页(不含参考文献)
