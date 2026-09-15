# Phase 1.5.2 (v1.5.2) 特征清单 — 141维

版本: v1.5.2 | 日期: 2026-09-03
来源: features.py (129维) + join_external.py (12维外部)

2026-09-08 更正：本目录保留 v1.5.2 的历史特征集合，已校正 top4、窗口点数与 NaN 的文字解释。修复后的完整 141 列数据集另存为 v1.5.5；OSI 精度、电力公司计数及数据入口改动见 [v1.5.5 变更记录](../versions/v1.5.5/README.md)。本文历史取值和实验结果不代表新版效果。

---

## 特征总览

| 类别 | 维数 | 来源 | 是否随时间变化 |
|---|---|---|---|
| A. 观测窗口停电摘要 | 31 | 3月11-13停电数据 | 否(每县固定) |
| B. 当前气象(t时刻) | 24 | URMA+ERA5原始+风向编码 | 是 |
| C. 目标时段气象统计 | 32 | [t,t+h]窗口聚合, 4horizon×8统计 | 是 |
| D. 派生气象 | 10 | 温度/结冰/交互等 | 是 |
| E. 气象变化率 | 6 | gust/pressure/temp差分 | 是 |
| F. 累计暴露 | 3 | 风暴开始以来累计 | 是 |
| G. 目标时刻精确气象 | 16 | t+h时刻的gust/wind/t2m/tp, 4horizon×4变量 | 是 |
| H. 时间特征 | 6 | 小时/天/风暴阶段 | 是 |
| I. 县级特征(原有) | 1 | log_customers | 是(微变) |
| J. 外部县级特征 | 12 | USDA/EIA/NLCD/TreeCanopy | 否(每县固定) |
| **合计** | **141** | | |

---

## A. 观测窗口停电摘要 (31维, 每县固定)

从3月11-13(hour_idx 0-71)的停电数据提取。对同一县的所有144个预测小时不变。

### A1. 末尾观测值 (hour_idx=71, 3月13日23:00)

| # | 特征 | 范围 | 含义 |
|---|---|---|---|
| 0 | `last_osi` | 0-0.55 | 观测窗口末尾OSI(持续性信号) |
| 1 | `last_P_t` | 0-0.82 | 末尾停电比例 |
| 2 | `last_N_t` | 0-0.07 | 末尾新增停电率 |
| 3 | `last_D_t` | 0-0.91 | 末尾6h滚动均值 |
| 4 | `last_R_t` | 0-0.06 | 末尾恢复率 |
| 5 | `last_outage_pct` | 0-82.4 | 末尾停电百分比 |

**OSI 分量结构与相关性分析**

OSI = 0.40×P_t + 0.35×N_t + 0.25×D_t − 0.10×R_t, 但四个分量对 OSI 的贡献严重不均:

| 分项 | 均值 | 占OSI均值 | 与OSI相关 | 零占比 |
|---|---|---|---|---|
| **0.40×P_t** | 0.00365 | **59.0%** | **0.995** | 43% |
| 0.35×N_t | 0.00034 | **5.6%** | 0.380 | 59% |
| **0.25×D_t** | 0.00230 | **37.1%** | **0.957** | 27% |
| 0.10×R_t | 0.00010 | **1.6%** | 0.473 | 52% |

P_t + D_t 占了 OSI 的 96.1%。N_t 和 R_t 合计仅 7.2%(且 R_t 是减项)。

**原因(物理必然, 非数据错误):**
- 停电有惯性: 一旦大面积停电, outageCount 变化缓慢(每小时新增/恢复仅占总量~10%)。所以 P_t(当前停电比例)和 D_t(6h均值)几乎相同(r=0.989)。
- 流量远小于存量: N_t 均值 0.001 vs P_t 均值 0.009 — 每小时新增停电仅为当前停电的~10%。R_t 同理。
- OSI 退化为"P_t 的放大版": OSI ≈ 0.40×P_t + 0.25×D_t ≈ 0.65×P_t(因为 D_t≈P_t)。

**N_t 和 R_t 携带了什么 OSI 没有的信息:**

虽然对 OSI 的数值贡献小, 但 N_t/R_t 编码的是**变化方向**:

| 场景 | P_t(存量) | N_t(恶化) | R_t(恢复) | OSI | 物理含义 |
|---|---|---|---|---|---|
| 停电刚开始 | 0.05 | 0.05 | 0 | 0.028 | 快速恶化中 |
| 停电持续 | 0.05 | 0.001 | 0.001 | 0.022 | 稳定状态 |
| 正在恢复 | 0.05 | 0 | 0.02 | 0.016 | 好转中 |

三个场景 P_t 相同(0.05), 但 OSI 差异不大(0.028 vs 0.022 vs 0.016), 而 N_t/R_t 能区分"恶化中"和"恢复中"。OSI 丢失了这个方向信息。



### A2. 72h窗口统计

| # | 特征 | 范围 | 含义 |
|---|---|---|---|
| 6 | `osi_mean_72h` | 0-0.27 | 72h平均OSI |
| 7 | `osi_max_72h` | 0-0.65 | 72h峰值OSI |
| 8 | `osi_std_72h` | 0-0.23 | 72h OSI标准差 |
| 9 | `outage_pct_mean_72h` | 0-39.6 | 72h平均停电% |
| 10 | `outage_pct_max_72h` | 0-100 | 72h峰值停电% |
| 11 | `N_t_mean_72h` | 0-0.019 | 72h平均恶化率 |
| 12 | `N_t_max_72h` | 0-0.379 | 72h最严重恶化时刻 |
| 13 | `R_t_mean_72h` | 0-0.017 | 72h平均恢复率 |
| 14 | `R_t_max_72h` | 0-0.187 | 72h最快恢复时刻 |

### A3. 预事件 vs 风暴起始

| # | 特征 | 范围 | 含义 |
|---|---|---|---|
| 15 | `pre_event_mean_osi` | 0-0.39 | 3月11-12平均OSI(2月遗留停电基线) |
| 16 | `pre_event_max_outage_pct` | 0-100 | 预事件峰值停电% |
| 17 | `storm_onset_max_osi` | 0-0.60 | 3月13峰值OSI(第一波) |
| 18 | `storm_onset_mean_osi` | 0-0.21 | 3月13平均OSI |

### A4. 趋势与模式

| # | 特征 | 范围 | 含义 |
|---|---|---|---|
| 19 | `osi_trend_last6h` | -0.06-0.02 | 末尾6h OSI线性斜率 |
| 20 | `is_worsening` | 0/1 | N_t R_t(正在恶化) |
| 21 | `frac_zero_72h` | 0-0.96 | 72h中OSI=0的小时占比 |
| 22 | `hours_since_peak` | 0-71 | 距72h内OSI峰值的小时数 |

### A5. 观测窗口气象摘要

| # | 特征 | 范围 | 含义 | 文献依据 |
|---|---|---|---|---|
| 23 | `gust_mean_obs` | 13-31 | 72h平均阵风 | — |
| 24 | `gust_max_obs` | 29-70 | 72h最大阵风 | — |
| 25 | `gust_gt30_obs` | 0-35 | 72h中gust>30mph小时数 | P1(Cerrai/Yang系列) |
| 26 | `gust_gt40_obs` | 0-18 | 72h中gust>40mph小时数 | P1 |
| 27 | `cumul_gust_obs` | 939-2251 | 72h累计风力暴露 | P5(Arora 2023) |
| 28 | `gust_last_obs` | 3.5-47.5 | 末尾阵风值 | P6(Alpay 2020) |
| 29 | `gust_max_last6h_obs` | 15-65 | 末尾6h最大阵风 | P6 |
| 30 | `gust_trend_last6h_obs` | -6.5-3.3 | 末尾6h阵风趋势 | P6 |

---

## B. 当前气象 (24维, 随t变化)

预测时刻t的气象变量。

### B1. URMA (2.5km, 多边形均值)

| # | 特征 | 范围 | 单位 | 含义 |
|---|---|---|---|---|
| 31 | `gust_t` | 0.6-58.3 | mph | 阵风(最强单预测因子) |
| 32 | `wind_speed_t` | 0-38 | mph | 持续风速 |
| 33 | `wind_sin_t` | -1~1 | — | 风向sin编码(圆形变量) |
| 34 | `wind_cos_t` | -1~1 | — | 风向cos编码 |
| 35 | `t2m_t` | 259-298 | K | 2m气温 |
| 36 | `d2m_t` | 257-290 | K | 2m露点 |

### B2. ERA5 (~31km, 县质心)

| # | 特征 | 范围 | 单位 | 含义 |
|---|---|---|---|---|
| 37 | `tp_t` | 0-11.3 | mm | 总降水 |
| 38 | `rain_t` | 0-11.3 | mm | 液态降雨 |
| 39 | `csnow_t` | 0-2.7 | cm | 降雪 |
| 40 | `sdwe_t` | 0-0.09 | m | 雪深(水当量) |
| 41 | `r2_t` | 15-100 | % | 相对湿度 |
| 42 | `vpd_t` | 0-2.5 | kPa | 蒸汽压差 |
| 43 | `soil_moist_t` | 0.17-0.52 | m³/m³ | 土壤湿度(树木根固力) |
| 44 | `blh_t` | 10-3090 | m | 边界层高度(对流活动) |
| 45-48 | `tcc_t`/`lcc_t`/`mcc_t`/`hcc_t` | 0-100 | % | 总/低/中/高云量 |
| 49 | `sp_t` | 885-1027 | hPa | 地面气压 |
| 50 | `mslma_t` | 987-1032 | hPa | 海平面气压 |
| 51 | `sdswrf_t` | 0-863 | W/m² | 短波辐射 |
| 52 | `direct_rad_t` | 0-737 | W/m² | 直接辐射 |
| 53 | `diffuse_rad_t` | 0-463 | W/m² | 散射辐射 |
| 54 | `et0_t` | 0-0.61 | mm/hr | 参考蒸散 |

---

## C. 目标时段气象统计 (32维, 随t和horizon变化)

对每个horizon h∈{1,6,24,48}, 从[t, t+h]窗口提取8个统计量。气象可用任意时刻(规则允许)。

每个horizon的8个统计:

| 统计 | 含义 | 文献依据 |
|---|---|---|
| `gust_max_next_{h}h` | 窗口最大阵风 | 三元组聚合(标准配方) |
| `gust_mean_next_{h}h` | 窗口平均阵风 | 三元组聚合 |
| `wind_speed_max_next_{h}h` | 窗口最大持续风速 | — |
| `total_tp_next_{h}h` | 窗口累计降水 | Cerrai事件总量 |
| `min_t2m_next_{h}h` | 窗口最低温度(结冰风险) | — |
| `gust_gt30_next_{h}h` | gust>30mph小时数 | P1(Cerrai/Yang系列, 最验证有效) |
| `gust_gt40_next_{h}h` | gust>40mph小时数 | P1(显著损害阈值) |
| `gust_peak4h_mean_next_{h}h` | 窗口内最高 min(4,有效点数) 个阵风值的均值，不要求连续 | 历史列名保留；不等于最大连续4小时均值 |

特征号: 65-96 (4 horizon × 8 统计 = 32维)

---

## D. 派生气象 (10维)

| # | 特征 | 范围 | 含义 | 文献依据 |
|---|---|---|---|---|
| 55 | `temp_c_t` | -14~25 | 摄氏温度(t2m-273.15) | 派生 |
| 56 | `dewpoint_c_t` | -16~17 | 摄氏露点(d2m-273.15) | 派生 |
| 57 | `icing_risk_t` | 0/1 | 温度<0°C且湿度>80% → 结冰 | 派生 |
| 58 | `gust_exceed_30` | 0-28 | 未来1h最大阵风超30mph幅度 | 派生 |
| 59 | `soil_moist_x_gust` | 0.2-24 | 土湿×风速(树木倒伏风险) | 派生交互 |
| 60 | `is_snowing` | 0/1 | 是否降雪 | 派生 |
| 61 | `gust_x_hso` | 1-3820 | 阵风×距观测小时数 | P4(拆解时间-气象共线性) |
| 62 | `gust_x_phase` | 0-108 | 阵风×风暴阶段 | P4 |
| 63 | `soil_gust_x_phase` | 0-47 | 土湿×阵风×阶段(三重交互) | P4 |
| 64 | `gust_exceed30_x_hso` | 0-1690 | 阵风超限×距观测小时数 | P4 |

---

## E. 气象变化率 (6维)

| # | 特征 | 范围 | 含义 | 文献依据 |
|---|---|---|---|---|
| 97 | `gust_change_1h` | -24~28 | 1h阵风变化 | P3(Alpay 2020, 互相关l=1) |
| 98 | `gust_change_3h` | -32~36 | 3h阵风变化 | P3 |
| 99 | `pressure_change_3h` | -6~11 | 3h气压变化(锋面信号) | P3 |
| 100 | `pressure_change_6h` | -11~15 | 6h气压变化 | P3 |
| 101 | `temp_change_6h` | -20~22 | 6h温变(冷锋) | P3 |
| 102 | `wind_change_1h` | -15~28 | 1h风速变化 | P3 |

---

## F. 累计暴露量 (3维)

| # | 特征 | 范围 | 含义 | 文献依据 |
|---|---|---|---|---|
| 103 | `cumul_gust_since_onset` | 347-4458 | 3/13以来累计风力 | P5(Arora 2023) |
| 104 | `cumul_tp_since_onset` | 0-53 | 3/13以来累计降水 | P5 |
| 105 | `gust_gt30_since_onset` | 0-61 | 3/13以来gust>30mph小时数 | P5 |

---

## G. 目标时刻精确气象 (16维)

对每个horizon h, 取t+h时刻的精确值(非窗口聚合)。

| # | 特征 | 范围 | 含义 | NaN% | 文献依据 |
|---|---|---|---|---|---|
| 106-109 | `gust_at_t{1,6,24,48}h` | 0.6-58.3 | 目标时刻精确阵风 | 0.7-33.3% | STO-CAST(SHAP: 未来精确>窗口统计) |
| 110-113 | `wind_speed_at_t{1,6,24,48}h` | 0-38 | 目标时刻精确风速 | 同上 | |
| 114-117 | `t2m_at_t{1,6,24,48}h` | 259-298 | 目标时刻精确温度 | 同上 | |
| 118-121 | `tp_at_t{1,6,24,48}h` | 0-11.3 | 目标时刻精确降水 | 同上 | |

NaN 出现在 t+h 超出数据末尾(3/19 23:00)时。仅相同 horizon 的目标也为 NaN、对应行不参与该 horizon 训练；其他 horizon 的有效训练行仍可能包含此列的 NaN，交由模型处理。聚合窗口则截短至数据末尾，不把未知气象填零。

---

## H. 时间特征 (6维)

| # | 特征 | 范围 | 含义 |
|---|---|---|---|
| 122 | `hour_of_day` | 0-23 | 小时 |
| 123 | `hour_sin` | -1~1 | 小时sin编码(日内周期) |
| 124 | `hour_cos` | -1~1 | 小时cos编码 |
| 125 | `days_since_onset` | 1-6 | 风暴开始(3/13)后天数 |
| 126 | `hours_since_obs` | 1-144 | 距最后观测(3/13 23:00)小时数(信息衰减) |
| 127 | `storm_phase` | 0-3 | 0=第一波(3/14), 1=间歇(3/15), 2=第二波(3/16-17), 3=消退(3/18-19) |

---

## I. 县级特征 (1维, 原有)

| # | 特征 | 范围 | 含义 |
|---|---|---|---|
| 128 | `log_customers` | 7.1-13.5 | log(1+customersTracked), 县规模 |

---

## J. 外部县级特征 (12维, 每县固定, v1.5.2新增)

通过 join_external.py 从 county_features.csv 拼接, 不重算已有特征。

### J1. 城市属性 (2维)

| # | 特征 | 范围 | 含义 | 数据源 | 与log_customers相关度 |
|---|---|---|---|---|---|
| 129 | `rural_urban_code` | 1-9 | 1=大都市→9=偏远农村 | USDA | 低-中(r≈0.3-0.5) |
| 130 | `is_metro` | 0/1 | 城市(code≤3) | USDA派生 | 低-中 |

### J2. 密度 (1维)

| # | 特征 | 范围 | 含义 | 数据源 | 与log_customers相关度 |
|---|---|---|---|---|---|
| 131 | `pop_density` | 8-11940 | 人口/县面积(人/sqmi) | USDA+Census | 低(r≈0.1-0.3) |

### J3. 电网效率 (2维)

| # | 特征 | 范围 | 含义 | 数据源 | 与log_customers相关度 |
|---|---|---|---|---|---|
| 132 | `n_utilities` | 1-20 | 县内电力公司数 | EIA Service_Territory | 低 |
| 133 | — | — | ~~line_miles_per_customer~~ | — | (EIA-861不含线路里程, 未实现) |

注: EIA Form 861 不含架空线里程数据, 故 `line_miles_per_customer` 未实现。`n_utilities` 是 EIA 的唯一可用特征。

### J4. 土地覆盖/植被 (7维)

| # | 特征 | 范围 | 含义 | 数据源 | 与log_customers相关度 |
|---|---|---|---|---|---|
| 133 | `pct_forest` | 0.6-88.7 | 森林占比 | NLCD土地覆盖 | **极低** |
| 134 | `pct_developed` | 2.2-86.6 | 已开发占比 | NLCD | 中(r≈0.3-0.5) |
| 135 | `pct_agriculture` | 0-87.5 | 农业占比 | NLCD | 低 |
| 136 | `pct_water` | 0-5.5 | 水体占比 | NLCD | 极低 |
| 137 | `pct_wetland` | 0-19.8 | 湿地占比 | NLCD | 极低 |
| 138 | `tree_canopy_pct` | 0.7-81.7 | 树冠覆盖率 | NLCD Tree Canopy | **极低** |
| 139 | `tree_canopy_std` | 6.9-40.4 | 县内树冠空间变异度 | NLCD Tree Canopy | 极低 |

### J5. 交互 (1维)

| # | 特征 | 范围 | 含义 | 计算 |
|---|---|---|---|---|
| 140 | `forest_x_customers` | 8-932 | 森林×客户数(树木倒伏暴露量) | pct_forest × log_customers |

---

## 共线性审查

### 完全冗余 (r=1.0, 数学恒等) — 6对

| 特征A | 特征B | 原因 | 影响 |
|---|---|---|---|
| `t2m_t` | `temp_c_t` | temp_c = t2m - 273.15 (线性变换) | 浪费2维 |
| `d2m_t` | `dewpoint_c_t` | dewpoint_c = d2m - 273.15 | 浪费2维 |
| `last_P_t` | `last_outage_pct` | outage_pct = P_t × 100 (仅差系数) | 浪费1维 |
| `osi_mean_72h` | `outage_pct_mean_72h` | OSI ≈ 0.40×outage_pct (近似线性) | 浪费1维 |
| `gust_mean_obs` | `cumul_gust_obs` | cumul = mean × 72 (窗口等长) | 浪费1维 |
| `gust_mean_next_1h` | `gust_peak4h_mean_next_1h` | h=1时窗口通常含 t、t+1 两点，末尾含一点；top4覆盖全部，等于mean | 浪费1维 |

**建议**: 下次改版时删除 temp_c_t, dewpoint_c_t, last_outage_pct, outage_pct_mean_72h, cumul_gust_obs, gust_peak4h_mean_next_1h (保留前者)。不影响LightGBM性能, 但节省6维采样预算。

### 近似冗余 (r>0.95) — 主要问题

| 特征组 | 典型 r | 问题 | 建议 |
|---|---|---|---|
| `last_osi` ↔ `last_P_t` ↔ `last_D_t` | 0.97-0.99 | 三者都是停电状态的度量, 高度共线 | 保留last_osi, 可删last_P_t和last_D_t |
| `days_since_onset` ↔ `hours_since_obs` ↔ `storm_phase` | 0.95-0.99 | 三个时间位置特征 | 保留hours_since_obs(最细粒度) |
| `pct_forest` ↔ `tree_canopy_pct` | 0.99 | 两个植被指标 | 保留pct_forest(更常用), tree_canopy可删 |
| `gust_max_next_{h}h` ↔ `gust_mean_next_{h}h` ↔ `gust_peak4h_mean_next_{h}h` | 0.93-1.0 | 短horizon下三者几乎相同 | 短horizon(h=1,6)保留gust_max即可 |
| `osi_max_72h` ↔ `osi_std_72h` | 0.98 | 峰值和标准差高度相关(只有峰值高才有大std) | 保留osi_max_72h |
| `storm_onset_max_osi` ↔ `storm_onset_mean_osi` | 0.97 | 3月13的max和mean | 保留storm_onset_max_osi |
| `N_t_mean_72h` ↔ `N_t_max_72h` ↔ `storm_onset_*` | 0.90-0.97 | 恶化率指标互相共线 | 保留N_t_mean_72h |

### 对LightGBM的影响

- **不损害RMSE**: LightGBM在分裂时随机选特征(feature_fraction=0.8), 高共线的特征只选一个用, 不影响预测精度
- **浪费采样预算**: 141维中约15-20维是冗余的, 20%采样丢的是有效信息位
- **特征重要性不可靠**: 共线特征的gain会被分散, 不能准确反映哪个维度真正重要
- **稳定性降低**: 不同fold可能选不同共线特征, 增大CV方差

### 结论

当前141维特征存在6对完全冗余和约15对近似冗余。对LightGBM不影响性能, 但浪费了约20维的采样预算。**不建议现在改**(会触发缓存重建+重训), 但在报告写作时应讨论此局限, 并在后续版本(如需进一步优化时)做特征剪枝。

---

## 冗余特征在 top 15 中的影响与逐对分析

### A. 冗余特征在 top 15 中的出现情况

#### t+1h — 问题最严重

top 15 中的共线对:

| 共线对 | r | gain A | gain B | 合计 | 真实维度 |
|---|---|---|---|---|---|
| `last_P_t` ↔ `last_osi` ↔ `last_outage_pct` | 0.997-1.0 | 23.1 | 19.4 | +4.9 = **47.4** | "末尾停电状态" |
| `hours_since_obs` ↔ `cumul_gust_since_onset` | 0.926 | 34.1 | 3.8 = **37.9** | | "时间位置" |
| `gust_x_hso` ↔ `soil_gust_x_phase` | 0.922 | 7.6 | 3.1 = **10.7** | | "阵风×时间交互" |

**gain 被分散后的真实排名 vs 显示排名:**

| 维度 | 真实合计 gain | 显示排名 | 显示中最高 gain |
|---|---|---|---|
| 末尾停电状态 | **47.4** | #2/3/5 (分散!) | 23.1 |
| 时间位置 | **37.9** | #1/6 (分散!) | 34.1 |
| 阵风×时间交互 | **10.7** | #4/7 (分散!) | 7.6 |

**结论: 显示的 #1(hours_since_obs=34.1) 是误导。** 真正 #1 应该是"末尾停电状态"(47.4), 但它的 gain 被三个 r≈1.0 的特征分摊了。如果只保留 `last_osi` 一个, 它的 gain 会接近 47.4, 超过 hours_since_obs。

#### t+6h — 与 t+1h 类似

top 15 中的共线对:

| 共线对 | r | 合计 gain |
|---|---|---|
| `last_P_t` ↔ `last_osi` ↔ `last_outage_pct` | 0.997-1.0 | 14.2+13.8+2.4 = **30.4** |
| `hours_since_obs` ↔ `cumul_gust_since_onset` | 0.926 | 25.7+3.7 = **29.4** |

同样: "停电状态"的真实 gain(30.4)略高于"时间"(29.4), 但显示排名中 hours_since_obs 排 #1。

#### t+24h — 影响中等

top 15 中的共线对:

| 共线对 | r | 合计 gain |
|---|---|---|
| `last_osi` ↔ `last_P_t` ↔ `last_D_t` | 0.976-0.997 | 2.9+0.5+0.4 = **3.8** |
| `cumul_gust_since_onset` ↔ `hours_since_obs` | 0.926 | 3.0+2.8 = **5.8** |

t+24h 中"时间"维度(cumul_gust + hso=5.8)仍高于"停电状态"(3.8), 此处时间确实更重要。

#### t+48h — 问题较轻

top 15 中的共线对:

| 共线对 | r | 合计 gain | 说明 |
|---|---|---|---|
| `pct_forest` ↔ `tree_canopy_pct` | 0.986 | 2.9+0.2=3.1 | 植被测量被分摊 |
| `gust_peak4h_mean_next_24h` ↔ `gust_mean_next_24h` | 0.945 | 0.6+0.4=1.0 | 24h阵风统计被分摊 |
| `gust_mean_next_48h` ↔ `gust_gt30_next_48h` | 0.887 | 3.7+0.3=4.0 | 48h阵风统计被分摊 |
| `gust_mean_next_48h` ↔ `gust_mean_next_24h` | 0.891 | (不同窗口) | 跨窗口共线, 见下方分析 |

t+48h 的 gain 分布更平(最高仅 3.7), 共线影响较小, 但 `pct_forest` 和 `tree_canopy_pct` 完全重复。

### B. 共线性如何影响特征重要性分析

**核心问题**: LightGBM 在每次分裂时随机选 80% 特征。当三个特征 r≈1.0 时, 模型在不同树/不同折叠中随机选不同的一个, 导致:

1. **gain 被分摊**: 同一维度的 gain 分散到多个共线特征上 → 单个特征的 gain 看起来比真实贡献低
2. **排名失真**: 真实最重要的维度可能因为 gain 分散而排不到 #1
3. **跨折不稳定**: 不同 CV fold 可能选不同的共线特征 → fold 间 gain 波动大(我们之前看到的 CV 变异系数 27-35% 部分来源于此)
4. **误导决策**: 我们之前根据"hours_since_obs #1"判断时间特征主导 → 实际上"停电状态"才是 #1, 时间只是 #2

### C. 每个近似冗余对分析

#### 第一类: 数学恒等 (r=1.0, 定义如此)

这些特征之间是纯线性变换关系, 不是巧合, 而是定义如此。

| 共线对 | 为什么 r=1.0 | 删谁 |
|---|---|---|
| `t2m_t` vs `temp_c_t` | temp_c = t2m - 273.15, 纯线性变换 | 删 temp_c_t |
| `d2m_t` vs `dewpoint_c_t` | dewpoint_c = d2m - 273.15 | 删 dewpoint_c_t |
| `last_P_t` vs `last_outage_pct` | outage_pct = P_t × 100, 仅差系数 | 删 last_outage_pct |
| `gust_mean_obs` vs `cumul_gust_obs` | cumul = mean × 72(窗口等长) | 删 cumul_gust_obs |
| `gust_mean_next_1h` vs `gust_peak4h_mean_next_1h` | h=1时通常含2值、末尾1值；top4覆盖全部，退化为mean | 删 peak4h_next_1h |
| `osi_mean_72h` vs `outage_pct_mean_72h` | OSI由P_t(权重0.40)主导 → 72h均值也近似线性 | 删 outage_pct_mean_72h |

#### 第二类: 设计必然 (r 0.92-1.0, 特征定义本身导致)

这些特征不是数学恒等, 但因为特征的定义方式(权重分配、物理惯性、低变异变量)导致高度相关。

| 共线对 | r | 为什么高相关 | 删谁 |
|---|---|---|---|
| `last_osi` ↔ `last_P_t` | 0.997 | OSI = 0.40×P_t + 0.35×N_t + 0.25×D_t - 0.10×R_t; P_t权重0.40且其他项小 → OSI ≈ 0.40×P_t | 删 last_P_t |
| `last_osi` ↔ `last_D_t` | 0.989 | D_t = rolling_mean(P_t, 6h); 停电有惯性(P_t变化慢) → D_t ≈ P_t → last_osi ≈ 0.40×D_t | 删 last_D_t |
| `pct_forest` ↔ `tree_canopy_pct` | 0.986 | 两者都是树密度度量(土地覆盖分类 vs 光谱分析), 底层物理量相同 | 删 tree_canopy_pct |
| `pct_forest` ↔ `forest_x_customers` | 0.986 | forest_x_customers = pct_forest × log_customers; log_customers变异小(CV=11%) → 交互 ≈ pct_forest × 10 ≈ pct_forest | 删 forest_x_customers |
| `days_since_onset` ↔ `hours_since_obs` | 0.986 | 两者都是t的线性函数(days=day-13, hso=hour_idx-71), 单调递增 | 删 days_since_onset |
| `hours_since_obs` ↔ `storm_phase` | 0.947 | storm_phase是hours_since_obs的粗粒度离散化 | 删 storm_phase |
| `hours_since_obs` ↔ `cumul_gust_since_onset` | 0.926 | 阵风恒正 → 累积和单调递增 → 与时间索引同步 | 删 cumul_gust_since_onset |
| `gust_x_hso` ↔ `gust_x_phase` | 0.949 | phase是hso的离散化 → gust×phase ≈ gust×(hso的粗版) | 删 gust_x_phase |
| `gust_x_hso` ↔ `soil_gust_x_phase` | 0.922 | soil_moist近常数(0.35±0.06) → soil×gust×phase ≈ 0.35×(gust×phase) ≈ 0.35×(gust×hso) | 删 soil_gust_x_phase |
| `gust_exceed_30` ↔ `gust_exceed30_x_hso` | 0.943 | hso在单个horizon内近常数 → 交互≈gust_exceed_30 × 常数 ≈ 线性放大 | 删 gust_exceed30_x_hso |
| `rural_urban_code` ↔ `is_metro` | 0.868 | is_metro = (code ≤ 3), 是code的离散化 | 删 is_metro |

#### 第三类: 物理相关 (r 0.85-0.95, 同一物理过程的不同度量)

这些特征测量的是同一物理现象的不同方面。相关性来自物理过程本身, 但它们并非完全冗余——在某些极端情况下可以提供不同信息。需要逐对判断是否保留。

| 共线对 | r | 为什么相关 | 是否冗余 | 处理 |
|---|---|---|---|---|
| `gust_max_next_h` ↔ `gust_mean_next_h` (h=1,6) | 0.93-1.0 | 短窗口(h≤6)内阵风变化幅度有限 → max≈mean | **冗余** | h=1,6删mean保留max(max信息更全); h=24,48保留两者(max测峰值强度, mean测持续水平, 物理含义不同) |
| `gust_peak4h_mean_next_h` ↔ `gust_mean_next_h` (h=1,6) | 0.93-0.99 | 短窗口时top4≈全部 → peak4h≈mean | **冗余**(同上, 短窗口退化) | h=1,6删peak4h; h=24,48保留(peak4h捕捉破坏性峰值, mean反映整体水平) |
| `gust_peak4h_mean_next_h` ↔ `gust_max_next_h` (h=1,6) | 0.93-0.99 | 短窗口时三者(max/mean/peak4h)趋同 | **冗余** | h=1,6只保留max; h=24,48保留max和peak4h(peak4h是4峰均值, 比单峰max更稳定) |
| `gust_mean_next_24h` ↔ `gust_mean_next_48h` | 0.891 | 24h是48h子集, 均值部分重叠 | **不冗余** | **都保留**: 24h可能只含第一波消退, 48h含第二波峰值 — 不同风暴阶段, 信息不同 |
| `gust_max_next_24h` ↔ `gust_max_next_48h` | 0.906 | 同上, 24h是48h子集 | **不冗余** | **都保留**: 理由同上, 不同窗口可能捕获不同风暴波次 |
| `gust_mean_next_48h` ↔ `gust_gt30_next_48h` | 0.887 | 平均高 → 超阈值小时数多 | **不冗余** | **都保留**: mean测平均强度(连续量), gt30测损害持续时间(计数), 物理含义不同; 极端情况: 短时强阵风(mean低但gt30高) vs 持续中等阵风(mean高但gt30也高) — 两种停电机制不同 |
| `gust_mean_next_24h` ↔ `gust_gt30_next_24h` | 0.864 | 同上 | **不冗余** | **都保留**: 理由同上 |
| `storm_onset_max_osi` ↔ `storm_onset_mean_osi` | 0.968 | 同一窗口的max和mean, 风暴起始OSI轨迹较平稳 | **冗余** | 保留storm_onset_max_osi(max更捕捉峰值破坏); 删mean |
| `N_t_mean_72h` ↔ `N_t_max_72h` | 0.965 | 平均高 → 峰值也高 | **冗余** | 保留N_t_mean_72h(mean更稳定); 删max |
| `osi_max_72h` ↔ `osi_std_72h` | 0.981 | 均值低时std由max决定 → max高=std大 | **冗余** | 保留osi_max_72h; 删std |
| `osi_max_72h` ↔ `outage_pct_max_72h` | 0.997 | OSI由P_t主导 → max也近似线性 | **冗余** | 保留osi_max_72h; 删outage_pct_max_72h |
| `osi_max_72h` ↔ `storm_onset_max_osi` | 0.890 | 3月13是72h窗口的后半段, 峰值常出现在3月13 | **不冗余** | **都保留**: osi_max_72h可能峰值在预事件(2月遗留), storm_onset_max_osi专测3月13 — 来源不同 |
| `pressure_change_3h` ↔ `pressure_change_6h` | 0.929 | 6h变化包含3h变化(嵌套窗口) | **冗余** | 保留pressure_change_6h(信息更全, 含更长趋势); 删3h |
| `tp_t` ↔ `rain_t` | 0.935 | tp = rain + snow; 3月事件以雨为主 → tp≈rain | **不冗余** | **都保留**: tp含降雪(冰冻停电机制), rain只含液态; 在近冰点时两者可以显著不同(csnow>0时tp≠rain) |
| `tp_t` ↔ `total_tp_next_1h` | 0.901 | 当前时刻降水 ≈ 未来1h累计(1h窗口=当前+下一小时) | **冗余** | 保留tp_t(更直接); total_tp_next_1h在h=1时与tp_t几乎相同 |
| `sdswrf_t` ↔ `direct_rad_t` | 0.925 | 总辐射含直接辐射 → 总高=直接高 | **不冗余** | **都保留**: 阴天时sdswrf≈diffuse(直接=0但总>0), 晴天时sdswrf≈direct — 云量信号不同 |
| `t2m_t` ↔ `d2m_t` | 0.861 | 温度和露点物理相关(都受气团控制) | **不冗余** | **都保留**: t2m-d2m = 薄球温度差, 是独立的物理量(湿度指标), Cerrai系列专门用比湿 |
| `wind_speed_t` ↔ `gust_t` | 0.901 | 持续风速高 → 阵风也高(同一风场) | **不冗余** | **都保留**: gust/ wind_speed = 阵风比(湍流强度), 物理含义不同; 雷暴gust远大于wind_speed, 温带风暴gust≈1.5×wind_speed |

### 总结

| 类别 | 对数 | 性质 | 处理 |
|---|---|---|---|
| 数学恒等 | 6对 | 纯线性变换, 定义如此 | 删后者 |
| 设计必然 | 11对 | 权重/惯性/低变异/离散化导致 | 删冗余者 |
| 物理相关-冗余 | 6对 | 同一过程, 信息重复 | 删冗余者 |
| 物理相关-保留 | 8对 | 同一过程, 但不同物理含义 | **都保留** |
