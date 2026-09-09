
本文档描述训练文件（DM_Train.csv，51,624行×63列）和测试文件（DM_Test.csv，13,608行×43列）中的每个变量。统计值基于训练数据计算。OSI公式及分量推导见 osi_methodology.pdf。气象变量来源与聚合方法见 weather_variables.pdf。

本文档所列所有变量在训练和测试文件中均存在，但有两个例外：（1）`split` 和 `severity_tier` 仅存在于训练文件中；（2）停电变量（outageCount、outage_pct、\(P_t\)、\(N_t\)、\(D_t\)、\(R_t\)、osi 及所有滞后列）在两个文件中均存在，但测试文件中3月14日至19日的对应值为NaN。气象变量和标识符在两个文件的所有216小时中均完整提供。OSI预测目标仅存在于训练文件中，测试文件中完全隐藏。

---

### 标识符与元数据

**请勿将这些列作为模型输入特征。**

| 列名 | 描述 |
|------|------|
| timestamp_et | 每小时时间戳（美国东部时间 EST，UTC-5）。范围从 2026-03-11 00:00 至 2026-03-19 23:00；共216个唯一时间戳，每县每小时一行。 |
| fipsCode | FIPS县级代码，唯一标识每个县。训练集239个县，测试集63个县，两文件无重叠。 |
| countyName | 县名称（可读形式）。302个县中共有187个唯一名称（部分名称在不同州重复）。 |
| stateName | 州全称：Indiana（印第安纳）、Ohio（俄亥俄）、Pennsylvania（宾夕法尼亚）、West Virginia（西弗吉尼亚）。 |
| stateAbbr | 两字母州缩写：IN、OH、PA、WV。 |
| in_event_window | False = 3月11-12日事件前滞后时段；True = 3月13-19日风暴窗口。 |
| split | 训练文件中始终为“TRAIN”。仅存在于训练文件。 |
| severity_tier | 州内县级严重等级（0-4），基于峰值OSI排名与平均停电持续时间得出。仅用于分层交叉验证，不作为模型输入特征。仅存在于训练文件。 |

---

### 原始停电变量

（注：以下变量在两个文件中均提供，但测试文件中3月14-19日值为NaN。）

| 列名 | 描述 |
|------|------|
| outageCount | 当前小时该县报告的用户停电数（原始计数）。 |
| customersTracked | 该县跟踪的用户总数（用作分母）。 |
| outage_pct | outageCount / customersTracked，即停电比例，截断至1.0。 |
| P_t | 当前停电分数（同 outage_pct），OSI状态分量。 |
| N_t | 新停电增长量：max(0, ΔoutageCount) / customersTracked，表示该小时新增停电比例。 |
| D_t | 6小时持续状态：rolling_mean(P_t, 6小时)，表示过去6小时平均停电分数。 |
| R_t | 主动恢复速率：max(0, -ΔoutageCount) / customersTracked，表示该小时减少的停电比例。 |
| osi | 复合OSI值，按公式 \(OSI_t = 0.40P_t + 0.35N_t + 0.25D_t - 0.10R_t\) 计算，并截断至≥0。 |

**滞后特征**：对于上述每个停电相关变量（outageCount、outage_pct、P_t、N_t、D_t、R_t、osi），均提供了1小时、3小时、6小时、24小时、48小时的滞后值，列名如 `outageCount_lag1h`、`P_t_lag24h` 等。

**OSI预测目标（仅训练文件）**：
- `osi_target_t01h`：t+1小时的OSI
- `osi_target_t06h`：t+6小时的OSI
- `osi_target_t24h`：t+24小时的OSI
- `osi_target_t48h`：t+48小时的OSI

测试文件中这些列为空（NaN）。

---

### 气象变量（URMA和ERA5）

以下所有气象变量在两个文件中全部216小时均可用。详细来源和单位见 weather_variables.pdf。

**URMA变量（2.5 km，县级多边形平均）**：
- `gust`：10米阵风（mph）
- `wind_speed_10m`：10米持续风速（mph）
- `wind_dir_10m`：10米风向（度，0-360）
- `t2m`：2米气温（K）
- `d2m`：2米露点温度（K）

**ERA5变量（~31 km，县中心点采样）**：
- `surface_pressure`：地面气压（hPa）
- `sea_level_pressure`：海平面气压（hPa）
- `boundary_layer_height`：边界层高度（m）
- `total_precipitation`：总降水量（mm）
- `liquid_precipitation`：液态降水量（mm）
- `snowfall`：降雪量（mm）
- `snow_depth`：雪深（m）
- `cloud_cover_total`：总云量（%）
- `cloud_cover_low`：低云量（%）
- `cloud_cover_mid`：中云量（%）
- `cloud_cover_high`：高云量（%）
- `shortwave_radiation`：短波辐射（W/m²）
- `relative_humidity`：相对湿度（%）
- `soil_moisture`：土壤湿度（m³/m³）
- `reference_evapotranspiration`：参考蒸散量（mm）

---

### 训练/测试划分统计

训练集239县，测试集63县，按州和严重等级分层划分。具体分布见 OSI_Methodology.pdf 中的表格。

