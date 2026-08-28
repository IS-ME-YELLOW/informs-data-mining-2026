**INFORMS 2026数据挖掘学会数据挑战赛——气象变量：来源、单位和聚合方法**

### 混合气象数据来源

气象特征来自两个互补来源，以平衡空间分辨率和变量广度：

| 来源 | 描述 |
|------|------|
| NOAA URMA（2.5 km） | 风和温度。非受限中尺度分析（Unrestricted Mesoscale Analysis）是2.5 km分辨率、经观测订正的分析场。用于驱动停电的高价值风和温度场。按县级多边形提取平均值（使用完整2.5 km分辨率，而非单点）。这修正了粗分辨率再分析资料中常见的风速阵风夸大伪影。 |
| Open-Meteo 提供的 ERA5（~31 km） | URMA未提供的补充场（云、辐射、降水相态、雪、土壤湿度、气压）。在县中心点采样。许可证：CC BY 4.0。 |

两种数据源均覆盖3月11日至19日全时段，风暴开始时无分辨率中断。

---

### URMA字段（2.5 km，多边形平均）

| 列名 | URMA字段 | 单位 | 描述 |
|------|----------|------|------|
| gust | GUST:10m | mph | 10米阵风 |
| wind_speed_10m | WIND:10m | mph | 10米持续风速 |
| wind_dir_10m | WDIR:10m | ° | 10米风向 |
| t2m | TMP:2m | K | 2米气温 |
| d2m | DPT:2m | K | 2米露点温度 |

注：URMA风场（阵风、持续风速）从原始m/s转换为mph（乘以2.2369）以符合美国业务惯例。风向单位为度；温度和露点保持为开尔文。

---

### 关于风向的编码重要提示

`wind_dir_10m` 是圆形变量，范围0-360°，其中359°和1°几乎指向同一方向（均接近正北），但在数值上相差358度。标准模型会错误地将其视为线性变量。正确编码方式：

\[
\text{wind\_sin} = \sin\left(\frac{2\pi \times \text{wind\_dir}}{360}\right), \quad
\text{wind\_cos} = \cos\left(\frac{2\pi \times \text{wind\_dir}}{360}\right)
\]

请使用 `wind_sin` 和 `wind_cos` 作为特征，而非原始度数。

---

### ERA5字段（~31 km，县中心点）

以下变量通过Open-Meteo获取：

- `surface_pressure`（地面气压，hPa）
- `sea_level_pressure`（海平面气压，hPa）
- `boundary_layer_height`（边界层高度，m）
- `total_precipitation`（总降水量，mm）
- `liquid_precipitation`（液态降水量，mm）
- `snowfall`（降雪量，mm）
- `snow_depth`（雪深，m）
- `cloud_cover_total`（总云量，%）
- `cloud_cover_low`（低云量，%）
- `cloud_cover_mid`（中云量，%）
- `cloud_cover_high`（高云量，%）
- `shortwave_radiation`（短波辐射，W/m²）
- `relative_humidity`（相对湿度，%）
- `soil_moisture`（土壤湿度，m³/m³）
- `reference_evapotranspiration`（参考蒸散量，mm）

---

### 数据源链接

| 来源 | 分辨率 | URL |
|------|--------|-----|
| NOAA URMA (AWS S3: noaa-urma-pds) | 2.5 km | https://registry.opendata.aws/noaa-rtma/ |
| Open-Meteo ERA5 | ~31 km | https://open-meteo.com/en/docs/historical-weather-api |
| NOAA HRRR（备选） | 3 km | https://registry.opendata.aws/noaa-hrrr-pds/ |

**静态基础设施特征（可选增强）**：

| 来源 | URL |
|------|-----|
| EIA Form 861：线路英里、用户数 | https://www.eia.gov/electricity/data/eia861/ |
| NLCD 树冠覆盖率（USGS） | https://www.mrlc.gov/data |
| USDA 城乡编码 | https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/ |

---

### 关于分辨率的说明

URMA阵风经观测订正，分辨率2.5 km，适合捕捉驱动停电的精细风场。ERA5场较粗（~31 km），适用于缓慢变化的大气背景变量（云、气压、湿度），但不适于精细风场；后者来自URMA。

---

### 时间对齐注意事项

URMA时间戳从UTC转换为固定偏移-5小时（EST）。通过Open-Meteo获取的ERA5数据使用本地时间 America/New_York，事件窗口期间（由于2026年3月8日开始夏令时）实际为EDT（-4小时）。两个序列均为逐小时密集序列，行标签相同，因此给定标签小时的URMA和ERA5值可能描述物理条件最多相差一小时。对于ERA5的缓慢变化背景场（如气压、湿度、土壤湿度）这种偏移无关紧要，不影响URMA风变量；但对于ERA5降水场（tp、rain）在风暴活跃期变化迅速时，可能略有影响。

---