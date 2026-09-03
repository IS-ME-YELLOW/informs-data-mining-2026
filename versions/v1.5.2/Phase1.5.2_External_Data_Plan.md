# 外部数据引入方案 (Phase 1.5.2)

> 基于文献调查报告中14篇论文的外部数据使用实践,
> 结合 Phase 1.5.1 的特征重要性分析(log_customers 是 t+48h #1 特征)制定。
> 日期: 2026-09-01

---

## 一、为什么需要外部数据

### 1.1 当前瓶颈

Phase 1.5.1 的特征重要性显示:

| Horizon | #1 特征 | gain | 含义 |
|---|---|---|---|
| t+1h | hours_since_obs | 28.8 | 信息衰减(时间) |
| t+48h | **log_customers** | **2.2** | **县规模(唯一的县级特征)** |

`log_customers` 在 t+48h 排名第一 — 说明**县级差异是长期预测的核心信号**。但我们目前只有 `log_customers` 一个县级特征(32个静态特征中其余都是观测窗口停电统计)。

### 1.2 文献证据

| 论文 | 外部数据 | 增益 | 关键发现 |
|---|---|---|---|
| McRoberts 2018 | 海拔+土地覆盖+土壤+降水+植被 | **+17%** | 环境因子显著提升精度 |
| Yang 2020b | SumAssets(基础设施数) | "通常是最重要变量" | 基础设施密度是核心 |
| Lee 2025 | 县人口 | **排名第一**, 超过所有气象变量 | 人口规模是主导(⚠️对应我们的log_customers, 非城市化梯度) |
| D'Amico 2019 | 树种+修剪频率 | 修剪频率#2, 树种+3% | 植被管理比树种更重要 |
| Cerrai 2019/2020 | NLCD土地覆盖+LAI | 一贯使用 | PercDeveloped/PercDecid/PercConif |
| Arora 2023 | 树木覆盖%+根区深度+土壤 | 关键变量 | 树木覆盖是飓风停电的主要调节因子 |

### 1.3 核心逻辑

停电 = f(气象驱动力, **县级脆弱性**)

我们已有丰富的气象驱动力特征(129维), 但县级脆弱性只有 `log_customers`。外部数据补充:
- 基础设施密度(线路越多→暴露越多)
- 植被覆盖(树越多→风倒树风险越高)
- 城市化程度(城市→地下电缆/维护好, 农村→架空线/维护弱)

---

## 二、数据源与特征设计

### 设计原则: 避免与 log_customers 共线

`log_customers` 已在模型中独占"规模"维度。引入外部数据时**只加捕获不同维度的特征**, 不加与规模共线的原始计数变量, 否则会:
1. 稀释特征重要性(模型随机选共线特征之一, gain分散)
2. 浪费 feature_fraction 配额(冗余特征挤占有效特征位置)
3. 增大CV不稳定性

| 维度 | 当前特征 | 引入的 | 不引入(共线) |
|---|---|---|---|
| 规模 | log_customers ✅ | — | ❌ population, customer_count, overhead_line_miles |
| 密度 | — | ✅ pop_density | ❌ customer_density(与pop_density几乎相同) |
| 城市属性 | — | ✅ rural_urban_code, is_metro | ❌ is_rural(is_metro的逆,冗余) |
| 电网效率 | — | ✅ line_miles_per_customer, n_utilities | — |
| 植被 | — | ✅ pct_forest, tree_canopy_pct | — |
| 土地利用 | — | ✅ pct_developed, pct_agriculture | — |

### 2.1 USDA Rural-Urban Continuum Codes

**数据源**: https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/
**格式**: Excel, 每县一行, 含 FIPS + code (1-9)
**获取难度**: 极低(下载即用)

| 特征 | 计算 | 文献依据 | 物理含义 | 与log_customers相关度 |
|---|---|---|---|---|
| `rural_urban_code` | 原始值 1-9 | Cerrai系列 PercDeveloped (间接); ⚠️Lee 2025的"人口#1"对应log_customers而非此变量 | 1=大都市核心 → 9=偏远农村 | 低-中(r≈0.3-0.5) |
| `is_metro` | code ≤ 3 ? 1 : 0 | Cerrai系列 PercDeveloped | 城市 vs 非城市(地下电缆vs架空线) | 低-中 |

### 2.2 EIA Form 861 — 电力基础设施

**数据源**: https://www.eia.gov/electricity/data/eia861/
**格式**: Excel, 按电力公司上报, 需按县聚合
**获取难度**: 中(需筛选4州 + 县级聚合)

| 特征 | 计算 | 文献依据 | 物理含义 | 与log_customers相关度 |
|---|---|---|---|---|
| `line_miles_per_customer` | overhead_line_miles / customers | D'Amico 2019 (customer density) | 每客户架空线长度(电网效率: 长线少客=农村脆弱) | 中(r≈0.4-0.6, 但比值维度不同) |
| `n_utilities` | 县内电力公司数量 | — | 管理复杂度(多公司协调恢复慢) | 低 |

> 注: ❌ 不加 `overhead_line_miles` 原始值(与 customers 共线 r≈0.7-0.8); ❌ 不加 `customer_density`(与 pop_density 几乎相同)
>
> EIA Form 861 中 overhead_line_miles 是按电力公司上报的。需要按电力公司→县服务区域映射聚合, 或用 customers 按比例分配。

### 2.3 NLCD 土地覆盖 (GIS库已就位)

**数据源**: https://www.mrlc.gov/data (NLCD 2019 Land Cover, 30m栅格)
**格式**: GeoTIFF栅格, 用 rasterstats.zonal_stats() 按县边界做分区统计
**获取难度**: 中(rasterio 1.4.3 + rasterstats 0.21.0 + geopandas 1.0.1 已安装)

| 特征 | 计算 | 文献依据 | 物理含义 | 与log_customers相关度 |
|---|---|---|---|---|
| `pct_forest` | (落叶林+针叶林+混合林) / 县面积 | Cerrai系列 PercDecid/PercConif | 树木倒伏风险(风倒树) | **极低** |
| `pct_developed` | (已开发4级) / 县面积 | Cerrai系列 PercDeveloped | 城市化程度(地下电缆+维护好) | 中(r≈0.3-0.5) |
| `pct_agriculture` | (耕地+牧场) / 县面积 | — | 开阔地形(风无遮挡但无树) | 低 |
| `tree_canopy_pct` | 树冠覆盖率(NLCD Tree Canopy产品) | Arora 2023 (树木覆盖%) | 风倒树直接风险(比pct_forest更精确) | **极低** |

### 2.4 US Census — 县面积与人口 (辅助)

**数据源**: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html
**格式**: CSV, 每县一行, 含 FIPS + 人口 + 土地面积
**获取难度**: 极低

| 特征 | 计算 | 文献依据 | 物理含义 | 与log_customers相关度 |
|---|---|---|---|---|
| `pop_density` | population / county_area | 派生(非Lee直接验证) | 人口集中度(影响维护响应速度) | 低(r≈0.1-0.3, 小县可能密集也可能稀疏) |

> 注: ❌ 不加 `population` 原始值(与 log_customers 共线, 美国电力覆盖率>99%, 人口≈客户数); ❌ 不加 `county_area_sqmi` 作为独立特征(仅用于计算 pop_density)

---

## 三、获取方式与代码设计

### 3.1 文件结构

```
INFORMS_DATA_MINING/
├── external_data/              # 外部数据存放目录
│   ├── raw/                    # 原始下载文件
│   │   ├── rural_urban_codes.csv
│   │   ├── eia861_county.xlsx
│   │   ├── census_county_pop.csv
│   │   └── nlcd_county_stats.csv  (或手动导出)
│   └── county_features.csv     # 拼接后的县级特征(每县一行, 含fipsCode)
├── phase1/
│   ├── features.py             # 修改: 加载county_features并拼接
│   └── config.py               # FEATURE_VERSION = 'v1.5.2'
```

### 3.2 获取步骤(人工+脚本)

**步骤1: USDA Rural-Urban Codes (10分钟)**
1. 访问 https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/
2. 下载 Excel → 保存为 `external_data/raw/rural_urban_codes.csv`
3. 筛选 IN/OH/PA/WV 四州, 保留 FIPS + Description + Code 列

**步骤2: US Census 人口与面积 (10分钟)**
1. 访问 https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html
2. 下载 County Population Totals → 保存为 `external_data/raw/census_county_pop.csv`
3. 保留 FIPS + Population + Land Area 列

**步骤3: EIA Form 861 (1-2小时)**
1. 访问 https://www.eia.gov/electricity/data/eia861/
2. 下载 `Sales_Utility_County.xlsx` → `external_data/raw/eia861_county.xlsx`
3. 筛选 IN/OH/PA/WV, 按县聚合: customers总和、utility数量
4. 下载 `Utility_Data.xlsx` 获取 overhead vs underground 线路里程

**步骤4: NLCD 土地覆盖 (处理方式待定)**
- 方案A: 用 MRLC 统计工具手动导出4州302县的统计 → `nlcd_county_stats.csv`
- 方案B: 写 Python 脚本用 rasterio 读取 NLCD 栅格 + county shapefile 做 zonal stats
- 方案C(最简): 用 USDA NASS Quick Stats 的县级 "Land in Farms" 作为农业代理

### 3.3 拼接脚本设计

```python
# external_data/build_county_features.py
# 职责: 读取 raw/ 下各数据源, 按fipsCode拼接, 输出 county_features.csv

def build_county_features():
    """拼接所有外部数据源 → county_features.csv (每县一行)"""
    # 1. 获取全部302个县的fipsCode (从DM_Train+DM_Test)
    all_fips = get_all_fips()  # 302个

    # 2. USDA Rural-Urban Codes
    ru = pd.read_csv('raw/rural_urban_codes.csv')
    ru = ru.rename(columns={'FIPS': 'fipsCode', ...})

    # 3. Census 人口与面积
    census = pd.read_csv('raw/census_county_pop.csv')

    # 4. EIA Form 861 (县级聚合)
    eia = pd.read_excel('raw/eia861_county.xlsx')
    eia_agg = eia.groupby('fipsCode').agg({'customers': 'sum', ...})

    # 5. NLCD 土地覆盖 (如有)
    nlcd = pd.read_csv('raw/nlcd_county_stats.csv')

    # 6. 拼接
    df = merge_all(all_fips, ru, census, eia_agg, nlcd)
    df.to_csv('county_features.csv', index=False)
```

### 3.4 features.py 改动

在 `build_feature_matrix()` 中, 加载 `county_features.csv` 并按 fipsCode 拼接:

```python
# 在 build_feature_matrix 开头加载县级外部特征
county_ext = pd.read_csv(os.path.join(DATA_DIR, 'external_data', 'county_features.csv'))

# 在 for fips, county_df 循环中:
county_row = county_ext[county_ext['fipsCode'] == int(fips)].iloc[0]
ext_feat = {
    # 城市属性(非规模)
    'rural_urban_code': county_row['rural_urban_code'],
    'is_metro': county_row['is_metro'],
    # 密度(非规模)
    'pop_density': county_row['pop_density'],
    # 电网效率(非规模)
    'line_miles_per_customer': county_row['line_miles_per_customer'],
    'n_utilities': county_row['n_utilities'],
    # 植被(全新维度)
    'pct_forest': county_row['pct_forest'],
    'pct_developed': county_row['pct_developed'],
    'pct_agriculture': county_row['pct_agriculture'],
    'tree_canopy_pct': county_row['tree_canopy_pct'],
    # 交互
    'forest_x_customers': county_row['pct_forest'] * county_row['log_customers'],
}
feat.update(ext_feat)
```

### 3.5 新增特征列名

```python
# M. 外部县级特征 (文献: McRoberts 2018 +17%, Yang 2020b SumAssets)
# 设计原则: 只引入与log_customers捕获不同维度的特征, 不加共线的原始计数变量
EXTERNAL_FEATURES = [
    # 维度: 城市属性(非规模)
    'rural_urban_code',       # 1-9序数, 与log_customers r≈0.3-0.5
    'is_metro',               # 二值, 城市vs非城市(地下电缆vs架空线)
    # 维度: 密度(非规模)
    'pop_density',            # population/area, 与log_customers r≈0.1-0.3
    # 维度: 电网效率(非规模)
    'line_miles_per_customer', # 架空线/客户, 比值不共线
    'n_utilities',            # 电力公司数量
    # 维度: 植被(全新, 与log_customers几乎不共线)
    'pct_forest',             # 森林占比
    'pct_developed',          # 已开发占比
    'pct_agriculture',        # 农业占比
    'tree_canopy_pct',        # 树冠覆盖率(比pct_forest更精确)
    # 维度: 交互(捕获"规模×植被"等复合效应)
    'forest_x_customers',     # pct_forest × log_customers (有多少客户暴露在森林中)
]
# ❌ 不引入: population(与log_customers共线), overhead_line_miles(与customers共线),
#           customer_density(与pop_density共线), is_rural(is_metro的逆),
#           county_area_sqmi(仅用于计算pop_density, 不独立入模)
```

---

## 四、合规要点

1. **fipsCode 仅作连接键**: 不作为模型输入特征, 仅用于拼接外部数据 → 合规(FEAT-5)
2. **外部数据是静态县级特征**: 不涉及时间因果规则, 不使用未来停电信息 → 合规(CAUSAL-1~4)
3. **不使用 severity_tier**: 外部数据与 severity_tier 无关 → 合规(FEAT-1)
4. **可复现性**: `external_data/build_county_features.py` 脚本化, 数据源URL公开 → 合规(REPRO-1~2)

---

## 五、预期影响

### 5.1 基于文献的增益估计

| 数据源 | 文献增益 | 预期对我们 |
|---|---|---|
| 基础设施(EIA) | Yang 2020b: "最重要变量" | 中(t+48h 的 log_customers 已#1, EIA 补充更细粒度) |
| 城市化(USDA Rural-Urban) | Cerrai系列PercDeveloped(间接); ⚠️Lee 2025的"人口#1"对应log_customers非此变量 | 低(0.5-2%, 与log_customers部分共线) |
| 人口(Census) | Lee 2025: #1特征 | 已有(log_customers覆盖); pop_density增量小 |
| 土地覆盖/植被(NLCD) | McRoberts 2018: +17%; D'Amico: +3% | 中-高(GIS库已就位, 可直接处理栅格) |
| 全部组合 | McRoberts 2018: +17% | 3-10% (我们已有 log_customers, 边际递减) |

### 5.2 对不同 horizon 的预期

| Horizon | 当前瓶颈 | 外部数据预期 |
|---|---|---|
| t+1h | 持续性信号强 | 低(短期靠last_osi) |
| t+6h | 持续性+气象 | 低-中 |
| t+24h | **弱(仅+24% vs Zero)** | **中-高**(县级脆弱性差异在长期预测中更重要) |
| t+48h | **弱(仅+11% vs Zero)** | **中-高**(同上, log_customers已#1) |

外部数据对长期 horizon(t+24h, t+48h)的预期收益最大。

---

## 六、执行计划

| 步骤 | 内容 | 预估时间 | 依赖 |
|---|---|---|---|
| 1 | 下载 USDA Rural-Urban + Census 数据 | 20分钟 | 无 |
| 2 | 写 build_county_features.py 拼接脚本 | 30分钟 | 步骤1 |
| 3 | 下载 EIA Form 861 + 县级聚合 | 1-2小时 | 无 |
| 4 | 获取 NLCD 县级统计(GIS库已就位: rasterio+rasterstats+geopandas) | 1-2小时 | 无 |
| 5 | 更新 features.py: 加载+拼接外部特征 | 30分钟 | 步骤2-4 |
| 6 | 更新 config.py: FEATURE_VERSION='v1.5.2' | 1分钟 | — |
| 7 | 运行 + 对比 v1.5.1 | 20分钟 | 步骤5-6 |
| 8 | 更新 Results.md + 日志 | 10分钟 | 步骤7 |

### 优先级

1. **先做 USDA + Census**(最简单, 10分钟下载, 立即可用)
2. **再做 EIA**(中等难度, 基础设施是文献中"最重要变量")
3. **最后做 NLCD**(最复杂, 但McRoberts证明增益最大)

每个数据源加入后可独立测试, 观察CV RMSE变化, 逐步叠加。


### 统计代码
```python
$ cd "/home/yongyi.xie/INFORMS_DATA_MINING" && python3 -c "
import numpy as np, pandas as pd

X = pd.read_parquet('cache/features_train_v1.5.1.parquet')
y = pd.read_parquet('cache/targets_train_v1.5.1.parquet')
meta = pd.read_parquet('cache/meta_train_v1.5.1.parquet')

# 对每个horizon: 分析县级方差 + log_customers的解释力 + 剩余空间
print('=== 县级方差分析: 外部数据的增益空间 ===')
print()
for h in ['osi_target_t01h','osi_target_t06h','osi_target_t24h','osi_target_t48h']:
    tgt = y[h].values
    valid = ~np.isnan(tgt)
    fips = meta['fipsCode'].values[valid]
    tgt_v = tgt[valid]
    log_c = X['log_customers'].values[valid]
    
    total_var = np.var(tgt_v)
    
    county_data = {}
    for f, v, lc in zip(fips, tgt_v, log_c):
        county_data.setdefault(f, {'tgt':[], 'lc':lc})
        county_data[f]['tgt'].append(v)
    
    county_means = np.array([np.mean(d['tgt']) for d in county_data.values()])
    county_lcs = np.array([d['lc'] for d in county_data.values()])
    
    between_var = np.var(county_means)
    icc = 100 * between_var / total_var
    
    # log_customers的R²
    coeffs = np.polyfit(county_lcs, county_means, 1)
    pred = np.polyval(coeffs, county_lcs)
    ss_res = np.sum((county_means - pred)**2)
    ss_tot = np.sum((county_means - county_means.mean())**2)
    r2_logc = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    # 也看gust_mean_obs的R²
    gmo = X['gust_mean_obs'].values[valid]
    county_gmos = {}
    for f, v in zip(fips, gmo):
        county_gmos.setdefault(f, []).append(v)
    county_gmo = np.array([np.mean(v) for v in county_gmos.values()])
    coeffs2 = np.polyfit(county_gmo, county_means, 1)
    pred2 = np.polyval(coeffs2, county_gmo)
    ss_res2 = np.sum((county_means - pred2)**2)
    r2_gmo = 1 - ss_res2/ss_tot if ss_tot > 0 else 0
    
    # 两个合在一起
    X_combined = np.column_stack([county_lcs, county_gmo])
    X_combined = np.column_stack([np.ones(len(county_lcs)), X_combined])
    coeffs3 = np.linalg.lstsq(X_combined, county_means, rcond=None)[0]
    pred3 = X_combined @ coeffs3
    ss_res3 = np.sum((county_means - pred3)**2)
    r2_both = 1 - ss_res3/ss_tot if ss_tot > 0 else 0
    
    remaining = 100 - r2_both
    print(f'{h}:')
    print(f'  总方差的{icc:.1f}%来自县级差异')
    print(f'  其中 log_customers 解释 {100*r2_logc:.1f}%')
    print(f'  其中 gust_mean_obs 解释 {100*r2_gmo:.1f}%')
    print(f'  两者合计解释 {100*r2_both:.1f}%')
    print(f'  → 外部数据可争取的空间: 县级方差的{remaining:.1f}%')
    print(f'  → 占总方差的 {icc*remaining/100:.1f}%')
    print()
```

我用方差分解分析了当前模型对县级差异的覆盖:
Horizon	县级差异占总方差	log_customers + gust_mean_obs 已解释	未解释
t+1h	20.4%	5.3%	95%
t+6h	20.4%	4.3%	96%
t+24h	21.3%	2.5%	98%
t+48h	25.2%	3.7%	96%

结论: 尽管模型有 32 个"静态"特征,但它们几乎全是观测窗口的停电统计,不包含真正的县级地理/基础设施/植被信息。log_customers 虽是 t+48h gain #1,但仅解释了县级方差的 3.3%——外部数据面对的是几乎空白的 20-25% 方差空间。

### 三类数据的预期增益
1. USDA Rural-Urban Codes (3特征)
捕获维度: 城市化梯度(1=大都市, 9=偏远农村)
独有信息:
- 城市→地下电缆(风不直接影响), 维护预算高
- 农村→架空线(直接暴露于风), 维护弱, 树木多
- 这与 log_customers 不完全共线: 一个人口 5 万的县可能是城市核心(地下电缆),也可能是农村分散(架空线)
- ⚠️ **纠错**: 原引"Lee 2025 — 人口在置换重要性中排名第一"有误。Lee 2025 的"人口#1"对应的是 county population(即我们的 log_customers), **不是**城市化梯度。USDA Rural-Urban Code 的文献依据应改为: Cerrai 系列用 PercDeveloped 作为城市化代理(间接支持), Lee 2025 的证据实际支持 log_customers 而非此变量。
预期增益:
- 单一序数变量(1-9), 可能解释县级方差的 5-15%
- 对总方差: 0.5-2%
- RMSE 改善: 0.5-2% **(修正下调: 原1-3% → 0.5-2%)**
- 最大受益 horizon: t+24h/t+48h(县级差异更重要)

2. EIA Form 861 (4特征)
捕获维度: 电力基础设施物理暴露量
独有信息:
- 架空线总里程 → 风倒树的直接物理暴露面(线路越长,树倒在线上的概率越高)
- 客户密度 → 与 Census pop_density 类似但专门针对电力客户
- line_miles_per_customer → 电网效率(长线少客=农村脆弱, 短线多客=城市健壮)
- n_utilities → 管理复杂度(多公司协调恢复慢)
文献: Yang 2020b — "SumAssets(基础设施数)通常是最重要变量"; Cerrai 系列用 cntPole/sumIso
与现有特征的关系:
- line_miles 与 customers 相关(r≈0.7-0.8), 但 line_miles_per_customer 是新维度
- 类似于"总量 vs 效率"的区别——两个客户数相同的县, 架空线密度可以差几倍
预期增益:
- 在 log_customers 基础上增量解释县级方差的 10-20%
- 对总方差: 2-4%
- RMSE 改善: 2-5%
3. NLCD 土地覆盖 (5-6特征)
捕获维度: 植被密度与土地利用结构——这是当前模型完全缺失的维度
独有信息:
- pct_forest → 树木密度,风倒树风险的直接度量
- pct_developed → 城市化(地下电缆+维护好)
- pct_agriculture → 开阔地形(风无遮挡但无树)
- tree_canopy_pct → 树冠覆盖(比 forest % 更精确)
与现有特征的关系:
- 几乎不共线 — 植被密度与客户数/停电历史无直接关系
- 两个客户数、gust 完全相同的县, 如果一个 60% 森林、一个 5% 森林, 停电严重度可以差几倍
- 这正是"县级脆弱性"的核心: 同样的风, 对树多的县杀伤大得多
文献:
- McRoberts 2018: 加入环境因子(海拔+土地覆盖+土壤+植被) → +17% 精度提升
- Cerrai 系列: PercDeveloped/PercDecid/PercConif 在所有模型中一贯使用
- D'Amico 2019: 树种仅 +3%, 但土地覆盖已在模型中才能让树种增益降到 3%——说明 land cover 本身贡献远大于 3%
- Arora 2023: 树木覆盖%是飓风停电的主要调节因子
预期增益:
- 解释县级方差的 20-40%(新维度,信息量大)
- 对总方差: 4-10%
- RMSE 改善: 3-8%
- 最大受益 horizon: 全部, 尤其是 t+24h/t+48h


三类合计预期
数据源	县级方差解释(估计)	总方差	RMSE改善	依据
USDA Rural-Urban	5-15%	0.5-2%	0.5-2%	Cerrai系列PercDeveloped(间接) ⚠️修正: Lee 2025对应log_customers而非此变量
EIA Form 861	10-20% (增量)	2-4%	2-5%	Yang 2020b
NLCD Land Cover	20-40%	4-10%	3-8%	McRoberts 2018 +17%
三者合计	35-55%	7-16%	5-10%	McRoberts 2018 综合+17%

不同 horizon 的增益分布
Horizon	当前 RMSE	县级ICC	外部数据可争取	预期改善后
t+1h	0.0126	20.4%	小(短期靠持续性)	0.012-0.0124
t+6h	0.0112	20.4%	小-中	0.0105-0.0110
t+24h	0.0091	21.3%	中-大	0.0080-0.0088
t+48h	0.0078	25.2%	大(最大)	0.0068-0.0074
外部数据对 t+24h/t+48h 的收益最大,正好是我们当前最弱的两个 horizon。