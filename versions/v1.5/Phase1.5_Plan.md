# Phase 1.5 特征工程改进方案

> 基于文献调查报告(`articles/气象与时间特征工程调查报告.md`)中14篇论文的特征工程实践,
> 结合 Phase 1 运行结果(CV RMSE、特征重要性、零预测偏差分析)制定。
> 日期: 2026-08-31

---

## 一、Phase 1 回顾与问题诊断

### 1.1 当前性能

| Horizon | 模型 RMSE | Zero基线 | 优于Zero | 主要特征(top 3) |
|---|---|---|---|---|
| t+1h | 0.0128 | 0.0227 | 44% | hours_since_obs, last_P_t, last_osi |
| t+6h | 0.0114 | 0.0192 | 41% | hours_since_obs, last_P_t, last_osi |
| t+24h | 0.0091 | 0.0119 | 24% | hours_since_obs, last_osi, wind_speed_max_next_48h |
| t+48h | 0.0081 | 0.0091 | 11% | gust_mean_obs, log_customers, total_tp_next_48h |

### 1.2 核心问题: 气象特征贡献度低

Phase 1 中 `hours_since_obs` 和 `last_osi` 占据了绝大多数 gain, 气象特征贡献很低。

**根本原因分析(通过数据验证):**

1. **风暴有固定时间结构**: 第一波3/14、第二波3/16-17、3/18-19平静。`hours_since_obs` 一个变量就编码了"风暴走到哪个阶段", 与气象变量高度共线。
2. **OSI 是停电状态的函数, 有惯性**: `last_osi` 直接编码当前停电状态, 比气象驱动更强。
3. **气象特征没有捕获阶段交互**: `gust_t=30` 是孤立数值, 模型无法区分"第一波的30mph"(树未倒)和"第二波的30mph"(树已倒)。

**数据验证**: t+48h中, target=0时gust反而更高(24.8 vs 21.7), 证明同gust在不同阶段效果完全不同。

### 1.3 零预测偏差

| Horizon | 预测零% | 目标零% | 差距 |
|---|---|---|---|
| t+1h | 6.3% | 43.9% | -37.6pp |
| t+06h | 11.9% | 43.0% | -31.1pp |
| t+24h | 16.6% | 43.6% | -27.0pp |
| t+48h | 0.7% | 42.3% | -41.6pp |

模型严重欠预测零值。但数据分析显示 **特征已能区分零/非零**:

| (t+1h) | target=0 | target>0 |
|---|---|---|
| gust mean | 14.9 mph | 22.8 mph |
| gust>30mph占比 | 4.5% | 24.8% |
| hours_since_obs | 86h | 61h |

结论: 问题不在特征区分力, 而在 Huber loss 的输出校准(不倾向预测精确零值)。

---

## 二、文献中反复验证有效的特征工程做法

### 调查报告提炼的16条最佳实践(与本文档相关的条目)

| # | 做法 | 文献来源 | 证据强度 |
|---|---|---|---|
| 1 | 阈值超越时长(wgtX, CowgtX, ggtX) | Cerrai 2019/2020, Yang 2020a/b (4篇) | 最高(4篇一脉相承) |
| 2 | 峰值窗口条件均值(最强风4h窗口) | Cerrai 2019/2020, Yang 2020a/b | 高(同上) |
| 3 | 事件极值+总量+峰值窗口三元组聚合 | Cerrai系列, Guikema 2014 | 高 |
| 4 | 前瞻性特征堆叠(未来气象比历史值贡献大) | STO-CAST 2026 (SHAP验证) | 高(已有) |
| 5 | 滞后停电特征(互相关选最优滞后l=1) | Alpay 2020 | 中 |
| 6 | 小时sin/cos周期编码 | Alpay 2020 (仅此一篇) | 低(已有) |
| 7 | 前期条件特征(SPI6, 前期土湿) | Arora 2023 | 中 |
| 8 | 植被-气象交互(LAI×降水类型) | Cerrai系列 | 中(需外部数据) |
| 9 | 基础设施/人口加权 | Yang 2020a(QWD), Yang 2020b(SumAssets offset), Lee 2025 | 中 |
| 10 | 物理一致性约束(阵风线性样条) | Arora 2023 | 低(建模约束) |
| 11 | 雪密度=雪水当量/积雪量 | Cerrai 2020 | 中(冬季适用) |
| 12 | 风-降水共现比+植被交互乘积 | Cerrai 2020 | 中 |
| 13 | 降水类型先分类后建模 | Cerrai 2020, Lee 2025 | 中 |
| 14 | 极端事件外推: 树模型无法外推, GLM可以 | Cerrai 2020 | 重要启示 |
| 15 | 植被种类(树种丰度) | D'Amico 2019 | 低(增益~3%, 需外部数据) |
| 16 | 植被管理/历史风暴时间(修剪频率, 距上次登陆) | D'Amico 2019 | 低(需外部数据) |

### 文献中阈值选择参考

| 文献阈值(m/s) | 换算(mph) | 数据中超此值占比 | 含义 |
|---|---|---|---|
| 5 m/s | 11.2 mph | 75.1% | 低阈值(几乎总有风) |
| 9 m/s | 20.1 mph | 47.7% | 中低阈值 |
| 13 m/s | 29.1 mph | 22.0% | **中阈值(损害开始)** |
| 17 m/s | 38.0 mph | 7.1% | **高阈值(显著损害)** |
| 20 m/s | 44.7 mph | 2.9% | 极高阈值(电线杆设计风速) |

> 我们选择 **30 mph** (≈13.4 m/s, 文献的中阈值) 和 **40 mph** (≈17.9 m/s, 文献的高阈值) 作为两个阈值。

---

## 三、Phase 1.5 特征工程改进方案

### 3.1 新增特征清单(按优先级)

#### P1: 阈值超越时长 (新增~10维)

文献依据: Cerrai 2019/2020, Yang 2020a/b — 4篇UConn系列一脉相承, 该领域被反复验证的最有效风特征构造。

```python
# 对每个horizon h ∈ {1, 6, 24, 48}:
'gust_gt30_next_{h}h'   # [t, t+h]中 gust>30mph 的小时数
'gust_gt40_next_{h}h'   # [t, t+h]中 gust>40mph 的小时数

# 观测窗口(每县固定):
'gust_gt30_obs'          # 72h观测窗口中 gust>30mph 的小时数
'gust_gt40_obs'          # 72h观测窗口中 gust>40mph 的小时数
```

物理含义: 损害性风力持续时间, 而非仅峰值。30mph是损害开始阈值(文献13m/s), 40mph是显著损害阈值(文献17m/s)。

数据验证: 30mph覆盖19.7%小时(有区分力), 40mph覆盖5.4%(捕捉极端)。

#### P2: 峰值窗口条件均值 (新增~4维, 替代现有简单mean)

文献依据: Cerrai/Yang系列的 "最强风4小时窗口均值" — 比简单全窗口均值保留更多破坏性信息。

```python
# 对每个horizon h ∈ {1, 6, 24, 48}:
'gust_peak4h_mean_next_{h}h'  # [t,t+h]中gust最高的min(4,h)小时的均值
```

注意: 当 h < 4 时(如 h=1), 退化为 gust 本身, 此时与 gust_mean_next_1h 相同。

#### P3: 气象变化率 (新增~6维)

文献依据: Alpay 2020 — 互相关分析确定 wind-outage 最优滞后为 l=1; 气象变化率捕捉风暴锋面过境动态。

```python
'gust_change_1h'        # gust(t) - gust(t-1)     1小时阵风变化
'gust_change_3h'        # gust(t) - gust(t-3)     3小时阵风变化
'pressure_change_3h'    # sp(t) - sp(t-3)         3小时气压变化(锋面信号)
'pressure_change_6h'    # sp(t) - sp(t-6)         6小时气压变化
'temp_change_6h'        # t2m(t) - t2m(t-6)       6小时温变(冷锋)
'wind_change_1h'        # wind_speed(t) - wind_speed(t-1)
```

数据验证: |Δgust|>5占15.9%, >10占2.4%, 携带风暴锋面信号。变化率与时间位置相关性低(任何时刻都可能有变化)。

#### P4: 气象×阶段交互 (新增~4维)

文献依据: 我的分析(数据验证)。Cerrai 2020的 LAI_snow (植被×降雪交互) 和 Snow_time (风雪共现比) 是同类思路。

```python
'gust_x_hso'                 # gust(t) × hours_since_obs
'gust_x_phase'               # gust(t) × storm_phase
'soil_gust_x_phase'          # soil_moist × gust × storm_phase  (三重交互)
'gust_exceed30_x_hso'        # gust_exceed_30 × hours_since_obs
```

物理含义: 同样风力在不同风暴阶段(树未倒 vs 已倒, 维修队未部署 vs 已部署)对停电的影响不同。交互项拆解共线性。

数据验证: t+48h中 target=0 时 gust 更高(24.8 vs 21.7), 证明阶段效应显著。

#### P5: 累计暴露量 (新增~4维)

文献依据: Arora 2023 (SPI6前期累积), Cerrai 2020 (事件总累积降水), Cerrai 2019 (MaxTotPrec事件总量)。

```python
'cumul_gust_since_onset'     # sum(gust from March 13 0:00 to t)   累计风力暴露
'cumul_tp_since_onset'       # sum(tp from March 13 0:00 to t)     累计降水
'gust_gt30_since_onset'      # count(gust>30 from March 13 0:00 to t)  累计损害性风力时长
'cumul_gust_obs'             # sum(gust in 72h observed window)    观测窗口总风力
```

物理含义: 基础设施疲劳累积。累计风/雨暴露 → 树木根系松动、线路老化。

#### P6: 观测窗口气象边界 (新增~3维)

文献依据: Alpay 2020 (滞后特征 l=1), STO-CAST 2026 (O(-6)是nowcast关键输入)。

```python
'gust_last_obs'              # gust at March 13 23:00 (观测窗口末尾气象边界条件)
'gust_max_last6h_obs'        # max gust in last 6h of observed window
'gust_trend_last6h_obs'      # gust linear slope in last 6h of observed window
```

物理含义: 观测窗口末尾的气象状态是连接观测期和预测期的边界条件。风在观测窗口末尾的走势预示了预测窗口初期的停电动态。

### 3.2 特征总数变化

| 类别 | Phase 1 | Phase 1.5 新增 | Phase 1.5 合计 |
|---|---|---|---|
| 观测窗口摘要 | 25 | +7 (P1观测2 + P5观测1 + P6气象边界3 + cumul_gust_obs 1) | 32 |
| 当前气象 | 24 | +6 (P3变化率) | 30 |
| 派生气象 | 6 | +4 (P4交互) | 10 |
| 目标时段气象统计 | 20 | +12 (P1阈值8 + P2峰值4) | 32 |
| 时间特征 | 6 | 0 | 6 |
| 县级特征 | 1 | 0 | 1 |
| 累计暴露 | 0 | +3 (P5累计3) | 3 |
| **合计** | **82** | **+32** | **114** |

### 3.3 代码改动范围

| 文件 | 改动 |
|---|---|
| `features.py` | 在 OBSERVED_FEATURES, DERIVED_WEATHER_FEATURES, HORIZON_STAT_FEATURES 中新增列名; 新增 compute_threshold_exceedance(), compute_peak_window_mean(), compute_weather_changes(), compute_interactions(), compute_cumulative_exposure(), compute_obs_gust_profile() 函数; 在 build_feature_matrix() 中调用新函数 |
| `config.py` | FEATURE_VERSION 从 'v1' 改为 'v1.5' |
| `cache.py` | 无改动(自动用新版本号) |
| `model.py` | 无改动(特征数自动适应) |
| `evaluate.py` | 新增: post_processing_threshold() 函数(预测后处理阈值) |
| `main.py` | 调用后处理阈值; 日志记录新特征数 |
| `Phase1_Plan.md` | 不改(Phase 1 已完成) |

### 3.4 后处理阈值(解决零预测偏差)

不引入两-stage, 改用简单后处理:

```python
def post_process(preds, threshold=0.001):
    """预测后处理: 低于阈值的预测设为0"""
    preds = np.where(preds < threshold, 0.0, preds)
    return preds
```

阈值 0.001 的选择依据: OSI 的最小非零值在训练数据中约为 0.0001(4位小数精度), 但从实际含义看, OSI < 0.001 意味着停电比例极低, 可视为无停电。

如果后处理后零%仍严重偏低, 再考虑两-stage。

---

## 四、何时切换到时序模型

### 4.1 文献启示

| 论文 | 模型 | 时序 vs 表格 | 关键条件 |
|---|---|---|---|
| Alpay 2020 | LSTM(回看9h) vs RF(lag=1) | LSTM **仅微弱优于** RF | 需要互相关分析选最优滞后 |
| STO-CAST 2026 | GRU(12h滑窗) | 效果好 | 需要**4个台风事件**训练 |
| Prieto 2025 | CNN+LSTM(2-5天回看) | 多模型集成才稳定 | 日尺度, 多事件 |

核心启示: 时序模型的优势在于自动学习最优回看窗口和非线性时序交互, 但需要**足够多的事件**来泛化。单事件下容易记忆特定时间模式。

### 4.2 切换条件(全部满足)

1. ⬜ Phase 1.5 特征已实现并测试
2. ⬜ CV RMSE 连续2轮迭代无改善(plateau)
3. ⬜ OOF分析发现系统性时序误差(如:无法捕捉第一波→第二波延迟,或气象→停电非线性滞后)
4. ⬜ 能明确指出时序模型能解决什么表格模型不能

### 4.3 预估时间线

| 阶段 | 内容 | 预估时间 |
|---|---|---|
| Phase 1.5 第1轮 | 实现P1+P2(阈值超越+峰值窗口), 测试 | 1天 |
| Phase 1.5 第2轮 | 实现P3+P4(变化率+交互), 测试 | 1天 |
| Phase 1.5 第3轮 | 实现P5+P6(累计+边界)+后处理阈值, 测试 | 0.5天 |
| 评估 | 对比3轮RMSE趋势, 判断是否plateau | 0.5天 |
| **决策点** | 若plateau → 启动Phase 2(GRU); 否则继续迭代 | — |

### 4.4 时序模型的风险

我们只有 **1次风暴事件**(239县×216h)。STO-CAST用了4个台风才训出GRU。单事件下时序模型容易:
- 记忆"3/14高→3/18低"的时间模式, 而非学习通用气象→停电动力学
- 在测试集上看似有效(同事件), 但物理可解释性差
- 过拟合风险高(序列模型参数远多于表格模型)

**缓解策略**: 若启动Phase 2, GRU的输入应包含气象序列(而非仅停电序列), 让模型学习 weather→outage 的因果关系, 而非纯时间模式。

---

## 五、是否需要两阶段预测

### 5.1 结论: 暂不引入

**理由:**
1. **特征已有区分力**: 数据验证显示 target=0 和 target>0 在 gust/hours_since_obs 上有显著差异
2. **简单后处理可解决**: 加阈值后处理(pred < 0.001 → 0)即可改善零%
3. **两-stage引入额外复杂度**: 分类器误差传播; 单事件下分类器本身不稳定
4. **文献中两-stage用于不同目的**: Yang 2020a按事件严重度分3级(低/中/高), 不是按零/非零分; Aljurbua 2025按是否停运二分类再5级, 是不同任务结构

### 5.2 何时重新考虑

- 加了阈值超越特征(P1)+后处理阈值后, 零%仍严重偏低(如<20% vs target 43%)
- 某个特定horizon的零%问题特别严重(当前t+48h只有0.7%为零)
- CV分析发现模型在"应该预测零但预测了正值"的样本上误差特别大

### 5.3 若引入两-stage的设计方案(备用)

```
Stage 1: LightGBM二分类器 (target > 0 ?)
  输入: 同Phase 1.5特征
  输出: P(nonzero)
  阈值: 0.5 (或CV优化)

Stage 2: LightGBM回归器 (仅在非零样本上训练)
  输入: 同上特征
  输出: OSI值

最终预测: pred = P(nonzero) × regression_value
```

注意: 不用硬阈值分类(会丢失梯度信息), 用概率加权更平滑。

---

## 六、详细代码改动清单

### 6.1 features.py 改动

#### 新增特征列名

```python
# P1: 阈值超越时长
THRESHOLD_FEATURES = []
for _h in ['1', '6', '24', '48']:
    THRESHOLD_FEATURES.extend([
        f'gust_gt30_next_{_h}h',
        f'gust_gt40_next_{_h}h',
    ])
THRESHOLD_FEATURES.extend(['gust_gt30_obs', 'gust_gt40_obs'])

# P2: 峰值窗口条件均值
PEAK_WINDOW_FEATURES = [f'gust_peak4h_mean_next_{_h}h' for _h in ['1','6','24','48']]

# P3: 气象变化率
WEATHER_CHANGE_FEATURES = [
    'gust_change_1h', 'gust_change_3h',
    'pressure_change_3h', 'pressure_change_6h',
    'temp_change_6h', 'wind_change_1h',
]

# P4: 气象×阶段交互
INTERACTION_FEATURES = [
    'gust_x_hso', 'gust_x_phase',
    'soil_gust_x_phase', 'gust_exceed30_x_hso',
]

# P5: 累计暴露量
CUMULATIVE_FEATURES = [
    'cumul_gust_since_onset', 'cumul_tp_since_onset',
    'gust_gt30_since_onset', 'cumul_gust_obs',
]

# P6: 观测窗口气象边界
OBS_GUST_FEATURES = [
    'gust_last_obs', 'gust_max_last6h_obs', 'gust_trend_last6h_obs',
]
```

#### 新增计算函数

```python
def compute_threshold_exceedance(county_df, t_idx, h):
    """P1: 计算 [t, t+h] 中 gust>30/40mph 的小时数"""
    end = min(t_idx + h + 1, len(county_df))
    window = county_df.iloc[t_idx:end]
    return {
        'gust_gt30': (window['gust'] > 30).sum(),
        'gust_gt40': (window['gust'] > 40).sum(),
    }

def compute_peak_window_mean(county_df, t_idx, h):
    """P2: [t,t+h]中gust最高的min(4,h)小时的均值"""
    end = min(t_idx + h + 1, len(county_df))
    window = county_df.iloc[t_idx:end]
    n_top = min(4, len(window))
    return window['gust'].nlargest(n_top).mean()

def compute_weather_changes(county_df, t_idx):
    """P3: 气象变化率(gust/pressure/temp/wind的1h/3h/6h差分)"""
    def safe_diff(col, lag):
        cur = county_df.iloc[t_idx][col]
        if t_idx - lag >= 0:
            prev = county_df.iloc[t_idx - lag][col]
            return cur - prev
        return np.nan
    return {
        'gust_change_1h': safe_diff('gust', 1),
        'gust_change_3h': safe_diff('gust', 3),
        'pressure_change_3h': safe_diff('sp', 3),
        'pressure_change_6h': safe_diff('sp', 6),
        'temp_change_6h': safe_diff('t2m', 6),
        'wind_change_1h': safe_diff('wind_speed_10m', 1),
    }

def compute_interactions(row, gust_exceed_30, hours_since_obs, storm_phase):
    """P4: 气象×阶段交互"""
    return {
        'gust_x_hso': row['gust'] * hours_since_obs,
        'gust_x_phase': row['gust'] * storm_phase,
        'soil_gust_x_phase': row['soil_moist'] * row['gust'] * storm_phase,
        'gust_exceed30_x_hso': gust_exceed_30 * hours_since_obs,
    }

def compute_cumulative_exposure(county_df, t_idx):
    """P5: 从风暴开始(March 13, hour_idx=48)到t的累计暴露"""
    start = 48  # March 13 0:00
    window = county_df.iloc[start:t_idx+1]
    return {
        'cumul_gust_since_onset': window['gust'].sum(),
        'cumul_tp_since_onset': window['tp'].sum(),
        'gust_gt30_since_onset': (window['gust'] > 30).sum(),
    }

def compute_obs_gust_profile(obs):
    """P6: 观测窗口末尾的气象边界条件"""
    last6 = obs.tail(6)
    gust_last6 = last6['gust'].dropna().values
    if len(gust_last6) >= 2:
        x = np.arange(len(gust_last6))
        trend = np.polyfit(x, gust_last6, 1)[0] if np.std(gust_last6) > 0 else 0.0
    else:
        trend = 0.0
    return {
        'gust_last_obs': obs.iloc[-1]['gust'],
        'gust_max_last6h_obs': gust_last6.max() if len(gust_last6) > 0 else 0.0,
        'gust_trend_last6h_obs': trend,
    }
```

#### build_feature_matrix() 中的调用

在现有特征拼装后, 追加:
```python
# P1: 阈值超越
for h_key, h_val in HORIZON_HOURS.items():
    te = compute_threshold_exceedance(county_df, t_idx, h_val)
    h_short = str(h_val)
    for name, val in te.items():
        feat[f'{name}_next_{h_short}h'] = val

# P2: 峰值窗口均值
for h_key, h_val in HORIZON_HOURS.items():
    feat[f'gust_peak4h_mean_next_{str(h_val)}h'] = compute_peak_window_mean(county_df, t_idx, h_val)

# P3: 气象变化率
feat.update(compute_weather_changes(county_df, t_idx))

# P4: 交互
feat.update(compute_interactions(row, feat['gust_exceed_30'],
           feat['hours_since_obs'], feat['storm_phase']))

# P5: 累计暴露(需在观测窗口特征中添加 cumul_gust_obs)
feat.update(compute_cumulative_exposure(county_df, t_idx))

# P6: 在 compute_observed_features() 中添加 obs_gust_profile
```

### 6.2 config.py 改动

```python
FEATURE_VERSION = 'v1.5'  # 从 'v1' 改为 'v1.5', 触发缓存重建
```

### 6.3 evaluate.py 改动

```python
def post_process(preds, threshold=0.001):
    """预测后处理: 低于阈值的预测设为0, 改善零预测偏差"""
    preds = np.where(preds < threshold, 0.0, preds)
    return preds
```

### 6.4 main.py 改动

在预测后调用后处理:
```python
for h in HORIZONS:
    preds[h] = predict(results[h]['model'], X_test)
    preds[h] = post_process(preds[h])  # 新增
```

### 6.5 执行顺序

```bash
# 1. 修改 features.py (新增函数+列名+调用)
# 2. 修改 config.py (FEATURE_VERSION='v1.5')
# 3. 修改 evaluate.py (新增post_process)
# 4. 修改 main.py (调用post_process)
# 5. 运行: python main.py --rebuild-features
#    (特征从v1.5缓存重新构建, 模型训练+日志)
# 6. 对比 Phase 1 (v1) 的日志结果
```

---

## 七、回溯复盘检查点

### 每轮迭代后记录

| 检查项 | 方法 |
|---|---|
| RMSE是否改善 | `grep "Mean RMSE" logs/experiments.log` 对比v1 vs v1.5 |
| 气象特征重要性是否提升 | `grep "gust" logs/experiments.log` 看 gain 排名变化 |
| 零预测偏差是否改善 | `grep "zero%" logs/experiments.log` 对比 |
| 新特征是否有贡献 | 日志中 feature importance top 15 是否出现新特征名 |
| 过拟合迹象 | CV各折RMSE标准差是否增大 |

### 决策树

```
v1.5 RMSE vs v1:
├── 显著改善(>5%) → 继续P3-P6, 进入Phase 1.5第2轮
├── 微弱改善(1-5%) → 继续P3-P6, 但降低期望
└── 无改善/恶化 → 分析特征重要性, 检查是否有数据泄露或冗余
    └── 若连续2轮无改善 → 评估是否切换Phase 2(时序模型)
```

---

## 八、Phase 1.5 运行结果与下一步方向

### 8.1 v1 → v1.5 结果对比

| Horizon | v1 RMSE | v1.5 RMSE | 改善 | v1 MAE | v1.5 MAE | 改善 |
|---|---|---|---|---|---|---|
| t+1h | 0.012824 | 0.012588 | **-1.8%** | 0.004188 | 0.004063 | **-3.0%** |
| t+6h | 0.011404 | 0.011299 | **-0.9%** | 0.003972 | 0.003846 | **-3.2%** |
| t+24h | 0.009075 | 0.008929 | **-1.6%** | 0.003314 | 0.003324 | +0.3% |
| t+48h | 0.008052 | 0.007861 | **-2.4%** | 0.002835 | 0.002746 | **-3.1%** |

全部 horizon RMSE 改善 0.9%–2.4%。MAE 在 3 个 horizon 上改善 3%+ (主要来自 post_process 零校准)。

特征重要性: `hours_since_obs` gain 从 45.7 降到 33.7 (-26%), 4 个新特征进入 top 15 (P4交互2个, P5累计2个)。零预测偏差从 6.3%→48.2% (target 43.9%), 彻底解决。

**结论: 微弱改善区间, 可继续迭代。**

### 8.2 数据验证的核心发现: 目标时刻精确气象 > 窗口统计

STO-CAST (Yang 2026) 的 SHAP 分析显示: "forward-looking variables W(+6), D(+6), R(+6) contribute more than historical variables W(-6), O(-6)"。

用我们的数据验证:

| 特征 | 与 t+48h target 相关系数 | 来源 |
|---|---|---|
| `gust_max_next_48h` (窗口max, 当前) | 0.1178 | Phase 1.5 已有 |
| `gust_at_t+48h` (精确值, **缺失**) | **0.1651** (+40%) | **应新增** |
| `gust_max_next_24h` (窗口max, 当前) | 0.1608 | Phase 1.5 已有 |
| `gust_at_t+24h` (精确值, **缺失**) | **0.1859** (+16%) | **应新增** |

这直接解释了为什么 t+24h/t+48h 性能弱——我们有未来窗口的聚合统计, 但缺少目标时刻本身的精确气象值。

### 8.3 下一步改进方向 (按 ROI 排序)

#### 方向 1: 目标时刻精确气象 (最高 ROI, 1天)

来源: STO-CAST 2026 — SHAP 证明未来精确值 > 窗口统计

做法: 对每个 horizon h, 新增 t+h 时刻的精确值:
- `gust_at_t+1h`, `gust_at_t+6h`, `gust_at_t+24h`, `gust_at_t+48h`
- 同理 `wind_speed_at_t+Xh`, `t2m_at_t+Xh`, `tp_at_t+Xh`, `r2_at_t+Xh`
- 共 ~16 新特征 (4 变量 × 4 horizon)

成本: features.py 中加一个 lookup (county_df.iloc[t_idx + h]), 极简单。

#### 方向 2: 加权 Huber 损失 (高 ROI, 半天)

来源: STO-CAST 2026 — 加权 Huber 损失 WHL: δ=10, w=1000, t=5 (停电数 > 5 的样本加权 1000 倍)

做法: 在 LightGBM 训练时传入 weight 参数:
```python
weight = np.where(y > 0, 10.0, 1.0)  # 非零样本加权10倍
dtr = lgb.Dataset(X_tr, label=y_tr, weight=weight)
```
效果: 让模型在训练时就关注非零样本, 比后处理阈值更根本。

#### 方向 3: 超参调优 (中等 ROI, 1天)

做法: 利用缓存特征秒级加载, 对以下参数做 grid/random search:
- `num_leaves`: [15, 31, 63, 127]
- `learning_rate`: [0.01, 0.03, 0.05, 0.1]
- `min_child_samples`: [20, 50, 100, 200]
- `feature_fraction`: [0.6, 0.8, 1.0]
- `lambda_l1`: [0, 0.1, 1.0]
- `lambda_l2`: [0.5, 1.0, 5.0]

#### 方向 4: 外部数据 (高潜力, 2-3天)

来源: McRoberts 2018 (+环境因子精度提升 17%), D'Amico 2019 (树种), Lee 2025 (人口#1)

做法: 用 fipsCode 作连接键拼接:
- EIA Form 861: 县级线路里程、客户密度
- NLCD Tree Canopy: 树冠覆盖率 (Cerrai系列反复验证有效)
- USDA Rural-Urban Codes: 城乡等级

#### 方向 5: LightGBM + XGBoost 集成 (中等 ROI, 1天)

来源: Cerrai 2019 (RF+BART+DT+ENS优化, MAPE降20%), Lee 2025 (XGBoost最优)

做法: 训练 XGBoost (不同算法+不同种子), 与 LightGBM 预测取加权平均。

#### 方向 6: GRU 时序模型 (高风险高回报, 3-5天)

来源: STO-CAST 2026 (GRU, 4个台风), Alpay 2020 (LSTM仅微弱优于RF)

适用条件: 方向 1-3 做完后, t+24h/t+48h 仍弱。

设计:
- 输入: 72h 观测序列 (停电+气象) + 目标时刻气象
- GRU 2层, hidden=128
- 解码: hidden state + 未来气象 → 全连接 → OSI
- 重点用于 t+24h/t+48h (tabular 最弱的 horizon)
- 风险: 单事件, 需强正则化

### 8.4 推荐执行顺序

```
Phase 1.5 Round 2 (v3, 1-2天):
  ├── 方向1: 目标时刻精确气象 (+16新特征)
  ├── 方向2: 加权Huber损失
  └── 方向3: 超参调优 (用缓存特征快速迭代)
         ↓
评估: RMSE是否继续改善?
  ├── 是 → Phase 1.5 Round 3: 方向4(外部数据) + 方向5(集成)
  └── 否(plateau) → Phase 2: 方向6(GRU) for t+24h/t+48h
```

### 8.5 版本号命名规范

当前存在的版本不一致问题:
- `submission_phase1.csv` 硬编码文件名, 不含版本号 → 每次运行覆盖
- 模型文件用 `lgbm_{horizon}_v1.5.txt` → 版本标签是 v1.5 但人读层面是 "Phase 1.5"
- 日志中记录的 `feature_version: v1.5` 与 Results.md 中的 "Phase 1.5" 不统一

**统一命名规范:**

| 阶段 | FEATURE_VERSION | 输出文件名 | 模型文件名 | 日志/文档标签 |
|---|---|---|---|---|
| Phase 1 (82维) | `v1` | `submission_v1.csv` | `lgbm_{horizon}_v1.txt` | Phase 1 (v1) |
| Phase 1.5 (113维) | `v1.5` | `submission_v1.5.csv` | `lgbm_{horizon}_v1.5.txt` | Phase 1.5 (v1.5) |
| Phase 1.5 R2 (方向1-3) | `v3` | `submission_v3.csv` | `lgbm_{horizon}_v3.txt` | Phase 1.5 R2 (v3) |
| Phase 2 (GRU) | `v4` | `submission_v4.csv` | (GRU模型另命名) | Phase 2 (v4) |

**需要修改的文件 (暂不改代码, 仅记录待改项):**

| 文件 | 当前 | 应改为 |
|---|---|---|
| `config.py` OUTPUT_FILE | `submission_phase1.csv` (硬编码) | `submission_{FEATURE_VERSION}.csv` (动态拼接) |
| `config.py` MODEL_DIR中模型名 | 由 `model.py` 用 `FEATURE_VERSION` 拼接 | 已正确, 但需确保与 OUTPUT_FILE 一致 |
| `main.py` 日志 notes | 写死 "Phase 1 baseline" | 改为含 `FEATURE_VERSION` 的动态描述 |
| `main.py` 打印标题 | 写死 "Phase 1: LightGBM Baseline" | 改为 `f"Phase {PHASE_LABEL}: LightGBM"` |

**待改项清单 (下次改代码时执行):**
1. config.py: 新增 `PHASE_LABEL = 'Phase 1.5'`; OUTPUT_FILE 改为 `submission_{FEATURE_VERSION}.csv`
2. main.py: 标题和 notes 用 PHASE_LABEL 替代硬编码字符串
3. 确保 submission/model/cache 三处文件名都用同一个 `FEATURE_VERSION`
