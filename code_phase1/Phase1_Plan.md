# Phase 1 构建方案: LightGBM 多目标回归基线

> 不使用外部数据。仅依赖 DM_Train.csv + DM_Test.csv + sample_submission.csv。

---

## 0. 总体架构

```
DM_Train.csv ──┐
               ├─→ 数据加载&预处理 ──→ 特征工程 ──→ 训练样本 ──→ 4×LightGBM ──→ 预测 ──→ submission.csv                
DM_Test.csv  ──┘                       
sample_submission.csv ─────────────────→ 填充
```

**4个独立模型**: t+1h, t+6h, t+24h, t+48h 各一个 LightGBM 回归器。
理由: t+48h 与 t+1h 相关仅 0.056, 各 horizon 动态差异大, 独立建模更合理。

---

## 1. 数据加载与预处理

### 1.1 加载
- 读取 DM_Train.csv (51,624行), DM_Test.csv (13,608行), sample_submission.csv (9,072行)
- 用 `csv.DictReader` 读取(不依赖 pandas, 减少环境依赖)

### 1.2 时间戳解析
- 格式: "3/14/2026 0:00" → `datetime.strptime(ts, '%m/%d/%Y %H:%M')`
- **必须用 datetime 排序, 不能用字符串排序**(字符串排序 "10:00" < "9:00" 是错的)
- 每个县按时间排序, 分配全局小时索引 hour_idx (0–215)
  - 0–47: March 11–12 (预事件, in_event_window=False)
  - 48–71: March 13 (观测窗口后半, 风暴第一波开始)
  - 72–215: March 14–19 (预测窗口)

### 1.3 测试集 OSI 重建
- 测试集无 `osi` 列, 但有 P_t, N_t, D_t, R_t (March 11–13 有值)
- 重建公式: `osi = max(0, 0.40*P_t + 0.35*N_t + 0.25*D_t - 0.10*R_t)`
- 用**存储的组件值直接代入**, 不从 outageCount 重新推导 N_t/R_t(验证发现实际为毛流量定义, 与PDF公式不符)

### 1.4 风向编码
- `wind_sin = sin(2 * pi * wind_dir_10m / 360)`
- `wind_cos = cos(2 * pi * wind_dir_10m / 360)`
- 不使用原始 wind_dir_10m

### 1.5 温度转换
- `temp_c = t2m - 273.15`
- `dewpoint_c = d2m - 273.15`
- 用于派生结冰风险等特征

---

## 2. 特征工程

### 设计原则
训练集中预测窗口(March 14–19)的停电数据**不作为输入特征**(模拟测试集条件, 测试集这些列为NaN)。仅使用:
- 观测窗口(March 11–13, 72h)的停电数据 → 提取县级静态摘要
- 气象数据(全216h可用) → 按预测时刻 t 和目标时刻 t+h 提取
- 时间特征 → 从 timestamp 派生

### 2.A 观测窗口摘要特征 (每县固定, 72h统计, ~25维)

从每县 March 11–13 (hour_idx 0–71) 的停电数据提取:

| # | 特征名 | 计算方式 | 含义 |
|---|---|---|---|
| A1 | `last_osi` | hour_idx=71 的 osi | 风暴起始时的最新严重度 |
| A2 | `last_P_t` | hour_idx=71 的 P_t | 当前停电比例 |
| A3 | `last_N_t` | hour_idx=71 的 N_t | 最新恶化率 |
| A4 | `last_D_t` | hour_idx=71 的 D_t | 6h持续停电趋势 |
| A5 | `last_R_t` | hour_idx=71 的 R_t | 最新恢复率 |
| A6 | `last_outage_pct` | hour_idx=71 的 outage_pct | 停电百分比 |
| A7 | `osi_mean_72h` | 72h osi 均值 | 整体严重度基线 |
| A8 | `osi_max_72h` | 72h osi 最大值 | 观测窗口峰值 |
| A9 | `osi_std_72h` | 72h osi 标准差 | 波动性 |
| A10 | `outage_pct_mean_72h` | 72h outage_pct 均值 | 平均停电比例 |
| A11 | `outage_pct_max_72h` | 72h outage_pct 最大值 | 峰值停电比例 |
| A12 | `N_t_mean_72h` | 72h N_t 均值 | 平均恶化速度 |
| A13 | `N_t_max_72h` | 72h N_t 最大值 | 最严重恶化时刻 |
| A14 | `R_t_mean_72h` | 72h R_t 均值 | 平均恢复速度 |
| A15 | `R_t_max_72h` | 72h R_t 最大值 | 最快恢复时刻 |
| A16 | `pre_event_mean_osi` | March 11–12 (idx 0–47) osi 均值 | 遗留停电基线 |
| A17 | `pre_event_max_outage_pct` | March 11–12 最大 outage_pct | 遗留停电峰值 |
| A18 | `storm_onset_max_osi` | March 13 (idx 48–71) osi 最大值 | 第一波峰值 |
| A19 | `storm_onset_mean_osi` | March 13 osi 均值 | 第一波平均 |
| A20 | `osi_trend_last6h` | idx 66–71 osi 线性回归斜率 | 临近预测时的趋势 |
| A21 | `is_worsening` | last_N_t > last_R_t ? 1 : 0 | 是否正在恶化 |
| A22 | `frac_zero_72h` | 72h中 osi=0 的小时占比 | 低严重度时段比例 |
| A23 | `hours_since_peak` | 71 − argmax(osi in 72h) | 距峰值已过去多久 |
| A24 | `gust_mean_obs` | 72h gust 均值 | 观测窗口平均风力 |
| A25 | `gust_max_obs` | 72h gust 最大值 | 观测窗口最强阵风 |

### 2.B 当前气象特征 (随 t 变化, ~22维)

预测时刻 t 的气象变量(直接从数据中取值):

| 特征 | 来源 |
|---|---|
| `gust_t` | URMA |
| `wind_speed_t` | URMA |
| `wind_sin_t`, `wind_cos_t` | 从 wind_dir 派生 |
| `t2m_t`, `d2m_t` | URMA (Kelvin) |
| `temp_c_t`, `dewpoint_c_t` | 派生 |
| `tp_t`, `rain_t`, `csnow_t`, `sdwe_t` | ERA5 |
| `r2_t`, `vpd_t`, `soil_moist_t` | ERA5 |
| `blh_t`, `tcc_t`, `lcc_t`, `mcc_t`, `hcc_t` | ERA5 |
| `sp_t`, `mslma_t` | ERA5 |
| `sdswrf_t`, `direct_rad_t`, `diffuse_rad_t`, `et0_t` | ERA5 |

### 2.C 目标时段气象特征 (随 t 和 horizon 变化, ~20维)

对每个 horizon h ∈ {1, 6, 24, 48}, 从气象数据中提取 [t, t+h] 窗口的统计量:

| 特征 | 计算 | 含义 |
|---|---|---|
| `gust_max_next_{h}h` | max(gust in [t, t+h]) | 未来最大阵风 |
| `gust_mean_next_{h}h` | mean(gust in [t, t+h]) | 未来平均阵风 |
| `wind_speed_max_next_{h}h` | max(wind_speed in [t, t+h]) | 未来最大持续风速 |
| `total_tp_next_{h}h` | sum(tp in [t, t+h]) | 未来累计降水 |
| `min_t2m_next_{h}h` | min(t2m in [t, t+h]) | 未来最低温度(结冰风险) |

> 这 5 个特征 × 4 个 horizon = 20 维。所有 4 个 horizon 的统计量都作为所有模型的输入, 让模型自行选择相关特征。

### 2.D 派生气象特征 (~5维)

| 特征 | 计算 | 含义 |
|---|---|---|
| `icing_risk_t` | 1 if temp_c_t < 0 and r2_t > 80 else 0 | 当前结冰风险 |
| `gust_exceed_30` | max(0, gust_max_next_1h - 30) | 阵风超过损害阈值(30mph)的幅度 |
| `gust_change_1h` | gust(t+1h) - gust(t) | 1小时内阵风变化 |
| `soil_moist_x_gust` | soil_moist_t * gust_t | 土湿×风速交互(树木倒伏风险) |
| `is_snowing` | 1 if csnow_t > 0 else 0 | 是否降雪 |

### 2.E 时间特征 (随 t 变化, 6维)

| 特征 | 计算 | 含义 |
|---|---|---|
| `hour_of_day` | t.hour (0–23) | 日内周期 |
| `hour_sin`, `hour_cos` | sin/cos(2π × hour/24) | 周期编码 |
| `days_since_onset` | (t - March 13 0:00).days | 风暴开始后天数 |
| `hours_since_obs` | t - March 13 23:00 (小时数, 1–144) | 距最后观测的小时数(信息衰减) |
| `storm_phase` | 0=第一波(3/14), 1=间歇(3/15), 2=第二波(3/16-17), 3=消退(3/18-19) | 风暴阶段 |

### 2.F 县级特征 (1维)

| 特征 | 计算 | 含义 |
|---|---|---|
| `log_customers` | log(customersTracked at t) | 县规模(对数, 全程有值) |

### 特征总计
~25 (观测摘要) + 22 (当前气象) + 20 (目标时段气象) + 5 (派生) + 6 (时间) + 1 (县级) = **~79 维**

---

## 3. 训练样本构造

### 3.1 训练集构造
对每个训练县 f (239个):
1. 提取观测窗口 (hour_idx 0–71) → 计算特征 2.A (25维, 对该县固定)
2. 提取观测窗口气象 → 计算 2.A 中的 gust_mean_obs, gust_max_obs
3. 对预测窗口每个小时 t (hour_idx 72–215, 共144小时):
   a. 从全量气象数据中提取 t 时刻气象 → 特征 2.B
   b. 计算 [t, t+h] 窗口气象统计 → 特征 2.C
   c. 计算派生特征 → 特征 2.D
   d. 计算时间特征 → 特征 2.E
   e. 取县级特征 → 特征 2.F
   f. 取目标值: osi_target_t01h, t06h, t24h, t48h
   g. 拼成一行 (79维特征 + 4个目标)

总样本数: 239县 × 144小时 = 34,416 行 (去掉各 horizon 末尾NaN后分别为 34,177 / 29,982 / 28,680 / 22,944)

### 3.2 关键合规点
- **不使用**预测窗口(March 14–19)的停电数据作为输入(包括 outageCount, osi, lag 特征等)
- 气象数据可使用任意时刻(包括 t 之后的未来气象)
- severity_tier 不作为特征, 仅用于CV分层
- stateAbbr/fipsCode 不作为特征, 仅用于CV分组和外部数据拼接(Phase 1不用外部数据)

---

## 4. 交叉验证

### 4.1 策略
- **5折 GroupKFold**: group = fipsCode (同一县的144行样本在同一折, 防止信息泄漏)
- **分层**: 按 (stateAbbr, severity_tier) 组合分层, 确保每折的州/严重度分布均衡
- 随机种子: 42

### 4.2 实现
1. 构造 group_id = fipsCode (239个group)
2. 构造 stratum = stateAbbr + '_' + str(severity_tier)
3. 用 `StratifiedGroupKFold` (sklearn) 或手动实现
4. 每折: 80% 县训练, 20% 县验证

---

## 5. 模型配置

### 5.1 LightGBM 参数 (初始值)
```python
params = {
    'objective': 'huber',        # Huber损失: 对零膨胀和极端值鲁棒
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 50,     # 239县×144h, 需防过拟合
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'verbose': -1,
    'seed': 42,
}
num_boost_round = 2000
early_stopping_rounds = 100
```

### 5.2 训练流程
对每个 horizon (t01h, t06h, t24h, t48h):
1. 取该 horizon 的非NaN样本
2. 5折CV: 每折用4折训练, 1折验证, early stopping
3. 记录每折验证RMSE/MAE
4. 用全量数据 + 最佳迭代数重训练最终模型
5. 保存模型和特征重要性

### 5.3 预测后处理
- clip 到 [0, 0.65] (OSI 观测最大值)
- 超出 March 19 23:00 的目标保持 NaN

---

## 6. 测试集预测

### 6.1 测试特征构造
对每个测试县 f (63个):
1. 提取观测窗口 (March 11–13, 全部72h有值)
2. 重建 OSI: `osi = max(0, 0.40*P_t + 0.35*N_t + 0.25*D_t - 0.10*R_t)`
3. 计算观测窗口摘要特征 (2.A, 与训练完全相同)
4. 对预测窗口每个小时 t (March 14–19, 144小时):
   a. 从测试集气象数据提取特征 (2.B–2.F, 与训练完全相同)
   b. 用4个模型分别预测 → 4个目标值

### 6.2 填充提交文件
1. 以 sample_submission.csv 为模板
2. 按行匹配 (fipsCode, timestamp_et)
3. 填入4个预测值
4. 超出数据末尾的行保持 NaN
5. **不改动**标识符列和行顺序

---

## 7. 评估与分析

### 7.1 CV指标
- 每折每horizon: RMSE, MAE
- 5折平均 ± 标准差
- 按horizon分别报告

### 7.2 基线对比
| 基线 | 方法 | 预期 |
|---|---|---|
| Zero | pred = 0 | RMSE ≈ osi的std ≈ 0.03 |
| Persistence | pred = last_osi | t+1h 应较好, t+48h 应很差 |
| Mean | pred = training mean osi | 最简单的基线 |

### 7.3 分析维度
- 特征重要性 (LightGBM gain importance)
- 按 horizon 对比性能 (预期 t+1h 最好, t+48h 最差)
- 按 storm_phase 对比 (第二波期间预测可能更难)
- 按 hours_since_obs 对比 (随时间推移精度下降)
- 按 county severity 对比 (高严重度县是否更难预测)
- 残差分析: 高估/低估模式

---

## 8. 文件结构

```
INFORMS_DATA_MINING/
├── fundamentals/          # 已有文档
│   ├── Compliance_Checklist.md
│   ├── PDF_Verification_Report.md
│   ├── verify_pdf_claims.py
│   └── email_to_organizers.md
├── phase1/
│   ├── config.py          # 路径、随机种子、超参数、特征版本号
│   ├── data_loader.py     # CSV加载、时间戳解析、OSI重建
│   ├── features.py        # 特征工程(观测摘要+气象+时间)
│   ├── cache.py           # 特征缓存: parquet读写、版本管理
│   ├── cv.py              # GroupKFold分层交叉验证
│   ├── model.py           # LightGBM训练/预测
│   ├── evaluate.py        # 指标计算、基线对比、分析
│   ├── logger.py          # 实验日志: 逐次运行结果记录
│   └── main.py            # 主流程编排
├── cache/                 # 特征缓存目录(gitignore)
│   ├── features_train_v1.parquet
│   ├── features_test_v1.parquet
│   └── feature_names_v1.json
├── logs/                  # 实验日志目录(gitignore)
│   └── experiments.log    # 追加写, 每次运行追加一段
├── DM_Train.csv
├── DM_Test.csv
├── sample_submission.csv
└── submission_phase1.csv  # 输出
```

### 各文件职责

**config.py**: 集中管理所有配置
- 文件路径(相对路径)
- 随机种子 (42)
- 时间窗口边界 (hour_idx: 0-47, 48-71, 72-215)
- LightGBM超参数
- 特征开关
- **FEATURE_VERSION = 'v1'** (改特征工程代码时递增, 触发缓存重建)
- 缓存目录路径 (CACHE_DIR = '../cache')
- 日志目录路径 (LOG_DIR = '../logs')
- 日志文件路径 (LOG_FILE = '../logs/experiments.log')

**data_loader.py**: 数据加载与预处理
- `load_train()`: 加载DM_Train.csv, 按县按时间排序, 返回 {fips: [rows]}
- `load_test()`: 加载DM_Test.csv, 同上
- `load_submission()`: 加载sample_submission.csv
- `reconstruct_osi(rows)`: 从P_t,N_t,D_t,R_t重建osi
- `encode_wind_direction(rows)`: 添加wind_sin, wind_cos
- `get_hour_idx(ts)`: 时间戳→全局小时索引 (0-215)

**features.py**: 特征工程
- `compute_observed_features(county_rows, obs_end_idx=71)`: 计算观测窗口摘要 (2.A, 25维)
- `compute_weather_at_t(county_rows, t_idx)`: t时刻气象 (2.B, 22维)
- `compute_weather_horizon_stats(county_rows, t_idx, horizons)`: 目标时段统计 (2.C, 20维)
- `compute_derived_weather(t_weather)`: 派生气象 (2.D, 5维)
- `compute_temporal_features(t_idx)`: 时间特征 (2.E, 6维)
- `compute_county_features(county_rows, t_idx)`: 县级特征 (2.F, 1维)
- `build_feature_row(...)`: 拼装单行特征
- `build_feature_matrix(data, is_train)`: 构建完整特征矩阵 (返回 X, y, meta)
- `get_feature_names()`: 返回特征列名顺序列表 (用于缓存对齐)

**cache.py**: 特征缓存管理
- `cache_path(is_train)`: 返回缓存文件路径 (如 `cache/features_train_v1.parquet`)
- `load_features(is_train)`: 若缓存存在则加载, 返回 (X, y, meta); 不存在返回 None
- `save_features(X, y, meta, feature_names, is_train)`: 保存到 parquet + json
- `get_or_build(data, is_train, force_rebuild=False)`: 主入口 — 有缓存则加载, 无则调 `build_feature_matrix` 计算并存盘
- 版本管理: 文件名含 `FEATURE_VERSION`, 改版本号即废弃旧缓存

**cv.py**: 交叉验证
- `get_cv_folds(train_fips, train_severity)`: 返回5折 (train_idx, val_idx) per fold
- `StratifiedGroupKFoldWrapper`: 封装sklearn或手动实现

**model.py**: 模型
- `train_lgbm(X_train, y_train, X_val, y_val, params)`: 训练单个模型
- `train_all_horizons(X, y_dict, folds)`: 训练4个模型
- `predict(model, X_test)`: 预测
- `save_model(model, path)` / `load_model(path)`: 保存/加载

**evaluate.py**: 评估
- `compute_metrics(y_true, y_pred)`: RMSE, MAE
- `persistence_baseline(...)`: 持续性基线
- `zero_baseline(...)`: 零基线
- `mean_baseline(...)`: 均值基线
- `feature_importance(model)`: 特征重要性
- `error_analysis(...)`: 分horizon/分阶段/分县分析

**logger.py**: 实验日志记录
- `init_logger(log_path)`: 初始化, 确保目录存在, 打开追加模式
- `log_run_header(config)`: 记录运行头(时间戳、feature_version、seed、参数摘要)
- `log_cv_results(horizon, fold_metrics, summary)`: 记录单horizon的逐折+汇总指标
- `log_baselines(baseline_metrics)`: 记录基线对比表
- `log_feature_importance(horizon, top_features)`: 记录top-15特征重要性
- `log_prediction_summary(horizon, pred_stats)`: 记录预测值分布(min/max/mean/zero%)
- `log_notes(text)`: 记录备注(手动传入的观察、错误、洞察)
- `log_run_footer()`: 记录运行尾(分隔符)
- 日志格式: 纯文本追加写, 段落式, 人读+grep友好

**main.py**: 主流程
```python
# 0. 初始化日志
log = init_logger(LOG_FILE)
log_run_header(config)  # 时间戳、feature_version、seed、超参摘要

# 1. 加载数据
train_data = load_train()
test_data = load_test()
submission = load_submission()

# 2. 预处理 (OSI重建, 风向编码)
preprocess(train_data)
preprocess(test_data)

# 3. 构建特征 (有缓存则加载, 无则计算并存盘)
X_train, y_train, meta_train = get_or_build(train_data, is_train=True)
X_test, meta_test = get_or_build(test_data, is_train=False)
log(f"特征: train {X_train.shape}, test {X_test.shape}, version={FEATURE_VERSION}")

# 4. CV设置
folds = get_cv_folds(meta_train)

# 5. 训练4个模型 + 记录CV结果
models = {}
for horizon in HORIZONS:
    model, oof_pred, fold_metrics = train_one_horizon(
        X_train, y_train[horizon], folds, params
    )
    models[horizon] = model
    log_cv_results(horizon, fold_metrics, summary=compute_metrics(y_train[horizon], oof_pred))

# 6. 基线对比 + 记录
baselines = compute_all_baselines(y_train, meta_train)
log_baselines(baselines)

# 7. 预测 + 记录预测分布
preds = {h: predict(models[h], X_test) for h in HORIZONS}
for h in HORIZONS:
    log_prediction_summary(h, pred_stats(preds[h]))
    log_feature_importance(h, feature_importance(models[h], top_n=15))

# 8. 填充提交文件
fill_submission(submission, preds, meta_test)

# 9. 保存
save_submission(submission, 'submission_phase1.csv')
log_notes("Phase 1 baseline run, default params.")
log_run_footer()
```

> **调参时**: 步骤1–3只执行一次(首次)。后续调参只需重复步骤4–5,直接从 parquet 加载特征,秒级启动。每次调参自动追加一段日志,含超参和CV结果,方便回溯对比。

> **重建特征时**: 在 config.py 中将 `FEATURE_VERSION` 从 `'v1'` 改为 `'v2'`,或运行 `python main.py --rebuild-features`。旧缓存保留不删,可回退。

> **查日志**: `grep "RUN" logs/experiments.log` 快速定位每次运行; `grep "Mean RMSE" logs/experiments.log` 横向对比各次CV。

---

## 9. 关键风险与对策

| 风险 | 对策 |
|---|---|
| 过拟合 (239县×1事件) | min_child_samples=50, L1/L2正则, feature_fraction=0.8, 限制num_leaves |
| hours_since_obs 增大后信息衰减 | 该特征让模型学习衰减; 后期主要靠气象特征驱动 |
| 第二波风暴(March 16)预测难 | 气象统计量(2.C)应能捕捉; 观测窗口无第二波信息 |
| t+48h 难度大(相关0.056) | 独立模型; 不依赖持续性; 依赖气象和县级基线 |
| 零膨胀(43%) | Huber loss天然处理; Phase 1不做两阶段; 若CV差则Phase 1.5加分类器 |
| 测试集OSI重建精度 | 直接用存储组件值代入公式(验证精度<0.00005) |

---

## 10. 预期性能

| Horizon | 预期RMSE | 主要信息来源 |
|---|---|---|
| t+1h | ~0.01–0.02 | 持续性(last_osi) + 当前气象 |
| t+6h | ~0.02–0.03 | last_osi + 未来6h气象统计 |
| t+24h | ~0.03–0.04 | 气象统计(第二波信号) + 县级基线 |
| t+48h | ~0.03–0.05 | 气象统计 + 县级基线 (最难) |

> OSI的std约为0.03, 零基线的RMSE即为0.03。模型需要在t+1h显著优于0.03, 在t+48h至少不差于0.03。

---

## 11. 日志格式示例

每次运行追加一段到 `logs/experiments.log`,格式如下:

```
================================================================================
RUN 2026-08-31 14:30:15 | FEATURE_VERSION=v1 | SEED=42
================================================================================

--- Config ---
feature_version: v1
feature_count: 79
train_samples: 34416 (t01h: 34177, t06h: 32982, t24h: 28680, t48h: 22944)
test_samples: 9072
lightgbm_params:
  objective: huber, lr: 0.05, num_leaves: 31, min_child_samples: 50
  feature_fraction: 0.8, bagging_fraction: 0.8, lambda_l1: 0.1, lambda_l2: 1.0

--- CV Results ---
[osi_target_t01h]
  Fold 1: RMSE=0.0123, MAE=0.0045  (best_iter=347)
  Fold 2: RMSE=0.0118, MAE=0.0042  (best_iter=412)
  Fold 3: RMSE=0.0125, MAE=0.0048  (best_iter=389)
  Fold 4: RMSE=0.0121, MAE=0.0044  (best_iter=401)
  Fold 5: RMSE=0.0119, MAE=0.0043  (best_iter=375)
  Mean RMSE=0.0121 ± 0.0003, Mean MAE=0.0044 ± 0.0002

[osi_target_t06h]
  Fold 1: RMSE=0.0215, MAE=0.0082  (best_iter=523)
  ...
  Mean RMSE=0.0210 ± 0.0004, Mean MAE=0.0080 ± 0.0003

[osi_target_t24h]
  ...
  Mean RMSE=0.0320 ± 0.0008, Mean MAE=0.0125 ± 0.0005

[osi_target_t48h]
  ...
  Mean RMSE=0.0380 ± 0.0012, Mean MAE=0.0150 ± 0.0007

--- Baselines ---
                  t01h    t06h    t24h    t48h
Zero:            0.0300  0.0300  0.0300  0.0300
Persistence:     0.0150  0.0250  0.0350  0.0400
Mean:            0.0300  0.0300  0.0300  0.0300
Model:           0.0121  0.0210  0.0320  0.0380

--- Feature Importance (top 10) ---
[osi_target_t01h]
  1. last_osi              gain=1523.5
  2. gust_max_next_1h      gain=845.2
  3. hours_since_obs        gain=612.3
  4. osi_max_72h            gain=489.1
  5. gust_t                 gain=423.7
  ...

[osi_target_t48h]
  1. gust_max_next_48h      gain=1820.3
  2. log_customers          gain=945.6
  3. osi_mean_72h           gain=723.4
  ...

--- Prediction Summary ---
[osi_target_t01h] min=0.0000, max=0.3200, mean=0.0060, zero%=42.1%
[osi_target_t06h] min=0.0000, max=0.2850, mean=0.0061, zero%=41.8%
[osi_target_t24h] min=0.0000, max=0.2500, mean=0.0063, zero%=41.5%
[osi_target_t48h] min=0.0000, max=0.2400, mean=0.0068, zero%=40.2%

--- Notes ---
Phase 1 first run, default params. t+1h beats persistence by 19%.
t+24h/t+48h barely better than zero baseline. Next: tune num_leaves, try two-stage.

================================================================================
```

> 快速查询:
> - `grep "RUN" logs/experiments.log` — 列出所有运行时间戳
> - `grep "Mean RMSE" logs/experiments.log` — 横向对比各次CV
> - `grep "Notes" -A2 logs/experiments.log` — 查看每次备注
