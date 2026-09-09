# v1.5.5：已确认特征错误修复与可复用数据集

日期：2026-09-08。基于 v1.5.2 完整 141 列特征修复，对应原基线模型 `versions/v1.5.4/v1.5.2` 所用的特征集合。没有采用 v1.5.3 的删列结果。

本版已实际生成训练、测试、目标及元信息缓存，默认构建方式为：复用已核验的 v1.5.2 缓存，重算观测摘要并替换确实改变的字段，重新构建外部县级表并修正 n_utilities。文件路径、匹配失败处理和缓存写入保护也已修复。

## 1. 实际改动及原因

### 1.1 n_utilities：跨州同名县与县名异写

原逻辑只按 County 汇总 EIA Utility Number，跨州同名县被合并；名称匹配失败还会默认为 1。

现在先按 State 和规范化县名分组，对 Utility Number 计数去重，再与 Census 县参考表连接到 FIPS。规范化仅处理大小写、空格、标点，不使用模糊匹配。匹配键在同一州内必须唯一，县名或公司编号缺失、县未匹配时直接报错，不能静默填 1。

实际结果：302/302 个县成功匹配，129 个县数值改变；其中训练 99 个县、14,256 行，测试 30 个县、4,320 行。

| 县 | FIPS | 旧值 | 新值 |
|---|---|---:|---:|
| IN Adams | 18001 | 10 | 4 |
| OH Adams | 39001 | 10 | 4 |
| PA Adams | 42001 | 10 | 2 |
| IN LaGrange（EIA: Lagrange） | 18087 | 1 | 6 |
| IN St. Joseph（EIA: St Joseph） | 18141 | 1 | 8 |

完整县级映射、原始 EIA 县名、旧值、新值与 changed 标志见 [eia_county_matches.csv](eia_county_matches.csv)。n_utilities 表示 EIA 2024 服务区域表中覆盖该县的不同公司数量，不能据此解释为输配电网络规模。

### 1.2 观测期 OSI：统一训练和测试的精度

两侧仅在 hour_idx=0..71 内执行相同计算：

```text
OSI_observed = round(max(0, 0.40*P_t + 0.35*N_t + 0.25*D_t - 0.10*R_t), 4)
```

使用文件中存储的 P/N/D/R，不从 outageCount 的净变化推导 N/R。训练预测窗口已有 osi、官方 target 列及原始 CSV 均未修改；测试预测窗口不会从将来组件生成可用的 osi。观测期组件缺失或非有限值会报错。

实际核对：训练 239 县前 72 小时，重建后的四位小数 OSI 与官方已有 osi **逐值完全一致**。模拟把全部训练县转换成测试格式、删除 osi 后再预处理，31 个观测摘要也完全一致。

测试数据原来重建后未舍入，修复改变 10 个 OSI 派生摘要。县平均零值比例从 **28.3289% → 50.8377%**。这是修复输入计算口径造成的变化，没有修改真实目标或填补停电观测。

`hours_since_peak` 在 4 个测试县发生变化：39047、42037、42071、42093。其中 PA Montour（42093）的 72 小时 OSI 舍入后全部为零，按已有公式不存在正值峰值，因此该县 144 行的 hours_since_peak 为 NaN。其余峰值时间变化来自舍入后并列最大值取首次出现的位置，保持原有 argmax 定义。

### 1.3 只修文字含义、保留数值的项目

- `gust_peak4h_mean_next_{h}h` 仍为窗口内最高 min(4,有效点数) 个风速的均值，**不要求连续**。本版保留列名和数值，修正文档与代码注释；没有暗中替换为连续 4 小时公式。
- `next_h` 聚合仍取闭区间 `[t,t+h]`，通常 h+1 个点，越过 hour_idx=215 时截短。h=1 通常包含两点，末尾一点。
- `*_at_t{h}h` 超出数据末尾仍为 NaN。只有相同 horizon 的目标必然也无效；其他 horizon 的有效训练行可能含这些缺失特征。未用零填补未知天气。
- 基础特征实际为 129 列、观测摘要为 31 列，修正旧注释中的 114/130/32 等计数。
- 土地覆盖和树冠表按用户确认的权威来源直接使用，只校验本地 FIPS、数值和连接；不重新追溯来源、不改变分类或分母。土地覆盖并非穷尽所有类别的百分比，pct_unmapped 的解释仍见此前审视报告。

## 2. 与 v1.5.2 的最终数值差异

列名、顺序均保持原版 141 列。以下为逐值比较，不只比较近似值。未列出的列全部保持原缓存数值，包括气象、时间、人口密度、土地覆盖、树冠及 forest_x_customers。

| 数据 | 特征 | 改变县数 | 改变行数 |
|---|---|---:|---:|
| train | n_utilities | 99 | 14,256 |
| test | n_utilities | 30 | 4,320 |
| test | last_osi | 58 | 8,352 |
| test | osi_mean_72h | 63 | 9,072 |
| test | osi_max_72h | 63 | 9,072 |
| test | osi_std_72h | 63 | 9,072 |
| test | pre_event_mean_osi | 62 | 8,928 |
| test | storm_onset_max_osi | 63 | 9,072 |
| test | storm_onset_mean_osi | 63 | 9,072 |
| test | osi_trend_last6h | 61 | 8,784 |
| test | frac_zero_72h | 61 | 8,784 |
| test | hours_since_peak | 4 | 576 |

机器可读明细见 [feature_changes.csv](feature_changes.csv)，含精确变化行数、rtol/atol=1e-12 下变化行数、最大绝对差及 NaN 状态改变数量。涉及 NaN 转换的最大差只统计两侧都有限的值，不能用该最大差代表 NaN 转换。

重算未变摘要时，当前数值库的 polyfit 可能产生约 1e-15 的浮点差异。默认构建逐列比较，完全在 rtol/atol=1e-12 内的列保留父缓存原值，避免将数值库舍入误差混入本轮修复。`--from-raw` 会重算这些值，与默认路径按上述容差一致，不承诺文件字节一致。

## 3. 数据文件

所有路径均相对项目根目录。

| 文件 | 规模/用途 |
|---|---|
| `cache/features_train_v1.5.5.parquet` | 34,416 × 141，239 县 × 144 小时 |
| `cache/features_test_v1.5.5.parquet` | 9,072 × 141，63 县 × 144 小时 |
| `cache/targets_train_v1.5.5.parquet` | 34,416 × 4，保留官方目标及 NaN |
| `cache/meta_train_v1.5.5.parquet` | 34,416 × 7，FIPS、时间、分组元信息等 |
| `cache/meta_test_v1.5.5.parquet` | 9,072 × 6 |
| `cache/feature_names_v1.5.5.json` | 固定 141 列名称与顺序 |
| `cache/manifest_v1.5.5.json` | 输入、构建代码、输出 SHA-256、运行环境及语义 |
| `data/county_features_v1.5.5.csv` | 302 县 × 11 个外部字段，另有 FIPS 索引列 |
| `versions/v1.5.5/validation.json` | 全量数据形状、缺失模式、目标和匹配检查结果 |
| `versions/v1.5.5/test_results.json` | 回归测试结果及其对应 manifest/test 脚本指纹 |

旧 `cache/*_v1.5.2.*`、`cache/*_v1.5.3.*`、`data/county_features.csv` 保留原样，未覆盖已有模型、实验结果或提交文件。

训练四个目标的有效行数分别为：1h **34,177**，6h **32,982**，24h **28,680**，48h **22,944**。元信息和官方标签逐值不变，可以沿用 `cv/cv_assignments_balanced_v1_seed42.csv`。

## 4. 后续直接调用

在项目根目录的 Python 环境中：

```python
from code_phase1.feature_dataset import load_feature_dataset

data = load_feature_dataset()  # 默认校验输出指纹、列名、顺序、行键、缺失模式
X_train, y_train, meta_train = data.X_train, data.y_train, data.meta_train
X_test, meta_test = data.X_test, data.meta_test
```

加载只读缓存，不触发模型训练或重新构建。若想自行读取 Parquet，也可以，但这样会绕过完整校验。

`code_phase1/config.py` 的 FEATURE_VERSION 已从 v1.5.3 改为 **v1.5.5**，现有主流程的 `get_or_build` 会通过新加载器读取完整 141 列；LightGBM 参数和 balanced_v1 CV 配置未改变。

历史实验脚本若把 v1.5.2 文件名写死（包括 `versions/v2/component_experiment.py`），仍读取旧版。本次未改写历史实验。后续直接 OSI 和 P/N/D/R 分别预测再合成 OSI 的新实验，应统一调用上述加载器，并单独保留模型结果目录。y_train 仍是四个官方 OSI 目标；P/N/D/R 分量目标需沿用分量实验的目标构造，不能把这里的 y_train 当成分量目标。

## 5. 构建、校验、依赖

当前工作环境使用 Python 3.13；本版运行依赖固定在 [requirements.txt](requirements.txt)。本机复用了已有 numpy/pandas，补装 pyarrow、openpyxl、et-xmlfile、dbfread 到项目 `.python_packages/`（已被 Git 忽略），专用入口会自动查找该目录。

其他环境可在自己的虚拟环境中执行：

```powershell
python -m pip install -r versions/v1.5.5/requirements.txt
```

以下命令在项目根目录执行；构建脚本本身用绝对项目根目录定位资源，不依赖当前工作目录。

```powershell
# 已经生成；反复使用通常只需校验或直接调用加载器
python versions/v1.5.5/build_features.py --verify

# 首次生成：复用已经核验的完整 v1.5.2 父缓存
python versions/v1.5.5/build_features.py

# 显式覆盖新版本，绝不覆盖旧版本
python versions/v1.5.5/build_features.py --overwrite

# 从原始 CSV + 县级来源表重算全部 141 列，不依赖父特征缓存生成数值
python versions/v1.5.5/build_features.py --from-raw --overwrite

# 回归与集成检查，写 test_results.json
python versions/v1.5.5/test_feature_fixes.py
```

原 `versions/v1.5.2/join_external.py` 已作为兼容入口转发到上述 v1.5.5 构建器，支持相同参数。它现在生成独立修复版，不会把 141 列写到全局版本指向的剪枝缓存。

单独重建外部县级表可使用 `data/external/scripts/build_county_features.py --output <独立CSV路径>`；默认也是版本化文件名，已存在时须 `--overwrite`。完整数据集建立后应通过完整构建器更新，单独改动县级输出会被加载器的指纹检查识别，不能作为一套一致的数据直接使用。

重建输入包括 DM_Train/Test.csv、USDA rural_urban_codes.csv、Census DBF 属性表、EIA 服务区域 XLSX、用户提供的土地覆盖/树冠 CSV；旧 county_features.csv 用于旧新计数审计。无需 shp 几何数据或重新下载 GIS 文件。存在父缓存时，全量重建还会比较父版元信息、标签及未授权改变的特征；若来源变化引起额外差异，流程会报错，要求审查新版本范围。

## 6. 构建和读取保障

1. 默认复用父缓存前，检查 [parent_inputs_sha256.json](parent_inputs_sha256.json) 中原始 CSV 和 5 个父缓存指纹。源数据变化不能继续拼接旧气象特征；应显式重算并审视差异。
2. 原始数据每县必须为从 2026-03-11 起连续 216 小时，无重复县/时间键；预测缓存每县必须完整覆盖 hour_idx=72..215。
3. 外部数据要求 FIPS 唯一、完整匹配、数值有限；RUCC 为 1..9 的整数、县陆地面积为正。去掉无差别 fillna(0)，避免未匹配被伪装为真实零值。
4. 只允许本轮确认的 OSI 派生字段和 n_utilities 出现实质差异；其余列若有超容差变化则中止。
5. 标签同时与原始 CSV、存在的旧缓存逐值比对；元信息也检查原始行键与旧版一致。
6. 输出先在临时目录中生成，再替换新版本文件，manifest 最后写入。多文件替换本身不是操作系统级事务，但中途失败会使旧 manifest 校验失败，阻止读取半套数据。
7. 加载时默认验证全部输出文件 SHA-256。来源/代码指纹用于可追溯性；加载已冻结数据不会因为后续源代码变动自动重建它。回归测试另行核对来源与代码是否仍匹配构建记录。
8. 通用缓存保存器不能把基础 129 列直接覆盖 v1.5.2/v1.5.3/v1.5.5。主流程旧 `--rebuild-features` 对这些版本给出专用构建命令，避免重建后丢掉外部 12 列。

## 7. 本次验证及边界

- 全量校验训练 34,416 行、测试 9,072 行的 141 列结构、排序、唯一键、允许的 NaN 模式及标签。
- 全量检查 302 个县的 EIA 映射，并对跨州 Adams、LaGrange、St. Joseph 的正确计数作回归断言。
- 全部 239 个训练县模拟测试格式后，31 个观测摘要与训练格式完全一致。
- 抽取 1 个训练县和 2 个测试县（含无正值峰值的 42093），共 432 行，从原始数据重算全部 141 列，和已生成缓存在 rtol/atol=1e-12 下相同。
- 对上述训练县，将预测窗口的停电变量、lag、target、delta 人为改成 999，再重算特征，141 列逐值不变。
- 验证现有缓存入口可用，并拒绝用部分特征构建器覆盖完整版本。
- 最终 **9 项测试全部通过**，结果记录于 test_results.json；git diff --check 通过。

本次交付采用默认复用路径；没有另跑全县慢速 `--from-raw` 全流程，已对其核心计算与外部拼接路径做上述 432 行重算验证。此前审视报告中的全量公式复核作为父缓存已核验的依据，不能混称为本次运行了全县原始重建。

本轮没有训练 LightGBM/GRU，也没有新增未来气象特征，因此不报告新版 RMSE 提升。因训练侧只有 n_utilities 改变，后续模型对照应明确区分修复前后的训练变化与测试 OSI 精度修复。四个分量分别预测再组合 OSI 的方向保留到后续模型实验。

## 8. 文件级修改清单

| 文件 | 本轮修改 |
|---|---|
| `code_phase1/data_loader.py` | 观测期 OSI 两侧统一重建、clip、round(4)；观测组件校验；保持未来训练值及目标 |
| `code_phase1/features.py` | 修正基础维数、观测维数、top4 的注释和函数说明，未改气象公式 |
| `data/external/scripts/build_county_features.py` | 修正项目路径；直接读取 DBF 属性；州内县名规范化与 EIA 去重；严格连接校验；版本化输出及审计 |
| `versions/v1.5.2/join_external.py` | 修复失效历史入口，转发到固定 v1.5.5 完整构建器 |
| `code_phase1/feature_dataset.py` | 新增版本化构建/加载、父输入指纹、差异范围检查、暂存输出和 manifest 校验 |
| `code_phase1/config.py` | 默认特征版本改为 v1.5.5；不改模型/CV参数 |
| `code_phase1/cache.py` | 新版完整缓存读取；禁止部分重建覆盖版本化数据；通用列名记录改为实际 X.columns |
| `code_phase1/main.py` | 更新重建命令注释及参数帮助，不改训练逻辑 |
| `fundamentals/Feature_Catalog_v1.5.2.md` | 纠正 top4、h=1窗口点数、跨horizon缺失说明；链接本修复版 |
| `versions/v1.5.5/build_features.py` | 新增构建/全量重算/校验命令入口 |
| `versions/v1.5.5/test_feature_fixes.py` | 新增9项公式回归及数据集集成检查 |
| `versions/v1.5.5/requirements.txt` | 记录本版依赖版本 |
| `versions/v1.5.5/parent_inputs_sha256.json` | 固定允许复用的父缓存及原始输入指纹 |
| 本目录 README/CSV/JSON 与 `cache/*v1.5.5*`、县级新版 CSV | 本轮生成的数据与记录，详见第3节 |

前一轮审视报告、已有 data/raw 和 tmp 内容保留，没有将其混作本轮新增特征。
