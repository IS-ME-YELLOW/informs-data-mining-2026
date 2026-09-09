# v1.5.6：土地覆盖 unmapped 与窗口质量特征

日期：2026-09-08。

本版直接在 v1.5.6 上继续更新：以 v1.5.5 的 141 列修复数据为父版本，加入土地覆盖 unmapped 相关列和现有未来气象窗口的质量说明列。没有训练模型，也没有改变 v1.5.5 的冻结文件。

## 新增特征

| 特征 | 公式 | 含义 |
|---|---|---|
| `pct_unmapped` | 来自 `data/county_landcover.csv` | 有效土地覆盖像元中，未进入现有主要类别的比例 |
| `pct_forest_classified` | `100 * pct_forest / (100 - pct_unmapped)` | 只在已分类像元内部计算的森林占比，用作敏感性特征 |
| `window_len_next_{h}h` | `min(h+1, 216-hour_idx)` | 现有 `[t,t+h]` 气象聚合窗口实际可用点数 |
| `is_full_next_{h}h` | `1[hour_idx+h < 216]` | 现有 `[t,t+h]` 窗口是否完整到达 `t+h` |
| `tp_rate_next_{h}h` | `total_tp_next_{h}h / window_len_next_{h}h` | 降水累计量按实际窗口长度归一后的每小时强度 |
| `gust_gt30_frac_next_{h}h` | `gust_gt30_next_{h}h / window_len_next_{h}h` | 阵风超过 30 mph 的小时占比 |
| `has_weather_at_t{h}h` | `1[hour_idx+h < 216]` | `*_at_t{h}h` 精确目标时刻气象是否存在 |

其中 `h` 取 `1, 6, 24, 48`，因此窗口质量特征一共 20 列。它们不引入新的天气来源，只解释旧窗口的可用长度和截短状态。`pct_forest` 保持原有“全部有效像元分母”口径；`forest_x_customers` 也保持不变，仍使用原始 `pct_forest`。

## 文件

| 文件 | 用途 |
|---|---|
| `cache/features_train_v1.5.6.parquet` | 34,416 行 x 163 列 |
| `cache/features_test_v1.5.6.parquet` | 9,072 行 x 163 列 |
| `cache/targets_train_v1.5.6.parquet` | v1.5.5 目标逐值复制 |
| `cache/meta_train_v1.5.6.parquet` | v1.5.5 训练元信息逐值复制 |
| `cache/meta_test_v1.5.6.parquet` | v1.5.5 测试元信息逐值复制 |
| `cache/feature_names_v1.5.6.json` | 固定 163 列顺序 |
| `cache/manifest_v1.5.6.json` | 输入、源码、输出指纹和语义说明 |
| `data/county_features_v1.5.6.csv` | v1.5.5 县级表加 `pct_unmapped` 和 `pct_forest_classified` |
| `versions/v1.5.6/feature_changes.csv` | 新增列的训练/测试统计摘要 |
| `versions/v1.5.6/validation.json` | 形状、目标、元信息和缺失模式检查 |

## 使用方式

```powershell
python versions/v1.5.6/build_features.py --verify
python versions/v1.5.6/build_features.py --overwrite
```

Programmatic loading:
代码调用：

```python
from code_phase1.feature_dataset_v156 import load_feature_dataset

data = load_feature_dataset()
X_train, y_train = data.X_train, data.y_train
```

`code_phase1/config.py` 指向 `v1.5.6`，`code_phase1/cache.py` 通过专用加载器读取这套冻结数据。历史 v1.5.5 文件保持不变。
