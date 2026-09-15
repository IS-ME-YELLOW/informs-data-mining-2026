# Phase 2-F：v1.5.6 LightGBM + DEM + 空间 GAT 残差模型

本目录是 `code_phase2_dem_eval` 的独立 v1.5.6 版本，不覆盖原 DEM-GAT 模型及其输出。
LightGBM 和 GAT 均使用 `cache/features_*_v1.5.6.parquet`，共 163 个 Phase-1 特征；LightGBM 直接预测 OSI，使用 v1.5.6 的 Huber + RMSE 配置。v1.5.6 在 v1.5.5 的 141 列基础上增加 `pct_unmapped`、`pct_forest_classified` 和每个 horizon 的窗口长度、完整性、降水率、强阵风比例、目标时刻天气可用性，共 22 列。
由于仓库中的 v1.5.6 文本 booster 在当前 LightGBM 4.6.0 环境中报 `Model format error`，本版本不加载这些冻结文件，而是使用完全相同的 v1.5.6 特征、Huber 参数、县级 CV 和 `<0.001` 置零后处理重新训练最终 booster，并保存到本目录。因此它是 v1.5.6 的兼容重训复现，不是原冻结文件本身。
它根据 `evaluation_procedure.pdf` 做了三项调整：

1. alpha 使用严格嵌套县级 CV 选择：外层验证县只用于最终评分，MAE 仅作报告；
2. direct LightGBM 使用 v1.5.6 冻结配置的 Huber objective 和 RMSE evaluation metric；
3. GAT 默认使用 MSE，并同时输出 RMSE、MAE、MSE、MedAE、Bias、误差标准差、R²、最大绝对误差和有效样本数。

官方排名规则：四个 horizon 分别计算 RMSE，最终排名是四个 horizon 的平均名次；若平均名次相同，用 t+1h RMSE 决胜。官方可评分行数为 9009、8694、7560、6048。本版本额外生成 `submission_audit.csv` 检查这些行数。

本文档对应 code_phase2_dem 当前实现，逐模块说明输入、内部结构、数学理论和输出。它是原 code_phase2 的独立版本，不覆盖原有模型及结果。

## 1. 总体结构

本模型不让 GAT 直接预测 OSI，而是先由 LightGBM 产生主预测，再由空间 GAT 学习主模型的空间残差：

~~~text
clean Phase-1 features + weather + observed outage summaries
                         ↓
                 LightGBM base b(v,t,h)
                         ↓
          residual r(v,t,h) = y(v,t,h) - b(v,t,h)
                         ↓
       county graph + DEM features + edge-aware GAT
                         ↓
                 correction c(v,t,h)
                         ↓
            prediction = clip(b + α c, 0, 0.65)
~~~

其中 v 是县节点，t 是预测起点，h 是 1/6/24/48 小时 horizon，y 是真实 OSI，b 是 base prediction，c 是 GAT 输出的残差修正。

最终公式为：

\[
\hat y(v,t,h)=\operatorname{clip}(b(v,t,h)+\alpha_hc(v,t,h),0,0.65).
\]

每个 horizon 单独选择 alpha。

## 2. 模块输入输出总览

| 文件 | 输入 | 内部职责 | 输出 |
|---|---|---|---|
| config.py | 路径、时间范围、超参数 | 全局配置 | 各模块使用的常量 |
| build_terrain.py | USGS DEM、县界 SHP/DBF | 下载瓦片、县级地形统计 | county_terrain.csv、manifest |
| data.py | parquet、meta、DEM CSV | 行对齐、构造县×时间节点张量 | GAT features、行索引 |
| base_model.py | clean features、OSI 标签、county folds | 训练或读取 LightGBM base | OOF/test base predictions |
| spatial.py | 县 DBF/SHP、FIPS | 构建 kNN 和共享边界图 | edge_index、edge_attr、坐标 |
| gat_model.py | 节点特征、图、residual | 两层 edge-aware multi-head GAT | residual correction |
| metrics.py | y_true、y_pred | RMSE、MAE、clip、alpha 评分 | 指标 |
| main.py | 上述所有输出 | 组织训练、OOF、alpha、submission | 模型、CV、提交文件 |

## 3. config.py：全局配置

### 3.1 数据路径

~~~text
cache/features_train_v1.5.6.parquet
cache/features_test_v1.5.6.parquet
cache/meta_train_v1.5.6.parquet
cache/meta_test_v1.5.6.parquet
cache/targets_train_v1.5.6.parquet
data/geo/c_16ap26.dbf
data/geo/c_16ap26.shp
data/geo/county_terrain.csv
~~~

v1.5.6 clean cache 已经过预测窗口因果审计。DEM CSV 按 fipsCode 连接，是静态县级信息。

### 3.2 时间设置

~~~python
PRED_START = 72
PRED_END = 216
HORIZON_HOURS = {1h: 1, 6h: 6, 24h: 24, 48h: 48}
OSI_MAX = 0.65
~~~

预测起点为 hour_idx=72,...,215，共 144 个时间片。对于 horizon h，只有：

\[
t+h\le215
\]

的起点有完整标签；最后 h 个起点不进入训练和评估，提交中相应位置为 NaN。

### 3.3 LightGBM 配置

~~~text
objective         = huber
learning_rate     = 0.05
num_leaves        = 31
min_child_samples = 50
feature_fraction  = 0.8
bagging_fraction  = 0.8
bagging_freq      = 5
lambda_l1         = 0.1
lambda_l2         = 1.0
rounds            = 2000
early stopping    = 100
~~~

Huber 损失为：

\[
L_\delta(r)=
\begin{cases}
\frac12r^2,&|r|\le\delta,\\
\delta(|r|-\frac12\delta),&|r|>\delta.
\end{cases}
\]

小误差使用平方损失，大误差近似绝对值损失，降低极端停电样本对树模型的破坏性。

### 3.4 GAT 配置

~~~text
kNN k             = 8
hidden            = 24
第一层 heads       = 4
dropout           = 0.12
learning rate     = 0.003
weight decay      = 2e-4
epochs            = 220
patience          = 35
alpha grid        = 0, .05, .10, .20, .35, .50, .75, 1.0
~~~

正式训练保持 5 个县折、全部 144 个时间片和 220 epochs 上限；patience 只是 early stopping。

## 4. build_terrain.py：DEM 特征构建

### 4.1 输入和下载

输入：

1. USGS TNM 的 National Elevation Dataset (NED) 1 arc-second，约 30m；
2. 现有县界 c_16ap26.shp/.shx/.dbf；
3. Indiana=18、Ohio=39、Pennsylvania=42、West Virginia=54。

代码按州查询 USGS，按 1°×1° tile 选择最新版本，当前覆盖 302 个县并下载 75 个 GeoTIFF。清单保存于：

~~~text
data/geo/dem_3dep_1arcsec/manifest.json
~~~

已经存在的完整瓦片会被跳过，所以默认命令支持断点续传。

### 4.2 海拔统计

对县 G_v 内有效 DEM 像元 z_i，代码在线累计 count、sum、sumsq、min、max：

\[
\bar z_v=\frac1n\sum_i z_i,
\qquad
\sigma_v=
\sqrt{\frac1n\sum_i z_i^2-\bar z_v^2}.
\]

输出：

~~~text
elevation_mean_m
elevation_std_m
elevation_min_m
elevation_max_m
~~~

单位为米，nodata 不参与统计。

### 4.3 坡度

GeoTIFF 是经纬度坐标，代码按纬度 phi 将像元大小转换为米：

\[
\Delta x\approx111320\cos(\phi)|\Delta\lambda|,
\qquad
\Delta y\approx110540|\Delta\phi|.
\]

高程梯度：

\[
s_x=\frac{\partial z}{\partial x},
\qquad
s_y=\frac{\partial z}{\partial y}.
\]

坡度：

\[
slope_{deg}
=\frac{180}{\pi}\arctan\sqrt{s_x^2+s_y^2}.
\]

输出：

~~~text
slope_mean_deg
slope_std_deg
~~~

### 4.4 terrain_ruggedness

代码把地形崎岖度定义为县内有效中心像元到有效 8 邻居的平均绝对高程差：

\[
R_v=
\operatorname{mean}_{i\in G_v}
\left[
\frac1{|N_8(i)|}
\sum_{j\in N_8(i)}|z_i-z_j|
\right].
\]

单位为米。最终 CSV：

~~~text
data/geo/county_terrain.csv
~~~

字段：

~~~text
fipsCode
elevation_mean_m
elevation_std_m
elevation_min_m
elevation_max_m
slope_mean_deg
slope_std_deg
terrain_ruggedness
~~~

DEM 为静态特征，不涉及预测期间的未来状态。

## 5. data.py：样本对齐和节点特征

### 5.1 load_cached_data()

输入五个 parquet：

~~~text
X_train, X_test, meta_train, meta_test, targets_train
~~~

内部统一 FIPS 为五位字符串，保留 hour_idx、timestamp_et、stateAbbr。

输出：

~~~text
X_train, X_test, meta_train, meta_test, y
~~~

### 5.2 load_terrain_features()

输入 county_terrain.csv。

内部检查：

- 7 个字段是否存在；
- FIPS 是否重复；
- 地形值是否全部有限；
- 以 fips_str 作为索引。

输出：

\[
T\in\mathbb R^{302\times7}.
\]

### 5.3 make_county_time_view()

输入：

~~~text
X_train, X_test
meta_train, meta_test
base_train, base_test
horizon
coords
    terrain
~~~

输出：

~~~text
features
base
train_rows
test_rows
all_fips
feature_names
~~~

其中 features 是县-时间规则张量。DEM 版本加入邻居统计前的维度：

\[
F_0=1+163+2+7=173.
\]

| 特征组 | 维度 | 说明 |
|---|---:|---|
| base prediction | 1 | 当前 horizon 的 OSI 主预测 |
| clean Phase-1 features | 163 | 天气、历史观测、静态特征等 |
| coordinates | 2 | 纬度、经度 |
| DEM | 7 | 海拔、坡度、崎岖度统计 |

州标识（`stateAbbr`）不进入模型。它只能用于数据整理或分层交叉验证；坐标是独立的数值空间位置，DEM 列是按 FIPS 连接后的地形数值。

时间索引：

\[
\tau=hour\_idx-72.
\]

缺失 clean feature 用 0 填充；地形缺失直接报错。

### 5.4 add_neighbor_feature_aggregates()

最多选择 16 个字段：

~~~text
last_osi, last_P_t, last_D_t, last_N_t, last_R_t
osi_mean_72h, osi_max_72h, osi_trend_last6h
gust_t, wind_speed_t, tp_t, rain_t
gust_max_next_h, gust_mean_next_h
wind_speed_max_next_h, total_tp_next_h
~~~

对边 u→v：

\[
\bar x_v(t)=\frac1{|N(v)|}\sum_{u\in N(v)}x_u(t),
\qquad
\Delta x_v(t)=\bar x_v(t)-x_v(t).
\]

同时加入邻居均值和自身差异，最多新增 32 维。因此：

\[
F=173+32=205.
\]

时间规则：

- 邻居 outage/OSI/P/N/D/R 只来自 hour_idx<72；
- 未来天气只使用比赛允许的 weather forecast；
- DEM 当前直接作为节点特征，不额外做 DEM 邻居统计；
- GAT 的消息传递会自然学习地形相近邻居的影响。

## 6. base_model.py：LightGBM 基模型

本版本以 v1.5.8 的 163 个特征和 `balanced_v1` 五折为基线协议。支持 `direct`（C0 直接 OSI）、`component_v158`（C1 分量路线）以及兼容旧命令的 `component_v21`。最后一个入口不再读取旧 v2.1 模型，而是转到严格的 v1.5.8 分量重训。

### 6.1 direct 模式

train_base_models() 对每个 horizon 单独训练 OSI LightGBM。

输入：

~~~text
 X_train    : [n_train, 163]
y[horizon] : [n_train]
X_test     : [n_test, 211]
5 个 county-grouped folds
~~~

每个外层 fold 只用外层训练县标签。外层训练县内部再使用其余四个固定县折生成 cross-fitted base prediction；外层验证县和测试县由只在外层训练县拟合的模型预测。这样整个外层 GAT 图中的 base prediction 都没有接触外层验证标签。保存：

~~~text
oof            : [n_train]
test           : [n_test]
train_by_fold  : [5, n_train]
test_by_fold   : [5, n_test]
fold_metrics
summary
~~~

OOF residual：

\[
r_i=y_i-\operatorname{OOF\_LGBM}(x_i).
\]

完成 5 折后，使用所有训练县和平均最佳迭代轮数训练 full model，保存：

~~~text
outputs/models/lightgbm_<horizon>.txt
~~~

### 6.2 component_v158 / component_v21 模式

命令：

~~~bash
--base-mode component_v158
~~~

`component_v21` 也可用于兼容旧命令，但实际执行同一套 v1.5.8 分量重训。它不读取：

~~~text
versions/v2/v2.1/oof_osi_predictions.csv
versions/v2/v2.1/test_component_predictions.csv
~~~

每个 horizon 独立训练四个 v1.5.8 component LightGBM，预测 `P_t/N_t/D_t/R_t` 后组合 OSI。组件标签来自 v1.5.8 的县内 horizon 平移文件；各组件也执行同样的外层/内层 cross-fitting。

\[
b=0.40P_t+0.35N_t+0.25D_t-0.10R_t.
\]

test prediction 再做：

\[
b\leftarrow\operatorname{clip}(b,0,0.65),
\qquad b<0.001\Rightarrow b=0.
\]

组件模型输出再统一执行 `[0, 0.65]` clip 和 `<0.001` 置零；该后处理与 direct、GAT OOF、提交文件一致。旧 v2.1 artifact 不再作为 Phase 2 基线，因为它的 141 个 v1.5.2 特征和已生成 OOF 不能保证当前外层图隔离。

## 7. spatial.py：空间图

### 7.1 节点

节点集合：

\[
V=\{\text{302 个县 FIPS}\}.
\]

其中 239 个训练县、63 个测试县。测试县无标签但保留在图中。

### 7.2 边

build_spatial_graph() 合并：

1. 每个县的 8 个中心点 kNN；
2. shapefile 中真实共享县界的县对；
3. self-loop。

kNN 和县界边均双向加入，最终 3,028 条有向边。

### 7.3 edge_attr

每条边有 4 维：

~~~text
distance_scaled
bearing_sin
bearing_cos
shared_county_border
~~~

距离：

\[
d_{uv}=111\sqrt{(\Delta lat)^2+
(\cos(\overline{lat})\Delta lon)^2}\quad km,
\]

\[
e^{(1)}_{uv}=\min(d_{uv}/500,2).
\]

方向：

\[
\theta_{uv}
=\operatorname{atan2}
(\cos(\overline{lat})\Delta lon,\Delta lat),
\]

\[
e^{(2)}_{uv}=\sin\theta_{uv},
\qquad
e^{(3)}_{uv}=\cos\theta_{uv}.
\]

第四维为共享县界标记，共享为 1，否则为 0；self-loop 标记为 0。

## 8. gat_model.py：edge-aware multi-head GAT

### 8.1 输入

~~~text
features  : [T,N,F] = [144,302,205]
residual  : [T,N]
train_mask: [T,N]
edge_index: [2,3028]
edge_attr  : [3028,4]
~~~

代码将 features 展平为：

\[
X\in\mathbb R^{(T N)\times F}.
\]

空间边在每个时间快照复制，形成 block-diagonal graph。没有跨时间 edge；时间动态由天气和时间片特征表达。

### 8.2 第一层

输入 205，hidden=24，heads=4，拼接输出 96。

\[
z_u^k=W^kx_u.
\]

edge-aware attention score：

\[
e_{uv}^k=
\operatorname{LeakyReLU}
\left(
(a_s^k)^Tz_u^k+
(a_d^k)^Tz_v^k+
(w_e^k)^Te_{uv}
\right).
\]

对目标节点入边 softmax：

\[
\alpha_{uv}^k=
\frac{\exp(e_{uv}^k)}
{\sum_{q\in N_{in}(v)}\exp(e_{qv}^k)}.
\]

消息聚合：

\[
h_v^k=\sum_{u\in N_{in}(v)}
\alpha_{uv}^kz_u^k.
\]

四个 head concat 后加 bias，使用 ELU。attention 权重 dropout=0.12。

### 8.3 第二层和 skip connection

第二层接收 96，1 head，输出 24：

\[
q_v=\operatorname{ELU}
\left(\sum_{u\in N_{in}(v)}
\alpha_{uv}Wz_u+b\right).
\]

然后：

\[
q_v\leftarrow q_v+W_{skip}x_v.
\]

因此 DEM 可以通过邻居 attention 传播，也可以通过节点自身 skip path 直接影响结果。

### 8.4 输出头

~~~text
Linear(24,24)
GELU
Dropout(0.12)
Linear(24,1)
~~~

输出：

\[
c_v=\operatorname{MLP}(q_v),
\]

reshape 为：

~~~text
correction_grid : [144,302]
~~~

这是 residual correction，不是最终 OSI。

## 9. GAT 残差学习

### 9.1 residual 标准化

每个 fold：

\[
s=\max(\operatorname{std}(r_{train}),0.003),
\qquad
r'=r/s.
\]

GAT 学习 r′，推理后乘回 s：

\[
c=s\cdot c'.
\]

### 9.2 加权 MSE

\[
w_i=1+4\cdot\operatorname{clip}(|r'_i|/2.5,0,1),
\]

\[
L=\frac1{|M|}
\sum_{i\in M}w_i(\hat r'_i-r'_i)^2.
\]

M 是当前 fold 允许监督的县-时间单元。大 residual 权重更高，强调严重停电误差。

### 9.3 mask

验证 fold 中：

- 训练县进入监督；
- 验证县移除；
- 没有完整标签的最后 h 个时间片移除；
- 非有限标签移除；
- 测试县永远不监督。

测试县可接收训练县消息，但测试标签不进入 loss。

## 10. main.py：训练、OOF 和推理

### 10.1 每个 horizon

依次对 1h、6h、24h、48h：

1. 取得 base OOF/test prediction；
2. 构造县×时间节点张量；
3. 加入邻居统计；
4. 构造 y_grid、b_grid；
5. 计算 residual：

\[
r_{grid}(t,v)=y_{grid}(t,v)-b_{grid}(t,v).
\]

### 10.2 五折外层 GAT OOF

每个县折：

1. 将验证县设为 held-out；
2. 使用该 outer fold 的 base prediction 构造 features 和 residual；
3. 只用 outer-train 县监督 GAT；
4. 对全图 inference；
5. 只提取 outer-validation 县 correction，写入 fold_oof_corr。

外层验证标签不会参与 alpha 选择。

### 10.3 alpha 选择

CV 折与 `versions/xyy/v1.5.8` 完全对齐：`balanced_v1`、seed=42、5 个县级折。
对每个 outer fold，使用其余 4 个固定县折作为 inner validation：

~~~text
outer fold i
├─ inner folds j != i: 训练 GAT并预测 inner validation correction
├─ 汇总其余4个inner validation预测
├─ 只在outer-train标签上选择 alpha_i
└─ outer validation: 固定 alpha_i，只评分
~~~

最终全训练集 GAT 使用 outer-fold 内部选择得到的 alpha 众数；不会再使用全部 outer OOF 标签重新选择 global alpha。

候选：

\[
\mathcal A=\{0,0.05,0.10,0.20,0.35,0.50,0.75,1.0\}.
\]

\[
\hat y_\alpha=\operatorname{postprocess}(b+\alpha c),
\qquad \operatorname{postprocess}(z)=\mathbf 1_{\operatorname{clip}(z,0,0.65)\ge 0.001}\operatorname{clip}(z,0,0.65).
\]

\[
RMSE=\sqrt{\frac1n\sum_i(y_i-\hat y_i)^2},
\qquad
MAE=\frac1n\sum_i|y_i-\hat y_i|.
\]

代码按比赛官方 RMSE 最小化；MAE、R² 等只用于报告：

\[
J(\alpha)=RMSE(\alpha).
\]

候选中包含 alpha=0，因此 GAT 没有稳定增益时会自动退回 base。每个 inner selection、outer evaluation 和最终 alpha 都写入：

~~~text
outputs/alpha_selection.csv
~~~

### 10.4 final GAT 和 test prediction

嵌套 OOF 完成后，使用全部训练县 OOF residual 训练 final GAT：

- 所有训练县可监督；
- 测试县无标签但在图中；
- mean/std 只由训练县-时间单元计算；
- 全图推理测试县 correction。

\[
\hat y_{test}
=\operatorname{clip}(b_{test}+\alpha c_{test},0,0.65).
\]

每个 horizon 保存：

~~~text
outputs/models/gat_residual_<horizon>.pt
~~~

其中包含 state_dict、feature mean/std、edge_index、edge_attr、嵌套选择后的 alpha、residual_scale、fit_info。

## 11. metrics.py

输入：

~~~text
y_true : 真实 OSI
y_pred : 预测 OSI
~~~

metrics() 返回 RMSE 和 MAE，只在两者同时有限的位置计算。post_process_osi() 统一执行 [0,0.65] 裁剪和小于 0.001 置零。alpha 只按官方 RMSE 选择，MAE 仅用于诊断。

## 12. 输出文件

~~~text
code_phase2_dem/outputs/
├─ models/
│  ├─ gat_residual_osi_target_t01h.pt
│  ├─ gat_residual_osi_target_t06h.pt
│  ├─ gat_residual_osi_target_t24h.pt
│  └─ gat_residual_osi_target_t48h.pt
├─ submission_phase2_dem_gat.csv
├─ cv_summary.csv
├─ alpha_selection.csv
└─ run_metadata.json
~~~

submission 的关键保证：

- 列结构和行顺序与 sample_submission 一致；
- 通过 fipsCode、timestamp_et 对齐；
- 预测窗口外的目标列为 NaN；
- 有效预测值在 [0,0.65]。

## 13. 时间因果与泄漏控制

1. 标签只进入 LightGBM/GAT 监督损失。
2. 预测窗口外 outage、OSI、P/N/D/R、lag、target 不作为输入。
3. 自身和邻居 outage summary 只来自 hour_idx<72。
4. 未来只使用比赛允许的天气特征。
5. DEM 是静态地形数据，没有比赛期间未来状态。
6. GAT fold 只监督训练县，验证县用于 OOF correction。
7. feature mean/std 只使用当前 fold 训练县-时间单元。
8. 测试县可参与消息传递，但测试标签不进入 loss 或 alpha 选择。

## 14. 运行命令

DEM 尚未生成时先运行：

~~~bash
python code_phase2_dem/build_terrain.py --workers 8
~~~

主训练
~~~bash
conda activate myenv
pip install -r code_phase2_dem/requirements.txt

python code_phase2_dem/main.py \
  --base-mode direct \
  --device cuda \
  --epochs 220 \
  --patience 35 \
  --time-stride 1 \
  --k 8
~~~
本机若没有 CUDA，可用 `--device cpu` 做验证，但正式结果建议使用 GPU。

使用自包含 direct LightGBM：

~~~bash
python code_phase2_dem/main.py \
  --base-mode direct \
  --device cuda \
  --epochs 220 \
  --patience 35 \
  --time-stride 1 \
  --k 8
~~~
