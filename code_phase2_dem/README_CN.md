# Phase 2-D：LGBM + 空间 GAT + DEM 残差模型

本目录是独立的 DEM 增强版本，不覆盖原有 `code_phase2` 模型。

## 1. 新增数据

已从 USGS TNM 官方接口下载 `National Elevation Dataset (NED) 1 arc-second`
（3DEP 约 30 m）覆盖 Indiana、Ohio、Pennsylvania、West Virginia 的 75 个
DEM 瓦片，并按现有县界计算 302 个县的静态地形特征：

```text
data/geo/county_terrain.csv
```

字段为：

```text
fipsCode, elevation_mean_m, elevation_std_m, elevation_min_m,
elevation_max_m, slope_mean_deg, slope_std_deg, terrain_ruggedness
```

其中 `terrain_ruggedness` 定义为县内有效像元到有效 8 邻域像元的平均绝对
高程差（米）。原始 DEM 下载清单和来源信息保存在：

```text
data/geo/dem_3dep_1arcsec/manifest.json
```

重新下载或重算：

```bash
conda activate myenv
python code_phase2_dem/build_terrain.py --workers 4
```

## 2. 模型结构

```text
Phase-1 clean features + weather + observed outage summaries
                         + county terrain (7 static DEM features)
                                      ↓
                         component LightGBM base
                                      ↓
             8-NN + county-border + edge-aware spatial GAT
                                      ↓
                         residual correction and OOF alpha
```

DEM 特征只作为静态县级输入，不含时间变化，也不涉及预测窗口外的 outage、
OSI、P/N/D/R 或 lag 信息。模型仍使用同一套 5 个县折、全部 144 个时间片和
`220` epochs 上限；服务器 GPU 上不要为了提速降低这些默认规模。

## 3. 运行

```bash
conda activate myenv
python code_phase2_dem/main.py \
  --base-mode component_v21 \
  --device cuda \
  --epochs 220 \
  --patience 35 \
  --time-stride 1 \
  --k 8
```

本机若没有 CUDA，可用 `--device cpu` 做验证，但正式结果建议使用 GPU。
所有 DEM 版本输出写入独立目录：

```text
code_phase2_dem/outputs/
```

其中最终提交文件为：

```text
code_phase2_dem/outputs/submission_phase2_dem_gat.csv
```

## 4. 依赖

除原 Phase 2 依赖外，DEM 构建需要 `rasterio>=1.3`。服务器环境安装：

```bash
pip install -r code_phase2_dem/requirements.txt
```

## 1. 模型结论

当前完整 GPU 运行版本为：

```text
8-NN + county-border + edge features + neighbor weather
+ neighbor observed outage state + static county features
+ component LightGBM base + spatial GAT residual
```

相对 `code_phase1` 的 direct-OSI LightGBM 基线，四个预测时域的 RMSE 和 MAE 均下降。GAT 的独立增益主要出现在 t+48h；t+1h、t+6h、t+24h 的 OOF blend 选择了 `alpha=0`，因此自动保留更强的 LightGBM 主模型，避免空间修正造成退化。

## 2. 完整流程

```text
DM_Train / DM_Test + v1.5.7 clean features
                  │
                  ├─ component LightGBM：预测 P_t/N_t/D_t/R_t
                  │                     ↓ 官方 OSI 公式组合
                  │                 LightGBM base
                  │
                  ├─ county graph：8-NN + 真实共享县界
                  │
                  ├─ edge-aware GAT：学习 base residual
                  │
                  └─ prediction = clip(base + alpha × GAT residual)
```

训练残差为：

```text
residual = observed_OSI_target - LightGBM_OOF_prediction
```

GAT 只在训练县上计算监督损失；测试县仅作为无标签图节点参与消息传递。

## 3. 空间图与节点特征

图节点为 302 个县，其中 239 个训练县、63 个测试县。图边包含每个县的 8 个地理近邻、真实共享县界边和自环，总边数为 3,028 条。每条边包含距离、方向 sin/cos 和是否共享县界四个属性。

当前节点特征共 250 维，包括 LightGBM base prediction、Phase 1 的 211 个 clean features、经纬度、州 one-hot、邻居平均特征以及自身减邻居平均的差异特征。邻居聚合包含前 72 小时的 `last_osi`、OSI 均值/峰值/趋势、`last_P_t/N_t/D_t/R_t`，以及当前和未来天气窗口统计。

## 4. 时间合规

1. 所有 outage、OSI、P/N/D/R 和 lag 空间特征只从 `hour_idx < 72` 的观测窗口计算；
2. `hour_idx >= 72` 不使用任何县的未来停电值；
3. 未来天气允许使用完整 216 小时，因此未来天气聚合可用于输入；
4. 训练标签只用于 LightGBM/GAT 的监督损失，不作为测试节点输入；
5. 提交中目标时间超过 3 月 19 日的行保留 NaN。

## 5. 环境与运行

```bash
conda activate phase2
cd /path/to/informs-data-mining-2026
python code_phase2/main.py \
  --base-mode component_v21 \
  --device cuda \
  --epochs 220 \
  --patience 35 \
  --time-stride 1 \
  --k 8
```

本次完整运行配置为：`seed=42`、`epochs=220`、`patience=35`、`device=cuda`、`time_stride=1`、`graph_k=8`、`feature_cache=v1.5.7`。

## 6. 输出文件

```text
code_phase2/outputs/submission_phase2_gat.csv
code_phase2/outputs/cv_summary.csv
code_phase2/outputs/alpha_selection.csv
code_phase2/outputs/run_metadata.json
code_phase2/outputs/models/
```

提交文件保持 `sample_submission.csv` 的列顺序、行顺序和标识符不变。

## 7. Base 模式

`component_v21` 使用仓库已有的 v2.1 P/N/D/R LightGBM artifact，并按官方 OSI 公式组合，是当前推荐提交模式。若需要完全自包含的 direct-OSI 对照，可运行：

```bash
python code_phase2/main.py --base-mode direct --device cuda
```

## 8. 后续可选数据

当前空间版本不需要新增数据。若继续提升，优先加入 USGS 3DEP 30m DEM 汇总的平均/标准差海拔、平均坡度和地形粗糙度，整理为 `data/geo/county_terrain.csv`，通过 `fipsCode` 连接。任何新增时变 outage 数据都不应加入预测输入。
