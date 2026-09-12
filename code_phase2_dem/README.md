# Phase 2-D — LightGBM + spatial GAT + DEM residual

This directory is an independent DEM-enhanced copy of the existing Phase 2
code. It never overwrites `code_phase2` outputs.

USGS 3DEP/NED 1 arc-second (~30 m) DEM tiles for Indiana, Ohio, Pennsylvania,
and West Virginia have been summarized into `data/geo/county_terrain.csv`.
The seven terrain fields are joined by `fipsCode`; `terrain_ruggedness` is the
mean valid 8-neighbour absolute elevation difference in meters. Download
metadata and tile URLs are in `data/geo/dem_3dep_1arcsec/manifest.json`.

Build or rebuild the terrain table with:

```bash
conda activate myenv
python code_phase2_dem/build_terrain.py --workers 4
```

Run the full DEM model with the original training scale:

```bash
conda activate myenv
python code_phase2_dem/main.py --base-mode component_v21 --device cuda --epochs 220 --patience 35 --time-stride 1 --k 8
```

Outputs are written to `code_phase2_dem/outputs/`, including
`submission_phase2_dem_gat.csv`, `cv_summary.csv`, `alpha_selection.csv`, and
the saved GAT models. `rasterio>=1.3` is the additional dependency.

中文完整文档：[`README_CN.md`](README_CN.md)；评估报告：[`评估报告_Phase2.md`](评估报告_Phase2.md)。

This directory contains a self-contained model 2.0 built on the latest Phase 1
feature cache (`v1.5.7`). For each horizon it:

1. trains the Phase-1-compatible LightGBM model with the persisted county-level
   5-fold assignment;
2. obtains out-of-fold LightGBM predictions and defines the county-hour residual
   `y - y_lightgbm`;
3. builds a 302-county graph from `data/geo/c_16ap26.dbf` (latitude/longitude,
   symmetric 8-nearest-neighbour edges);
4. builds a hybrid spatial graph from exact county-border adjacency plus kNN
   edges, and appends neighbour weather/observed-window summary aggregates;
5. trains a two-layer multi-head GAT only on training-county residuals. The
   current iteration uses all 211 clean Phase-1 features and weighted residual
   loss to emphasize rare severe outage counties. Residual targets are
   standardized per horizon/fold and converted back before blending; and
6. adds the GAT residual correction to the LightGBM prediction, with the blend
   weight selected from held-out counties and clipped to the physical OSI range.

The GAT feature set uses only Phase-1 features that are available at prediction
time. In particular, no outage field, lag field, OSI field, or target from the
March 14–19 prediction window is read by Phase 2. Weather fields may use the
full supplied weather trajectory, as permitted by the challenge rules.

Run from the project root with the requested environment:

```powershell
conda activate myenv
python code_phase2/main.py
```

If the shell's conda installation does not switch environments on Windows, use
the environment interpreter directly:

```powershell
D:\app\anacnda1\envs\myenv\python.exe code_phase2/main.py
```

The default run keeps the full training scale: all 144 forecast snapshots,
5 county-grouped folds, k=8 neighbours, and up to 220 GAT epochs with early
stopping. `--device auto` uses CUDA when the remote environment provides it.
For a local smoke test only, use `--epochs 10 --patience 3 --time-stride 4`.

Outputs are written to `code_phase2/outputs/`, including the submission,
county-grouped CV comparison, alpha selection table, and saved GAT/LightGBM
models.

An optional stronger base is available from the existing repository artifact:

```powershell
python code_phase2/main.py --base-mode component_v21 --device auto
```

This mode uses the frozen v2.1 component LightGBM predictions (`P_t/N_t/D_t/R_t`)
and composes OSI with the official formula before fitting the GAT residual.
The default `direct` mode remains fully self-contained and is the cleanest
direct comparison with the Phase-1 LightGBM baseline.

## 当前空间特征与时间合规

当前迭代不需要新增下载数据，直接使用已有 `data/geo/c_16ap26.shp`：

- 图结构：8-nearest-neighbour 边 + 真实县界共享边；
- 边特征：距离、方向 sin/cos、是否共享县界；
- 节点空间特征：邻居平均天气和自身-邻居天气差异；
- 观测空间特征：邻居 `last_osi`、72 小时均值/峰值、趋势、`last_P_t/N_t/D_t/R_t`。

时间规则严格执行：

1. 邻居 outage/OSI/P/N/D/R 特征只从 `hour_idx < 72` 生成；
2. 预测窗口 `hour_idx >= 72` 只使用天气、静态空间和县级元数据；
3. 邻居未来天气统计可以使用完整 216 小时，因为比赛允许使用未来气象；
4. 训练标签只进入 LightGBM/GAT 的监督损失，不作为测试节点输入。

若继续增加外部数据，优先下载 USGS 3DEP 30m DEM，并整理为
`data/geo/county_terrain.csv`，字段包括 `fipsCode`、平均/标准差海拔、平均坡度和地形粗糙度。
