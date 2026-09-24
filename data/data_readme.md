# 额外空间数据说明

本文 GAT 残差模型额外使用县级空间边界、县级坐标和地形信息。相关数据用于构建县域空间图，并补充节点的静态地形特征。

## 1. 县级空间边界与坐标数据

### 本地文件

```text
data/geo/c_16ap26.dbf
data/geo/c_16ap26.shp
data/geo/c_16ap26.shx
data/geo/c_16ap26.prj
```

`.dbf` 和 `.shp` 是模型构图所需的核心文件；`.shx` 和 `.prj` 是 ESRI Shapefile 的配套索引与投影文件，随数据一并保留。

### 数据来源

该数据来自 NOAA/National Weather Service (NWS) GIS County Boundaries 数据，文件版本标识为 `c_16ap26`。直接下载地址为 [c_16ap26.zip](https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip)，数据平台入口为 [NWS GIS Counties](https://www.weather.gov/gis/Counties)。DBF 文件包含以下字段：

```text
CWA, FE_AREA, TIME_ZONE, FIPS, LON, LAT
```

论文中建议表述为：

> 本研究使用 NOAA/National Weather Service (NWS) GIS County Boundaries 数据，文件版本为 `c_16ap26`，通过 [https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip](https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip) 下载。该文件包含 `CWA`、`FE_AREA`、`TIME_ZONE`、`FIPS`、`LON` 和 `LAT` 等字段。

英文论文表述：

> County boundary and coordinate data were obtained from the NOAA/National Weather Service (NWS) GIS County Boundaries product, version `c_16ap26`, downloaded from [https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip](https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip). The accompanying DBF table contains county identifiers, including `FIPS`, as well as longitude and latitude attributes (`LON` and `LAT`). The corresponding county polygons were used to construct the spatial graph.

### 处理方式与模型用途

模型从 DBF 文件中读取县级 FIPS、经度和纬度，从 SHP 文件中读取县级多边形。上述信息用于：

1. 构建 302 个县级节点；
2. 根据县级坐标构建对称 kNN 空间邻接关系；
3. 根据县界多边形识别共享边界的县对；
4. 为图边计算归一化距离、方向编码和共享边界标识；
5. 将纬度和经度作为 GAT 节点输入中的两个静态特征。

空间图的实现见 [`code_phase2_final/m2_source/spatial.py`](../code_phase2_final/m2_source/spatial.py)，最终模型的调用见 [`code_phase2_final/final_model.py`](../code_phase2_final/final_model.py)。

## 2. USGS DEM 数据

### 本地文件

```text
data/geo/dem_3dep_1arcsec/*.tif
data/geo/dem_3dep_1arcsec/manifest.json
```

### 数据来源

原始高程栅格来自 U.S. Geological Survey (USGS) 3D Elevation Program (3DEP)，通过 The National Map (TNM) Access API 查询并下载。数据集为 **National Elevation Dataset (NED) 1 arc-second**，空间分辨率约为 30 m。

API 端点为 [https://tnmaccess.nationalmap.gov/api/v1/products](https://tnmaccess.nationalmap.gov/api/v1/products)。本研究实际使用的查询 URL 如下：

```text
https://tnmaccess.nationalmap.gov/api/v1/products?datasets=National+Elevation+Dataset+%28NED%29+1+arc-second&bbox=-88.09779357899998%2C37.77191162100007%2C-84.78469848599997%2C41.760711670000035&max=10000
https://tnmaccess.nationalmap.gov/api/v1/products?datasets=National+Elevation+Dataset+%28NED%29+1+arc-second&bbox=-84.82019805899995%2C38.40321350100004%2C-80.51869964599996%2C41.97760772700008&max=10000
https://tnmaccess.nationalmap.gov/api/v1/products?datasets=National+Elevation+Dataset+%28NED%29+1+arc-second&bbox=-80.52029418899997%2C39.71961212200006%2C-74.68949890099998%2C42.26982345500005&max=10000
https://tnmaccess.nationalmap.gov/api/v1/products?datasets=National+Elevation+Dataset+%28NED%29+1+arc-second&bbox=-82.64439392099996%2C37.20221328700006%2C-77.71949768099995%2C40.637912750000055&max=10000
```

API 返回的每个 DEM 瓦片的直接下载 URL、瓦片编号、文件大小、发布日期、更新时间和本地文件路径记录在 [`dem_3dep_1arcsec/manifest.json`](geo/dem_3dep_1arcsec/manifest.json) 中。

论文中建议表述为：

> 本研究使用 U.S. Geological Survey (USGS) 3D Elevation Program (3DEP) 的 National Elevation Dataset (NED) 1 arc-second 产品。DEM 产品通过 The National Map Access API v1（[https://tnmaccess.nationalmap.gov/api/v1/products](https://tnmaccess.nationalmap.gov/api/v1/products)）查询，并根据 API 返回的产品记录下载；空间分辨率约为 30 m。具体查询范围、DEM 瓦片 URL、发布日期和下载记录见随附的 `manifest.json` 文件。

英文论文表述：

> Elevation data were obtained from the U.S. Geological Survey (USGS) 3D Elevation Program (3DEP), specifically the National Elevation Dataset (NED) at 1 arc-second resolution (approximately 30 m). DEM products were queried through The National Map Access API v1 ([https://tnmaccess.nationalmap.gov/api/v1/products](https://tnmaccess.nationalmap.gov/api/v1/products)) using four bounding-box requests covering the study region. The returned DEM tiles were downloaded from their product-specific USGS URLs. Tile identifiers, download URLs, publication dates, and local file records are provided in the accompanying `manifest.json` file.

## 3. 县级地形特征数据

### 本地文件及构建代码

```text
data/geo/county_terrain.csv
```

`county_terrain.csv` 不是独立下载的数据源，而是由上述 USGS 3DEP/NED DEM 和 NWS 县级边界文件计算生成的派生数据。生成代码为 [`code_phase2_final\build_county_terrain_features\build_terrain.py`](..code_phase2_final\build_county_terrain_features\build_terrain.py)，从仓库根目录运行：

```bash
python code_phase2_final/build_county_terrain_features/build_terrain.py
```

该脚本读取 `data/geo/c_16ap26.dbf/.shp` 中的县级边界，通过上述 TNM Access API 查询并下载覆盖目标区域的 DEM 瓦片，再对每个县进行空间裁剪和分区统计。

生成的 7 个县级特征为：

| 特征 | 含义 | 单位 |
|---|---|---|
| `elevation_mean_m` | 县内平均高程 | m |
| `elevation_std_m` | 县内高程标准差 | m |
| `elevation_min_m` | 县内最低高程 | m |
| `elevation_max_m` | 县内最高高程 | m |
| `slope_mean_deg` | 县内平均坡度 | degree |
| `slope_std_deg` | 县内坡度标准差 | degree |
| `terrain_ruggedness` | 有效八邻域绝对高程差的平均值 | m |

坡度由 DEM 高程梯度计算；`terrain_ruggedness` 定义为县内有效像元与其有效八邻域之间的绝对高程差的平均值。统计仅使用位于县界内且在 DEM 中有效的像元。

最终 GAT 运行时直接读取 `county_terrain.csv`，不再直接读取原始 DEM TIFF。该文件为每个县节点提供 7 个随县变化、但在时间维度上保持不变的静态地形特征。

英文论文表述：

> The county-level terrain table (`county_terrain.csv`) is a derived product rather than an independently downloaded dataset. It was generated by masking the USGS 3DEP/NED DEM tiles with the NWS county polygons and computing county-level elevation, slope, and terrain-ruggedness statistics. The resulting seven static terrain variables were joined to the county nodes and supplied as time-invariant node features to the GAT.

## 4. 数据来源链

```text
NWS 县级边界/坐标数据
          │
          ├── 县级多边形 → 共享边界图边
          ├── 县级经纬度 → kNN 图边、距离/方向边属性、节点坐标特征
          └── 县界掩膜
                  │
USGS 3DEP/NED DEM ──┘
          │
          └── county_terrain.csv → 7 个静态地形特征
```

## 5. 可复现性说明

若只复现最终 GAT 模型，必须提供：

```text
data/geo/c_16ap26.dbf
data/geo/c_16ap26.shp
data/geo/county_terrain.csv
```

若需要从原始高程数据重新构建 `county_terrain.csv`，还必须提供：

```text
data/geo/dem_3dep_1arcsec/*.tif
data/geo/dem_3dep_1arcsec/manifest.json
code_phase2_final/build_county_terrain_features/build_terrain.py
```

原始 DEM TIFF 不会在最终 GAT 训练阶段直接作为输入；它们仅用于生成 `county_terrain.csv`。
