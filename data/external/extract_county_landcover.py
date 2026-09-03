"""
提取县级土地覆盖统计 — 本地运行脚本
============================================================
在你的本地机器运行(TIF文件所在的位置), 不需要上传整个TIF。

用途: 从 NLCD Land Cover 分类栅格中提取每个县的:
  - 各土地覆盖类别的面积百分比
  - 主要类别占比(forest/developed/agriculture/water)

前提: 需要安装 rasterio + rasterstats + geopandas
  pip install rasterio rasterstats geopandas

用法:
  python3 extract_county_landcover.py \
    --raster "path/to/lcnext-1.0-stratum-map-Clipped.tif" \
    --counties "path/to/tl_2024_us_county/tl_2024_us_county.shp" \
    --output "county_landcover.csv"

输出: county_landcover.csv (约302行, <50KB), 上传到服务器即可。

注意: 本脚本计算土地覆盖类别的面积比例，不计算 Tree Canopy Cover
（树冠覆盖率）。树冠覆盖率是连续百分比栅格，需要使用均值等连续型统计。
"""

import argparse
import numpy as np
import rasterio
import geopandas as gpd
from rasterstats import zonal_stats
import csv
import sys


NLCD_CLASSES = {11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95}


def sample_classes(src, nodata, window_size=512, grid_size=5):
    """从全图均匀分布的窗口取样，避免左上角样本不具代表性。"""
    row_starts = np.linspace(0, max(0, src.height - window_size), grid_size, dtype=int)
    col_starts = np.linspace(0, max(0, src.width - window_size), grid_size, dtype=int)
    classes = set()

    for row_start in row_starts:
        for col_start in col_starts:
            window = rasterio.windows.Window(
                col_start,
                row_start,
                min(window_size, src.width - col_start),
                min(window_size, src.height - row_start),
            )
            values = src.read(1, window=window)
            if nodata is not None:
                values = values[values != nodata]
            classes.update(np.unique(values).tolist())

    return sorted(classes)


def main():
    parser = argparse.ArgumentParser(description='提取县级土地覆盖统计')
    parser.add_argument('--raster', required=True, help='NLCD TIF栅格文件路径')
    parser.add_argument('--counties', required=True, help='县边界shapefile路径(tl_2024_us_county.shp)')
    parser.add_argument('--output', default='county_landcover.csv', help='输出CSV路径')
    args = parser.parse_args()

    # 1. 读取TIF, 检查有哪些类别值
    print(f'[1/4] 读取栅格: {args.raster}')
    with rasterio.open(args.raster) as src:
        raster_crs = src.crs
        nodata = src.nodata
        print(f'  CRS: {raster_crs}')
        print(f'  NoData: {nodata}')
        print(f'  Size: {src.width} x {src.height}')

    # 取全图分散样本，避免左上角恰好都是 NoData 或单一类别。
    with rasterio.open(args.raster) as src:
        unique_vals = sample_classes(src, nodata)
    print(f'  栅格值样本(全图分散取样): {unique_vals[:30]}')
    print(f'  总共{len(unique_vals)}种值')
    unknown_sample_classes = sorted(set(unique_vals) - NLCD_CLASSES)
    if unknown_sample_classes:
        print(
            '  警告: 检测到非标准 NLCD 编码 '
            f'{unknown_sample_classes[:30]}。它们将保留在 cls_* 列和 '
            'pct_unmapped 中，但不会被归入任何 NLCD 大类。'
        )

    # 2. 读取县边界, 筛选4州
    print(f'\n[2/4] 读取县边界: {args.counties}')
    counties = gpd.read_file(args.counties)
    counties['STATEFP'] = counties['STATEFP'].astype(str).str.zfill(2)
    counties_4s = counties[counties['STATEFP'].isin(['18', '39', '42', '54'])].copy()
    print(f'  4州县数: {len(counties_4s)}')

    # 确保CRS匹配
    if counties_4s.crs != raster_crs:
        print(f'  CRS不匹配, 重投影县边界: {counties_4s.crs} → {raster_crs}')
        counties_4s = counties_4s.to_crs(raster_crs)

    # 3. 对每个县做zonal statistics: 计算各类别的像素数
    print(f'\n[3/4] 计算县级土地覆盖统计(zonal_stats)...')
    print(f'  这可能需要几分钟, 取决于TIF大小和县数量...')

    # 用zonal_stats计算每个类别的面积(像素数)
    # categorical=True 会为每个类别输出count
    stats = zonal_stats(
        counties_4s,
        args.raster,
        categorical=True,
        nodata=nodata,
        all_touched=False,
    )

    # 4. 计算百分比并输出
    print(f'\n[4/4] 计算百分比并输出: {args.output}')

    # 收集所有出现的类别
    all_classes = set()
    for s in stats:
        all_classes.update(s.keys())
    all_classes = sorted(all_classes)
    print(f'  所有类别: {all_classes}')
    unknown_classes = sorted(set(all_classes) - NLCD_CLASSES)
    if unknown_classes:
        print(f'  警告: 县级统计中含未映射编码: {unknown_classes}')

    # NLCD标准分类映射(如果适用)
    # 如果TIF的值不是标准NLCD编码, 需要你根据实际值调整
    class_groups = {
        'forest': set(),      # 落叶林41, 针叶林42, 混合林43
        'developed': set(),   # 21,22,23,24
        'agriculture': set(), # 81,82
        'water': set(),       # 11
        'wetland': set(),     # 90,95
        'barren': set(),      # 31
        'shrub': set(),       # 52
        'grassland': set(),   # 71
    }

    for v in all_classes:
        v_int = int(v)
        if v_int == 11:
            class_groups['water'].add(v)
        elif v_int == 12:
            class_groups['water'].add(v)  # perennial ice/snow
        elif 21 <= v_int <= 24:
            class_groups['developed'].add(v)
        elif v_int == 31:
            class_groups['barren'].add(v)
        elif 41 <= v_int <= 43:
            class_groups['forest'].add(v)
        elif v_int == 52:
            class_groups['shrub'].add(v)
        elif v_int == 71:
            class_groups['grassland'].add(v)
        elif v_int in (81, 82):
            class_groups['agriculture'].add(v)
        elif v_int in (90, 95):
            class_groups['wetland'].add(v)

    print(f'  类别分组: {", ".join(f"{k}={sorted(v)}" for k, v in class_groups.items() if v)}')

    # 输出CSV: 每县一行, 含各类别百分比
    rows_out = []
    for idx, row in counties_4s.reset_index(drop=True).iterrows():
        fips = str(row['GEOID']).zfill(5)
        s = stats[idx]
        total_pixels = sum(s.values()) if s else 1

        out = {'fipsCode': fips, 'county_name': row['NAME'], 'total_pixels': total_pixels}

        # 各大类百分比
        for group_name, group_vals in class_groups.items():
            if group_vals:
                count = sum(s.get(v, 0) for v in group_vals)
                out[f'pct_{group_name}'] = round(100.0 * count / total_pixels, 2) if total_pixels > 0 else 0.0

        unmapped_count = sum(s.get(v, 0) for v in unknown_classes)
        out['pct_unmapped'] = round(100.0 * unmapped_count / total_pixels, 2) if total_pixels > 0 else 0.0

        # 各具体类别百分比(可选, 详细记录)
        for v in all_classes:
            count = s.get(v, 0)
            out[f'cls_{int(v)}'] = round(100.0 * count / total_pixels, 2) if total_pixels > 0 else 0.0

        rows_out.append(out)

    # 写CSV
    if rows_out:
        fieldnames = list(rows_out[0].keys())
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        print(f'\n完成! 输出{len(rows_out)}行到 {args.output}')
        print(f'文件大小: 约{sys.getsizeof(rows_out[0]) * len(rows_out) / 1024:.0f}KB')
        print(f'\n请将 {args.output} 上传到服务器:')
        print(f'  INFORMS_DATA_MINING/data/raw/county_landcover.csv')
    else:
        print('错误: 没有输出行')


if __name__ == '__main__':
    main()
