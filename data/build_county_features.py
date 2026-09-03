"""
build_county_features.py — 拼接所有外部数据源, 输出 county_features.csv
============================================================
输入(在 data/raw/ 下):
  - tl_2025_us_county/tl_2025_us_county.shp  (县边界+面积)
  - rural_urban_codes.csv                     (USDA: RUCC + Population)
  - county_landcover.csv                       (NLCD: pct_forest等)
  - county_tree_canopy_2025.csv                (NLCD: tree_canopy_pct)
  - eia861_2024/Service_Territory_2024.xlsx   (EIA: 县-电力公司映射)

输出:
  - data/county_features.csv (302行, 每县一行)

设计原则: 只引入与log_customers不共线的维度(比值/分类/植被), 不加原始计数
"""

import os
import csv
import numpy as np
import pandas as pd
import geopandas as gpd

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'county_features.csv')

TARGET_STATES = ['18', '39', '42', '54']  # IN, OH, PA, WV


def load_county_area():
    """从shapefile提取县陆地面积(平方米→平方英里)"""
    shp_path = os.path.join(RAW_DIR, 'tl_2025_us_county', 'tl_2025_us_county.shp')
    gdf = gpd.read_file(shp_path)
    gdf['STATEFP'] = gdf['STATEFP'].astype(str).str.zfill(2)
    gdf = gdf[gdf['STATEFP'].isin(TARGET_STATES)].copy()
    gdf['fipsCode'] = gdf['GEOID'].astype(str).str.zfill(5)
    gdf['county_area_sqmi'] = gdf['ALAND'].astype(float) / 2589988.11  # m² → sqmi
    result = gdf[['fipsCode', 'county_area_sqmi']].set_index('fipsCode')
    print(f'  县面积: {len(result)} 县')
    return result


def load_usda_rural_urban():
    """从USDA CSV提取RUCC + Population, 透视为每县一行"""
    csv_path = os.path.join(RAW_DIR, 'rural_urban_codes.csv')
    df = pd.read_csv(csv_path, encoding='latin-1')
    df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)
    df = df[df['FIPS'].str[:2].isin(TARGET_STATES)]
    # 透视: 每个Attribute变一列
    pivot = df.pivot_table(index='FIPS', columns='Attribute', values='Value', aggfunc='first')
    pivot.index.name = 'fipsCode'
    result = pd.DataFrame({
        'rural_urban_code': pivot['RUCC_2023'].astype(int),
        'Population_2020': pivot['Population_2020'].astype(float),
    })
    print(f'  USDA: {len(result)} 县')
    return result


def load_landcover():
    """读取NLCD县级土地覆盖百分比"""
    csv_path = os.path.join(RAW_DIR, 'county_landcover.csv')
    df = pd.read_csv(csv_path)
    df['fipsCode'] = df['fipsCode'].astype(str).str.zfill(5)
    cols = ['fipsCode', 'pct_forest', 'pct_developed', 'pct_agriculture',
            'pct_water', 'pct_wetland']
    df = df[cols].set_index('fipsCode')
    for c in cols[1:]:
        df[c] = df[c].astype(float)
    print(f'  NLCD土地覆盖: {len(df)} 县')
    return df


def load_tree_canopy():
    """读取NLCD县级树冠覆盖"""
    csv_path = os.path.join(RAW_DIR, 'county_tree_canopy_2025.csv')
    df = pd.read_csv(csv_path)
    df['fipsCode'] = df['fipsCode'].astype(str).str.zfill(5)
    cols = ['fipsCode', 'tree_canopy_pct', 'tree_canopy_std']
    df = df[cols].set_index('fipsCode')
    for c in cols[1:]:
        df[c] = df[c].astype(float)
    print(f'  NLCD树冠: {len(df)} 县')
    return df


def load_eia_n_utilities():
    """从EIA Service_Territory计算每县电力公司数量"""
    xlsx_path = os.path.join(RAW_DIR, 'eia861_2024', 'Service_Territory_2024.xlsx')
    st = pd.read_excel(xlsx_path)
    st_4s = st[st['State'].isin(['IN', 'OH', 'PA', 'WV'])]

    # 按县名统计电力公司数
    n_utils_by_name = st_4s.groupby('County')['Utility Number'].nunique()

    # 需要shapefile的县名来做匹配
    shp_path = os.path.join(RAW_DIR, 'tl_2025_us_county', 'tl_2025_us_county.shp')
    gdf = gpd.read_file(shp_path)
    gdf['STATEFP'] = gdf['STATEFP'].astype(str).str.zfill(2)
    gdf_4s = gdf[gdf['STATEFP'].isin(TARGET_STATES)].copy()
    gdf_4s['fipsCode'] = gdf_4s['GEOID'].astype(str).str.zfill(5)

    # 县名匹配
    result = {}
    for _, row in gdf_4s.iterrows():
        fips = row['fipsCode']
        name = row['NAME']
        n = n_utils_by_name.get(name, 1)  # 默认1个电力公司
        result[fips] = int(n)

    df = pd.DataFrame.from_dict(result, orient='index', columns=['n_utilities'])
    df.index.name = 'fipsCode'
    print(f'  EIA n_utilities: {len(df)} 县 (匹配{n_utils_by_name.shape[0]}个县名)')
    return df


def main():
    print('=== 构建县级外部特征 ===')

    print('\n[1/5] 县面积 (shapefile)')
    area = load_county_area()

    print('\n[2/5] USDA Rural-Urban Codes')
    usda = load_usda_rural_urban()

    print('\n[3/5] NLCD 土地覆盖')
    lc = load_landcover()

    print('\n[4/5] NLCD 树冠覆盖')
    tc = load_tree_canopy()

    print('\n[5/5] EIA 电力公司数量')
    n_utils = load_eia_n_utilities()

    # 合并
    print('\n=== 合并 ===')
    merged = area.join(usda, how='outer')
    merged = merged.join(lc, how='outer')
    merged = merged.join(tc, how='outer')
    merged = merged.join(n_utils, how='outer')

    # 填充缺失
    merged['n_utilities'] = merged['n_utilities'].fillna(1).astype(int)  # 默认1个电力公司
    merged = merged.fillna(0)  # 其他缺失填0

    # 派生特征
    merged['is_metro'] = (merged['rural_urban_code'] <= 3).astype(int)
    merged['pop_density'] = merged['Population_2020'] / merged['county_area_sqmi']
    merged['pop_density'] = merged['pop_density'].replace([np.inf, -np.inf], 0)

    # 最终列(不含population和area, 因为它们是中间变量)
    # 注意: 不含 county_area_sqmi(仅用于计算pop_density) 和 Population_2020(与log_customers共线)
    final_cols = [
        'rural_urban_code',      # 城市属性(1-9)
        'is_metro',              # 城市属性(二值)
        'pop_density',           # 密度(非规模)
        'n_utilities',           # 电网管理复杂度
        'pct_forest',            # 植被(全新维度)
        'pct_developed',         # 土地利用
        'pct_agriculture',       # 土地利用
        'pct_water',             # 土地利用
        'pct_wetland',           # 土地利用
        'tree_canopy_pct',      # 植被(精确)
        'tree_canopy_std',       # 植被空间变异
    ]
    final = merged[final_cols].copy()
    final.index.name = 'fipsCode'

    # 验证
    print(f'\n输出: {len(final)} 县 × {len(final.columns)} 特征')
    print(f'列: {list(final.columns)}')
    print(f'\n缺失值: {final.isna().sum().sum()}')
    print(f'\n各特征统计:')
    for col in final.columns:
        vals = final[col]
        print(f'  {col}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}')

    # 保存
    final.to_csv(OUT_PATH)
    print(f'\n保存到: {OUT_PATH}')
    print(f'文件大小: {os.path.getsize(OUT_PATH)} bytes')


if __name__ == '__main__':
    main()
