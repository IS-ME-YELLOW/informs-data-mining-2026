"""
build_v157_f1.py — 构建 v1.5.7: v1.5.6 + F1 目标时刻天气轨迹 (12列)
============================================================
F1特征: 以目标时刻 s=t+h 为终点的天气窗口统计, 而非以预测原点 t 为起点。
让 h=1 模型也能看到目标前 24/48h 的天气积累。

新增12列:
  gust_mean/max_W{6,24,48}_s  (6列)
  total_tp_W{24,48}_s          (2列)
  gust_change_s{3,6}          (2列)
  pressure_change_s{3,6}      (2列)

窗口定义: W_L(s) = {s-L+1, ..., s}, s = hour_idx + h
超出 [0, 215] 时设 NaN (与目标 NaN 模式一致)。
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path = [p for p in sys.path if '.python_packages' not in p]

CACHE = 'cache'
SRC_VER = 'v1.5.6'
DST_VER = 'v1.5.7'

# F1 特征列名
F1_COLUMNS = [
    'gust_mean_W6_s', 'gust_max_W6_s',
    'gust_mean_W24_s', 'gust_max_W24_s',
    'gust_mean_W48_s', 'gust_max_W48_s',
    'total_tp_W24_s', 'total_tp_W48_s',
    'gust_change_s3', 'gust_change_s6',
    'pressure_change_s3', 'pressure_change_s6',
]

# 需要从原始CSV读取的气象列
WEATHER_COLS = ['gust', 'sp', 'tp']
HORIZON_HOURS = {'osi_target_t01h': 1, 'osi_target_t06h': 6,
                  'osi_target_t24h': 24, 'osi_target_t48h': 48}
HORIZON_VALUES = [1, 6, 24, 48]


def load_raw_weather(csv_path):
    """加载原始CSV, 返回 {fips: {col: np.array(216)}} """
    df = pd.read_csv(csv_path)
    df['fipsCode'] = df['fipsCode'].astype(str).str.zfill(5)
    weather = {}
    for fips, grp in df.groupby('fipsCode'):
        grp = grp.sort_values('timestamp_et').reset_index(drop=True)
        weather[fips] = {col: grp[col].to_numpy(dtype=float) for col in WEATHER_COLS}
    return weather


def compute_f1_for_row(hour_idx, horizon_h, weather_arrays):
    """对单行计算F1的12个特征值。s = hour_idx + horizon_h"""
    s = hour_idx + horizon_h
    gust = weather_arrays['gust']  # np.array(216)
    sp = weather_arrays['sp']
    tp = weather_arrays['tp']

    result = {}

    # W6(s) = [s-5, s], 需要 s >= 5 且 s < 216
    for L, name_L in [(6, 'W6'), (24, 'W24'), (48, 'W48')]:
        start = s - L + 1
        if start >= 0 and s < 216:
            window = gust[start:s+1]
            result[f'gust_mean_{name_L}_s'] = float(np.mean(window))
            result[f'gust_max_{name_L}_s'] = float(np.max(window))
        else:
            result[f'gust_mean_{name_L}_s'] = np.nan
            result[f'gust_max_{name_L}_s'] = np.nan

    # total_tp for W24, W48
    for L, name_L in [(24, 'W24'), (48, 'W48')]:
        start = s - L + 1
        if start >= 0 and s < 216:
            result[f'total_tp_{name_L}_s'] = float(np.sum(tp[start:s+1]))
        else:
            result[f'total_tp_{name_L}_s'] = np.nan

    # gust change at s
    for lag, name in [(3, 's3'), (6, 's6')]:
        if s - lag >= 0 and s < 216:
            result[f'gust_change_{name}'] = float(gust[s] - gust[s - lag])
            result[f'pressure_change_{name}'] = float(sp[s] - sp[s - lag])
        else:
            result[f'gust_change_{name}'] = np.nan
            result[f'pressure_change_{name}'] = np.nan

    return result


def main():
    print(f'=== Build {DST_VER}: {SRC_VER} + F1 ({len(F1_COLUMNS)} cols) ===')

    # 1. 加载原始天气数据
    print('[1/4] Loading raw weather...')
    train_weather = load_raw_weather('data/DM_Train.csv')
    test_weather = load_raw_weather('data/DM_Test.csv')
    print(f'  Train: {len(train_weather)} counties, Test: {len(test_weather)} counties')

    # 2. 加载v1.5.6缓存
    print('[2/4] Loading v1.5.6 cache...')
    X_train = pd.read_parquet(f'{CACHE}/features_train_{SRC_VER}.parquet')
    y_train = pd.read_parquet(f'{CACHE}/targets_train_{SRC_VER}.parquet')
    meta_train = pd.read_parquet(f'{CACHE}/meta_train_{SRC_VER}.parquet')
    X_test = pd.read_parquet(f'{CACHE}/features_test_{SRC_VER}.parquet')
    meta_test = pd.read_parquet(f'{CACHE}/meta_test_{SRC_VER}.parquet')
    print(f'  Train: {X_train.shape}, Test: {X_test.shape}')

    # 3. 计算F1特征
    print('[3/4] Computing F1 features...')

    def compute_f1_batch(X, meta, weather_source, tag):
        fips_arr = meta['fipsCode'].astype(str).str.zfill(5).values
        hour_arr = meta['hour_idx'].values

        rows_out = []
        for i in range(len(meta)):
            fips = fips_arr[i]
            t_idx = int(hour_arr[i])
            w = weather_source[fips]
            # 对每个horizon都计算(所有horizon模型都接收全部列)
            # 但F1特征与horizon无关(s依赖h), 所以我们为4个h分别计算
            # 实际上: 不同horizon模型看到的是同一行, s=t+h不同
            # 但特征矩阵每行只有一个t, 模型知道自己的h
            # 解决: 为每个h生成一组F1特征(带h后缀)
            row = {}
            for h_val in HORIZON_VALUES:
                s = t_idx + h_val
                f1 = compute_f1_for_row(t_idx, h_val, w)
                h_suffix = str(h_val)
                for col, val in f1.items():
                    row[f'{col}_{h_suffix}h'] = val
            rows_out.append(row)

        f1_df = pd.DataFrame(rows_out)
        print(f'  {tag}: {f1_df.shape} (per-row F1, {len(HORIZON_VALUES)} horizons × {len(F1_COLUMNS)} cols)')
        return f1_df

    # 等等 - 这里有个设计问题。F1特征依赖horizon h(因为s=t+h), 但特征矩阵每行
    # 对应一个固定的t, 4个horizon模型共享同一特征矩阵。
    # 解决方案: 为每个h生成一组F1特征, 列名带h后缀
    # 这样4个horizon模型都接收全部4×12=48列, 但只用自己的12列

    F1_ALL_COLUMNS = []
    for h_val in HORIZON_VALUES:
        for col in F1_COLUMNS:
            F1_ALL_COLUMNS.append(f'{col}_{h_val}h')

    print(f'  F1 total: {len(F1_ALL_COLUMNS)} cols (4 horizons × 12 features)')
    print(f'  NOTE: 每个h一组F1特征, 模型只用自己h的那组')

    # 重新计算
    def compute_f1_for_all_h(t_idx, w):
        row = {}
        for h_val in HORIZON_VALUES:
            f1 = compute_f1_for_row(t_idx, h_val, w)
            h_suffix = str(h_val)
            for col, val in f1.items():
                row[f'{col}_{h_suffix}h'] = val
        return row

    train_rows = []
    for i in range(len(meta_train)):
        fips = meta_train['fipsCode'].iloc[i]
        fips_str = str(fips).zfill(5)
        t_idx = int(meta_train['hour_idx'].iloc[i])
        w = train_weather[fips_str]
        train_rows.append(compute_f1_for_all_h(t_idx, w))

    test_rows = []
    for i in range(len(meta_test)):
        fips = meta_test['fipsCode'].iloc[i]
        fips_str = str(fips).zfill(5)
        t_idx = int(meta_test['hour_idx'].iloc[i])
        w = test_weather[fips_str]
        test_rows.append(compute_f1_for_all_h(t_idx, w))

    f1_train = pd.DataFrame(train_rows, columns=F1_ALL_COLUMNS)
    f1_test = pd.DataFrame(test_rows, columns=F1_ALL_COLUMNS)
    print(f'  Train F1: {f1_train.shape}, Test F1: {f1_test.shape}')
    print(f'  Train NaN: {f1_train.isna().sum().sum()} (expected: tail rows where s>215)')

    # 4. 拼接并保存
    print('[4/4] Joining and saving...')
    X_train_new = pd.concat([X_train.reset_index(drop=True), f1_train], axis=1)
    X_test_new = pd.concat([X_test.reset_index(drop=True), f1_test], axis=1)

    assert list(X_train_new.columns) == list(X_test_new.columns), 'Column mismatch!'
    assert len(X_train_new) == len(X_train)
    assert len(X_test_new) == len(X_test)

    # 保存
    for tag, X, y, meta in [('train', X_train_new, y_train, meta_train),
                              ('test', X_test_new, None, meta_test)]:
        X.to_parquet(f'{CACHE}/features_{tag}_{DST_VER}.parquet', index=False)
        meta.to_parquet(f'{CACHE}/meta_{tag}_{DST_VER}.parquet', index=False)
        if y is not None:
            y.to_parquet(f'{CACHE}/targets_train_{DST_VER}.parquet', index=False)

    with open(f'{CACHE}/feature_names_{DST_VER}.json', 'w') as f:
        json.dump(list(X_train_new.columns), f)

    print(f'  Saved: {X_train_new.shape} (train), {X_test_new.shape} (test)')
    print(f'  Total: {X_train_new.shape[1]} cols ({X_train.shape[1]} + {len(F1_ALL_COLUMNS)})')

    # 验证NaN模式
    print(f'  NaN train: {X_train_new.isna().sum().sum()}')

    # 看F1样本值
    print(f'\n  Sample (train row 0, h=48):')
    for col in F1_COLUMNS:
        full_col = f'{col}_48h'
        val = f1_train[full_col].iloc[0]
        print(f'    {full_col}: {val:.3f}' if not np.isnan(val) else f'    {full_col}: NaN')


if __name__ == '__main__':
    main()
