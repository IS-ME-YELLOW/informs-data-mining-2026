import csv, math
from datetime import datetime
from collections import defaultdict

# ============================================================
# INFORMS 2026 Data Mining Challenge — PDF 数字论断验证脚本
# 用法: python3 verify_pdf_claims.py
# 依赖: 仅标准库 (csv, math, datetime, collections)
# ============================================================

def load_csv(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames

def to_float(v):
    if v is None or v.strip() == '':
        return None
    try:
        return float(v)
    except:
        return None

def compute_stats(rows, col):
    vals = [to_float(r[col]) for r in rows if to_float(r[col]) is not None]
    n = len(vals)
    if n == 0:
        return {'n': 0, 'min': None, 'max': None, 'mean': None,
                'zeros': None, 'zero_pct': None, 'median': None}
    mn = min(vals); mx = max(vals); mean = sum(vals) / n
    zeros = sum(1 for v in vals if v == 0)
    s = sorted(vals)
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {'n': n, 'min': mn, 'max': mx, 'mean': mean,
            'zeros': zeros, 'zero_pct': 100 * zeros / n, 'median': median}

results = []

def check(id, claim, actual, tol=None, match=None):
    if match is not None:
        status = 'PASS' if match else 'FAIL'
    elif tol is not None:
        status = 'PASS' if abs(actual - claim) <= tol else 'FAIL'
    else:
        status = 'PASS' if str(claim) == str(actual) else 'FAIL'
    results.append((id, claim, actual, status))

# ============================================================
# Load data
# ============================================================
train, train_cols = load_csv('DM_Train.csv')
test, test_cols = load_csv('DM_Test.csv')
sub, sub_cols = load_csv('sample_submission.csv')

train_fips = set(row['fipsCode'] for row in train)
test_fips = set(row['fipsCode'] for row in test)
all_fips = train_fips | test_fips

# ============================================================
# 1. Dimensions
# ============================================================
print('=' * 70)
print('1. Data Dimensions')
print('=' * 70)

check(1, 51624, len(train), match=(len(train) == 51624))
check(2, 63, len(train_cols), match=(len(train_cols) == 63))
check(3, 13608, len(test), match=(len(test) == 13608))
check(4, 43, len(test_cols), match=(len(test_cols) == 43))
check(5, 9072, len(sub), match=(len(sub) == 9072))
check(6, 8, len(sub_cols), match=(len(sub_cols) == 8))
check(7, 20, len(set(train_cols) - set(test_cols)),
      match=(len(set(train_cols) - set(test_cols)) == 20))

# ============================================================
# 2. County counts and state distribution
# ============================================================
print('\n' + '=' * 70)
print('2. County Counts and State Distribution')
print('=' * 70)

check(8, 239, len(train_fips), match=(len(train_fips) == 239))
check(9, 63, len(test_fips), match=(len(test_fips) == 63))
check(10, 302, len(all_fips), match=(len(all_fips) == 302))
check(11, 0, len(train_fips & test_fips), match=(len(train_fips & test_fips) == 0))

for state, exp_tr, exp_te in [('IN', 72, 20), ('OH', 70, 18),
                               ('PA', 52, 15), ('WV', 45, 10)]:
    tr_ct = len(set(r['fipsCode'] for r in train if r['stateAbbr'] == state))
    te_ct = len(set(r['fipsCode'] for r in test if r['stateAbbr'] == state))
    check(f'12-{state}-train', exp_tr, tr_ct, match=(tr_ct == exp_tr))
    check(f'12-{state}-test', exp_te, te_ct, match=(te_ct == exp_te))

# ============================================================
# 3. Time window structure
# ============================================================
print('\n' + '=' * 70)
print('3. Time Window Structure')
print('=' * 70)

train_ts = sorted(set(row['timestamp_et'] for row in train))
check(13, 216, len(train_ts), match=(len(train_ts) == 216))
check(14, 216, len(set(row['timestamp_et'] for row in test)),
      match=(len(set(row['timestamp_et'] for row in test)) == 216))

first_train_rows = [r for r in train if r['fipsCode'] == list(train_fips)[0]]
check(15, 216, len(first_train_rows), match=(len(first_train_rows) == 216))

durations = set(to_float(r['event_duration_h']) for r in train)
check(16, 168, durations, match=(durations == {168.0}))

pre_event = [r for r in train if r['in_event_window'].strip().lower() == 'false']
storm = [r for r in train if r['in_event_window'].strip().lower() == 'true']
check(17, 48, len(pre_event) / len(train_fips), tol=0.001)
check(18, 168, len(storm) / len(train_fips), tol=0.001)

# ============================================================
# 4. OSI formula verification
# ============================================================
print('\n' + '=' * 70)
print('4. OSI Formula Verification')
print('=' * 70)

osi_diffs = []
osi_neg_count = 0
osi_total = 0
for r in train:
    pt = to_float(r['P_t'])
    nt = to_float(r['N_t'])
    dt = to_float(r['D_t'])
    rt = to_float(r['R_t'])
    osi = to_float(r['osi'])
    if all(v is not None for v in [pt, nt, dt, rt, osi]):
        computed = 0.40 * pt + 0.35 * nt + 0.25 * dt - 0.10 * rt
        if computed < 0:
            computed = 0
            osi_neg_count += 1
        osi_total += 1
        if abs(computed - osi) > 1e-6:
            osi_diffs.append(abs(computed - osi))

check(19, 0, len(osi_diffs), match=(len(osi_diffs) == 0))
print(f'  OSI formula: {osi_total} rows checked, {len(osi_diffs)} mismatches, '
      f'{osi_neg_count} clipped to 0')
if osi_diffs:
    print(f'  Diff range: min={min(osi_diffs):.6f}, max={max(osi_diffs):.6f}, '
          f'mean={sum(osi_diffs) / len(osi_diffs):.6f}')
    print('  NOTE: Tiny diffs are precision artifacts (osi stored at 4dp, '
          'components at ~8dp)')

# N_t / R_t mutual exclusivity
both_pos = sum(1 for r in train
               if to_float(r['N_t']) and to_float(r['R_t'])
               and to_float(r['N_t']) > 0 and to_float(r['R_t']) > 0)
print(f'  N_t>0 AND R_t>0: {both_pos}/{osi_total} ({100*both_pos/osi_total:.1f}%)')
print('  NOTE: PDF formula implies mutual exclusivity, but data shows '
      '26.8% both positive')

# ============================================================
# 5. Raw outage variable statistics
# ============================================================
print('\n' + '=' * 70)
print('5. Raw Outage Variable Statistics')
print('=' * 70)

var_specs = {
    'outageCount':       {'min': 0, 'max': 97457, 'mean': 379, 'zero_pct': 43},
    'customersTracked':  {'min': 1232, 'max': 696427},
    'outage_pct':        {'min': 0, 'max': 100, 'mean': 0.91, 'median': 0.01},
    'P_t':               {'min': 0, 'max': 1, 'mean': 0.0091},
    'N_t':               {'min': 0, 'max': 0.379, 'zero_pct': 59},
    'D_t':               {'min': 0, 'max': 1, 'mean': 0.0092},
    'R_t':               {'min': 0, 'max': 0.204, 'zero_pct': 52},
    'osi':               {'min': 0, 'max': 0.65, 'mean': 0.0062, 'zero_pct': 44},
}

for col, spec in var_specs.items():
    s = compute_stats(train, col)
    print(f'\n  {col}: n={s["n"]}, min={s["min"]}, max={s["max"]}, '
          f'mean={s["mean"]}, zeros={s["zeros"]}({s["zero_pct"]:.1f}%), '
          f'median={s["median"]}')
    if 'min' in spec:
        check(f'5-{col}-min', spec['min'], s['min'], tol=0.01)
    if 'max' in spec:
        check(f'5-{col}-max', spec['max'], s['max'], tol=0.01)
    if 'mean' in spec:
        check(f'5-{col}-mean', spec['mean'], s['mean'], tol=0.01)
    if 'zero_pct' in spec:
        check(f'5-{col}-zero%', spec['zero_pct'], round(s['zero_pct']), tol=1)
    if 'median' in spec:
        check(f'5-{col}-median', spec['median'], s['median'], tol=0.01)

# ============================================================
# 6. Target statistics and NaN
# ============================================================
print('\n' + '=' * 70)
print('6. Target Statistics and NaN')
print('=' * 70)

target_specs = {
    'osi_target_t01h': {'max': 0.599, 'mean': 0.0062,
                        'nan_per_county': 1, 'total_nan': 239},
    'osi_target_t06h': {'max': 0.599, 'mean': 0.0063,
                        'nan_per_county': 6, 'total_nan': 1434},
    'osi_target_t24h': {'nan_per_county': 24, 'total_nan': 5736},
    'osi_target_t48h': {'max': 0.598, 'mean': 0.0071,
                        'nan_per_county': 48, 'total_nan': 11472},
}

for col, spec in target_specs.items():
    s = compute_stats(train, col)
    nan_count = sum(1 for r in train if to_float(r[col]) is None)
    fips_nan = defaultdict(int)
    for r in train:
        if to_float(r[col]) is None:
            fips_nan[r['fipsCode']] += 1
    nan_set = set(fips_nan.values())
    print(f'\n  {col}: n={s["n"]}, min={s["min"]}, max={s["max"]}, '
          f'mean={s["mean"]}, zeros={s["zeros"]}({s["zero_pct"]:.1f}%), '
          f'NaN={nan_count}, NaN/county={nan_set}')
    if 'max' in spec:
        check(f'6-{col}-max', spec['max'], s['max'], tol=0.01)
    if 'mean' in spec:
        check(f'6-{col}-mean', spec['mean'], s['mean'], tol=0.01)
    if 'total_nan' in spec:
        check(f'6-{col}-nan', spec['total_nan'], nan_count,
              match=(nan_count == spec['total_nan']))

# Correlation t+48h vs t+1h
pairs = [(to_float(r['osi_target_t01h']), to_float(r['osi_target_t48h']))
         for r in train
         if to_float(r['osi_target_t01h']) is not None
         and to_float(r['osi_target_t48h']) is not None]
n = len(pairs)
ma = sum(p[0] for p in pairs) / n
mb = sum(p[1] for p in pairs) / n
cov = sum((p[0] - ma) * (p[1] - mb) for p in pairs) / n
va = sum((p[0] - ma) ** 2 for p in pairs) / n
vb = sum((p[1] - mb) ** 2 for p in pairs) / n
corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0
check('6-corr-t48-t01', 0.056, corr, tol=0.01)
print(f'\n  corr(t+48h, t+1h) = {corr:.4f} (PDF: 0.056)')

# ============================================================
# 7. Delta statistics
# ============================================================
print('\n' + '=' * 70)
print('7. OSI Delta Statistics')
print('=' * 70)

delta_specs = {
    'osi_delta_t01h': {'min': -0.369, 'max': 0.596},
    'osi_delta_t06h': {'min': -0.369, 'max': 0.596},
    'osi_delta_t48h': {'min': -0.640, 'max': 0.593},
}

for col, spec in delta_specs.items():
    s = compute_stats(train, col)
    print(f'  {col}: min={s["min"]}, max={s["mean"]}')
    if 'min' in spec:
        check(f'7-{col}-min', spec['min'], s['min'], tol=0.01)
    if 'max' in spec:
        check(f'7-{col}-max', spec['max'], s['max'], tol=0.01)

# Verify delta = target - osi
for dc, tc in [('osi_delta_t01h', 'osi_target_t01h'),
               ('osi_delta_t03h', 'osi_target_t03h'),
               ('osi_delta_t06h', 'osi_target_t06h'),
               ('osi_delta_t24h', 'osi_target_t24h'),
               ('osi_delta_t48h', 'osi_target_t48h')]:
    mism = sum(1 for r in train
               if to_float(r[dc]) and to_float(r[tc]) and to_float(r['osi'])
               and abs(to_float(r[dc]) - (to_float(r[tc]) - to_float(r['osi']))) > 1e-6)
    check(f'7-{dc}=def', 0, mism, match=(mism == 0))

# ============================================================
# 8. Lag feature NaN patterns
# ============================================================
print('\n' + '=' * 70)
print('8. Lag Feature NaN Patterns')
print('=' * 70)

lag_specs = {
    'outage_pct_lag1h': 1, 'outage_pct_lag3h': 3, 'outage_pct_lag6h': 6,
    'outage_pct_lag24h': 24, 'outage_pct_lag48h': 48,
    'osi_lag1h': 1, 'osi_lag3h': 3, 'osi_lag6h': 6,
    'osi_lag24h': 24, 'osi_lag48h': 48,
}

for col, exp_nan in lag_specs.items():
    fips = list(train_fips)[0]
    rows = [r for r in train if r['fipsCode'] == fips]
    rows.sort(key=lambda r: datetime.strptime(r['timestamp_et'], '%m/%d/%Y %H:%M'))
    nan_start = 0
    for r in rows:
        if to_float(r[col]) is None:
            nan_start += 1
        else:
            break
    check(f'8-{col}', exp_nan, nan_start, match=(nan_start == exp_nan))

# ============================================================
# 9. Data quality
# ============================================================
print('\n' + '=' * 70)
print('9. Data Quality')
print('=' * 70)

# 9a. Capping
exceed = [(r['fipsCode'],) for r in train
          if to_float(r['outageCount']) and to_float(r['customersTracked'])
          and to_float(r['customersTracked']) > 0
          and to_float(r['outageCount']) / to_float(r['customersTracked']) > 1.0]
capped_fips = set(exceed)
check('9a-capped-counties', 2, len(capped_fips), tol=0.5)

# 9b. customersTracked variation
ct_varying = 0
ct_ranges = []
for fips in train_fips:
    vals = [to_float(r['customersTracked']) for r in train
            if r['fipsCode'] == fips and to_float(r['customersTracked'])]
    if len(set(vals)) > 1:
        ct_varying += 1
        rng = 100 * (max(vals) - min(vals)) / min(vals) if min(vals) > 0 else 0
        ct_ranges.append(rng)
check('9b-varying', 204, ct_varying, tol=2)
if ct_ranges:
    ct_sorted = sorted(ct_ranges)
    check('9b-median-range', 0.05, ct_sorted[len(ct_sorted) // 2], tol=0.05)

# 9c. Pre-event outages
pre_outage = set()
for r in train:
    if r['in_event_window'].strip().lower() == 'false':
        oc = to_float(r['outageCount'])
        if oc and oc > 0:
            pre_outage.add(r['fipsCode'])
check('9c-pre-event-outage', 236, len(pre_outage), tol=2)

# 9d. FIPS 18111
fips18111_pre = [r for r in train
                 if r['fipsCode'] == '18111'
                 and r['in_event_window'].strip().lower() == 'false']
fips18111_100 = any(to_float(r['outage_pct']) and to_float(r['outage_pct']) >= 100
                    for r in fips18111_pre)
check('9d-fips18111-100', True, fips18111_100, match=fips18111_100)

# 9e. countyName uniqueness
train_names = set(r['countyName'] for r in train)
test_names = set(r['countyName'] for r in test)
all_names = train_names | test_names
check('9e-names-train', 187, len(train_names), match=(len(train_names) == 187))
check('9e-names-all', 187, len(all_names), match=(len(all_names) == 187))

# ============================================================
# 10. Test set NaN patterns
# ============================================================
print('\n' + '=' * 70)
print('10. Test Set NaN Patterns')
print('=' * 70)

test_fips_list = list(test_fips)
test_county = [r for r in test if r['fipsCode'] == test_fips_list[0]]
test_county.sort(key=lambda r: datetime.strptime(r['timestamp_et'], '%m/%d/%Y %H:%M'))

first_nan = None
for i, r in enumerate(test_county):
    if to_float(r['outageCount']) is None:
        first_nan = i
        break
check('10-nan-start', 72, first_nan, match=(first_nan == 72))

ct_nan = sum(1 for r in test_county if to_float(r['customersTracked']) is None)
check('10-ct-nan', 0, ct_nan, match=(ct_nan == 0))

for wc in ['gust', 'wind_speed_10m', 't2m', 'soil_moist']:
    wc_nan = sum(1 for r in test_county if to_float(r[wc]) is None)
    check(f'10-weather-{wc}', 0, wc_nan, match=(wc_nan == 0))

# ============================================================
# 11. Submission file
# ============================================================
print('\n' + '=' * 70)
print('11. Submission File')
print('=' * 70)

check('11-sub-counties', 63, len(set(r['fipsCode'] for r in sub)),
      match=(len(set(r['fipsCode'] for r in sub)) == 63))
check('11-sub-ts', 144, len(set(r['timestamp_et'] for r in sub)),
      match=(len(set(r['timestamp_et'] for r in sub)) == 144))

# ============================================================
# 12. severity_tier
# ============================================================
print('\n' + '=' * 70)
print('12. severity_tier')
print('=' * 70)

st_vals = set(to_float(r['severity_tier']) for r in train
              if to_float(r['severity_tier']) is not None)
check('12-st-range', True, st_vals == {0.0, 1.0, 2.0, 3.0, 4.0},
      match=(st_vals == {0.0, 1.0, 2.0, 3.0, 4.0}))
check('12-st-in-test', False, 'severity_tier' in test_cols,
      match=('severity_tier' not in test_cols))

# ============================================================
# 13. Supplementary checks
# ============================================================
print('\n' + '=' * 70)
print('13. Supplementary Checks')
print('=' * 70)

# D_t = rolling_mean(P_t, 6h)
dt_mism = 0
for fips in train_fips:
    rows = [r for r in train if r['fipsCode'] == fips]
    rows.sort(key=lambda r: datetime.strptime(r['timestamp_et'], '%m/%d/%Y %H:%M'))
    for i, r in enumerate(rows):
        pt = to_float(r['P_t'])
        dt_stored = to_float(r['D_t'])
        if pt is None or dt_stored is None:
            continue
        pt_vals = [to_float(rows[j]['P_t']) for j in range(max(0, i - 5), i + 1)]
        pt_vals = [v for v in pt_vals if v is not None]
        if not pt_vals:
            continue
        dt_comp = sum(pt_vals) / len(pt_vals)
        if abs(dt_comp - dt_stored) > 1e-6:
            dt_mism += 1
check('13-D_t-formula', 0, dt_mism, match=(dt_mism == 0))

# outage_pct formula
pct_mism = sum(1 for r in train
               if to_float(r['outageCount']) and to_float(r['customersTracked'])
               and to_float(r['outage_pct'])
               and to_float(r['customersTracked']) > 0
               and abs(min(to_float(r['outageCount']) /
                          to_float(r['customersTracked']) * 100, 100) -
                       to_float(r['outage_pct'])) > 0.01)
check('13-outage_pct-formula', 0, pct_mism, match=(pct_mism == 0))

# P_t formula
pt_mism = sum(1 for r in train
              if to_float(r['outageCount']) and to_float(r['customersTracked'])
              and to_float(r['P_t'])
              and to_float(r['customersTracked']) > 0
              and abs(min(to_float(r['outageCount']) /
                         to_float(r['customersTracked']), 1.0) -
                      to_float(r['P_t'])) > 1e-6)
check('13-P_t-formula', 0, pt_mism, match=(pt_mism == 0))

# Weather no NaN in train
weather_cols = ['gust', 'wind_speed_10m', 'wind_dir_10m', 't2m', 'd2m',
                'sp', 'mslma', 'blh', 'tp', 'rain', 'csnow', 'sdwe',
                'tcc', 'lcc', 'mcc', 'hcc', 'sdswrf', 'direct_rad',
                'diffuse_rad', 'r2', 'vpd', 'et0', 'soil_moist']
weather_nan = sum(sum(1 for r in train if to_float(r[wc]) is None)
                 for wc in weather_cols)
check('13-weather-no-nan-train', 0, weather_nan, match=(weather_nan == 0))

# ============================================================
# Summary
# ============================================================
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
total = len(results)
passed = sum(1 for r in results if r[3] == 'PASS')
failed = sum(1 for r in results if r[3] == 'FAIL')
print(f'Total: {total}, Passed: {passed}, Failed: {failed}')
if failed > 0:
    print('\nFailed items:')
    for id, claim, actual, status in results:
        if status == 'FAIL':
            print(f'  {id}: PDF={claim}, Actual={actual}')
