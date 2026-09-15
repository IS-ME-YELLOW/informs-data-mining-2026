"""Read-only model diagnosis. Writes summaries and figures, never fits a model."""
from pathlib import Path
import json
import struct

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HORIZONS = [1, 6, 24, 48]
FOCUS = ['42053', '54015', '54013', '39115', '39117', '18013', '54007']


def parquet(path):
    return pd.read_parquet(ROOT / path, engine='fastparquet')


raw = pd.read_csv(ROOT / 'data/DM_Train.csv', dtype={'fipsCode': str})
raw['time'] = pd.to_datetime(raw.timestamp_et)
raw['hour'] = raw.groupby('fipsCode').cumcount()
oof = parquet('versions/v1.12_tail_protected_ensemble/artifacts/oof_predictions.parquet')
six = parquet('versions/v1.11_tree_ensemble/artifacts/oof_predictions.parquet')
components = parquet('versions/xyy/v1.5.8/oof_component_predictions.parquet')
assert oof[['fipsCode', 'hour_idx']].equals(six[['fipsCode', 'hour_idx']])
assert oof[['fipsCode', 'hour_idx']].equals(components[['fipsCode', 'hour_idx']])
assert not raw.duplicated(['fipsCode', 'time']).any()
assert raw.groupby('fipsCode').size().eq(216).all()
names = raw.groupby('fipsCode').first()[['countyName', 'stateAbbr']]
external = pd.read_csv(ROOT / 'data/county_features_v1.5.6.csv', dtype={'fipsCode': str})
long_rows, county_rows, gate_rows, diagnostic_rows = [], [], [], []
for h in HORIZONS:
    key = f'osi_target_t{h:02d}h'
    work = oof.loc[oof['actual_' + key].notna(), ['fipsCode', 'timestamp_et', 'hour_idx', 'fold']].copy()
    work['horizon'] = h
    work['target_time'] = work.timestamp_et + pd.Timedelta(hours=h)
    for col, prefix in [('actual', 'actual_'), ('base', 'pred_v1.8_rule_'),
                        ('simple_average', 'pred_simple_average_'), ('prediction', 'pred_tail_protected_')]:
        work[col] = oof.loc[work.index, prefix + key]
    actual = work[['fipsCode', 'target_time']].merge(raw[['fipsCode', 'time', 'osi']], left_on=['fipsCode', 'target_time'], right_on=['fipsCode', 'time'], validate='one_to_one').osi.to_numpy()
    assert np.allclose(work.actual, actual, atol=0, rtol=0)
    work['sse'] = (work.actual - work.prediction) ** 2
    work['base_sse'] = (work.actual - work.base) ** 2
    work['average_sse'] = (work.actual - work.simple_average) ** 2
    work['true_q95'] = float(work.actual.quantile(.95))
    work['true_top5'] = work.actual >= work.true_q95
    work['protected'] = False
    work['gate_threshold'] = np.nan
    if h >= 24:
        for fold in range(5):
            model = json.loads((ROOT / f'versions/v1.12_tail_protected_ensemble/models/outer_fold{fold}_{key}.json').read_text())
            mask = work.fold == fold
            work.loc[mask, 'gate_threshold'] = model['threshold']
            work.loc[mask, 'protected'] = work.loc[mask, 'base'] > model['threshold']
        expected = np.where(work.protected, work.base, work.simple_average)
    else:
        expected = work.base
    assert np.allclose(expected, work.prediction, atol=1e-15)
    matrix = six.loc[work.index, [f'pred_{m}_{key}' for m in ['v1.8_rule', 'lgbm_direct', 'xgboost_direct', 'catboost_direct', 'xgboost_component', 'catboost_component']]].to_numpy()
    work['candidate_min'] = matrix.min(axis=1)
    work['candidate_max'] = matrix.max(axis=1)
    # Label-informed lower bounds are diagnostic only, not usable predictors.
    hull_oracle = np.clip(work.actual, work.candidate_min, work.candidate_max)
    gate_oracle = np.where(work.base_sse <= work.average_sse, work.base, work.simple_average)
    total = work.sse.sum()
    diagnostic_rows.append({'horizon': h, 'n': len(work), 'total_sse': total,
                            'rmse': np.sqrt(work.sse.mean()),
                            'underprediction_sse_share': work.loc[work.prediction < work.actual, 'sse'].sum() / total,
                            'gate_oracle_rmse': np.sqrt(np.mean((gate_oracle - work.actual) ** 2)),
                            'six_model_hull_oracle_rmse': np.sqrt(np.mean((hull_oracle - work.actual) ** 2)),
                            'true_q95': work.actual.quantile(.95)})
    for code, group in work.groupby('fipsCode'):
        peak = group.loc[group.actual.idxmax()]
        predpeak = group.loc[group.prediction.idxmax()]
        county_rows.append({'fipsCode': code, **names.loc[code].to_dict(), 'horizon': h,
                            'fold': int(group.fold.iloc[0]), 'n': len(group),
                            'sse': group.sse.sum(), 'sse_share': group.sse.sum() / total,
                            'rmse': np.sqrt(group.sse.mean()), 'bias': (group.prediction - group.actual).mean(),
                            'truth_peak': peak.actual, 'truth_peak_time': peak.target_time,
                            'prediction_at_truth_peak': peak.prediction,
                            'base_at_truth_peak': peak.base, 'average_at_truth_peak': peak.simple_average,
                            'max_candidate_at_truth_peak': peak.candidate_max,
                            'protected_at_truth_peak': bool(peak.protected),
                            'prediction_peak': predpeak.prediction, 'prediction_peak_time': predpeak.target_time,
                            'actual_sum': group.actual.sum(), 'prediction_sum': group.prediction.sum(),
                            'hours_actual_ge005': int((group.actual >= .05).sum()),
                            'hours_actual_ge005_protected': int(((group.actual >= .05) & group.protected).sum()),
                            'largest_6h_sse_share_within_county': group.sse.rolling(6).sum().max() / group.sse.sum()})
    if h >= 24:
        for truehigh in [True, False]:
            for protected in [True, False]:
                group = work[(work.true_top5 == truehigh) & (work.protected == protected)]
                gate_rows.append({'horizon': h, 'true_top5': truehigh, 'protected': protected,
                                  'n': len(group), 'sse_share': group.sse.sum() / total,
                                  'rmse': np.sqrt(group.sse.mean()), 'base_rmse': np.sqrt(group.base_sse.mean()),
                                  'average_rmse': np.sqrt(group.average_sse.mean())})
    long_rows.append(work)

long = pd.concat(long_rows, ignore_index=True)
county = pd.DataFrame(county_rows)
county.to_csv(OUT / 'county_errors.csv', index=False)
pd.DataFrame(gate_rows).to_csv(OUT / 'gate_error_decomposition.csv', index=False)
pd.DataFrame(diagnostic_rows).to_csv(OUT / 'diagnostic_bounds.csv', index=False)
long[long.fipsCode.isin(FOCUS)].to_csv(OUT / 'focus_predictions.csv', index=False)
raw[raw.fipsCode.isin(FOCUS)][['fipsCode', 'countyName', 'stateAbbr', 'time', 'hour', 'outageCount', 'customersTracked', 'P_t', 'N_t', 'D_t', 'R_t', 'osi', 'gust', 'wind_speed_10m', 'tp', 'rain', 'soil_moist']].to_csv(OUT / 'focus_observations.csv', index=False)

raw_summary = []
for code in FOCUS:
    group = raw[raw.fipsCode == code]
    obs = group[group.hour < 72]
    last = obs.iloc[-1]
    formula = np.maximum(0, .4 * group.P_t + .35 * group.N_t + .25 * group.D_t - .1 * group.R_t)
    raw_summary.append({'fipsCode': code, **names.loc[code].to_dict(),
                        'observed_last_osi': last.osi, 'observed_last_P': last.P_t,
                        'observed_peak_P': obs.P_t.max(), 'observed_peak_osi': obs.osi.max(),
                        'observed_last_N': last.N_t, 'observed_last_R': last.R_t,
                        'customers_min': group.customersTracked.min(), 'customers_max': group.customersTracked.max(),
                        'P_formula_max_error': np.max(np.abs(group.P_t - np.minimum(group.outageCount / group.customersTracked, 1))),
                        'osi_formula_max_error': np.max(np.abs(group.osi - formula))})
pd.DataFrame(raw_summary).merge(external, on='fipsCode', validate='one_to_one').to_csv(OUT / 'county_context.csv', index=False)
long[long.fipsCode.isin(FOCUS)].assign(target_day=lambda x:x.target_time.dt.strftime('%m-%d')).groupby(['fipsCode', 'horizon', 'target_day']).agg(sse=('sse', 'sum'), actual_max=('actual', 'max'), actual_mean=('actual', 'mean'), predicted_max=('prediction', 'max'), predicted_mean=('prediction', 'mean')).reset_index().to_csv(OUT / 'daily_error_patterns.csv', index=False)

# D is a six-hour trailing mean. Its observed terms give a deterministic lower bound.
d_rolling = raw.groupby('fipsCode', sort=False).P_t.transform(lambda x: x.rolling(6, min_periods=1).mean())
assert np.max(np.abs(d_rolling - raw.D_t)) < 1e-6
bound_violations = []
for code, group in raw.groupby('fipsCode'):
    observed_p = group[group.hour < 72].set_index('hour').P_t
    for target_hour in range(73, 77):
        lower = observed_p.loc[range(target_hour - 5, 72)].sum() / 6
        component_row = components[(components.fipsCode == code) & (components.hour_idx == target_hour - 1)].iloc[0]
        predicted = component_row.pred_D_t_target_t01h
        actual = component_row.actual_D_t_target_t01h
        assert actual + 1e-6 >= lower
        if lower - predicted > 1e-6:
            bound_violations.append({'fipsCode': code, 'target_hour': target_hour,
                                     'known_D_lower_bound': lower, 'predicted_D': predicted,
                                     'actual_D': actual, 'gap': lower - predicted})
pd.DataFrame(bound_violations).to_csv(OUT / 'known_history_D_bound_violations.csv', index=False)

# State geography is only used to describe neighboring counties, never as a learned input.
buf = (ROOT / 'data/external/tl_2025_us_county/tl_2025_us_county.dbf').read_bytes()
num = struct.unpack_from('<I', buf, 4)[0]
header, record_size = struct.unpack_from('<HH', buf, 8)
fields, offset = [], 32
while buf[offset] != 13:
    desc = buf[offset:offset+32]
    fields.append((desc[:11].split(b'\0')[0].decode(), desc[16]))
    offset += 32
geo_rows = []
for j in range(num):
    record = buf[header+j*record_size:header+(j+1)*record_size]
    values, at = {}, 1
    for name, size in fields:
        values[name] = record[at:at+size].decode('latin1').strip()
        at += size
    geo_rows.append(values)
geo = pd.DataFrame(geo_rows).set_index('GEOID')[['INTPTLAT', 'INTPTLON']].astype(float)
test = pd.read_csv(ROOT / 'data/DM_Test.csv', dtype={'fipsCode': str})
test['time'] = pd.to_datetime(test.timestamp_et)
test['hour'] = test.groupby('fipsCode').cumcount()
both = pd.concat([raw, test], ignore_index=True)
both['split_label'] = np.where(both.fipsCode.isin(set(raw.fipsCode)), 'train', 'test')
ctx = both[both.hour == 71][['fipsCode', 'countyName', 'stateAbbr', 'split_label', 'customersTracked', 'P_t']].rename(columns={'P_t': 'observed_last_P'})
ctx = ctx.merge(external, on='fipsCode', validate='one_to_one')
ctx.to_csv(OUT / 'all_county_observable_context.csv', index=False)
neighbor_rows = []
for code in FOCUS[:4]:
    eligible = geo.reindex(ctx.fipsCode)
    lat = np.deg2rad(eligible.INTPTLAT.to_numpy())
    lon = np.deg2rad(eligible.INTPTLON.to_numpy())
    lat0, lon0 = np.deg2rad(geo.loc[code].to_numpy())
    hav = np.sin((lat-lat0)/2)**2 + np.cos(lat)*np.cos(lat0)*np.sin((lon-lon0)/2)**2
    km = 6371 * 2 * np.arcsin(np.minimum(1, np.sqrt(hav)))
    nearest = ctx.assign(distance_km=km).query('fipsCode != @code').nsmallest(6, 'distance_km')
    for _, row in nearest.iterrows():
        group = raw[(raw.fipsCode == row.fipsCode) & (raw.hour >= 120)]
        neighbor_rows.append({'focus_fips': code, 'neighbor_fips': row.fipsCode,
                              'countyName': row.countyName, 'split': row.split_label,
                              'distance_km': row.distance_km,
                              'observed_last_P': row.observed_last_P,
                              'future_peak_osi_diagnostic_train_only': group.osi.max() if len(group) else np.nan})
pd.DataFrame(neighbor_rows).to_csv(OUT / 'geographic_neighbors.csv', index=False)

# Standalone research figures: hours on the x axis always refer to target time.
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
COLORS = {'actual':'#172b4d', 'prediction':'#ca5b2e', 'average':'#188c94', 'gust':'#687483'}


def trajectory_figure(cases, filename):
    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 3.05 * len(cases)), gridspec_kw={'width_ratios': [1.7, 1]})
    axes = np.atleast_2d(axes)
    for row, (code, h) in enumerate(cases):
        ax, weather = axes[row]
        obs = raw[(raw.fipsCode == code) & (raw.hour >= 48)]
        pred = long[(long.fipsCode == code) & (long.horizon == h)]
        ax.plot(obs.time, obs.osi, color=COLORS['actual'], lw=1.7, label='Actual OSI')
        ax.plot(pred.target_time, pred.prediction, color=COLORS['prediction'], lw=1.6, label='v1.12 OOF')
        if h >= 24:
            ax.plot(pred.target_time, pred.simple_average, color=COLORS['average'], lw=1.2, ls='--', label='Six-model average')
        ax.axvspan(pd.Timestamp('2026-03-13'), pd.Timestamp('2026-03-13 23:00'), color='#dedede', alpha=.5)
        ax.axvline(pd.Timestamp('2026-03-13 23:00'), color='#8c8c8c', ls=':', lw=1)
        name = names.loc[code]
        ax.set_title(f'{name.countyName}, {name.stateAbbr} ({code}) | t+{h}h', loc='left', fontweight='bold')
        ax.set_ylabel('OSI')
        ax.set_ylim(bottom=0)
        ax.set_xlim(pd.Timestamp('2026-03-13'), pd.Timestamp('2026-03-20'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='upper right', fontsize=8, frameon=False)
        weather.plot(obs.time, obs.gust, color=COLORS['gust'], lw=1.4, label='Gust')
        rain_ax = weather.twinx()
        rain_ax.bar(obs.time, obs.tp, width=.033, color='#2c8fb7', alpha=.45, label='Precipitation')
        weather.set_ylabel('Gust (mph)')
        rain_ax.set_ylabel('Precipitation (mm/hour)')
        weather.set_xlim(pd.Timestamp('2026-03-13'), pd.Timestamp('2026-03-20'))
        weather.set_ylim(bottom=0)
        rain_ax.set_ylim(bottom=0)
        weather.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        weather.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        weather.grid(axis='y', alpha=.2)
        weather.set_title('Provided weather', loc='left')
        if row == len(cases)-1:
            ax.set_xlabel('Target date (dataset timestamp labels)')
            weather.set_xlabel('Date (dataset timestamp labels)')
    fig.suptitle('Large errors have different temporal patterns', fontsize=16, y=.995)
    fig.text(.02, .005, 'Gray area: observed outage window. Predictions stop receiving outage observations after March 13 23:00. Each row uses its stated horizon.', fontsize=9)
    fig.tight_layout(rect=[0, .024, 1, .975])
    fig.savefig(OUT / filename, dpi=170)
    plt.close(fig)


trajectory_figure([('42053',1), ('54015',48), ('54013',48), ('39115',48)], 'county_trajectories.png')
trajectory_figure([('39117',1), ('18013',24), ('54007',48)], 'additional_trajectories.png')

fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))
for ax, (code, h) in zip(axes.flat, [('42053',1), ('54015',48), ('54013',48), ('39115',48)]):
    pred = long[(long.fipsCode == code) & (long.horizon == h)]
    if code == '42053':
        start, end = pd.Timestamp('2026-03-13 18:00'), pd.Timestamp('2026-03-15 00:00')
    elif code == '39115':
        start, end = pd.Timestamp('2026-03-18 06:00'), pd.Timestamp('2026-03-18 20:00')
    else:
        start, end = pd.Timestamp('2026-03-16'), pd.Timestamp('2026-03-19')
    visible = pred[pred.target_time.between(start, end)]
    real = raw[(raw.fipsCode == code) & raw.time.between(start,end)]
    ax.plot(real.time, real.osi, color=COLORS['actual'], lw=2, label='Actual OSI')
    ax.plot(visible.target_time, visible.prediction, color=COLORS['prediction'], lw=2, label='v1.12 OOF')
    peak = visible.loc[visible.actual.idxmax()]
    ax.scatter([peak.target_time], [peak.actual], color=COLORS['actual'], s=20)
    ax.annotate(f'{peak.actual:.4f}', (peak.target_time, peak.actual), xytext=(6, 6), textcoords='offset points', fontsize=11)
    ax.set_title(f'{names.loc[code,"countyName"]} | t+{h}h', loc='left', fontsize=12)
    ax.set_xlim(start, end)
    ax.set_ylim(0, real.osi.max()*1.2)
    ax.set_ylabel('OSI', fontsize=11)
    ax.grid(axis='y', alpha=.2)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=10, loc='upper right')
fig.suptitle('Amplitude misses, prolonged outages, and a sudden spike', fontsize=14)
fig.supxlabel('Target timestamp (all outage inputs end at March 13 23:00)', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'peak_comparison.png', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
for ax, h in zip(axes, [24,48]):
    group = pd.DataFrame(gate_rows).query('horizon == @h')
    selected = group[group.true_top5]
    labels = ['High actual OSI,\nprotected', 'High actual OSI,\nnot protected', 'Other rows']
    values = [selected.loc[selected.protected,'sse_share'].iloc[0]*100, selected.loc[~selected.protected,'sse_share'].iloc[0]*100, group.loc[~group.true_top5,'sse_share'].sum()*100]
    bars = ax.bar(labels, values, color=['#ca5b2e','#188c94','#9ba5ae'])
    ax.bar_label(bars, fmt='%.1f%%', padding=4)
    ax.set_ylim(0, max(values)*1.23)
    ax.set_title(f't+{h}h: share of total squared error')
    ax.set_ylabel('SSE share (%)')
    ax.grid(axis='y', alpha=.2)
fig.suptitle('Protected peaks can still be badly underestimated', fontsize=14)
fig.tight_layout()
fig.savefig(OUT / 'gate_error_shares.png', dpi=170)
plt.close(fig)

summary = {'source_model': 'v1.12 saved OOF', 'training_performed': False,
           'scope': 'diagnosis; no future outage information becomes a model input',
           'oracle_warning': 'Label-informed lower bounds quantify limitations; not deployable or estimated test performance.',
           'focus': FOCUS, 'checks': {'target_alignment': True, 'gate_reconstruction': True, 'county_hour_uniqueness': True}}
(OUT / 'analysis_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Saved diagnosis to', OUT)
print(county[county.fipsCode.isin(FOCUS)].pivot(index='fipsCode', columns='horizon', values='sse_share').mul(100).round(2).to_string())
