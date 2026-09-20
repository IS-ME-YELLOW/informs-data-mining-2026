"""Three-CV P6 confirmation summary; no prediction averaging or model fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;BASE=PARENT.parent
SEEDS=(42,20260917,20260918)
HS=('osi_target_t01h','osi_target_t06h','osi_target_t24h','osi_target_t48h')
HOURS=(1,6,24,48)


def readjson(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def write_json(p,obj):
    p=Path(p).resolve();assert p.is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_frame(p,df):
    p=Path(p).resolve();assert p.is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False)


def load(seed,name):
    root=PARENT if seed==42 else HERE/f'split{seed}'
    d=pd.read_csv(root/f'summary/{name}.csv',float_precision='round_trip',dtype={'fipsCode':str} if 'county' in name else {})
    if 'split_seed' not in d:d.insert(0,'split_seed',seed)
    else:assert d.split_seed.eq(seed).all()
    if 'case' in d:d=d[d['case'].isin(['B0','AB_P06'])].copy()
    if name in ('branch_diagnostics','branch_loss_decomposition'):
        d=d[d.horizon=='osi_target_t06h'].copy()
    return d


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    complete_parent=readjson(PARENT/'completion.json');assert complete_parent['status']=='COMPLETE'
    for seed in SEEDS[1:]:
        root=HERE/f'split{seed}';run=root/f'runs/v18_ptransfer_ab_v1_split{seed}_model42'
        assert readjson(run/'logs/independent_verification.json')['status']=='PASS'
        assert readjson(root/'analysis_verification.json')['status']=='PASS'
    names=['primary_scores','P_component_metrics','P71_bins','county_metrics','fold_metrics','window_metrics',
           'bootstrap_intervals','branch_diagnostics','branch_loss_decomposition','OSI_change_decomposition']
    all_tables={name:pd.concat([load(s,name) for s in SEEDS],ignore_index=True) for name in names}
    for name,d in all_tables.items():write_frame(HERE/f'summary/{name}.csv',d)
    primary=all_tables['primary_scores'];candidate=primary[primary['case']=='AB_P06']
    pm=all_tables['P_component_metrics'];boots=all_tables['bootstrap_intervals'];county=all_tables['county_metrics']
    folds=all_tables['fold_metrics'];windows=all_tables['window_metrics'];effect=all_tables['OSI_change_decomposition']
    consistency=[]
    for (h,f),g in county[county['case']=='AB_P06'].groupby(['horizon','fipsCode']):
        assert len(g)==3
        consistency.append(dict(horizon=h,fipsCode=f,countyName=g.countyName.iloc[0],stateAbbr=g.stateAbbr.iloc[0],
            improved_splits=int((g.sse_reduction>0).sum()),worsened_splits=int((g.sse_reduction<0).sum()),
            unchanged_splits=int((g.sse_reduction==0).sum()),
            split42_sse_reduction=float(g.loc[g.split_seed==42,'sse_reduction'].iloc[0]),
            split20260917_sse_reduction=float(g.loc[g.split_seed==20260917,'sse_reduction'].iloc[0]),
            split20260918_sse_reduction=float(g.loc[g.split_seed==20260918,'sse_reduction'].iloc[0])))
    consistency=pd.DataFrame(consistency);write_frame(HERE/'summary/county_consistency.csv',consistency)
    focus=pd.read_csv(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv',dtype={'fipsCode':str}).fipsCode.unique()
    write_frame(HERE/'summary/frozen_focus_counties.csv',county[(county['case']=='AB_P06')&county.fipsCode.isin(focus)])
    short=candidate[candidate.horizon.isin(HS[:2])];long=candidate[candidate.horizon.isin(HS[2:])]
    assert (long.rmse_delta==0).all()
    all_improve=bool((short.rmse_delta<0).all())
    ci=boots[(boots.domain=='OSI')&boots.horizon.isin(HS[:2])]
    all_ci=bool((ci.ci_high<0).all())
    verdict='支持将P6 a/b纳入后续实验的优先组合' if all_improve else '暂不冻结P6 a/b，先检查跨CV不一致'
    write_json(HERE/'summary/decision.json',dict(all_short_horizon_point_estimates_improve=all_improve,
        all_short_horizon_intervals_below_zero=all_ci,long_horizon_predictions_unchanged=True,
        recommendation=verdict,model_recipe=dict(P1='a/b + F2 neighbors',P6='a/b; original 163 features',
            D1='existing K>0 U reconstruction',P24_P48_N_R='existing same-CV strict sources'),
        underlying_events=1,independent_heldout_test_evaluations=0,additional_confirmation_fits=100,additional_outer_models=20,
        P6_three_CV_total_fits=150,P6_three_CV_total_outer_models=30,full_data_submission_models_trained=False))
    artifacts=[]
    for seed in SEEDS:
        root=PARENT if seed==42 else HERE/f'split{seed}'
        run=root/f'runs/v18_ptransfer_ab_v1_split{seed}_model42'
        m=readjson(run/'run_manifest.json')
        artifacts.append(dict(split_seed=seed,model_seed=42,case='AB_P06',reference_case='B0',
            run_directory=str(run),run_identity_hash=m['identity_hash'],
            files={name:dict(path=str(run/name),sha256=sha(run/name)) for name in
                ('run_manifest.json','component_predictions.parquet','control_predictions.parquet',
                 'aligned_unique_components.parquet','MODEL_CV_COMPLETE','EVALUATION_COMPLETE','VERIFIED_COMPLETE')}))
    write_json(HERE/'summary/candidate_manifest.json',dict(status='recommended_CV_recipe_not_deployed' if all_improve else 'unconfirmed_candidate',
        candidate='P1_ab_F2_plus_D_G_plus_P6_ab_163',source_P24_P48='original_strict',
        no_cross_split_prediction_averaging=True,full_data_models_exist=False,splits=artifacts))
    mainrows=[];metricrows=[];p_rows=[];countyrows=[];wrows=[];e_rows=[]
    for seed in SEEDS:
        g=candidate[candidate.split_seed==seed].set_index('horizon')
        mainrows.append([seed]+[f'{g.loc[h,"rmse_change_pct"]:+.3f}%' for h in HS])
        for h in HS[:2]:
            v=g.loc[h];b=boots[(boots.split_seed==seed)&(boots.domain=='OSI')&(boots.horizon==h)].iloc[0]
            metricrows.append([seed,f'{HOURS[HS.index(h)]}h',f'{v.baseline_rmse:.12f}',f'{v.rmse:.12f}',
                f'{v.rmse_delta:+.9f}',f'[{b.ci_low:+.9f}, {b.ci_high:+.9f}]'])
        for space in ['component_clipped','aligned']:
            v=pm[(pm.split_seed==seed)&(pm['case']=='AB_P06')&(pm.horizon==HS[1])&(pm.space==space)&(pm['group']=='full')].iloc[0]
            b=boots[(boots.split_seed==seed)&(boots.domain=='P')&(boots.horizon==HS[1])&(boots.space==space)].iloc[0]
            p_rows.append([seed,space,f'{v.baseline_rmse:.9f}',f'{v.rmse:.9f}',f'{v.rmse_change_pct:+.3f}%',f'[{b.ci_low:+.9f}, {b.ci_high:+.9f}]'])
        c=county[(county.split_seed==seed)&(county['case']=='AB_P06')&(county.horizon==HS[1])]
        f=folds[(folds.split_seed==seed)&(folds['case']=='AB_P06')&(folds.horizon==HS[1])]
        countyrows.append([seed,f'{int((c.sse_reduction>0).sum())}/{int((c.sse_reduction<0).sum())}',
            f'{int((f.sse_reduction>0).sum())}/{int((f.sse_reduction<0).sum())}',f'{c.sse_reduction.sum():+.6f}'])
        for r in windows[(windows.split_seed==seed)&(windows['case']=='AB_P06')&(windows.horizon==HS[1])&(~windows.common_target)&(windows.n>0)].itertuples():
            wrows.append([seed,f'{max(78,int(r.start))}–{int(r.end)}',f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
        e=effect[(effect.split_seed==seed)&(effect['case']=='AB_P06')&(effect.horizon==HS[1])].iloc[0]
        e_rows.append([seed,f'{e.P_self_sse_gain:+.9f}',f'{e.P_other_cross_sse_gain:+.9f}',
            f'{e.postprocessing_and_precision_sse_gain:+.9f}',f'{e.actual_OSI_sse_gain:+.9f}'])
    agreed=consistency[consistency.horizon==HS[1]]
    focus_rows=[]
    for f in focus:
        g=county[(county['case']=='AB_P06')&(county.horizon==HS[1])&(county.fipsCode==f)]
        focus_rows.append([f'{g.countyName.iloc[0]}, {g.stateAbbr.iloc[0]} ({f})']+
            [f'{g.loc[g.split_seed==s,"sse_reduction"].iloc[0]:+.6f}' for s in SEEDS])
    sign_note='三套的1h/6h点估计均改善' if all_improve else '三套的短时点估计未能全部同向改善'
    interval_note='六个短时比较的95%区间均低于0' if all_ci else f'六个短时比较中{int((ci.ci_high<0).sum())}个95%区间完全低于0：seed42和seed20260917的两项区间均低于0，seed20260918的两项区间跨零'
    report=['# P6 a/b迁移：两套CV确认与三套汇总（2026-09-19）','',
        f'**结论：{verdict}。{sign_note}；24h/48h逐行不变。**','',
        '本轮新增seed20260917、seed20260918的P6训练，每套50次拟合、10个模型，共100次拟合、20个外层模型。三套P6累计150次拟合、30个模型。每套参照为自己的完整F2，P6始终只有原163列，未加入邻县特征。','',
        '## 1. 对整体OSI的影响','',
        '相对各自完整F2的pooled RMSE变化，负数改善：','',
        table(['CV seed','1h','6h','24h','48h'],mainrows),'',
        table(['CV seed','输出h','B0 RMSE','P6 a/b后RMSE','ΔRMSE','县轨迹95%区间'],metricrows),'',
        f'{interval_note}。区间来自固定2,000次县轨迹配对bootstrap，不表示已经获得隐藏测试集提升。','',
        '1h和6h的净SSE改善来自相同的重叠目标小时；最早的73–77小时不变。两项百分比不同是因为评分范围和基准RMSE不同，不能当作两份独立复现证据。','',
        '## 2. P分量与整体收益的关系','',
        table(['CV seed','P6空间','B0 RMSE','新RMSE','变化','95%区间'],p_rows),'',
        'component_clipped是P6来源本身的重建P，aligned是C3对齐后的P。P6来源本身在三套均改善；C3后P分量在seed42和seed20260918略差、seed20260917略好，六个P分量比较的区间均跨零。因此可称整体短时收益方向复现，不能称三套P分量都显著变准。','',
        '6h整体SSE改善分解，正数代表减少误差：','',
        table(['CV seed','P自身加权平方项','P与固定N/D/R交叉项','后处理/精度余项','OSI净改善'],e_rows),'',
        '各行分解与逐行预测的SSE变化闭合。交叉项的改善属于当前组合的实际收益，但不等于N/D/R模型被重新训练，也不能据此保证改变其他分量后P6收益保持相同。','',
        '## 3. 县与时段稳定性','',
        table(['CV seed','改善县/恶化县（6h）','改善折/恶化折','净SSE改善'],countyrows),'',
        f'6h有{int((agreed.improved_splits==3).sum())}县三套都改善、{int((agreed.worsened_splits==3).sum())}县三套都恶化，其余方向随划分变化。保留全部239县评分，没有按县筛选新模型。','',
        table(['CV seed','有效目标小时','6h RMSE变化','SSE改善'],wrows),'',
        '三套的主要收益都来自目标小时78–95。后期并非全面改善：seed20260917的96–143及144–215均有退化，seed20260918的144–215也略差。仍以全部有效行的整体RMSE为主，没有根据这些已观察结果加时间门控。','',
        '原先冻结的7个关注县，6h净SSE改善如下：','',
        table(['县','seed42','seed20260917','seed20260918'],focus_rows),'',
        'SSE是有符号净增量；较大的正值可以被其他县抵消。上述县和时段表用于解释，没有据此创建县级规则或时间门控。','',
        '## 4. 当前建议组合','',
        f'当前判断：**{verdict}**。若纳入后续参照，配置为：','',
        '- P1h：已确认a/b结构＋F2邻县信息。','- P6h：本次确认的a/b结构，原163列特征。',
        '- D1h：既有K>0时U重建规则。','- P24/P48、其他D和全部N/R：保持原strict来源；主24h/48h输出不变。','',
        '没有重启P24/P48结构实验，没有联合迁移其他来源，没有新增特征或调整C3。此次完成的是CV确认与组合方案判断，尚未训练全量提交模型，也未覆盖旧基线预测。','',
        '## 5. 严格性、适配与验证','',
        '- 两套入口都显式指定CV；baseline loader和独立verifier均使用对应CSV，不能回退seed42。F2 identity及行级fold逐一核对。',
        '- 数学、权重、训练fit、评价与独立对齐函数与首轮保持一致，AST适配审计PASS；仅修改来源/输出路径、CV身份、候选范围和计数。',
        '- 每套完成40次内层早停＋10次refit，独立重载10个真实模型，检查40个probe的作用域、分支标签、权重和最佳轮数；复算40组控制指标、6组区间及全部县/折/时段表。',
        '- P71、p端点、允许训练范围和外折标签扰动检查通过；P1/N/D/R来源不变，四horizon有效行仍为34,177 / 32,982 / 28,680 / 22,944。',
        '- 既有seed42文件与原来源只读；本轮新产物限定在confirmation。三套结果按CV分别报告，没有把三套OOF平均或拼接成新候选。',
        '- 三套均来自同一事件和同239县，且都已参与开发；重复CV与县bootstrap不能消除空间相关或模型选择偏差。','',
        '## 6. 文件入口','',
        f'- [整体指标]({HERE}/summary/primary_scores.csv)',
        f'- [P分量指标]({HERE}/summary/P_component_metrics.csv)',
        f'- [区间]({HERE}/summary/bootstrap_intervals.csv)',
        f'- [县方向一致性]({HERE}/summary/county_consistency.csv)',
        f'- [误差交叉项]({HERE}/summary/OSI_change_decomposition.csv)',
        f'- [适配核验]({HERE}/reference/adaptation_verification.json)',
        f'- [后续可读取的三CV候选来源清单]({HERE}/summary/candidate_manifest.json)',
        f'- [执行说明]({HERE}/README.md)','']
    (HERE/'Results_2026-09-19.md').write_text('\n'.join(report))
    for seed in SEEDS[1:]:
        rows=candidate[candidate.split_seed==seed]
        sub=['# P6确认单CV结果','',f'CV seed={seed}，model seed=42；仅B0与AB_P06。独立核验PASS。','',
             table(['输出h','B0 RMSE','候选RMSE','变化'],[[f'{HOURS[HS.index(r.horizon)]}h',f'{r.baseline_rmse:.12f}',f'{r.rmse:.12f}',f'{r.rmse_change_pct:+.3f}%'] for r in rows.itertuples()]),'',
             f'完整解释与三套比较见[汇总]({HERE}/Results_2026-09-19.md)。','']
        (HERE/f'split{seed}/Results_2026-09-19.md').write_text('\n'.join(sub))
    (HERE/'README.md').write_text(f'''# P6 a/b迁移确认

**状态：seed20260917、seed20260918的P6确认完成，独立核验PASS。** [三套结果]({HERE}/Results_2026-09-19.md)。

{verdict}。首轮seed42产物保持只读；本轮每套50次拟合、10个外层模型，共100次拟合、20个模型。只对P6使用原163列a/b，模型seed固定42，没有联合候选或新特征。

`confirmation_runtime.py`要求显式`--split-seed`，每个CV输出到自己的split目录；代码共用，模型和配置不共用。适配审计记录在reference，数值函数保持原样。

首次执行顺序：

```bash
cd {BASE.parents[2]}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/prepare_confirmation.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/train_transfer.py --stage preflight --split-seed 20260917
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/train_transfer.py --stage preflight --split-seed 20260918
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/run_confirmations.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_structure_horizon_transfer/confirmation/summarize_confirmation.py
```

已完成训练/评估stage拒绝覆盖。重新核验某套可单独运行`verify_transfer.py --split-seed 20260917`（或20260918）；不拟合新模型。

- `reference/`：父产物哈希和AST适配审计。
- `split*/reference/`：各CV来源哈希、预检；`split*/experiment_config.json`：完整配置。
- `split*/runs/v18_ptransfer_ab_v1_split*_model42/`：模型、内层逐行预测和曲线、全OOF、所有控制、指标与独立验证。
- `split*/summary/`：本CV指标及误差交叉项；`summary/`：三CV按split汇总。
- `final_integrity_check.json`、`completion.json`：本轮完整性与完成清单。

没有改写父目录README、结果或完成标记；确认后的状态以本目录入口为准。没有全量模型或提交文件。
''')
    print(candidate[['split_seed','horizon','baseline_rmse','rmse','rmse_change_pct']].to_string(index=False))
    print(verdict)


if __name__=='__main__':main()
