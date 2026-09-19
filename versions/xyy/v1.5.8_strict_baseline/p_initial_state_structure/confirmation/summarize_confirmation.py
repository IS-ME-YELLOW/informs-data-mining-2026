"""Read immutable runs, summarize the three fixed CV splits, and audit lineage."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
SEEDS=(42,20260917,20260918)
H1='osi_target_t01h';H6='osi_target_t06h'
FOCUS={'39117':'Morrow OH','42053':'Forest PA','18013':'Brown IN','54015':'Clay WV',
       '54013':'Calhoun WV','39119':'Muskingum OH','39115':'Morgan OH'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def csv(path):return pd.read_csv(path,dtype={'slice_value':str,'fipsCode':str},float_precision='round_trip')


def md(frame,columns,formats=None):
    formats=formats or {}
    lines=['| '+' | '.join(columns.values())+' |','|'+'|'.join('---' for _ in columns)+'|']
    for _,r in frame.iterrows():
        cells=[]
        for key in columns:
            value=r[key]
            if pd.isna(value):s='—'
            elif key in formats:s=formats[key].format(value)
            elif isinstance(value,(float,np.floating)) and value.is_integer():s=str(int(value))
            else:s=str(value)
            cells.append(s)
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def loaded_run(seed):
    run=EXP/'runs'/f'v18_pstate_nested_v1_split{seed}_model42'
    marker=read_json(run/'P_CV_COMPLETE');manifest=read_json(run/'run_manifest.json')
    assert marker['identity_hash']==manifest['identity_hash']
    assert manifest['identity']['split_seed']==seed
    for relative,digest in marker['files'].items():
        path=(run/relative).resolve();assert path.is_relative_to(run.resolve())
        assert sha(path)==digest,(seed,relative)
    for path,digest in manifest['identity']['source_sha256'].items():assert sha(path)==digest,path
    for name,digest in manifest['identity']['code_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else EXP/name
        assert sha(path)==digest,path
    cfg=EXP/'experiment_config.json' if seed==42 else HERE/'experiment_config.json'
    assert sha(cfg)==manifest['identity']['config_sha256']
    verification=read_json(run/'logs/independent_verification.json')
    assert verification['status']=='PASS' and verification['identity_hash']==manifest['identity_hash']
    assert verification['new_outer_models']==15 and verification['inner_probes']==60
    assert verification['split_seed']==seed
    execution=read_json(run/'execution.json');assert execution['attempts'][-1]['status']=='completed'
    tables={}
    for name in ('pooled','slices','bootstrap','d_interaction','branch_diagnostics','branch_loss_decomposition','error_interactions'):
        frame=csv(run/f'metrics/{name}.csv');frame.insert(0,'split_seed',seed);tables[name]=frame
    rounds=csv(run/'round_selection.csv');rounds.insert(0,'split_seed',seed);tables['rounds']=rounds
    # Required long-horizon invariants are verified again directly in saved predictions.
    oof=pd.read_parquet(run/'control_predictions.parquet')
    for h in ('osi_target_t24h','osi_target_t48h'):
        for case in ('E1','E2'):
            np.testing.assert_array_equal(oof[f'pred_E0_v18_rule_{h}'],oof[f'pred_{case}_v18_rule_{h}'])
    assert oof.fipsCode.nunique()==239 and len(oof)==34416
    reference=EXP/'runs/v18_pstate_nested_v1_split42_model42/run_manifest.json'
    old=read_json(reference)['identity']
    for field in ('settings','params','features','p_sha256','P_label_sha256'):
        assert manifest['identity'][field]==old[field],(seed,field)
    source={'split_seed':seed,'run':str(run),'identity_hash':manifest['identity_hash'],
        'complete_marker_sha256':sha(run/'P_CV_COMPLETE'),
        'independent_verification_sha256':sha(run/'logs/independent_verification.json'),
        'baseline_identity_hash':manifest['identity']['baseline_identity_hash'],
        'D_G_identity_hash':manifest['identity']['g_identity_hash'],
        'formal_fits':sum(a['fits_this_attempt'] for a in execution['attempts']),
        'wall_seconds':sum(a['wall_seconds'] for a in execution['attempts']),
        'new_models':15,'verification':'PASS'}
    return tables,source


def main():
    collected={};sources=[]
    for seed in SEEDS:
        tables,source=loaded_run(seed);sources.append(source)
        for name,frame in tables.items():collected.setdefault(name,[]).append(frame)
    tables={name:pd.concat(frames,ignore_index=True) for name,frames in collected.items()}
    dest=HERE/'summary';dest.mkdir(exist_ok=True);outputs={}
    def save(name,frame):
        path=dest/name;frame.to_csv(path,index=False);outputs[name]=sha(path)
    for name,frame in tables.items():save(f'{name}_all_splits.csv',frame)
    primary=tables['pooled'].query("domain=='OSI' and mode=='v18_rule'")
    scores=[];p_scores=[]
    for (seed,h),group in primary.groupby(['split_seed','horizon'],sort=False):
        a=group.loc[group.comparison=='E1_minus_E0'].iloc[0]
        b=group.loc[group.comparison=='E2_minus_E0'].iloc[0]
        c=group.loc[group.comparison=='E2_minus_E1'].iloc[0]
        scores.append({'split_seed':seed,'horizon':h,'n':int(b.n),'E0_rmse':b.baseline_rmse,
            'E1_rmse':a.candidate_rmse,'E2_rmse':b.candidate_rmse,
            'E1_vs_E0_pct':a.rmse_change_pct,'E2_vs_E0_pct':b.rmse_change_pct,'E2_vs_E1_pct':c.rmse_change_pct})
    scores=pd.DataFrame(scores);save('primary_scores.csv',scores)
    p_source=tables['pooled'].query("domain=='P' and horizon=='osi_target_t01h'")
    for (seed,space),group in p_source.groupby(['split_seed','mode'],sort=False):
        a=group.loc[group.comparison=='E1_minus_E0'].iloc[0]
        b=group.loc[group.comparison=='E2_minus_E0'].iloc[0]
        c=group.loc[group.comparison=='E2_minus_E1'].iloc[0]
        p_scores.append({'split_seed':seed,'space':space,'E0_rmse':b.baseline_rmse,'E1_rmse':a.candidate_rmse,
            'E2_rmse':b.candidate_rmse,'E1_vs_E0_pct':a.rmse_change_pct,'E2_vs_E0_pct':b.rmse_change_pct,
            'E2_vs_E1_pct':c.rmse_change_pct})
    p_scores=pd.DataFrame(p_scores);save('P_component_scores.csv',p_scores)
    slices=tables['slices'];boots=tables['bootstrap']
    county=slices.query("domain=='OSI' and mode=='v18_rule' and slice_type=='county'").copy()
    county['fraction_of_net_sse_reduction']=county.sse_reduction/county.groupby(['split_seed','horizon','comparison']).sse_reduction.transform('sum')
    save('county_effects.csv',county)
    focus=county.loc[county.slice_value.isin(FOCUS)].copy();focus['focus_name']=focus.slice_value.map(FOCUS)
    save('focus_counties.csv',focus)
    counts=[]
    for (seed,h,pair),group in county.groupby(['split_seed','horizon','comparison']):
        if h not in (H1,H6):continue
        folds=slices.loc[(slices.split_seed==seed)&(slices.domain=='OSI')&(slices['mode']=='v18_rule')&
            (slices.horizon==h)&(slices.comparison==pair)&(slices.slice_type=='fold')]
        morrow=group.loc[group.slice_value=='39117'].iloc[0]
        early=slices.loc[(slices.split_seed==seed)&(slices.domain=='OSI')&(slices['mode']=='v18_rule')&
            (slices.horizon==h)&(slices.comparison==pair)&(slices.slice_type=='window')&slices.slice_value.isin(['73-76','77-95'])]
        net=group.sse_reduction.sum()
        counts.append({'split_seed':seed,'horizon':h,'comparison':pair,'better_counties':int((group.sse_reduction>0).sum()),
            'worse_counties':int((group.sse_reduction<0).sum()),'unchanged_counties':int((group.sse_reduction==0).sum()),
            'better_folds':int((folds.sse_reduction>0).sum()),'net_sse_reduction':net,
            'Morrow_fraction_pct':100*morrow.sse_reduction/net if net!=0 else np.nan,
            'early_to95_fraction_pct':100*early.sse_reduction.sum()/net if net!=0 else np.nan})
    counts=pd.DataFrame(counts);save('improvement_counts.csv',counts)
    for typ,name in [('window','window_effects.csv'),('p_bin','initial_state_groups.csv'),
                     ('population','exclude_Morrow.csv'),('target_hour','target_hour_curve_data.csv'),('fold','fold_effects.csv')]:
        save(name,slices.loc[slices.slice_type==typ])
    consistency=[]
    for (h,pair,code),group in county.groupby(['horizon','comparison','slice_value']):
        assert set(group.split_seed)==set(SEEDS)
        consistency.append({'horizon':h,'comparison':pair,'fipsCode':code,'countyName':group.countyName.iloc[0],
            'improved_splits':int((group.sse_reduction>0).sum()),'worsened_splits':int((group.sse_reduction<0).sum()),
            'minimum_rmse_change_pct':group.rmse_change_pct.min(),'maximum_rmse_change_pct':group.rmse_change_pct.max(),
            'mean_sse_reduction_diagnostic_only':group.sse_reduction.mean()})
    consistency=pd.DataFrame(consistency);save('county_consistency.csv',consistency)
    save('source_runs.csv',pd.DataFrame(sources))
    short=scores.loc[scores.horizon.isin([H1,H6])]
    all_gain=bool((short.E2_vs_E0_pct<0).all() and (short.E2_vs_E1_pct<0).all())
    if all_gain:
        recommendation='三套固定分折中，E2 的 1h/6h 均优于当前 E0 与直接 L2 对照 E1，支持将这版 P 初始状态结构作为当前优先 P 候选保留，并在后续实验中沿用同一固定定义。这里没有自动替换原基线或训练最终提交模型。'
    else:
        recommendation='三套结果未在所有短时指标和对照上保持一致改善，应保留为混合证据，逐项说明退化；本轮不通过挑选分折、县或时段决定采用，也不自动训练最终提交模型。'
    effect=counts.query("comparison=='E2_minus_E0'")
    interval=boots.query("comparison=='E2_minus_E0' and population=='all_counties'").copy()
    interval['interval']=interval.apply(lambda r:f'[{r.ci_low:+.12f}, {r.ci_high:+.12f}]',axis=1)
    excluded=boots.query("domain=='OSI' and comparison=='E2_minus_E0' and population=='exclude_Morrow'").copy()
    excluded['interval']=excluded.apply(lambda r:f'[{r.ci_low:+.12f}, {r.ci_high:+.12f}]',axis=1)
    windows=slices.query("domain=='OSI' and mode=='v18_rule' and horizon=='osi_target_t01h' and comparison=='E2_minus_E0' and slice_type=='window'")
    bins=slices.query("domain=='OSI' and mode=='v18_rule' and horizon=='osi_target_t01h' and comparison=='E2_minus_E0' and slice_type=='p_bin'")
    selected_focus=focus.query("horizon=='osi_target_t01h' and comparison=='E2_minus_E0'")
    # Verify all old files; new confirmation outputs and the two new runs are additions only.
    snapshot=read_json(HERE/'reference/initial_workspace_snapshot.json')
    changed=[path for path,digest in snapshot['sha256'].items() if not Path(path).is_file() or sha(path)!=digest]
    assert not changed,changed
    integrity={'status':'PASS','checked_historical_files':len(snapshot['sha256']),'changed_files':changed,
        'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    (HERE/'final_integrity_check.json').write_text(json.dumps(integrity,indent=2)+'\n')
    fmt={'E0_rmse':'{:.12f}','E1_rmse':'{:.12f}','E2_rmse':'{:.12f}',
         'E1_vs_E0_pct':'{:+.5f}%','E2_vs_E0_pct':'{:+.5f}%','E2_vs_E1_pct':'{:+.5f}%'}
    primary_columns={'split_seed':'分折','horizon':'horizon','E0_rmse':'E0 RMSE','E1_rmse':'E1 RMSE','E2_rmse':'E2 RMSE',
                     'E1_vs_E0_pct':'E1 / E0 变化','E2_vs_E0_pct':'E2 / E0 变化','E2_vs_E1_pct':'E2 / E1 变化'}
    component_columns={'split_seed':'分折','space':'P 空间','E0_rmse':'E0 RMSE','E1_rmse':'E1 RMSE','E2_rmse':'E2 RMSE',
                       'E2_vs_E0_pct':'E2 / E0 变化','E2_vs_E1_pct':'E2 / E1 变化'}
    consistent=consistency.query("horizon=='osi_target_t01h' and comparison=='E2_minus_E0'")
    both_new=[s for s in sources if s['split_seed']!=42]
    new_fits=sum(s['formal_fits'] for s in both_new);assert new_fits==150
    total_fits=sum(s['formal_fits'] for s in sources);assert total_fits==225
    gain1=-short.loc[short.horizon==H1,'E2_vs_E0_pct']
    gain6=-short.loc[short.horizon==H6,'E2_vs_E0_pct']
    gain_p=-p_scores.loc[p_scores.space=='aligned','E2_vs_E0_pct']
    boot_osi=interval.loc[interval.domain=='OSI']
    synthesis=(f'相对当前 E0，三套 1h RMSE 改善 {gain1.min():.2f}%–{gain1.max():.2f}%，'
        f'6h 改善 {gain6.min():.2f}%–{gain6.max():.2f}%；C3 后 P1h RMSE 改善 '
        f'{gain_p.min():.2f}%–{gain_p.max():.2f}%。直接换为 L2 的 E1 变化远小于 E2，'
        '支持结构候选具有额外增量。' if all_gain else '')
    interval_note=('三套的最终 OSI1h/6h 与 C3 后 P1h 区间均位于零以下；原始 P1h 的 seed42 区间仍跨零，'
        '另外两套位于零以下。因此，当前证据在最终评分口径上比单看原始 P 分量更一致。'
        if (boot_osi.ci_high<0).all() and (interval.loc[(interval.domain=='P')&(interval['mode']=='aligned'),'ci_high']<0).all()
        else '各口径的区间方向需分别判断，不能从分量点估计推断最终评分的不确定性。')
    report=f'''# P 初始状态结构：两套确认与三分折汇总（2026-09-18）

**seed20260917、seed20260918 均已完整训练并通过新进程独立验证。** 本轮新增 150 次拟合、30 个外层模型；连同 seed42，共 225 次拟合、45 个外层模型。未训练全县最终模型，未生成测试提交。

{recommendation}

{synthesis}

## 1. 固定对照与路径

E0＝同分折严格 P-Huber＋D_G；E1＝仅 P1h 改为直接 L2；E2＝仅 P1h 改为 a/b 分解重建。所有候选保持同分折其他模型、163 列特征、模型 seed42、损失/权重定义、严格内层选轮、C3 和主规则。

原 seed42 代码已被其运行身份冻结。本轮使用一个共用确认入口，仅适配分折和路径；训练、权重、评价、预检与独立数值模块直接复用原文件。适配范围和数学函数 AST 核对见 `reference/adaptation_verification.json`，完整差异见 `reference/runner_adaptation.diff`。数值实验定义没有更改。

模型统一保存在父实验 `runs/v18_pstate_nested_v1_split{{seed}}_model42/`。确认配置、日志、预检和本报告位于 `confirmation/`；原 seed42 的所有文件均未修改。

## 2. 最终 OSI：三套短时主结果

每套包括全部 239 县；1h 34,177 行、6h 32,982 行。下表负的相对变化表示改善。

{md(short,primary_columns,fmt)}

24/48h 的主预测三套均逐行精确不变，完整数值见 `summary/primary_scores.csv`。主规则仍为 1/6h 用 C3、24/48h 用 C1；没有根据成绩更换路由。

E2−E1 用来判断结构候选超过“只换 L2”的部分；E2−E0 是当前组合的净增量。E2 含分解、贡献权重及两个分支，容量也不同，不能把全部收益归于单一公式。

## 3. P 分量精度

raw 是分支截断/重建后、C3 前；aligned 是 C3 后实际参与短时主结果的 P1h。D、N/R 和原始 P6/24/48 逐行不变。

{md(p_scores,component_columns,fmt)}

以下是 E2−E0 的县轨迹配对 bootstrap 95% 百分位区间（2,000 次，seed20260910）。raw P、aligned P 和最终 OSI 分开判断，不以单个分量区间代替比赛指标：

{md(interval,{'split_seed':'分折','domain':'指标','mode':'空间/规则','horizon':'horizon','rmse_delta':'ΔRMSE','interval':'95% 区间'},{'rmse_delta':'{:+.12f}'})}

E2−E1 和 E1−E0 的相同区间也完整保存在 `summary/bootstrap_all_splits.csv`。没有将三套重复使用的县拼成一个独立样本来缩小区间。

{interval_note}

## 4. 是否由少数县驱动

{md(effect,{'split_seed':'分折','horizon':'horizon','better_counties':'改善县','worse_counties':'恶化县','unchanged_counties':'不变县',
    'better_folds':'改善折/5','net_sse_reduction':'净 SSE 改善','Morrow_fraction_pct':'Morrow 占净改善'},{'net_sse_reduction':'{:+.9f}','Morrow_fraction_pct':'{:.2f}%'})}

固定排除 Morrow 后的诊断：

{md(excluded,{'split_seed':'分折','horizon':'horizon','baseline_rmse':'E0 RMSE','candidate_rmse':'E2 RMSE','rmse_change_pct':'相对变化','interval':'95% 区间'},
    {'baseline_rmse':'{:.12f}','candidate_rmse':'{:.12f}','rmse_change_pct':'{:+.5f}%'})}

主 1h 中，三套均改善的县有 **{int((consistent.improved_splits==3).sum())}** 个，三套均恶化的县有 **{int((consistent.worsened_splits==3).sum())}** 个。完整逐县方向一致性见 `county_consistency.csv`。净改善占比反映正负效果相抵后的贡献，不能当作获益县比例。

此前关注县的主 1h 变化（全部保留，无推理例外）：

{md(selected_focus,{'split_seed':'分折','focus_name':'县','baseline_rmse':'E0 RMSE','candidate_rmse':'E2 RMSE','rmse_change_pct':'相对变化'},
    {'baseline_rmse':'{:.9f}','candidate_rmse':'{:.9f}','rmse_change_pct':'{:+.4f}%'})}

## 5. 时段与初始状态

预先固定的四段目标小时，主 1h 的 E2−E0：

{md(windows,{'split_seed':'分折','slice_value':'目标小时','baseline_rmse':'E0 RMSE','candidate_rmse':'E2 RMSE','rmse_change_pct':'相对变化','sse_reduction':'SSE 改善量'},
    {'baseline_rmse':'{:.9f}','candidate_rmse':'{:.9f}','rmse_change_pct':'{:+.4f}%','sse_reduction':'{:+.9f}'})}

目标小时 73–95 占主 1h 净改善的比例：

{md(effect.loc[effect.horizon==H1],{'split_seed':'分折','early_to95_fraction_pct':'早期占净改善'},{'early_to95_fraction_pct':'{:.2f}%'})}

初始 P71 分组：

{md(bins,{'split_seed':'分折','slice_value':'P71 分组','counties':'县数','rmse_change_pct':'E2 相对 E0','sse_reduction':'SSE 改善量'},
    {'rmse_change_pct':'{:+.4f}%','sse_reduction':'{:+.9f}'})}

这些切片用于解释结构适用范围。没有据此增加县名单、近零阈值、时间开关或新路由。完整逐目标小时数据见 `target_hour_curve_data.csv`。

从三套切片看，较高初始停电组和早期窗口的收益最一致。目标小时 144–215 的变化分别为 seed42 +0.3048%、seed20260917 +1.0295%、seed20260918 −0.5349%；p=0 组也不是三套一致改善。该结构改善了初始状态的利用，但尚未证明能普遍解决晚期新增停电或低初始停电县的异常事件。保留这些边界，不继续用已看过的 OOF 搜索开关。

## 6. 误差抵消、分支与 D 交互

`branch_diagnostics_all_splits.csv` 保存原始/截断分支误差及截断数量；`branch_loss_decomposition_all_splits.csv` 核对 a、b 贡献 MSE 和交叉项；`error_interactions_all_splits.csv` 保存 P 与 N/D/R 合成误差的交叉项、官方发布精度和最终后处理影响。

将所有候选共同改用旧 D_B 后，主 1h 的 E2−E0：

{md(tables['d_interaction'].query("mode=='v18_rule' and horizon=='osi_target_t01h' and comparison=='E2_minus_E0'"),
    {'split_seed':'分折','baseline_rmse':'E0＋D_B RMSE','candidate_rmse':'E2＋D_B RMSE','rmse_change_pct':'相对变化'},
    {'baseline_rmse':'{:.12f}','candidate_rmse':'{:.12f}','rmse_change_pct':'{:+.5f}%'})}

主结果始终使用 D_G，未在 D_B/D_G 中挑选更有利版本。D/N/R 本身未重训，P 带来的端到端差异须结合误差抵消理解。

## 7. 验证与运行记录

{md(pd.DataFrame(sources),{'split_seed':'分折','formal_fits':'正式拟合','new_models':'外层模型','wall_seconds':'单套进程用时/秒','verification':'独立验证'}, {'wall_seconds':'{:.2f}'})}

本轮两套并行运行，每个 LightGBM 模型保持单线程，因此单套进程用时不能相加当作用户等待时长。新进程分别重载真实模型，复算标签/权重/内层轮数、重建 P、全部 C0–C3 与主规则、逐行误差、所有切片和 bootstrap；两套 `P_CV_COMPLETE` 都在验证 PASS 后生成。

数值预检和模型 receipt 证明：先按允许县、时间和支持条件选行再取标签；权重均值只用当前训练支持集；外折未来标签不参与训练/早停；未来停电数据不进入 P71；p=0 县完整参与最终评分；跨 horizon 对齐没有更改来源分折。

完成后核对开工前 **{len(snapshot['sha256'])}** 个历史文件，全部未改动；包含原 seed42 数值代码、模型、配置、文档和完成标记。来源/运行身份详见 `summary/source_runs.csv`；完整性结果见 `final_integrity_check.json`。

## 8. 决策与边界

{recommendation}

采用与否以最终 OSI RMSE 为主，P 分量改善是机制证据；恶化县和晚期窗口仍需要保留在后续评价中。本轮不继续围绕这三套 OOF 搜索县级或时间阈值。

三套是同一批 239 县、同一场事件的重复开发分折，已经被此前分析使用，并非三个未见过的外部验证集。县 bootstrap 未消除空间相关及反复模型选择偏差。不能把这里的改善解释为隐藏测试集保证，也不能把四个原始 RMSE 的均值称为官方排名分数。

## 9. 文件索引

- `summary/primary_scores.csv`、`P_component_scores.csv`：三套主要结果。
- `summary/pooled_all_splits.csv`、`bootstrap_all_splits.csv`：全部对照与区间。
- `summary/county_effects.csv`、`county_consistency.csv`、`focus_counties.csv`：县级效果与一致性。
- `summary/window_effects.csv`、`initial_state_groups.csv`、`target_hour_curve_data.csv`：固定时段、初始状态与逐时数据。
- `split20260917/Results_2026-09-18.md`、`split20260918/Results_2026-09-18.md`：各套简报。
- 父目录 `runs/v18_pstate_nested_v1_split{{seed}}_model42/`：模型、逐行 OOF、receipt 和独立验证。

CSV 复读用 `float_precision="round_trip"`；没有平均或拼合三套 OOF 预测构造新的候选成绩。
'''
    (HERE/'Results_2026-09-18.md').write_text(report)
    for seed in (20260917,20260918):
        section=f'''# P 初始状态结构：seed{seed}（2026-09-18）

75 次正式拟合、15 个外层模型和新进程独立验证均完成。使用同分折 E0＝严格 P-Huber＋D_G，E1＝直接 P-L2，E2＝初始状态结构。

{md(scores.loc[scores.split_seed==seed],primary_columns,fmt)}

P1h 分量：

{md(p_scores.loc[p_scores.split_seed==seed],component_columns,fmt)}

{md(effect.loc[effect.split_seed==seed],{'horizon':'horizon','better_counties':'改善县','worse_counties':'恶化县','better_folds':'改善折/5','Morrow_fraction_pct':'Morrow 占净改善'},{'Morrow_fraction_pct':'{:.2f}%'})}

完整三套比较、区间、时段分析与结论见 [三分折汇总]({HERE/'Results_2026-09-18.md'})。24/48h 主预测逐行不变，没有改路由、增加门控或生成测试提交。

运行目录：`{EXP/'runs'/f'v18_pstate_nested_v1_split{seed}_model42'}`。
'''
        (HERE/f'split{seed}/Results_2026-09-18.md').write_text(section)
    report_files=[HERE/'Results_2026-09-18.md',*(HERE/f'split{s}/Results_2026-09-18.md' for s in (20260917,20260918))]
    summary={'status':'PASS','source_runs':sources,'outputs':outputs,'reports':{str(p):sha(p) for p in report_files},
        'summary_script_sha256':sha(__file__),'same_numerical_settings_across_splits':True,
        'all_source_markers_and_code_hashes':'PASS','all_long_horizon_invariants':'PASS'}
    (dest/'summary_manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    completion={'status':'COMPLETE','new_split_seeds':[20260917,20260918],'summary_split_seeds':list(SEEDS),
        'new_formal_fits':150,'new_outer_models':30,'all_three_formal_fits':225,'all_three_outer_models':45,
        'independent_verification':'PASS','historical_integrity':'PASS','recommendation':recommendation,
        'final_training':False,'test_submission':False,'finished_at_utc':datetime.now(timezone.utc).isoformat()}
    (HERE/'completion.json').write_text(json.dumps(completion,indent=2,ensure_ascii=False)+'\n')
    print(short.to_string(index=False));print(p_scores.to_string(index=False));print(recommendation)


if __name__=='__main__':main()
