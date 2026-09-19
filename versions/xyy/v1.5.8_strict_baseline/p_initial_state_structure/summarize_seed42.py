"""Summarize only independently verified seed42 artifacts; never fit/select models."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/v18_pstate_nested_v1_split42_model42'


def read_csv(path):
    return pd.read_csv(path,dtype={'slice_value':str,'fipsCode':str},float_precision='round_trip')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def table(frame,columns,formats=None):
    formats=formats or {}
    lines=['| '+' | '.join(columns.values())+' |','|'+'|'.join('---' for _ in columns)+'|']
    for _,row in frame.iterrows():
        cells=[]
        for key in columns:
            value=row[key]
            if pd.isna(value):cell='—'
            elif key in formats:cell=formats[key].format(value)
            else:cell=str(value)
            cells.append(cell)
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def main():
    marker=json.loads((RUN/'P_CV_COMPLETE').read_text())
    for name,digest in marker['files'].items():assert sha(RUN/name)==digest,name
    verification=json.loads((RUN/'logs/independent_verification.json').read_text())
    assert verification['status']=='PASS' and verification['identity_hash']==marker['identity_hash']
    pooled=read_csv(RUN/'metrics/pooled.csv');slices=read_csv(RUN/'metrics/slices.csv')
    bootstrap=read_csv(RUN/'metrics/bootstrap.csv');rounds=read_csv(RUN/'round_selection.csv')
    dest=HERE/'summary';dest.mkdir(exist_ok=True)
    outputs={}
    def save(name,df):
        path=dest/name;df.to_csv(path,index=False);outputs[name]=sha(path)
    primary=pooled.query("domain=='OSI' and mode=='v18_rule'").copy()
    save('primary_comparisons.csv',primary)
    scores=[]
    for horizon,g in primary.groupby('horizon',sort=True):
        e1=g.loc[g.comparison=='E1_minus_E0'].iloc[0];e2=g.loc[g.comparison=='E2_minus_E0'].iloc[0]
        scores.append({'horizon':horizon,'n':int(e2.n),'E0_rmse':e2.baseline_rmse,'E1_rmse':e1.candidate_rmse,'E2_rmse':e2.candidate_rmse,
            'E1_change_pct':e1.rmse_change_pct,'E2_change_pct':e2.rmse_change_pct,
            'E2_vs_E1_pct':g.loc[g.comparison=='E2_minus_E1','rmse_change_pct'].iloc[0]})
    scores=pd.DataFrame(scores);save('primary_scores.csv',scores)
    p_scores=pooled.query("domain=='P' and horizon=='osi_target_t01h'").copy();save('P_component_comparisons.csv',p_scores)
    save('primary_bootstrap.csv',bootstrap)
    county=slices.query("domain=='OSI' and mode=='v18_rule' and slice_type=='county'").copy()
    county['fraction_of_net_sse_reduction']=county.sse_reduction/county.groupby(['horizon','comparison']).sse_reduction.transform('sum')
    county=county.sort_values(['horizon','comparison','sse_reduction'],ascending=[True,True,False]);save('county_effects.csv',county)
    # Previously discussed counties, with state disambiguation through FIPS.
    focus_codes=['39117','42053','18013','54015','54013','39119','39115']
    focus=county.loc[county.slice_value.isin(focus_codes)].copy()
    assert set(focus.slice_value)==set(focus_codes)
    save('focus_counties.csv',focus)
    save('window_effects.csv',slices.query("slice_type=='window'"))
    save('initial_state_groups.csv',slices.query("slice_type=='p_bin'"))
    save('target_hour_curve_data.csv',slices.query("slice_type=='target_hour'"))
    save('fold_effects.csv',slices.query("slice_type=='fold'"))
    save('exclude_Morrow.csv',slices.query("slice_type=='population'"))
    h1='osi_target_t01h';h6='osi_target_t06h'
    r2=primary.query('horizon==@h1 and comparison=="E2_minus_E0"').iloc[0]
    r21=primary.query('horizon==@h1 and comparison=="E2_minus_E1"').iloc[0]
    r1=primary.query('horizon==@h1 and comparison=="E1_minus_E0"').iloc[0]
    g6=primary.query('horizon==@h6').set_index('comparison')
    if r2.rmse_delta<0 and r21.rmse_delta<0 and g6.loc['E2_minus_E0','rmse_delta']<=1e-12 and g6.loc['E2_minus_E1','rmse_delta']<=1e-12:
        recommendation='E2 达到预定首轮晋级条件：建议在两套固定分折同时确认 E1/E2。当前仍只是 seed42 开发证据，不能直接认定隐藏测试集获益。'
    elif r1.rmse_delta<0 and g6.loc['E1_minus_E0','rmse_delta']<=1e-12 and r21.rmse_delta>=0:
        recommendation='本轮优先支持直接 P-L2（E1），没有证据支持 E2 相对 E1 的结构增量。建议优先确认 E1，不宣称分解结构成功。'
    elif min(r1.rmse_delta,r2.rmse_delta)<0:
        recommendation='本轮为混合结果，未满足结构候选的完整晋级条件。应结合 6h、P 精度和县级损益解释后再决定，不自动开始另两套或新增门控。'
    else:
        recommendation='E1/E2 均未改善当前参照的主 1h RMSE，本轮不建议自动推进这版结构；保留负结果，先解释误差来源。'
    pd_summary=[]
    for space,g in p_scores.groupby('mode'):
        e1=g.query('comparison=="E1_minus_E0"').iloc[0];e2=g.query('comparison=="E2_minus_E0"').iloc[0]
        pd_summary.append({'space':space,'E0':e2.baseline_rmse,'E1':e1.candidate_rmse,'E2':e2.candidate_rmse,
            'E1_pct':e1.rmse_change_pct,'E2_pct':e2.rmse_change_pct})
    pd_summary=pd.DataFrame(pd_summary)
    bootshow=bootstrap.query("population=='all_counties'").copy()
    bootshow['interval']=bootshow.apply(lambda r:f"[{r.ci_low:+.12f}, {r.ci_high:+.12f}]",axis=1)
    countrows=[]
    for (h,pair),g in county.groupby(['horizon','comparison']):
        if h not in (h1,h6):continue
        folds=slices.query('domain=="OSI" and mode=="v18_rule" and slice_type=="fold" and horizon==@h and comparison==@pair')
        countrows.append({'horizon':h,'comparison':pair,'better_counties':int((g.sse_reduction>0).sum()),
            'worse_counties':int((g.sse_reduction<0).sum()),'unchanged_counties':int((g.sse_reduction==0).sum()),
            'better_folds':int((folds.sse_reduction>0).sum()),'net_sse_reduction':g.sse_reduction.sum()})
    counts=pd.DataFrame(countrows);save('improvement_counts.csv',counts)
    model_summary=rounds.groupby(['outer_fold','kind']).agg(inner_best=('best_iteration',lambda x:','.join(str(int(v)) for v in x)),
        requested=('refit_rounds','first'),actual=('actual_trees','first'),weight_mean_min=('weight_mean','min'),weight_mean_max=('weight_mean','max')).reset_index()
    save('model_rounds.csv',model_summary)
    # Historical files are checked against the snapshot from before implementation.
    snapshot=json.loads((HERE/'reference/initial_workspace_snapshot.json').read_text())
    changed=[p for p,d in snapshot['sha256'].items() if not Path(p).is_file() or sha(p)!=d]
    integrity={'status':'PASS' if not changed else 'FAIL','checked_files':len(snapshot['sha256']),'changed_files':changed,
               'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    (HERE/'final_integrity_check.json').write_text(json.dumps(integrity,indent=2)+'\n');assert not changed
    # Focus trajectories for user inspection, including all hours, not only improved rows.
    branch=pd.read_parquet(RUN/'p_branch_oof_predictions.parquet');control=pd.read_parquet(RUN/'control_predictions.parquet')
    component=pd.read_parquet(RUN/'component_predictions.parquet');audit=pd.read_parquet(RUN/'target_audit.parquet')
    fips=set(focus.slice_value);take=branch.fipsCode.isin(fips)&branch.scoreable
    trajectory=branch.loc[take,['fipsCode','timestamp_et','hour_idx','fold','p71','pred_E1_P','pred_E2_P','a_P_contribution','b_P_contribution']].copy()
    trajectory['official_P']=audit.loc[take,'official_P'];trajectory['E0_P']=component.loc[take,'pred_E0_P_t_target_t01h']
    trajectory['official_OSI']=control.loc[take,f'actual_{h1}']
    for case in ('E0','E1','E2'):trajectory[f'{case}_OSI']=control.loc[take,f'pred_{case}_v18_rule_{h1}']
    save('focus_trajectories.csv',trajectory)
    def top(pair,good):
        g=county.loc[(county.horizon==h1)&(county.comparison==pair)].sort_values('sse_reduction',ascending=not good).head(8)
        return table(g,{'slice_value':'FIPS','countyName':'县','baseline_rmse':'参照 RMSE','candidate_rmse':'候选 RMSE',
            'sse_reduction':'SSE 改善量'}, {'baseline_rmse':'{:.9f}','candidate_rmse':'{:.9f}','sse_reduction':'{:+.9f}'})
    w=slices.query('domain=="OSI" and mode=="v18_rule" and slice_type=="window" and horizon==@h1')
    ex=slices.query('domain=="OSI" and mode=="v18_rule" and slice_type=="population" and horizon==@h1')
    inter=read_csv(RUN/'metrics/error_interactions.csv');cross=read_csv(RUN/'metrics/branch_loss_decomposition.csv')
    db=read_csv(RUN/'metrics/d_interaction.csv').query('mode=="v18_rule" and horizon==@h1')
    log=json.loads((RUN/'execution.json').read_text())
    fits=sum(x['fits_this_attempt'] for x in log['attempts']);seconds=sum(x['wall_seconds'] for x in log['attempts'])
    c1=county.loc[(county.horizon==h1)&(county.comparison=='E2_minus_E0')]
    morrow=c1.loc[c1.slice_value=='39117'].iloc[0]
    early=w.loc[(w.comparison=='E2_minus_E0')&w.slice_value.isin(['73-76','77-95']),'sse_reduction'].sum()
    interpretation=(f"结构候选的收益并非主要来自 Huber 改为 L2：E1 的 1h 仅变化 {r1.rmse_change_pct:+.4f}%，"
        f"而 E2 相对 E1 改善 {-r21.rmse_change_pct:.4f}%。E2 相对 E0 的五个外折 1h 均改善，"
        f"但仍有 {int((c1.sse_reduction<0).sum())} 个县恶化。Morrow 贡献净 SSE 改善的 "
        f"{100*morrow.sse_reduction/c1.sse_reduction.sum():.2f}%；排除它后 1h 仍改善 "
        f"{-ex.loc[ex.comparison=='E2_minus_E0','rmse_change_pct'].iloc[0]:.4f}%。"
        f"目标小时 73–95 贡献净改善的 {100*early/c1.sse_reduction.sum():.2f}%，后段 144–215 小幅恶化。"
        "这与方法强调初始状态的假设相符，但尚不能认定为普遍的后期增长/恢复建模能力。")
    body=f'''# P 初始状态结构实验：seed42 结果（2026-09-18）

**seed42 的 75 次正式拟合、15 个外层模型及新进程独立验证已完成。** 未训练另外两套分折，未训练全县最终模型，未生成测试提交。全部 239 县、34,177 个有效 1h 行进入主评分。

{recommendation}

## 1. 实验定义与主结果

E0 为严格 P-Huber＋D_G；E1 仅将 P1h 换成直接 L2；E2 仅将 P1h 换成初始状态 a/b 分解再重建。全部使用同一 D_G、163 列特征、CV seed42、模型 seed42、C3 和 v18_rule。E2−E1 检验结构候选的额外价值；E2−E0 检验对当前方案的净增量。

{table(scores,{'horizon':'horizon','E0_rmse':'E0 RMSE','E1_rmse':'E1 RMSE','E2_rmse':'E2 RMSE','E1_change_pct':'E1 相对 E0','E2_change_pct':'E2 相对 E0','E2_vs_E1_pct':'E2 相对 E1'},{'E0_rmse':'{:.12f}','E1_rmse':'{:.12f}','E2_rmse':'{:.12f}','E1_change_pct':'{:+.6f}%','E2_change_pct':'{:+.6f}%','E2_vs_E1_pct':'{:+.6f}%'})}

1/6h 仍固定使用 C3，24/48h 仍用 C1；未按实验结果改路由。24/48h 主预测逐行精确不变。P1h 的整段变化可以经同目标小时 C3 影响 6h，所以不能要求 6h 不变。

{interpretation}

## 2. P 分量是否更准确

raw 指分量截断/重建后、C3 前；aligned 指 C3 对齐后实际参与短时主结果的 P。

{table(pd_summary,{'space':'P 空间','E0':'E0 RMSE','E1':'E1 RMSE','E2':'E2 RMSE','E1_pct':'E1 相对 E0','E2_pct':'E2 相对 E0'},{'E0':'{:.12f}','E1':'{:.12f}','E2':'{:.12f}','E1_pct':'{:+.6f}%','E2_pct':'{:+.6f}%'})}

D、N/R 及其他 horizon 的原始 P 逐行保持原值。本实验没有改进或重新训练它们。P 改善是否转化为 OSI 改善，须看组合误差，不能从分量指标直接推断。

本轮 E2 的原始 P1h 点估计改善，但其配对 bootstrap 区间仍跨零；C3 后 P1h、最终 OSI1h/6h 的相应区间均位于零以下。应区分这些口径，不能笼统声称所有 P 结果均显著。

## 3. 区间与县级集中度

以下为候选−参照 RMSE 的县轨迹配对 bootstrap 95% 百分位区间（2,000 次，seed20260910）：

{table(bootshow,{'domain':'指标','mode':'空间/规则','horizon':'horizon','comparison':'比较','rmse_delta':'ΔRMSE','interval':'95% 区间'},{'rmse_delta':'{:+.12f}'})}

{table(counts,{'horizon':'horizon','comparison':'比较','better_counties':'改善县','worse_counties':'恶化县','unchanged_counties':'不变县','better_folds':'改善折/5','net_sse_reduction':'净 SSE 改善'},{'net_sse_reduction':'{:+.9f}'})}

固定排除 Morrow 的 1h 敏感性分析（仅诊断，主结果始终包括它）：

{table(ex,{'comparison':'比较','baseline_rmse':'参照 RMSE','candidate_rmse':'候选 RMSE','rmse_change_pct':'相对变化'},{'baseline_rmse':'{:.12f}','candidate_rmse':'{:.12f}','rmse_change_pct':'{:+.6f}%'})}

完整县表和净 SSE 贡献比例见 `summary/county_effects.csv`；负的 SSE 改善量表示该县变差。全部恶化案例均保留。

## 4. 变化来自哪些时段与县

固定四段目标小时的主 1h 结果：

{table(w,{'slice_value':'目标小时','comparison':'比较','baseline_rmse':'参照 RMSE','candidate_rmse':'候选 RMSE','rmse_change_pct':'相对变化'},{'baseline_rmse':'{:.9f}','candidate_rmse':'{:.9f}','rmse_change_pct':'{:+.4f}%'})}

E2 相对 E0 的最大收益县：

{top('E2_minus_E0',True)}

E2 相对 E0 的最大损失县：

{top('E2_minus_E0',False)}

E2 相对直接 L2 的最大损失县：

{top('E2_minus_E1',False)}

`initial_state_groups.csv` 保存预定 P71 分组结果；`target_hour_curve_data.csv` 保存完整逐目标小时曲线数据；`focus_trajectories.csv` 保存关注县逐行 P、a/b 贡献及 OSI 轨迹。分组和县名单仅用于解释，没有加入训练权重或推理开关。

P71≥0.5 的两县组收益突出，但 p=0 组总体小幅变差。Forest、Clay 仍有损失，说明当前结构主要帮助一部分已有较高初始停电的预测，尚未解决所有极端县。不能据这一次分组结果立即关闭某组分支或给县设例外。

## 5. 分支损失与组合误差

E2 的分解平方误差含交叉项，不能简单相加两条分支 RMSE。全部有效行的独立复算如下：

{table(cross.query('population=="all"'),{'a_mse':'a 贡献 MSE','b_mse':'b 贡献 MSE','twice_cross_mean':'交叉项 2E[ea·eb]','P_mse':'重建 P MSE'},{'a_mse':'{:.12f}','b_mse':'{:.12f}','twice_cross_mean':'{:+.12f}','P_mse':'{:.12f}'})}

C3 后 1h 的组合误差诊断：

{table(inter.query('space=="aligned" and horizon==@h1'),{'case':'候选','P_weighted_rmse':'0.4P 误差 RMSE','other_weighted_rmse':'其余分量合成误差 RMSE','two_mean_P_other_product':'P×其余分量交叉项','postprocessed_osi_rmse':'最终 OSI RMSE'},{'P_weighted_rmse':'{:.12f}','other_weighted_rmse':'{:.12f}','two_mean_P_other_product':'{:+.12f}','postprocessed_osi_rmse':'{:.12f}'})}

诊断同时保留官方分量重组与发布 OSI 的精度差异、上下界/置零后处理影响，详见 `metrics/error_interactions.csv`。a/b 是相对 P71 的数学分解，不是官方 N/R，也不代表实际客户群的恢复/故障概率。

将所有候选共同换回旧 D_B 后，1h 增量为：

{table(db,{'comparison':'比较','baseline_rmse':'参照 RMSE','candidate_rmse':'候选 RMSE','rmse_change_pct':'相对变化'},{'baseline_rmse':'{:.12f}','candidate_rmse':'{:.12f}','rmse_change_pct':'{:+.6f}%'})}

这是预定交互诊断，没有选择更好看的 D 版本替代主结果。

## 6. 训练、隔离与独立验证

- 共 {fits} 次正式拟合：60 次内层三折训练/一折早停、15 次外层四折重训；没有额外正式调参。运行累计约 {seconds/60:.2f} 分钟，实际过程见 `execution.json`。
- 原始 P 标签、a/b 监督和权重仅在允许县/支持行选取后构造；训练权重仅用该次训练支持集均值归一化。外折未来标签扰动不改变训练/选轮输入；未来停电数据不进入 P71。
- 分支使用原始预测上的贡献 RMSE 早停；直接 P 使用 LightGBM 原生 RMSE。LightGBM 内部监督为 float32，直接分支早停复算使用同一内部精度；最终 P/OSI 评分始终使用原冻结 float64 官方标签。分支自定义指标使用原始 float64 变换标签。
- p=0 的 53 个县不进入 a 拟合，但完整参与 P/OSI 评分；b 的零标签仍参与训练。没有县级例外、近零阈值、时间门控或混合系数。
- 新进程重载 15 个模型，独立复算标签/权重、内层最优轮指标、P 重建、C0–C3、逐行误差、所有指标和 bootstrap。预测/指标绝对容差 1e-12；大数值权重 ESS 诊断另用 1e-12 相对容差。
- 其他分量、C0、C1/C2 的 6/24/48h、主 24/48h 均通过逐行不变量检查。来源文件哈希和父实验历史 {len(snapshot['sha256'])} 个文件核对通过，未改写旧产物。

模型轮数：

{table(model_summary,{'outer_fold':'外折','kind':'目标','inner_best':'四次最优轮','requested':'重训轮数','actual':'实际树数'})}

`P_CV_COMPLETE` 在新进程验证通过后写入；`logs/independent_verification.json` 是真实保存模型的验证结果，不是 mock 验收。

## 7. 结论与限制

{recommendation}

E2 是“分解、贡献权重和两个树分支”的组合候选；相对 E1 的容量不同，不能把全部差异归因于单一公式。高初始停电县仍然少，结构不能补充不存在的灾害或抢修信息。

本次只完成 seed42，且该分折已经用于此前开发分析。bootstrap 条件于同一场事件和固定 OOF，未消除空间相关和反复选择带来的偏差，不能解释为隐藏测试集获益概率。官方按各 horizon RMSE 名次平均排名，1h 用于平局优先；本报告没有虚构四个原始 RMSE 的官方加权分数。

## 8. 文件索引

- `runs/{RUN.name}/`：15 个新模型、60 次探测的逐行最佳预测、receipt、完整分量/控制 OOF、指标与完成标记。
- `summary/primary_scores.csv`、`P_component_comparisons.csv`、`primary_bootstrap.csv`：主要数值。
- `summary/county_effects.csv`、`focus_counties.csv`、`focus_trajectories.csv`：完整县级损益与关注县轨迹。
- `summary/window_effects.csv`、`initial_state_groups.csv`、`target_hour_curve_data.csv`：固定切片和逐时曲线数据。
- `reference/approved_plan.md`、`experiment_config.json`：批准时的设计与实际执行范围。
- `final_integrity_check.json`：原历史文件未变；`completion.json`：本轮完成与未执行范围。

CSV 复读使用 `float_precision="round_trip"`。运行身份：`{marker['identity_hash']}`。
'''
    report=HERE/'Results_2026-09-18.md';report.write_text(body)
    summary_manifest={'identity_hash':marker['identity_hash'],'input_marker_sha256':sha(RUN/'P_CV_COMPLETE'),
        'outputs':outputs,'report_sha256':sha(report),'summary_script_sha256':sha(__file__)}
    (dest/'summary_manifest.json').write_text(json.dumps(summary_manifest,indent=2)+'\n')
    completion={'status':'COMPLETE','split_seed':42,'identity_hash':marker['identity_hash'],
        'independent_verification':'PASS','formal_fits':75,'saved_outer_models':15,
        'other_splits_trained':False,'final_models_trained':False,'test_submission_created':False,
        'historical_integrity':'PASS','recommendation':recommendation,'report':str(report),
        'finished_at_utc':datetime.now(timezone.utc).isoformat()}
    (HERE/'completion.json').write_text(json.dumps(completion,indent=2,ensure_ascii=False)+'\n')
    print(scores.to_string(index=False));print(pd_summary.to_string(index=False));print(recommendation)


if __name__=='__main__':main()
