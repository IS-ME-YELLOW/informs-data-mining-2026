"""Summarize F2 only across three frozen CV splits, with no model selection/refit."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;EXP=HERE.parent
SEEDS=(42,20260917,20260918);PAIR='F2_minus_F0';H1='osi_target_t01h';H6='osi_target_t06h'
FOCUS={'42053':'Forest PA','39117':'Morrow OH','18013':'Brown IN','54015':'Clay WV','54013':'Calhoun WV','39119':'Muskingum OH','39115':'Morgan OH'}


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text())
def csv(p):return pd.read_csv(p,dtype={'slice_value':str,'fipsCode':str},float_precision='round_trip')


def md(frame,cols,fmt=None):
    fmt=fmt or {};lines=['| '+' | '.join(cols.values())+' |','|'+'|'.join('---' for _ in cols)+'|']
    for _,r in frame.iterrows():
        row=[]
        for k in cols:
            v=r[k]
            if pd.isna(v):s='—'
            elif k in fmt:s=fmt[k].format(v)
            elif isinstance(v,(float,np.floating)) and v.is_integer():s=str(int(v))
            else:s=str(v)
            row.append(s)
        lines.append('| '+' | '.join(row)+' |')
    return '\n'.join(lines)


def load(seed):
    run=EXP/'runs'/f'v18_pinfo_v1_split{seed}_model42'
    marker=read(run/'INFO_CV_COMPLETE');m=read(run/'run_manifest.json');v=read(run/'logs/independent_verification.json')
    assert marker['identity_hash']==m['identity_hash']==v['identity_hash']
    assert m['identity']['split_seed']==seed and v['split_seed']==seed and v['status']=='PASS'
    assert v['feature_verification']['augmented_features_bitwise_exact']
    for n,h in marker['files'].items():assert sha(run/n)==h,(seed,n)
    for n,h in m['identity']['source_sha256'].items():assert sha(n)==h,n
    for n,h in m['identity']['code_sha256'].items():assert sha(Path(n) if Path(n).is_absolute() else EXP/n)==h,n
    config=EXP/'experiment_config.json' if seed==42 else HERE/'experiment_config.json'
    assert sha(config)==m['identity']['config_sha256']
    audit_root=EXP if seed==42 else HERE
    assert sha(audit_root/'verify_information_exact.py')==v['audit_identity']['adapter_sha256']
    assert sha(audit_root/'verify_information.py')==v['audit_identity']['core_verifier_sha256']
    assert sha(EXP/'independent_features.py')==v['audit_identity']['original_feature_verifier_sha256']
    ref=read(EXP/'runs/v18_pinfo_v1_split42_model42/run_manifest.json')['identity']
    for field in ('feature_package_identity','feature_manifest_sha256','params','settings','p_sha256','P_label_sha256'):
        assert m['identity'][field]==ref[field],(seed,field)
    assert m['identity']['feature_names']['F2']==ref['feature_names']['F2']
    records=read(run/'new_model_manifest.json');f2=[r for r in records if r['spec']['candidate']=='F2']
    assert len(f2)==10 and sum(r['fit']['formal_fit_count'] for r in f2)==50
    if seed!=42:
        assert len(records)==10 and m['identity']['trained_cases']==['F2']
        assert not (run/'models/F1').exists()
    tables={}
    for name in ('pooled','slices','bootstrap','branch_diagnostics','branch_loss_decomposition','error_interactions'):
        frame=csv(run/f'metrics/{name}.csv')
        if 'comparison' in frame:frame=frame.loc[frame.comparison==PAIR].copy()
        elif 'case' in frame:frame=frame.loc[frame['case'].isin(['F0','F2'])].copy()
        frame.insert(0,'split_seed',seed);tables[name]=frame
    r=csv(run/'round_selection.csv');r=r.loc[r.candidate=='F2'].copy();r.insert(0,'split_seed',seed);tables['rounds']=r
    oof=pd.read_parquet(run/'control_predictions.parquet')
    for h in ('osi_target_t24h','osi_target_t48h'):
        np.testing.assert_array_equal(oof[f'pred_F0_v18_rule_{h}'],oof[f'pred_F2_v18_rule_{h}'])
    assert len(oof)==34416 and oof.fipsCode.nunique()==239
    execution=read(run/'execution.json');assert execution['attempts'][-1]['status']=='completed'
    source={'split_seed':seed,'run':str(run),'identity_hash':m['identity_hash'],'audit_identity_hash':v['audit_identity_hash'],
        'complete_marker_sha256':sha(run/'INFO_CV_COMPLETE'),'independent_verification_sha256':sha(run/'logs/independent_verification.json'),
        'reference_E2_identity_hash':m['identity']['reference_E2_identity_hash'],'feature_package_identity':m['identity']['feature_package_identity'],
        'F2_fits':50,'F2_outer_models':10,'whole_source_run_fits':sum(a['fits_this_attempt'] for a in execution['attempts']),
        'whole_source_run_seconds':sum(a['wall_seconds'] for a in execution['attempts']),'verification':'PASS'}
    return tables,source


def main():
    collection={};sources=[]
    for seed in SEEDS:
        tables,source=load(seed);sources.append(source)
        for name,table in tables.items():collection.setdefault(name,[]).append(table)
    tables={n:pd.concat(fs,ignore_index=True) for n,fs in collection.items()}
    dest=HERE/'summary';dest.mkdir(exist_ok=True);outputs={}
    def save(n,table):
        table.to_csv(dest/n,index=False);outputs[n]=sha(dest/n)
    for n,table in tables.items():save(n+'_all_splits.csv',table)
    primary=tables['pooled'].query("domain=='OSI' and mode=='v18_rule'").copy();save('primary_scores.csv',primary)
    ps=tables['pooled'].query("domain=='P' and horizon=='osi_target_t01h'").copy();save('P_component_scores.csv',ps)
    boot=tables['bootstrap'];slices=tables['slices']
    county=slices.query("domain=='OSI' and mode=='v18_rule' and slice_type=='county'").copy()
    denominator=county.groupby(['split_seed','horizon']).sse_reduction.transform('sum')
    county['fraction_of_net_sse_reduction']=county.sse_reduction/denominator.where(denominator!=0)
    save('county_effects.csv',county)
    focus=county.loc[county.slice_value.isin(FOCUS)].copy();focus['focus_name']=focus.slice_value.map(FOCUS);save('focus_counties.csv',focus)
    for typ,name in [('fold','fold_effects.csv'),('window','window_effects.csv'),('p_bin','initial_state_groups.csv'),('population','exclude_Morrow.csv'),('target_hour','target_hour_curve_data.csv')]:
        save(name,slices.loc[slices.slice_type==typ])
    counts=[]
    for (seed,h),g in county.groupby(['split_seed','horizon']):
        if h not in (H1,H6):continue
        f=slices.loc[(slices.split_seed==seed)&(slices.domain=='OSI')&(slices['mode']=='v18_rule')&(slices.horizon==h)&(slices.slice_type=='fold')]
        net=g.sse_reduction.sum()
        counts.append({'split_seed':seed,'horizon':h,'better_counties':int((g.sse_reduction>0).sum()),'worse_counties':int((g.sse_reduction<0).sum()),
            'unchanged_counties':int((g.sse_reduction==0).sum()),'better_folds':int((f.sse_reduction>0).sum()),'net_sse_reduction':net,
            'Forest_net_share_pct':100*g.loc[g.slice_value=='42053','sse_reduction'].iloc[0]/net if net else np.nan,
            'Morrow_net_share_pct':100*g.loc[g.slice_value=='39117','sse_reduction'].iloc[0]/net if net else np.nan})
    counts=pd.DataFrame(counts);save('improvement_counts.csv',counts)
    consistency=[]
    for (h,fips),g in county.groupby(['horizon','slice_value']):
        assert set(g.split_seed)==set(SEEDS)
        consistency.append({'horizon':h,'fipsCode':fips,'countyName':g.countyName.iloc[0],
            'improved_splits':int((g.sse_reduction>0).sum()),'worsened_splits':int((g.sse_reduction<0).sum()),
            'minimum_rmse_change_pct':g.rmse_change_pct.min(),'maximum_rmse_change_pct':g.rmse_change_pct.max()})
    consistency=pd.DataFrame(consistency);save('county_consistency.csv',consistency)
    save('source_runs.csv',pd.DataFrame(sources))
    windows=slices.query("domain=='OSI' and mode=='v18_rule' and horizon=='osi_target_t01h' and slice_type=='window'")
    bins=slices.query("domain=='OSI' and mode=='v18_rule' and horizon=='osi_target_t01h' and slice_type=='p_bin'")
    interval=boot.query("population=='all_counties'").copy();interval['interval']=interval.apply(lambda r:f'[{r.ci_low:+.12f}, {r.ci_high:+.12f}]',axis=1)
    excluded=boot.query("domain=='OSI' and population=='exclude_Morrow'").copy();excluded['interval']=excluded.apply(lambda r:f'[{r.ci_low:+.12f}, {r.ci_high:+.12f}]',axis=1)
    short=primary.loc[primary.horizon.isin([H1,H6])]
    all_gain=bool((short.rmse_delta<0).all())
    if all_gain:
        recommendation='F2 在三套固定分折的主 1h/6h 均优于各自完整 E2，支持将“E2＋固定近邻摘要”作为当前优先输入方案保留。后续实验可用它作对照，但本轮没有自动替换旧主方案或训练最终提交模型。'
    elif (primary.loc[primary.horizon==H1,'rmse_delta']<0).all():
        recommendation='F2 的三套 1h 方向一致改善，但 6h 存在退化，属于短时指标间的混合结果。保留逐套证据，先评估退化幅度，不自动改路由或采用。'
    else:
        recommendation='F2 未在三套的主 1h 保持一致改善，首轮收益尚不能视为稳定增量。建议暂保留原 E2，并说明分折、县和时段差异，不挑最佳 CV 或新增门控补救。'
    decisions=[]
    for seed in SEEDS:
        rows=short.loc[short.split_seed==seed]
        one=rows.loc[rows.horizon==H1].iloc[0];six=rows.loc[rows.horizon==H6].iloc[0]
        decisions.append({'split_seed':seed,'1h_change_pct':one.rmse_change_pct,'6h_change_pct':six.rmse_change_pct,
            'both_short_horizons_improved':bool(one.rmse_delta<0 and six.rmse_delta<0)})
    save('split_decisions.csv',pd.DataFrame(decisions))
    snapshot=read(HERE/'reference/initial_workspace_snapshot.json')
    changed=[p for p,h in snapshot['sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    assert not changed,changed
    integrity={'status':'PASS','checked_historical_files':len(snapshot['sha256']),'changed_files':[],
        'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    (HERE/'final_integrity_check.json').write_text(json.dumps(integrity,indent=2)+'\n')
    fmt={'baseline_rmse':'{:.12f}','candidate_rmse':'{:.12f}','rmse_change_pct':'{:+.5f}%'}
    focal=focus.loc[focus.horizon==H1]
    stable=consistency.loc[consistency.horizon==H1]
    new_sources=[s for s in sources if s['split_seed']!=42]
    assert sum(s['F2_fits'] for s in new_sources)==100
    one_gain=-short.loc[short.horizon==H1,'rmse_change_pct'];six_gain=-short.loc[short.horizon==H6,'rmse_change_pct']
    overview=(f'相对完整 E2，F2 的三套 1h 改善 {one_gain.min():.2f}%–{one_gain.max():.2f}%，'
        f'6h 改善 {six_gain.min():.2f}%–{six_gain.max():.2f}%。四个预定时段的 1h 点估计在三套均改善，'
        '支持存在超出最初几个小时的空间上下文增量；但增益幅度较小，仍有县和初始状态组退化。'
        if all_gain and (windows.rmse_change_pct<0).all() else '应结合逐套分折、区间和时段结果判断，不以最好的一套替代整体证据。')
    def top(seed,ascending):
        g=county.loc[(county.split_seed==seed)&(county.horizon==H1)].sort_values('sse_reduction',ascending=ascending).head(6)
        return md(g,{'slice_value':'FIPS','countyName':'县','baseline_rmse':'F0 RMSE','candidate_rmse':'F2 RMSE','sse_reduction':'SSE 改善量'},
            {'baseline_rmse':'{:.9f}','candidate_rmse':'{:.9f}','sse_reduction':'{:+.9f}'})
    report=f'''# F2 近邻信息：两套确认与三分折汇总（2026-09-19）

**seed20260917、seed20260918 的 F2 均完整完成，并通过独立特征重建、模型重载和指标复算。** 本轮新增 100 次拟合、20 个外层模型。单计 F2，三套累计 150 次拟合、30 个模型；首轮另有 50 次 F1 拟合，本轮没有继续 F1。

{recommendation}

{overview}

## 1. 固定实验与整体成绩

F0＝同分折完整 E2（P1h 的 a/b 初始状态结构＋D_G）；F2＝只在 P1h 的 a/b 中增加固定二十项近邻历史/阵风摘要。每套使用自己的 F0，模型 seed42、权重、早停、C3 和主规则固定。输入 183 列、302 县几何网络、八邻居及等权聚合与首轮逐位一致。

{md(short,{'split_seed':'分折','horizon':'horizon','baseline_rmse':'F0 RMSE','candidate_rmse':'F2 RMSE','rmse_change_pct':'相对变化'},fmt)}

24/48h 的主预测三套均逐行精确不变，完整值见 `summary/primary_scores.csv`。每套全体 239 县都评分，1h/6h 分别 34,177/32,982 行。

## 2. P 分量与不确定性

raw 指截断/重建后、C3 前的 P1h；aligned 指 C3 后 P1h。

{md(ps,{'split_seed':'分折','mode':'P 空间','baseline_rmse':'F0 RMSE','candidate_rmse':'F2 RMSE','rmse_change_pct':'相对变化'},fmt)}

县轨迹配对 bootstrap（2,000 次，seed20260910）95% 百分位区间，ΔRMSE＝F2−F0：

{md(interval,{'split_seed':'分折','domain':'指标','mode':'空间/规则','horizon':'horizon','rmse_delta':'ΔRMSE','interval':'95% 区间'},{'rmse_delta':'{:+.12f}'})}

区间按每套固定预测独立重抽样县轨迹计算，不将三套同一批县拼成更大的独立样本。P 分量和最终 OSI 分开判断。

seed42 和 seed20260917 的主 1h/6h 及 P1h 两种空间区间都位于零以下；seed20260918 的对应区间均跨零。因此可称“点估计方向复现”，不能称“三套均显著改善”。

## 3. 县级损益与一致性

{md(counts,{'split_seed':'分折','horizon':'horizon','better_counties':'改善县','worse_counties':'恶化县','unchanged_counties':'不变县','better_folds':'改善折/5','net_sse_reduction':'净 SSE 改善','Forest_net_share_pct':'Forest 占净改善','Morrow_net_share_pct':'Morrow 占净改善'},
    {'net_sse_reduction':'{:+.9f}','Forest_net_share_pct':'{:.2f}%','Morrow_net_share_pct':'{:.2f}%'})}

主 1h 三套均改善的县有 **{int((stable.improved_splits==3).sum())}** 个，三套均恶化的县有 **{int((stable.worsened_splits==3).sum())}** 个；完整方向一致性表见 `county_consistency.csv`。净 SSE 占比可能因其他县抵消而超过 100%，不能解释成获益概率。

Forest 在 seed20260918 的主 1h 略有退化，但该套总体仍改善；这与“收益只来自修复 Forest”不符。Morrow 三套均改善，排除它后也仍有净收益，不过第三套的区间仍跨零。

排除 Morrow 的预定诊断：

{md(excluded,{'split_seed':'分折','horizon':'horizon','rmse_change_pct':'相对变化','interval':'95% 区间'},{'rmse_change_pct':'{:+.5f}%'})}

此前关注县的 1h 结果：

{md(focal,{'split_seed':'分折','focus_name':'县','baseline_rmse':'F0 RMSE','candidate_rmse':'F2 RMSE','rmse_change_pct':'相对变化'},
    {'baseline_rmse':'{:.9f}','candidate_rmse':'{:.9f}','rmse_change_pct':'{:+.4f}%'})}

seed20260917 的最大收益县：

{top(20260917,False)}

seed20260917 的最大损失县：

{top(20260917,True)}

seed20260918 的最大收益县：

{top(20260918,False)}

seed20260918 的最大损失县：

{top(20260918,True)}

以上县名单仅用于解释，未加入任何模型或推理开关。

## 4. 时段和初始状态

固定目标小时窗口的主 1h：

{md(windows,{'split_seed':'分折','slice_value':'目标小时','rmse_change_pct':'相对变化','sse_reduction':'SSE 改善量'},{'rmse_change_pct':'{:+.4f}%','sse_reduction':'{:+.9f}'})}

固定 P71 分组：

{md(bins,{'split_seed':'分折','slice_value':'初始状态','counties':'县数','rmse_change_pct':'相对变化'},{'rmse_change_pct':'{:+.4f}%'})}

完整逐小时数据保存为 `target_hour_curve_data.csv`。本轮没有用切片结果调整窗口、k、距离权重、列或门控。

## 5. 严格隔离、路径与验证

- 原 163 列、P71 和标签保持不变，183 列来自同一冻结特征包：`{sources[0]['feature_package_identity']}`。
- 近邻仅用各县已公开的小时 0–71 历史和比赛允许天气；不读取预测期真实停电、残差或 OOF 预测。外折/测试县的合法历史输入可参与聚合，未来标签不参与梯度、早停或特征。
- 两套均先通过独立预检，再按外层四折内三折训练/一折早停选轮，最终用完整四折重训。每套 40 次探测、10 次外层重训。
- 新进程按邻居排名逐项求和，确认增强特征逐位精确一致，再重载十个模型复算所有预测、来源 ID、分支损失、C0–C3、主指标、县/折/时段指标和 bootstrap。未放宽 1e-12 的预测/指标容差。
- 训练/特征数学复用首轮冻结文件；共用确认入口只改变分折、路径和 F2 单候选范围。新运行身份从开始就包含修正后的审计入口，不修改原 seed42 代码。
- 首轮 F1 矩阵只随整个来源包被检查，没有 F1 拟合上下文、模型或确认成绩。不得把“源包含 F1”写成“本轮确认了 F1”。
- 两个新 run 的 `INFO_CV_COMPLETE` 均在真实独立验收和历史完整性检查后生成。开工前 {len(snapshot['sha256'])} 个历史文件逐项核对未变。

{md(pd.DataFrame(sources),{'split_seed':'分折','F2_fits':'F2 拟合数','F2_outer_models':'F2 外层模型','whole_source_run_fits':'来源 run 总拟合数','whole_source_run_seconds':'来源 run 用时/秒','verification':'独立验证'},{'whole_source_run_seconds':'{:.2f}'})}

seed42 的来源 run 同时包含 F1，所以其总拟合数为 100；F2 本身为 50。本轮两套并行，各模型保持单线程，进程时长相加不代表用户等待时长。

## 6. 决策与边界

{recommendation}

F1 的证据仍仅限此前 seed42；本报告不确认 DEM、不合并 DEM＋近邻，也不将新成绩归因于图神经网络。本实验检验的是给固定树结构加入已知空间上下文的整组增量。

三套是同一事件、同一批县的重复开发分折，且已用于此前分析；bootstrap 未消除空间相关及模型选择偏差。官方按四个 horizon 的 RMSE 名次平均排序，平局优先 1h；不把原始 RMSE 均值当作官方成绩。没有平均或拼合三套 OOF 预测来构造额外候选。

## 7. 文件索引

- `summary/primary_scores.csv`、`P_component_scores.csv`、`bootstrap_all_splits.csv`：F2 三套核心结果。
- `summary/county_effects.csv`、`county_consistency.csv`、`focus_counties.csv`：全部县和方向一致性。
- `summary/window_effects.csv`、`initial_state_groups.csv`、`target_hour_curve_data.csv`：固定切片与逐时数据。
- `summary/source_runs.csv`、`summary_manifest.json`：同分折来源、运行/审计身份与汇总哈希。
- `split20260917/Results_2026-09-19.md`、`split20260918/Results_2026-09-19.md`：单套简报。
- 父目录 `runs/v18_pinfo_v1_split{{seed}}_model42/`：F2 模型、完整 F0/F2 OOF、receipt 和独立验证。

CSV 复读使用 `float_precision="round_trip"`。本轮未训练全县最终模型或生成测试提交。
'''
    report_path=HERE/'Results_2026-09-19.md';report_path.write_text(report)
    reports=[report_path]
    for seed in (20260917,20260918):
        p=HERE/f'split{seed}/Results_2026-09-19.md'
        text=f'''# F2 近邻信息确认：seed{seed}（2026-09-19）

50 次拟合、10 个 F2 外层模型及独立验证完成；同分折完整 E2 为 F0。

{md(primary.loc[primary.split_seed==seed],{'horizon':'horizon','baseline_rmse':'F0 RMSE','candidate_rmse':'F2 RMSE','rmse_change_pct':'相对变化'},fmt)}

P1h：

{md(ps.loc[ps.split_seed==seed],{'mode':'P 空间','baseline_rmse':'F0 RMSE','candidate_rmse':'F2 RMSE','rmse_change_pct':'相对变化'},fmt)}

完整结果、区间与结论见 [三分折汇总]({report_path})。未训练 F1、组合版本或测试提交模型。
'''
        p.write_text(text);reports.append(p)
    manifest={'status':'PASS','source_runs':sources,'outputs':outputs,'reports':{str(p):sha(p) for p in reports},
        'summary_script_sha256':sha(__file__),'same_feature_package_and_training_parameters':True,
        'all_long_horizon_predictions_unchanged':True}
    (dest/'summary_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    completion={'status':'COMPLETE','trained_cases':['F2'],'new_split_seeds':[20260917,20260918],'summary_split_seeds':list(SEEDS),
        'new_formal_fits':100,'new_outer_models':20,'F2_three_split_fits':150,'F2_three_split_outer_models':30,
        'independent_verification':'PASS','historical_integrity':'PASS','recommendation':recommendation,
        'F1_confirmation_performed':False,'combined_model_trained':False,'final_training':False,'test_submission':False,
        'finished_at_utc':datetime.now(timezone.utc).isoformat()}
    (HERE/'completion.json').write_text(json.dumps(completion,indent=2,ensure_ascii=False)+'\n')
    print(short[['split_seed','horizon','baseline_rmse','candidate_rmse','rmse_change_pct']].to_string(index=False))
    print(ps[['split_seed','mode','rmse_change_pct']].to_string(index=False));print(recommendation)


if __name__=='__main__':main()
