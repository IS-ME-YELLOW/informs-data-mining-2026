"""Summarize the frozen first-round information experiments, without fitting."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/v18_pinfo_v1_split42_model42'
H1='osi_target_t01h';H6='osi_target_t06h'
FOCUS={'42053':'Forest PA','39117':'Morrow OH','18013':'Brown IN','54015':'Clay WV','54013':'Calhoun WV','39119':'Muskingum OH','39115':'Morgan OH'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def csv(path):return pd.read_csv(path,dtype={'slice_value':str,'fipsCode':str},float_precision='round_trip')


def md(frame,columns,formats=None):
    formats=formats or {};lines=['| '+' | '.join(columns.values())+' |','|'+'|'.join('---' for _ in columns)+'|']
    for _,r in frame.iterrows():
        row=[]
        for k in columns:
            v=r[k]
            if pd.isna(v):s='—'
            elif k in formats:s=formats[k].format(v)
            elif isinstance(v,(float,np.floating)) and v.is_integer():s=str(int(v))
            else:s=str(v)
            row.append(s)
        lines.append('| '+' | '.join(row)+' |')
    return '\n'.join(lines)


def main():
    marker=json.loads((RUN/'INFO_CV_COMPLETE').read_text());verification=json.loads((RUN/'logs/independent_verification.json').read_text())
    assert verification['status']=='PASS' and marker['identity_hash']==verification['identity_hash']
    audit_identity=verification['audit_identity']
    assert sha(HERE/'verify_information_exact.py')==audit_identity['adapter_sha256']
    assert sha(HERE/'verify_information.py')==audit_identity['core_verifier_sha256']
    assert sha(HERE/'independent_features.py')==audit_identity['original_feature_verifier_sha256']
    for n,h in marker['files'].items():assert sha(RUN/n)==h,n
    manifest=json.loads((RUN/'run_manifest.json').read_text())
    for n,h in manifest['identity']['source_sha256'].items():assert sha(n)==h,n
    for n,h in manifest['identity']['code_sha256'].items():assert sha(HERE/n)==h,n
    feat=HERE/'features/v1';fm=json.loads((feat/'feature_manifest.json').read_text())
    assert fm['identity_hash']==manifest['identity']['feature_package_identity']
    assert sha(feat/'feature_manifest.json')==manifest['identity']['feature_manifest_sha256']
    for n,h in fm['files'].items():assert sha(feat/n)==h,n
    pooled=csv(RUN/'metrics/pooled.csv');slices=csv(RUN/'metrics/slices.csv');boot=csv(RUN/'metrics/bootstrap.csv')
    primary=pooled.query("domain=='OSI' and mode=='v18_rule'")
    dest=HERE/'summary';dest.mkdir(exist_ok=True);outputs={}
    def save(name,frame):
        frame.to_csv(dest/name,index=False);outputs[name]=sha(dest/name)
    save('primary_comparisons.csv',primary)
    score=[];p_score=[]
    for h,g in primary.groupby('horizon',sort=True):
        f1=g.query("comparison=='F1_minus_F0'").iloc[0];f2=g.query("comparison=='F2_minus_F0'").iloc[0]
        score.append({'horizon':h,'n':int(f1.n),'F0_rmse':f1.baseline_rmse,'F1_rmse':f1.candidate_rmse,
            'F2_rmse':f2.candidate_rmse,'F1_change_pct':f1.rmse_change_pct,'F2_change_pct':f2.rmse_change_pct})
    score=pd.DataFrame(score);save('primary_scores.csv',score)
    for space,g in pooled.query("domain=='P' and horizon=='osi_target_t01h'").groupby('mode',sort=False):
        f1=g.query("comparison=='F1_minus_F0'").iloc[0];f2=g.query("comparison=='F2_minus_F0'").iloc[0]
        p_score.append({'space':space,'F0_rmse':f1.baseline_rmse,'F1_rmse':f1.candidate_rmse,
            'F2_rmse':f2.candidate_rmse,'F1_change_pct':f1.rmse_change_pct,'F2_change_pct':f2.rmse_change_pct})
    p_score=pd.DataFrame(p_score);save('P_component_scores.csv',p_score);save('bootstrap.csv',boot)
    county=slices.query("domain=='OSI' and mode=='v18_rule' and slice_type=='county'").copy()
    denominator=county.groupby(['horizon','comparison']).sse_reduction.transform('sum')
    county['fraction_of_net_sse_reduction']=county.sse_reduction/denominator.where(denominator!=0)
    save('county_effects.csv',county)
    focus=county.loc[county.slice_value.isin(FOCUS)].copy();focus['focus_name']=focus.slice_value.map(FOCUS);save('focus_counties.csv',focus)
    for kind,name in [('fold','fold_effects.csv'),('window','window_effects.csv'),('p_bin','initial_state_groups.csv'),('population','exclude_Morrow.csv'),('target_hour','target_hour_curve_data.csv')]:
        save(name,slices.loc[slices.slice_type==kind])
    counts=[]
    for (h,pair),g in county.groupby(['horizon','comparison']):
        if h not in (H1,H6):continue
        fold=slices.loc[(slices.domain=='OSI')&(slices['mode']=='v18_rule')&(slices.horizon==h)&(slices.comparison==pair)&(slices.slice_type=='fold')]
        counts.append({'horizon':h,'comparison':pair,'better_counties':int((g.sse_reduction>0).sum()),
            'worse_counties':int((g.sse_reduction<0).sum()),'unchanged_counties':int((g.sse_reduction==0).sum()),
            'better_folds':int((fold.sse_reduction>0).sum()),'net_sse_reduction':g.sse_reduction.sum()})
    counts=pd.DataFrame(counts);save('improvement_counts.csv',counts)
    decisions=[]
    for case in ('F1','F2'):
        pair=case+'_minus_F0';one=primary.loc[(primary.horizon==H1)&(primary.comparison==pair)].iloc[0]
        six=primary.loc[(primary.horizon==H6)&(primary.comparison==pair)].iloc[0]
        if one.rmse_delta<0 and six.rmse_delta<=1e-12:
            decision='达到预定晋级条件，建议在另外两套冻结分折确认'
        elif one.rmse_delta<0:
            decision='1h 改善、6h 恶化，属于混合结果；不自动晋级或改路由'
        else:
            decision='未改善主 1h，不建议自动进入确认或纳入主方案'
        decisions.append({'candidate':case,'1h_change_pct':one.rmse_change_pct,'6h_change_pct':six.rmse_change_pct,'decision':decision})
    decisions=pd.DataFrame(decisions);save('decisions.csv',decisions)
    # Inspect fitted feature use; gain is descriptive, not causal feature value.
    records=json.loads((RUN/'new_model_manifest.json').read_text());importance=[]
    for r in records:
        model=lgb.Booster(model_file=str(RUN/r['model_path']));names=model.feature_name();gain=model.feature_importance(importance_type='gain');splits=model.feature_importance(importance_type='split')
        case=r['spec']['candidate'];kind=r['spec']['kind'];outer=r['spec']['outer_fold'];total=gain.sum()
        for n,g,n_split in zip(names,gain,splits):
            group='DEM' if n.startswith('dem_') else 'neighbor' if n.startswith(('nbr8_','self_minus_nbr8_')) else 'original_163'
            importance.append({'candidate':case,'kind':kind,'outer_fold':outer,'feature':n,'group':group,
                'gain':g,'gain_fraction':g/total if total else 0.,'split_count':int(n_split)})
    importance=pd.DataFrame(importance);save('feature_importance.csv',importance)
    usage=importance.groupby(['candidate','kind','outer_fold','group'],as_index=False).agg(gain_fraction=('gain_fraction','sum'),split_count=('split_count','sum'))
    save('feature_group_usage.csv',usage)
    mean_usage=usage.groupby(['candidate','kind','group'],as_index=False).agg(mean_gain_fraction=('gain_fraction','mean'),minimum_gain_fraction=('gain_fraction','min'),maximum_gain_fraction=('gain_fraction','max'))
    save('feature_group_usage_summary.csv',mean_usage)
    rounds=csv(RUN/'round_selection.csv')
    round_summary=rounds.groupby(['candidate','outer_fold','kind'],as_index=False).agg(inner_best=('best_iteration',lambda v:','.join(str(int(x)) for x in v)),requested=('refit_rounds','first'),actual=('actual_trees','first'))
    save('model_rounds.csv',round_summary)
    nn=pd.read_csv(feat/'neighbors_k8.csv',dtype={'target_fips':str,'neighbor_fips':str},float_precision='round_trip')
    geo=nn.groupby('target_fips',as_index=False).agg(nearest_km=('distance_km','min'),eighth_km=('distance_km','max'))
    cv=pd.read_csv(RUN/'cv_assignments.csv',dtype={'fipsCode':str}).set_index('fipsCode').fold
    nn['target_fold']=nn.target_fips.map(cv);nn['neighbor_fold']=nn.neighbor_fips.map(cv)
    nn['neighbor_is_test']=nn.neighbor_fold.isna();nn['same_outer_fold']=nn.target_fold.notna()&nn.neighbor_fold.eq(nn.target_fold)
    save('neighbor_role_audit.csv',nn);save('neighbor_distances.csv',geo)
    # Include all valid hours for previously discussed counties, with no outcome-based row filtering.
    branch=pd.read_parquet(RUN/'branch_oof_predictions.parquet');control=pd.read_parquet(RUN/'control_predictions.parquet');audit=pd.read_parquet(RUN/'target_audit.parquet')
    take=control.fipsCode.isin(FOCUS)&control[f'scoreable_{H1}']
    trajectory=control.loc[take,['fipsCode','timestamp_et','hour_idx','fold',f'actual_{H1}',*[f'pred_{c}_v18_rule_{H1}' for c in ('F0','F1','F2')]]].copy()
    trajectory['official_P']=audit.loc[take,'official_P'];trajectory['p71']=audit.loc[take,'p71']
    for case in ('F0','F1','F2'):
        b=branch.loc[branch['case']==case].reset_index(drop=True)
        for field in ('pred_P','a_P_contribution','b_P_contribution'):trajectory[f'{case}_{field}']=b.loc[take,field].to_numpy()
    save('focus_trajectories.csv',trajectory)
    # Finish with a new check of every pre-existing research file and source snapshot.
    snapshot=json.loads((HERE/'reference/initial_workspace_snapshot.json').read_text())
    changed=[p for p,h in snapshot['sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    integrity={'status':'PASS' if not changed else 'FAIL','checked_historical_files':len(snapshot['sha256']),'changed_files':changed,
        'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    (HERE/'final_integrity_check.json').write_text(json.dumps(integrity,indent=2)+'\n');assert not changed
    execution=json.loads((RUN/'execution.json').read_text());fits=sum(a['fits_this_attempt'] for a in execution['attempts']);assert fits==100
    wall=sum(a['wall_seconds'] for a in execution['attempts'])
    interval=boot.query("population=='all_counties'").copy();interval['interval']=interval.apply(lambda r:f'[{r.ci_low:+.12f}, {r.ci_high:+.12f}]',axis=1)
    excluded=boot.query("domain=='OSI' and population=='exclude_Morrow'").copy();excluded['interval']=excluded.apply(lambda r:f'[{r.ci_low:+.12f}, {r.ci_high:+.12f}]',axis=1)
    windows=slices.query("domain=='OSI' and mode=='v18_rule' and horizon=='osi_target_t01h' and slice_type=='window' and comparison!='F2_minus_F1'")
    bins=slices.query("domain=='OSI' and mode=='v18_rule' and horizon=='osi_target_t01h' and slice_type=='p_bin' and comparison!='F2_minus_F1'")
    focal=focus.query("horizon=='osi_target_t01h' and comparison!='F2_minus_F1'")
    fmt={'F0_rmse':'{:.12f}','F1_rmse':'{:.12f}','F2_rmse':'{:.12f}','F1_change_pct':'{:+.5f}%','F2_change_pct':'{:+.5f}%'}
    verdict='；'.join(f'{r.candidate}：{r.decision}' for r in decisions.itertuples(index=False))+'。'
    forest_shares={}
    for case in ('F1','F2'):
        cg=county.loc[(county.horizon==H1)&(county.comparison==case+'_minus_F0')]
        forest_shares[case]=100*cg.loc[cg.slice_value=='42053','sse_reduction'].iloc[0]/cg.sse_reduction.sum()
    interpretation=(f"首轮更支持优先确认 F2：其主 1h/6h 的县 bootstrap 区间均位于零以下，四个预定时段的 1h 点估计均改善，"
        f"Forest 占净 SSE 改善的 {forest_shares['F2']:.2f}%。F1 的 1h/6h 区间仍跨零，"
        f"Forest 占净改善的 {forest_shares['F1']:.2f}%，后两个时段略有退化。F1 达到预定点估计晋级条件，但证据弱于 F2。"
        "这不是隐藏测试集显著获益的保证，也不支持此时直接叠加两组特征。")
    def top_table(case,good):
        g=county.loc[(county.horizon==H1)&(county.comparison==case+'_minus_F0')].sort_values('sse_reduction',ascending=not good).head(6)
        return md(g,{'slice_value':'FIPS','countyName':'县','baseline_rmse':'F0 RMSE','candidate_rmse':'候选 RMSE','sse_reduction':'SSE 改善量'},
            {'baseline_rmse':'{:.9f}','candidate_rmse':'{:.9f}','sse_reduction':'{:+.9f}'})
    body=f'''# P 信息增量实验：seed42 首轮结果（2026-09-18）

**F1（DEM）和 F2（近邻摘要）均完成 50 次正式拟合，共 100 次拟合、20 个外层模型；独立特征重建、模型重载与指标复算全部 PASS。** 未执行其他分折、组合特征、门控搜索、最终训练或测试提交。

{verdict}

{interpretation}

## 1. 固定对照与整体 RMSE

F0 为此前完整 E2（P 初始状态 a/b＋D_G）。F1 只给 a/b 加七项 DEM，170 列；F2 只加二十项固定八近邻摘要，183 列。标签、权重、参数、严格 CV、其他分量和主规则固定。没有将原 P-Huber 误用为 F0。

{md(score,{'horizon':'horizon','F0_rmse':'F0 RMSE','F1_rmse':'F1 DEM RMSE','F2_rmse':'F2 近邻 RMSE','F1_change_pct':'F1 相对 F0','F2_change_pct':'F2 相对 F0'},fmt)}

主 1/6h 使用 C3，24/48h 使用 C1；24/48h 主预测逐行精确不变。P1h 变化可经 C3 传播到 6h。全部 239 县及 34,177/32,982/28,680/22,944 行分别进入四个 horizon 的 pooled 评分。

## 2. P 分量及区间

raw 为分支截断/重建后、C3 前；aligned 为 C3 后。D_G、N/R、原始 P6/24/48 和直接 OSI 的预测逐行未变。

{md(p_score,{'space':'P1h 空间','F0_rmse':'F0 RMSE','F1_rmse':'F1 RMSE','F2_rmse':'F2 RMSE','F1_change_pct':'F1 相对 F0','F2_change_pct':'F2 相对 F0'},fmt)}

县轨迹配对 bootstrap（2,000 次，seed20260910）95% 百分位区间，ΔRMSE＝候选−F0：

{md(interval,{'comparison':'比较','domain':'指标','mode':'空间/规则','horizon':'horizon','rmse_delta':'ΔRMSE','interval':'95% 区间'},{'rmse_delta':'{:+.12f}'})}

两个特征组是两次预定比较。不能把其中最好的一次及其未经选择校正的区间当作独立外部确认。

## 3. 县级一致性与关注县

{md(counts.loc[counts.comparison!='F2_minus_F1'],{'comparison':'比较','horizon':'horizon','better_counties':'改善县','worse_counties':'恶化县','unchanged_counties':'不变县','better_folds':'改善折/5','net_sse_reduction':'净 SSE 改善量'},{'net_sse_reduction':'{:+.9f}'})}

排除 Morrow 的固定敏感性分析，仅用于诊断，主结果始终包含它：

{md(excluded,{'comparison':'比较','horizon':'horizon','rmse_change_pct':'相对变化','interval':'95% 区间'},{'rmse_change_pct':'{:+.5f}%'})}

关注县的主 1h：

{md(focal,{'comparison':'比较','focus_name':'县','baseline_rmse':'F0 RMSE','candidate_rmse':'候选 RMSE','rmse_change_pct':'相对变化'},
    {'baseline_rmse':'{:.9f}','candidate_rmse':'{:.9f}','rmse_change_pct':'{:+.4f}%'})}

F1 最大收益县：

{top_table('F1',True)}

F1 最大损失县：

{top_table('F1',False)}

F2 最大收益县：

{top_table('F2',True)}

F2 最大损失县：

{top_table('F2',False)}

全部县、SSE 变化及净变化占比保存在 `summary/county_effects.csv`。净改善为负或很小时，占比不应解释为获益概率；所有恶化案例均保留。

## 4. 固定时段和初始状态

{md(windows,{'comparison':'比较','slice_value':'目标小时','rmse_change_pct':'主 1h 相对变化','sse_reduction':'SSE 改善量'},{'rmse_change_pct':'{:+.4f}%','sse_reduction':'{:+.9f}'})}

{md(bins,{'comparison':'比较','slice_value':'P71 分组','counties':'县数','rmse_change_pct':'主 1h 相对变化'},{'rmse_change_pct':'{:+.4f}%'})}

完整逐目标小时数据和关注县轨迹见 `target_hour_curve_data.csv`、`focus_trajectories.csv`。没有根据上述时段或县结果新增门控、改变邻居数或挑选特征。

## 5. 模型是否使用了新特征

下面是各分支五个外层模型中，新增特征组占训练分裂 gain 的比例：

{md(mean_usage.loc[mean_usage.group!='original_163'],{'candidate':'候选','kind':'分支','group':'特征组','mean_gain_fraction':'平均 gain 占比','minimum_gain_fraction':'最小','maximum_gain_fraction':'最大'},
    {'mean_gain_fraction':'{:.2%}','minimum_gain_fraction':'{:.2%}','maximum_gain_fraction':'{:.2%}'})}

这些指标只描述模型使用情况，不是特征的因果价值，也不能替代外折 OSI 成绩。全量逐模型分裂次数/gain 见 `feature_importance.csv`。特征组增大也改变固定 0.8 列采样时的候选集合，本轮未额外调参。

`metrics/branch_diagnostics.csv`、`branch_loss_decomposition.csv` 和 `error_interactions.csv` 分别保存分支贡献/截断、a/b 交叉项和 P 与其余分量的组合误差；不能将单项 RMSE 相加解释整体损失。

## 6. 近邻输入与无泄漏边界

八近邻仅按 302 个官方参赛县的坐标和 Haversine 距离确定，不含自身、接壤附加边、距离权重或标签信息。平均分母固定为八。302 县、2,416 条边、347,904 条县/起点/邻居依赖记录均已保存。

近邻可以属于外折或官方测试县，只贡献截止小时 71 的历史和赛事允许的天气；不读取任何县预测期的真实停电、标签、残差或 OOF 预测。这样的共享已公开输入符合本实验的固定节点场景，不等同于缺少历史输入的新地区泛化。

独立核验从原始数据重建九个白名单字段：测试 CSV 没有 osi，观测期 OSI 从已知 P/N/D/R 重组并取四位小数，再求末六小时斜率，与原缓存一致。`next_6h` 保持 `[t,t+6]` 最多七个点；未同时修复降水时间对齐。

未来停电列改为 NaN/999 或删除未来停电历史行，合法输入不变；改变官方测试县的已知 P71 或允许天气，邻居均值按预期变化；常数场、自身排除、行/节点打乱、并列距离、缺失和重复键均有预检。学习监督及早停仍严格排除当前外折未来标签。

DEM 使用批准时冻结的七列成品表，三百零二县覆盖和数值检查通过，未下载或重算原始栅格。这里不把信息增量效果称为 DEM 提取算法的逐像元验收。

## 7. 训练和独立验证

共 {fits} 次正式拟合（80 次三折探测、20 次四折重训），模型 seed42、CPU 单线程，进程累计约 {wall/60:.2f} 分钟。每条分支独立按既定贡献指标早停，再对四次最优轮数均值向下取整；没有按外折成绩选择轮数。

独立进程重新计算距离和聚合，**用独立重建的增强特征重载预测全部 20 个真实模型**，并复算标签/权重、最佳轮指标、P 重建、C0–C3、全部指标和 bootstrap。预测/指标绝对容差 1e-12；特征/距离独立计算容差 1e-10；大数值权重 ESS 使用 1e-12 相对容差。

{md(round_summary,{'candidate':'候选','outer_fold':'外折','kind':'分支','inner_best':'四次最优轮','requested':'重训轮数','actual':'实际树数'})}

历史 {len(snapshot['sha256'])} 个文件及所有来源快照哈希核对通过。`INFO_CV_COMPLETE` 在真实独立验证及历史完整性通过后写入，没有覆盖原 E2/D_G/严格基线产物。预检阶段处理了测试 osi 字段差异并收窄构造接口，保留了未训练的开发特征包记录；正式训练只使用最终冻结 v1 包，输入定义未改变。

初次独立审计还发现了均值求和顺序问题：连续数组的 `np.mean` 与训练特征的固定排名逐项求和存在浮点尾差，少数树预测因此跨过分裂阈值。修正只作用于独立复算的加法顺序；模型、输入特征、OOF、参数和误差容差均未改变，也没有新增拟合。`verify_information_exact.py` 使用独立县级循环，并要求重建的 170/183 列输入逐位精确一致，再调用完整核心验收。原失败记录、修正后日志、独立审计身份及全部原模型/特征/预测未变的哈希证据保存于 run 内；训练身份和审计修订身份分别保留，不冒充一次从未失败的验证。

当前复验入口是 `verify_information_exact.py`；原 `verify_information.py` 保留为冻结的核心审计版本，由新入口复用。完成记录的审计身份为 `{verification['audit_identity_hash']}`。

## 8. 下一步

{verdict}

{interpretation}

只按批准规则提出建议，本轮没有执行后两套分折或调整主方案。候选若进入确认，沿用完全相同特征定义并完整运行两套固定分折；若首轮不支持，则保留负结果，避免围绕当前 OOF 连续搜索邻居数、权重、县或时间阈值。

seed42 已用于此前开发，县 bootstrap 条件于同一事件和固定 OOF，未消除空间相关及模型选择偏差。官方按各 horizon 的 RMSE 名次平均排名；本报告未把原始 RMSE 均值当作官方分数。

## 9. 文件索引

- `features/v1/`：170/183 列训练和测试输入、九列白名单、2,416 条边和逐行近邻依赖；不是测试预测。
- `runs/{RUN.name}/`：20 个模型、80 次内层最佳预测、receipt、完整 F0/F1/F2 分量/控制 OOF、指标和完成标记。
- `summary/primary_scores.csv`、`P_component_scores.csv`、`bootstrap.csv`、`decisions.csv`：主要数值与预定决策。
- `summary/county_effects.csv`、`focus_trajectories.csv`、`window_effects.csv`、`initial_state_groups.csv`：损益定位。
- `summary/neighbor_role_audit.csv`、`feature_group_usage.csv`：合法输入关系及模型使用情况。
- `reference/approved_plan.md`、`experiment_config.json`、`preflight/split42/checks.json`：设计和真实输入预检。
- `final_integrity_check.json`、`completion.json`：本轮完成状态。

CSV 使用 `float_precision="round_trip"` 复读。运行身份：`{marker['identity_hash']}`。
'''
    report=HERE/'Results_2026-09-18.md';report.write_text(body)
    summary={'status':'PASS','identity_hash':marker['identity_hash'],'audit_identity_hash':verification['audit_identity_hash'],'input_marker_sha256':sha(RUN/'INFO_CV_COMPLETE'),
        'outputs':outputs,'report_sha256':sha(report),'summary_script_sha256':sha(__file__)}
    (dest/'summary_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    completion={'status':'COMPLETE','split_seed':42,'identity_hash':marker['identity_hash'],'formal_fits':100,'new_outer_models':20,
        'independent_verification':'PASS','audit_identity_hash':verification['audit_identity_hash'],'extra_fits_for_audit_correction':0,'historical_integrity':'PASS','decisions':decisions.to_dict(orient='records'),
        'other_splits_trained':False,'combined_features_trained':False,'final_training':False,'test_submission':False,
        'finished_at_utc':datetime.now(timezone.utc).isoformat()}
    (HERE/'completion.json').write_text(json.dumps(completion,indent=2,ensure_ascii=False)+'\n')
    print(score.to_string(index=False));print(p_score.to_string(index=False));print(decisions.to_string(index=False))


if __name__=='__main__':main()
