"""Summarize three P24 CVs without averaging predictions or hiding county tradeoffs."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;BASE=PARENT.parent
SEEDS=(42,20260917,20260918)
HS=('osi_target_t01h','osi_target_t06h','osi_target_t24h','osi_target_t48h')
HOURS=(1,6,24,48)


def read_json(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def write_json(p,v):
    p=Path(p).resolve();assert p.is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_frame(p,d):
    p=Path(p).resolve();assert p.is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False)


def load(seed,name):
    root=PARENT if seed==42 else HERE/f'split{seed}'
    df=pd.read_csv(root/f'summary/{name}.csv',float_precision='round_trip',dtype={'fipsCode':str} if 'county' in name else {})
    if 'split_seed' not in df:df.insert(0,'split_seed',seed)
    else:assert df.split_seed.eq(seed).all()
    if 'case' in df:df=df[df['case'].isin(['B0','NB_P24'])].copy()
    if name=='branch_diagnostics':df=df[df.horizon==HS[2]].copy()
    return df


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    assert read_json(PARENT/'completion.json')['status']=='COMPLETE'
    for seed in SEEDS[1:]:
        root=HERE/f'split{seed}';run=root/f'runs/v18_pneighbor_v1_split{seed}_model42'
        assert read_json(run/'logs/independent_verification.json')['status']=='PASS'
        assert read_json(root/'analysis_verification.json')['status']=='PASS'
    names=['primary_scores','P_component_metrics','P71_bins','county_metrics','fold_metrics','window_metrics',
           'bootstrap_intervals','branch_diagnostics','OSI_change_decomposition']
    data={name:pd.concat([load(s,name) for s in SEEDS],ignore_index=True) for name in names}
    for name,d in data.items():write_frame(HERE/f'summary/{name}.csv',d)
    primary=data['primary_scores'];cand=primary[primary['case']=='NB_P24']
    pm=data['P_component_metrics'];county=data['county_metrics'];fold=data['fold_metrics'];win=data['window_metrics'];ci=data['bootstrap_intervals']
    target=cand[cand.horizon==HS[2]];short=cand[cand.horizon.isin(HS[:2])]
    assert cand.loc[cand.horizon==HS[3],'rmse_delta'].eq(0).all()
    improves=int((target.rmse_delta<0).sum());short_improves=int((short.rmse_delta<0).sum())
    if improves==3 and short_improves==6:verdict='24h与短时方向均复现，支持将P24直接树＋邻县信息纳入后续优先组合'
    elif improves==3:verdict='24h方向复现，但短时有取舍，暂不直接替换当前组合'
    else:verdict='24h收益未能在三套CV稳定复现，暂不纳入当前组合'
    clay=county[(county['case']=='NB_P24')&(county.fipsCode=='54015')].copy()
    write_frame(HERE/'summary/Clay_54015_metrics.csv',clay)
    focus=pd.read_csv(BASE/'n_four_horizon_diagnosis/summary/case_selection.csv',dtype={'fipsCode':str}).fipsCode.unique()
    write_frame(HERE/'summary/frozen_focus_counties.csv',county[(county['case']=='NB_P24')&county.fipsCode.isin(focus)])
    consist=[]
    for (h,f),g in county[county['case']=='NB_P24'].groupby(['horizon','fipsCode']):
        assert len(g)==3
        consist.append(dict(horizon=h,fipsCode=f,countyName=g.countyName.iloc[0],stateAbbr=g.stateAbbr.iloc[0],
            improves_splits=int((g.sse_reduction>0).sum()),worsens_splits=int((g.sse_reduction<0).sum()),
            **{f'split{s}_sse_reduction':float(g.loc[g.split_seed==s,'sse_reduction'].iloc[0]) for s in SEEDS}))
    consist=pd.DataFrame(consist);write_frame(HERE/'summary/county_consistency.csv',consist)
    decision=dict(recommendation=verdict,target_improves_splits=improves,short_improved_comparisons=short_improves,
        long48_unchanged=True,current_baseline_changed=False,feature_package_reused_readonly=True,
        new_fits=50,new_outer_models=10,P24_three_CV_fits=75,P24_three_CV_outer_models=15,
        underlying_events=1,independent_test_evaluations=0)
    write_json(HERE/'summary/decision.json',decision)
    artifacts=[];traces=[]
    for seed in SEEDS:
        root=PARENT if seed==42 else HERE/f'split{seed}'
        run=root/f'runs/v18_pneighbor_v1_split{seed}_model42'
        m=read_json(run/'run_manifest.json')
        artifacts.append(dict(split_seed=seed,model_seed=42,case='NB_P24',reference_case='B0',
            run_directory=str(run),run_identity_hash=m['identity_hash'],
            files={name:dict(path=str(run/name),sha256=sha(run/name)) for name in
                ('run_manifest.json','component_predictions.parquet','control_predictions.parquet','aligned_unique_components.parquet',
                 'MODEL_CV_COMPLETE','EVALUATION_COMPLETE','VERIFIED_COMPLETE')}))
        d=pd.read_parquet(run/'control_predictions.parquet');d=d[d.fipsCode.astype(str).str.zfill(5)=='54015'].copy()
        columns=['fipsCode','timestamp_et','hour_idx','fold']+[c for c in d if c.startswith(('true_','pred_B0_v18_rule','pred_NB_P24_v18_rule'))]
        d=d[columns];d.insert(0,'split_seed',seed);traces.append(d)
    (HERE/'summary').mkdir(exist_ok=True)
    pd.concat(traces,ignore_index=True).to_parquet(HERE/'summary/Clay_54015_predictions_all_splits.parquet',index=False)
    write_json(HERE/'summary/candidate_manifest.json',dict(status='recommended_CV_recipe_not_deployed' if improves==3 and short_improves==6 else 'unconfirmed_candidate',
        candidate='P1_ab_F2_plus_P6_ab163_plus_D_G_plus_P24_direct_neighbor183',P48_N_R='original_strict',
        P24_feature_manifest=dict(path=str(PARENT/'features/v1/feature_manifest.json'),sha256=sha(PARENT/'features/v1/feature_manifest.json')),
        no_cross_split_prediction_averaging=True,full_data_submission_models_generated=False,splits=artifacts))
    matrix=[];scores=[];prows=[];countrows=[];windowrows=[];clayrows=[];erows=[];top=[]
    for seed in SEEDS:
        g=cand[cand.split_seed==seed].set_index('horizon')
        matrix.append([seed]+[f'{g.loc[h,"rmse_change_pct"]:+.3f}%' for h in HS])
        for h in HS[:3]:
            r=g.loc[h];b=ci[(ci.split_seed==seed)&(ci.domain=='OSI')&(ci.horizon==h)].iloc[0]
            scores.append([seed,f'{HOURS[HS.index(h)]}h',f'{r.baseline_rmse:.12f}',f'{r.rmse:.12f}',f'[{b.ci_low:+.9f}, {b.ci_high:+.9f}]'])
        for space in ('component_clipped','aligned'):
            r=pm[(pm.split_seed==seed)&(pm['case']=='NB_P24')&(pm.horizon==HS[2])&(pm.space==space)&(pm['group']=='full')].iloc[0]
            prows.append([seed,space,f'{r.baseline_rmse:.9f}',f'{r.rmse:.9f}',f'{r.rmse_change_pct:+.3f}%'])
        c=county[(county.split_seed==seed)&(county['case']=='NB_P24')&(county.horizon==HS[2])]
        f=fold[(fold.split_seed==seed)&(fold['case']=='NB_P24')&(fold.horizon==HS[2])]
        countrows.append([seed,f'{int((c.sse_reduction>0).sum())}/{int((c.sse_reduction<0).sum())}',
            f'{int((f.sse_reduction>0).sum())}/{int((f.sse_reduction<0).sum())}',f'{c.sse_reduction.sum():+.6f}'])
        r=clay[(clay.split_seed==seed)&(clay.horizon==HS[2])].iloc[0]
        clayrows.append([seed,int(r.fold),f'{r.baseline_rmse:.9f}',f'{r.rmse:.9f}',f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}',f'{c.sse_reduction.sum():+.6f}'])
        for r in win[(win.split_seed==seed)&(win['case']=='NB_P24')&(win.horizon==HS[2])&(~win.common_target)&(win.n>0)].itertuples():
            windowrows.append([seed,f'{max(96,int(r.start))}–{int(r.end)}',f'{r.rmse_change_pct:+.3f}%',f'{r.sse_reduction:+.6f}'])
        e=data['OSI_change_decomposition'];e=e[(e.split_seed==seed)&(e['case']=='NB_P24')&(e.horizon==HS[2])].iloc[0]
        erows.append([seed,f'{e.P_self_sse_gain:+.9f}',f'{e.P_other_cross_sse_gain:+.9f}',f'{e.postprocessing_sse_gain:+.9f}',f'{e.OSI_sse_gain:+.9f}'])
        ranked=c.sort_values('sse_reduction')
        for label,v in [('worsened',ranked.head(5)),('improved',ranked.tail(5))]:
            for r in v.itertuples():top.append(dict(split_seed=seed,direction=label,fipsCode=r.fipsCode,countyName=r.countyName,stateAbbr=r.stateAbbr,sse_reduction=r.sse_reduction))
    write_frame(HERE/'summary/top_county_changes.csv',pd.DataFrame(top))
    clay24=clay[clay.horizon==HS[2]]
    clay_note=f'Clay在{int((clay24.sse_reduction<0).sum())}/3套中恶化、{int((clay24.sse_reduction>0).sum())}/3套中改善。'
    c24=consist[consist.horizon==HS[2]]
    ci_target=ci[(ci.domain=='OSI')&(ci.horizon==HS[2])]
    ci_note='24h仅seed20260917的区间完全低于0；seed42和seed20260918的区间跨零。'
    report=['# P24邻县信息：两套CV确认与三套汇总（2026-09-20）','',
        f'**结论：{verdict}。** 本轮新增50次拟合、10个外层模型，两套独立特征/模型验收均PASS；P24三套累计75次拟合、15个模型。','',
        '各套B0均为自己的P1 a/b＋F2、P6 a/b（163列）、D_G和其他strict来源。本轮只给直接P24-Huber加入首轮冻结20列邻县特征，不更改P6/P48、损失、邻居数或天气窗口。','',
        '## 1. 四个最终OSI输出','',
        '下表是相对各自B0的pooled RMSE变化，负数改善、正数恶化。','',
        table(['CV seed','1h','6h','24h','48h'],matrix),'',
        table(['CV seed','输出h','B0 RMSE','候选RMSE','ΔRMSE的95%区间'],scores),'',
        ci_note+' 每项bootstrap为固定2000次县整条轨迹配对抽样；三套来自同一事件和同239县，并已参与开发，不能当成三份独立隐藏测试集。','',
        '48h三套逐行不变；P24通过C3影响1h/6h，但目标小时96以前不变。1h/6h在重叠目标时刻共用C3预测，不是两份独立证据。没有将三套OOF平均或拼接成新候选。','',
        '## 2. P分量、县和时段','',
        table(['CV seed','P24空间','B0 RMSE','候选RMSE','变化'],prows),'',
        'P24来源本身与C3后的P误差在三套均改善，方向与整体OSI一致。这里仍是点估计复现，不能把区间跨零的比较称为显著提升。','',
        'component_clipped是直接P24输出截断后的分量，aligned是C3后的诊断分量。最终24h使用前者对应的C1，不使用长期C3。','',
        table(['CV seed','改善县/恶化县（24h）','改善折/恶化折','全县SSE改善'],countrows),'',
        f'24h有{int((c24.improves_splits==3).sum())}县三套均改善、{int((c24.worsens_splits==3).sum())}县三套均恶化。所有县均保留计分，没有按结果筛县。','',
        table(['CV seed','目标小时','24h RMSE变化','SSE改善'],windowrows),'',
        '## 3. Clay预先指定的风险复核','',
        table(['CV seed','外折','Clay B0 RMSE','Clay候选RMSE','Clay RMSE变化','Clay SSE改善','全县SSE改善'],clayrows),'',
        clay_note+' 该县是根据首轮误差预先指定的诊断对象，不是可部署的选择开关；不能为改善本次结果而只对Clay回退旧模型或将其删除。','',
        '首轮Clay的24h RMSE上升约4.51%，两套确认分别下降约2.53%和0.67%；首轮的恶化未在确认中复现，不能认定该特征组必然损害Clay，也不能承诺每个划分都改善该县。','',
        '三套Clay的四horizon逐行轨迹统一保存在summary/Clay_54015_predictions_all_splits.parquet；新增两套也在各自split目录summary保存单套轨迹。全部239县、固定7县和每套改善/恶化前五县完整保存。这里的SSE增减是有符号净贡献，不是事件原因或跨事件风险概率。','',
        '## 4. 整体误差变化拆解','',
        table(['CV seed','P自身平方项改善','P与N/D/R交叉项改善','后处理/精度余项改善','OSI净SSE改善'],erows),'',
        '各项与实际逐行SSE变化闭合，P的变化和最终OSI的变化分别报告。没有重新训练N/D/R，也不能把分量改善直接当作整体改善。','',
        '## 5. 后续采用边界','',
        f'当前判断：**{verdict}**。完整已确认基线没有被覆盖；此次没有生成全量提交模型。',
        '后续优先组合为P1 a/b＋F2、P6 a/b（163列）、P24直接Huber＋邻县（183列）、D_G，其余P48/N/R及其他D保持原strict。P6/P48的邻县增量仍未确认，不顺带加入。',
        '当前证据支持方向复现，收益强度仍有不确定性。继续保留不加邻县的P24对照；后续若其他分量或组合方式改变，需重新检查完整OSI收益，不能机械相加不同实验的百分比。本轮没有拟合新的权重、县级规则或时间门控。','',
        '## 6. 实施与验收','',
        '- 每套入口显式指定CV，loader和verifier使用对应CSV；P6增强参照通过候选来源清单、run identity、行级fold及哈希确认。',
        '- 首轮183列包只读复用；每套独立重建h24邻居排序、原始时间语义、float64 rank顺序聚合和全部183列，模型重载使用重建数组。没有向原features目录写文件。',
        '- 每套20次内层probe＋5次完整四折refit；仅直接P24-Huber，全部模型seed42。外折监督不进入梯度、分箱或早停；外折标签扰动不改变训练输入。',
        '- 每套独立重载5个模型，检查20个probe、40组控制指标、6组区间、全部县/折/时段表和源预测不变量；有效行仍为34,177 / 32,982 / 28,680 / 22,944。',
        '- 新增文件仅在本confirmation目录；父实验和特征包、既有参照、模型与报告只读。统计区间不消除空间相关与重复模型选择偏差。','',
        '## 7. 阅读入口','',
        f'- [四输出指标]({HERE}/summary/primary_scores.csv)',f'- [P分量指标]({HERE}/summary/P_component_metrics.csv)',
        f'- [区间]({HERE}/summary/bootstrap_intervals.csv)',f'- [Clay三套指标]({HERE}/summary/Clay_54015_metrics.csv)',
        f'- [Clay三套逐行轨迹]({HERE}/summary/Clay_54015_predictions_all_splits.parquet)',
        f'- [县方向一致性]({HERE}/summary/county_consistency.csv)',f'- [误差变化拆解]({HERE}/summary/OSI_change_decomposition.csv)',
        f'- [后续实验可读取的候选来源清单]({HERE}/summary/candidate_manifest.json)',
        f'- [适配记录]({HERE}/reference/adaptation_verification.json)',f'- [执行说明]({HERE}/README.md)','']
    (HERE/'Results_2026-09-20.md').write_text('\n'.join(report))
    for seed in SEEDS[1:]:
        g=cand[cand.split_seed==seed]
        text=['# P24邻县确认单CV结果','',f'CV seed={seed}，model seed=42；独立核验PASS。','',
            table(['输出h','B0 RMSE','候选RMSE','变化'],[[f'{HOURS[HS.index(r.horizon)]}h',f'{r.baseline_rmse:.12f}',f'{r.rmse:.12f}',f'{r.rmse_change_pct:+.3f}%'] for r in g.itertuples()]),'',
            f'三套与Clay复核见[汇总]({HERE}/Results_2026-09-20.md)。','']
        (HERE/f'split{seed}/Results_2026-09-20.md').write_text('\n'.join(text))
    (HERE/'README.md').write_text(f'''# P24邻县信息确认

状态：seed20260917、seed20260918完成，独立验收PASS。新增50次拟合、10个外层模型；[三套结果]({HERE}/Results_2026-09-20.md)。

{verdict}。每套使用自己的P6增强完整B0，只重训直接P24-Huber，原冻结183列输入。首轮特征包只读复用。

首次执行：

```bash
cd {BASE.parents[2]}
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/confirmation/prepare_confirmation.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/confirmation/run_confirmations.py
.venv/bin/python -B versions/xyy/v1.5.8_strict_baseline/p_neighbor_horizon_increment/confirmation/summarize_confirmation.py
```

所有单套入口要求显式`--split-seed 20260917`或`20260918`。训练入口内部先完成预检再拟合；已完成stage拒绝覆盖，--resume仅限同身份未完成run。可单独运行verify_neighbors.py --split-seed相应值重新核验，不训练模型。

- reference：父文件哈希及AST适配核验。
- split*/reference和experiment_config.json：对应CV来源身份与预检。
- split*/runs：5个P24模型、20个probe的逐行输出/曲线、全OOF与控制、指标和独立验证。
- split*/summary：单套指标、误差分解、Clay逐行轨迹。
- summary：三套分开报告的指标、Clay和全县稳定性；没有平均OOF。
- completion.json、final_integrity_check.json：完成清单和原来源/特征包不变证明。

父目录README、报告和模型不回写。没有确认P6/P48邻县候选、没有联合候选、没有全量模型或提交。
''')
    print(cand[['split_seed','horizon','baseline_rmse','rmse','rmse_change_pct']].to_string(index=False))
    print(verdict);print(clay_note)


if __name__=='__main__':main()
