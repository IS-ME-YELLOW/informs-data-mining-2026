# 字段口径

所有CSV为无图表的数值结果。`case`是候选，`reference`是本行参考（不一定B0）；`pair`=case_minus_reference。`horizon`为最终输出或分量来源h（小时），具体以文件及space为准。`fold`为seed42外折0–4。`fipsCode`应按五位字符串读取。CSV复算建议`float_precision="round_trip"`。

- `n`：本行分组内真实标签有限的县小时数。有效目标截止215，四h全县n分别34177/32982/28680/22944。
- `rmse/sse/mae/bias`：候选在对应范围的指标；bias=预测−真实。`baseline_rmse`是同组reference的RMSE。`rmse_delta`=候选−参考；`rmse_change_pct`=100×(候选RMSE/参考RMSE−1)，负数改善。`sse_reduction`=参考SSE−候选SSE，正数改善。空组指标为NaN、SSE=0。
- `primary_scores/comparisons/group_metrics/county_metrics/fold_metrics/day_metrics/row_changes`均评价最终v18_rule OSI；`all_controls`另列C0/C1/C2/C3。1/6h最终用C3，24/48h最终用C1。
- `component_metrics`：单位为N或R比例。`space=source`指各h独立来源，`aligned`指按同县同目标小时C3平均。group=`zero/small_positive/tail_gt_001`分别Z=0、0<Z≤.01、Z>.01；互斥。`positive`是后两组并集，与它们不能相加。`baseline_group_sse`是本组参考SSE；`baseline_group_sse_share`分母为同component/space/horizon/reference的全部有效行参考SSE。
- `test_like`：冻结旧context下，按h/目标日被至少一测试县选为前5近邻的训练县并集；分组是输入定义，不用测试标签。`test_votes`是选中该训练县的不同测试县数，不作训练权重。
- `bootstrap`：ci_low/ci_high为候选−参考RMSE（绝对量纲）的95%区间，2000次县轨迹配对重采样，seed20260910。完全不变的输出区间为[0,0]、replicates=0。未校正历史模型选择，不是测试胜率。
- `classification`：binary真值=1[官方Z>0]；q_model是概率头，training_prior是允许四训练折发生率常数。logloss使用自然对数，数值裁剪[1e−15,1−1e−15]仅防log(0)；Brier=mean((q−真值)²)。两者越低越好。`fold=all`是全部外层OOF汇总。
- `calibration`：固定概率箱[0,.1),...,[.9,1]，n为箱内行数；mean_q与observed_positive_fraction比较校准。均不用于事后校准或阈值选择。
- `amplitude_diagnostics`：在group指定行上求true/q/raw_m/clipped_m/product均值。`conditional_m_rmse`为clip(m)直接对真实Z的组内RMSE，只有正值组才可解释为条件幅度诊断；all/zero只是反事实“不给q缩放”诊断，不能当部署模型成绩。raw_m_below_zero/above_one是该组原始幅度越界行数。模型输出始终q×clip(m)。
- `OSI_change_decomposition`：只针对最终1/6h，令旧未后处理有符号组合误差e=.4P+.35N+.25D−.1R−官方OSI，δ=.35ΔN−.1ΔR。old_error_cross_reduction=−2Σeδ；change_squared_reduction=−Σδ²；两者和是latent_sse_reduction。N_R_change_interaction=−2Σ(.35ΔN)(−.1ΔR)，已经包含在−Σδ²内，不能再加一次。postprocess_remainder=实际最终OSI SSE减少量−latent减少量，含非负/上界/.001后处理的影响；不是额外模型贡献。
- `benefit_concentration`：gross_gain=所有改善县ΔSSE之和，gross_loss=所有恶化县−ΔSSE之和，net=两者差。top5_share_gross_gain分母为gross_gain；forest_share_net仅net>0时计算，可负或超过1。县改进判定容差±1e−12仅用于计数，SSE汇总保留原始浮点数。`top_county_changes`的排名由本次结果产生，仅用于描述。
- `NR1_oof_predictions`：*_q为概率，*_m为原始幅度，*_clipped_m为[0,1]幅度，*_product为逐行乘积；*_prior_q为该外折训练四折的发生率；两个model_id及product_id提供来源。无效尾行均NaN/空身份。`target_hour=hour_idx+1`。
- `source_*`列中的新product_id由q、m两个模型身份及公式生成；旧分量身份从NB_P24逐行原样继承。它不是单个模型文件名。
