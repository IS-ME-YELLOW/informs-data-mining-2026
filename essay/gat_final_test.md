已整理完成，报告文件为：
[evaluation_report.md](D:/academic/dataminingINFORMS/informs-data-mining-2026/code_phase2_final_evaluate/outputs/runs/i3_m2_nested_cv_seed42/evaluation_report.md)
核心 pooled 外层 OOF 结果：
Horizon	Baseline RMSE	GAT RMSE	RMSE改善	Baseline MAE	GAT MAE	MAE改善
t01h	0.01084069	0.01061279	2.10%	0.00322729	0.00317294	1.68%
t06h	0.00990906	0.00985497	0.55%	0.00310047	0.00308711	0.43%
t24h	0.00841429	0.00821388	2.38%	0.00268345	0.00259282	3.38%
t48h	0.00766153	0.00749934	2.12%	0.00210679	0.00207955	1.29%


R² 变化：
t01h: 0.773414 → 0.782841
t06h: 0.733750 → 0.736649
t24h: 0.480975 → 0.505405
t48h: 0.365774 → 0.392341
结论：
- 四个 horizon 的 pooled RMSE 和 MAE 均优于 I3 baseline；
- t24h 的提升最明显；
- 但县级 bootstrap 的四个 95% RMSE 差异区间均跨过 0；
- 因此目前应表述为“GAT 有小幅改善趋势”，不能宣称达到 95% 统计显著性。
详细原始结果位于同一目录：
outer_oof.parquet
fold_metrics.csv
cv_summary.csv
county_metrics.csv
county_bootstrap.csv
alpha_selection.csv
run_manifest.json