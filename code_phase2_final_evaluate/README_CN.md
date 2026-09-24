# final 模型嵌套五折评价

本目录评价 `code_phase2_final` 的最终模型：I3 scope-aware baseline + `m2_robust_input` 残差 GAT。

评价入口直接导入并复用 `code_phase2_final/final_model.py`，因此不会另写一套简化 GAT。每个外层 fold 的流程是：

1. 将一个县级 fold 作为外层验证集；
2. 只用其余四个 fold 的标签训练 GAT；
3. 只在这四个 fold 内部进行四折 OOF alpha 选择；
4. 用选出的 alpha 对外层 fold 生成 baseline 和 GAT 预测；
5. 在同一批外层行上计算 pooled、逐 fold 和县级指标。

CV 文件固定为 `code_phase2_compare` 的 `balanced_v1_seed42`，SHA-256 必须为：

```text
2ee47b74590051cec9d0d1ed0ac7d7a640250c8370433b1dfdac9b281ffe4021
```

如果仓库中没有 `code_phase2_compare/cv/cv_assignments_balanced_v1_seed42.csv`，程序会使用 I3 包中的 `inputs/cv_seed42.csv`，并仍然强制检查上述哈希及逐县 fold 映射。

## 运行

从仓库根目录执行：

```bat
conda activate phase2
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
python -B code_phase2_final\I3_prediction_package_v1\verify_package.py
python -u code_phase2_final_evaluate\evaluate_nested_cv.py ^
  --package-dir D:\inform2026\informs-data-mining-2026\code_phase2_final\I3_prediction_package_v1 ^
  --output-dir D:\inform2026\informs-data-mining-2026\code_phase2_final_evaluate\outputs\runs ^
  --run-id i3_m2_nested_cv_seed42 ^
  --seed 42 ^
  --device cuda ^
  --epochs 220 ^
  --patience 35 ^
  --bootstrap-replicates 2000
```

评估会训练嵌套 CV 所需的 scope 模型，预计比最终全五折训练耗时更长。它不会覆盖 `code_phase2_final/outputs/runs` 中的最终提交模型。

## 输出

结果目录：

```text
code_phase2_final_evaluate\outputs\runs\i3_m2_nested_cv_seed42\
```

主要文件：

- `outer_oof.parquet`：同一批外层行上的 `fipsCode`、`timestamp_et`、`outer_fold`、`y_true`、`baseline_pred`、`gat_pred`、`horizon`，以及 alpha、scoreable 标记和审计字段；
- `fold_metrics.csv`：每个 horizon、outer fold、model 的 RMSE、MAE、bias、R² 等；
- `cv_summary.csv`：每个 horizon 的 pooled baseline/GAT 指标及相对 baseline 的 RMSE/MAE improvement；
- `county_metrics.csv`：县级 paired SSE/RMSE 比较；
- `county_bootstrap.csv`：按县重采样的 GAT-baseline RMSE 差异及 95% 区间；
- `alpha_selection.csv`：每个外层 fold 内部 alpha OOF 候选及选择结果；
- `run_manifest.json`：包哈希、CV 哈希、模型配置和输出哈希；
- `CV_COMPLETE`：评价流程完成标志。

`submission_i3_m2_gat.csv` 不能用于计算真实 RMSE/MAE，因为测试标签不可用；正式效果应以 `outer_oof.parquet` 和 `cv_summary.csv` 为准。
