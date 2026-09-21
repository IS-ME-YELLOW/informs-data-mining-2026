# Direct component GAT

This is a separate protocol from `code_phase2_dem_eval_v158` and
`code_phase2_compare`.

It does not import or train LightGBM. For every horizon and every component
`P_t`, `N_t`, `D_t`, and `R_t`, it trains a direct `DirectGAT` model. The five
outer county folds are fixed to `balanced_v1_seed42`: the model for outer fold
`k` is trained only on the other four folds and predicts fold `k`. The final
model is trained on all five folds and predicts the 63 test counties.

The input schema is the current 205-dimensional GAT schema. Since this
protocol has no LightGBM prediction, its first `base` channel is an explicit
constant-zero placeholder. The other 204 columns are unchanged: the frozen
163 v1.5.6 features, coordinates, seven terrain features, and 32 approved
neighbour aggregates. This preserves the GAT feature contract without using a
tree learner.

Each component prediction is clipped to `[0,1]`, then OSI is reconstructed as
`0.40 P + 0.35 N + 0.25 D - 0.10 R`, followed by the official `[0,0.65]`
post-processing rule. There is no residual target, correction, or alpha.

Run from the repository root on the remote Windows server:

```bat
python -u code_GAT\main.py --stage all --run-id direct_component_gat_v1_seed42_cuda --seed 42 --device cuda --epochs 220 --patience 35 --time-stride 1 --k 8
```

After `FINAL_READY`, independently verify and create `COMPLETE`:

```bat
python -u code_GAT\verify_artifacts.py code_GAT\outputs\runs\direct_component_gat_v1_seed42_cuda
```

Important artifacts are `outer_oof.parquet`, `cv_summary.csv`,
`county_metrics.csv`, `test_predictions.parquet`,
`submission_phase2_dem_gat.csv`, and the 96 direct component GAT checkpoints
under `models/gat/` (80 outer-fold models plus 16 full-training models).
