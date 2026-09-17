# dem_v158_nested_v2

- run_id: `dem_v158_p2_component_seed42_cuda_log`
- base_mode: `component_v158`
- experiment_version: `v1.5.8`
- feature_version: `v1.5.6`
- graph_input_dim: `205`
- CV evidence: `CV_COMPLETE`

## CV summary

```text
        horizon model     n      mse     rmse      mae    medae      bias  error_std       r2  max_abs_error
osi_target_t01h  base 34177 0.000143 0.011970 0.003567 0.000600 -0.000119   0.011970 0.723734       0.453308
osi_target_t01h   gat 34177 0.000320 0.017875 0.004368 0.000521  0.000754   0.017859 0.383982       0.453308
osi_target_t06h  base 32982 0.000114 0.010669 0.003423 0.000600 -0.000062   0.010668 0.691373       0.386931
osi_target_t06h   gat 32982 0.000112 0.010560 0.003393 0.000500 -0.000064   0.010560 0.697619       0.386931
osi_target_t24h  base 28680 0.000073 0.008560 0.002812 0.000300 -0.000091   0.008560 0.462834       0.327700
osi_target_t24h   gat 28680 0.000072 0.008500 0.002761 0.000300 -0.000107   0.008500 0.470322       0.327700
osi_target_t48h  base 22944 0.000061 0.007831 0.002240 0.000200 -0.000202   0.007828 0.337429       0.327700
osi_target_t48h   gat 22944 0.000061 0.007790 0.002233 0.000200 -0.000185   0.007787 0.344381       0.327700
```

## County-clustered paired bootstrap

```text
        horizon  replicates     seed  n_counties  base_rmse  gat_rmse  delta_rmse_gat_minus_base  delta_ci_low  delta_ci_high  gat_better_probability
osi_target_t01h        2000 20260910         239   0.011970  0.017875                   0.005904     -0.000399       0.015331                  0.3575
osi_target_t06h        2000 20260910         239   0.010669  0.010560                  -0.000109     -0.000341       0.000081                  0.8590
osi_target_t24h        2000 20260910         239   0.008560  0.008500                  -0.000060     -0.000404       0.000336                  0.6435
osi_target_t48h        2000 20260910         239   0.007831  0.007790                  -0.000041     -0.000230       0.000105                  0.6500
```

County metrics and bootstrap are recomputed from `outer_oof.parquet`; the bootstrap preserves all scoreable hours within sampled counties. Spatial dependence is therefore an interval limitation, not an independent-row assumption.
Final alpha selection is stored in `alpha_selection_final.parquet`; its internal CV score is not an independent outer score.
This report is generated from the run manifest and saved row-level artifacts.
