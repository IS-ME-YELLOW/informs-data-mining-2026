# `code_phase2_compare`

This directory is a separate, preregistered comparison protocol. It reuses
the v1.5.6 feature package, v1.5.8 component targets, balanced five-county-fold
split, LightGBM base training, nested alpha selection, and strict independent
reload checks from `code_phase2_dem_eval_v158`. It never reads historical v158
OOF files or checkpoints.

## Variants

| ID | Definition |
|---|---|
| `m0_full` | Current residual GAT: two graph-message layers plus the 205-dimensional raw-input skip path. |
| `m1_no_skip` | Same graph-message network and residual target, with the skip layer absent. |
| `m2_robust_input` | Current GAT after scope-fitted standardisation and preregistered clipping of every scaled input to `[-5, 5]`. The clipping value is not selected from outer labels. |
| `m3_bounded` | Current GAT with correction constrained to `[-0.25, 0.25]` OSI units by a differentiable `tanh` output bound. |

All variants use the same LightGBM base, graph, labels, horizons, folds,
alpha grid, post-processing, and nested county-level CV. The variant, policy,
ordered schema, upstream hashes, and architecture are included in the run
identity and every GAT checkpoint. A checkpoint from another variant is
rejected.

## Remote execution

Run from the repository root. Each variant must have its own run ID and run
directory. The following performs one complete variant run; repeat it with a
different `--variant` and `--run-id` for the other variants:

```bash
python -u code_phase2_compare/main.py --stage all --base-mode component_v158 --variant m0_full --run-id dem_compare_v1_m0_full_component_seed42_cuda --seed 42 --device cuda --epochs 220 --patience 35 --time-stride 1 --k 8
```

Before training, the same command can be run with `--stage preflight`. To
resume an incomplete run, use the identical arguments plus `--resume`; an
existing complete run is immutable. After `FINAL_READY`, run verification in
a new process:

```bash
python -u code_phase2_compare/verify_artifacts.py code_phase2_compare/outputs/runs/dem_compare_v1_m0_full_component_seed42_cuda
```

Compare verified variants only; this reads their frozen `outer_oof.parquet`
files and does not select a winner or read test labels:

```bash
python code_phase2_compare/compare_results.py code_phase2_compare/outputs/runs/dem_compare_v1_m0_full_component_seed42_cuda code_phase2_compare/outputs/runs/dem_compare_v1_m1_no_skip_component_seed42_cuda code_phase2_compare/outputs/runs/dem_compare_v1_m2_robust_input_component_seed42_cuda code_phase2_compare/outputs/runs/dem_compare_v1_m3_bounded_component_seed42_cuda --output-dir code_phase2_compare/outputs/comparisons/component_seed42
```

The comparison report contains `compare_summary.csv`,
`compare_county_metrics.csv`, and `compare_worst_counties.csv`. The outer OOF
score remains the fixed evaluation evidence for each preregistered variant;
the comparison utility does not turn post hoc winner selection into an
independent estimate.
