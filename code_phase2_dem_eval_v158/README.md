# dem_v158_nested_v2

This directory contains the isolated `dem_v158_nested_v2` protocol. The v1.5.8
experiment configuration deliberately pins its feature package to the frozen
v1.5.6 Phase-1 schema (163 columns); v1.5.8 supplies the component-target and
experiment route, not a separate feature table. GAT inputs therefore have 205
ordered columns. Historical 209-dimensional checkpoints, OOF files, and v2.1
artifacts are never loaded.

Run from the project root:

```bash
python code_phase2_dem_eval_v158/main.py --stage preflight --base-mode direct
python code_phase2_dem_eval_v158/main.py --stage cv \
  --base-mode component_v158 --run-id dem_v158_nested_v2_component_seed42 \
  --seed 42 --device auto --epochs 220 --patience 35 --time-stride 1 --k 8
python code_phase2_dem_eval_v158/main.py --stage final \
  --base-mode component_v158 --run-id dem_v158_nested_v2_component_seed42 \
  --resume --seed 42 --device auto --epochs 220 --patience 35 --time-stride 1 --k 8
```

Outputs are isolated under `outputs/runs/<run_id>/`. The independent verifier
must pass before `COMPLETE` is created:

```bash
python code_phase2_dem_eval_v158/verify_artifacts.py \
  code_phase2_dem_eval_v158/outputs/runs/<run_id>
```

See `README_CN.md` for the data contract, scope-isolated nested CV, causal
feature boundary, artifact schema, and historical-result policy.
