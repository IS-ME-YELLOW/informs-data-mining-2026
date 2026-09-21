# dem_v158_nested_v2

This directory contains the isolated `dem_v158_nested_v2` protocol. The v1.5.8
experiment configuration deliberately pins its feature package to the frozen
v1.5.6 Phase-1 schema (163 columns); v1.5.8 supplies the component-target and
experiment route, not a separate feature table. This no-neighbor-summary variant
uses 173 GAT input features
ordered columns. Historical 209-dimensional checkpoints, OOF files, and v2.1
artifacts are never loaded.

Run from the project root:

```bash
python code_phase2_dem_eval_v158_limit/main.py --stage preflight --base-mode direct
python code_phase2_dem_eval_v158_limit/main.py --stage cv \
  --base-mode component_v158 --run-id dem_v158_nested_v2_component_seed42 \
  --seed 42 --device auto --epochs 220 --patience 35 --time-stride 1 --k 8
python code_phase2_dem_eval_v158_limit/main.py --stage final \
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

The September 17 fixes for review items 1–6 have a no-training regression suite:

```bash
python -B code_phase2_dem_eval_v158/tests/test_repairs.py
```

CV base identities stay in the immutable `base_fit_manifest.parquet`; final
training publishes the complete superset in `base_fit_manifest_final.parquet`.
Numerical artifact verification is now implemented in `artifact_checks.py`,
including exact keyed coverage, official labels, alpha recomputation, metrics,
lineage, bootstrap, and submission values/NaN positions. `verify_artifacts.py`
then reloads real models with supervision reads and training calls blocked, and
checks the reconstructed graph and predictions against the final submission.
Only both successful phases can create `COMPLETE`.

Run the no-training verification regressions with:

```bash
python -B code_phase2_dem_eval_v158/tests/test_artifact_checks.py
```

Model-level resume now uses atomic completion records and an immutable identity
covering code, configuration, inputs and runtime. Requested and actual booster
iterations are recorded separately. Use a fresh run ID for this artifact format.

Run `python -B code_phase2_dem_eval_v158/tests/test_p2_contract.py` for the local
no-training regressions. `REMOTE_ACCEPTANCE_CN.md` gives the complete remote
commands for `validate_runtime.py`, including real-estimator label perturbations,
interruption/reload checks and short-booster validation. Local stand-ins cannot
mark real-model acceptance as passed; real GAT checks must run remotely.
