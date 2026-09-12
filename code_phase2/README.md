# Phase 2 — LightGBM + spatial GAT residual

This directory contains a self-contained model 2.0 built on the latest Phase 1
feature cache (`v1.5.7`). For each horizon it:

1. trains the Phase-1-compatible LightGBM model with the persisted county-level
   5-fold assignment;
2. obtains out-of-fold LightGBM predictions and defines the county-hour residual
   `y - y_lightgbm`;
3. builds a 302-county graph from `data/geo/c_16ap26.dbf` (latitude/longitude,
   symmetric 8-nearest-neighbour edges);
4. trains a two-layer multi-head GAT only on training-county residuals; and
5. adds the GAT residual correction to the LightGBM prediction, with the blend
   weight selected from held-out counties and clipped to the physical OSI range.

The GAT feature set uses only Phase-1 features that are available at prediction
time. In particular, no outage field, lag field, OSI field, or target from the
March 14–19 prediction window is read by Phase 2. Weather fields may use the
full supplied weather trajectory, as permitted by the challenge rules.

Run from the project root with the requested environment:

```powershell
conda activate myenv
python code_phase2/main.py
```

If the shell's conda installation does not switch environments on Windows, use
the environment interpreter directly:

```powershell
D:\app\anacnda1\envs\myenv\python.exe code_phase2/main.py
```

The default run keeps the full training scale: all 144 forecast snapshots,
5 county-grouped folds, k=8 neighbours, and up to 220 GAT epochs with early
stopping. `--device auto` uses CUDA when the remote environment provides it.
For a local smoke test only, use `--epochs 10 --patience 3 --time-stride 4`.

Outputs are written to `code_phase2/outputs/`, including the submission,
county-grouped CV comparison, alpha selection table, and saved GAT/LightGBM
 models.
