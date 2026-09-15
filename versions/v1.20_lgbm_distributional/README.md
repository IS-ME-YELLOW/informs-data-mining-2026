# v1.20 LightGBM Distributional Components

This experiment tests distribution-sensitive LightGBM objectives for replacing the same-horizon v1.8 C1 `P_t` and/or `D_t` predictions at `t+24` and `t+48`. It is isolated under this directory and does not modify earlier versions or shared scripts.

## Fixed candidates

Each component/horizon has six candidates: `dart_huber`, Tweedie powers 1.3/1.5/1.7, and quantiles 0.50/0.90. The primary `balanced_v1` run trains 2 components × 2 horizons × 6 candidates × 5 county folds = 120 models. DART uses 600 fixed rounds. Other candidates use at most 2000 rounds with early stopping patience 100. `--quick` changes these to 100 and 300/patience 30 respectively and is only for smoke testing.

The v1.8 baseline components are clipped and recomposed as same-horizon C1; C3 alignment is never used. The comparison/blending baseline is `pred_tail_protected_{horizon}` from v1.12.

## Commands

Use the project environment explicitly when desired:

```bash
PYTHON="C:/Users/Stella/.conda/envs/best-route/python.exe"
$PYTHON versions/v1.20_lgbm_distributional/run.py prepare
$PYTHON versions/v1.20_lgbm_distributional/run.py dry-run --num-threads 1
$PYTHON versions/v1.20_lgbm_distributional/run.py train-primary --num-threads 8
$PYTHON versions/v1.20_lgbm_distributional/run.py train-primary --num-threads 8 --resume
$PYTHON versions/v1.20_lgbm_distributional/run.py evaluate-primary --bootstrap-replicates 2000
$PYTHON versions/v1.20_lgbm_distributional/run.py train-repeats --num-threads 8 --resume
$PYTHON versions/v1.20_lgbm_distributional/run.py evaluate-repeats
$PYTHON versions/v1.20_lgbm_distributional/run.py verify --split balanced_v1
$PYTHON versions/v1.20_lgbm_distributional/run.py finalize --num-threads 8
```

`--resume` skips a fold only when the model, OOF shard, hashes, fold assignment, inputs, complete parameters, quick/formal mode, and sidecar fingerprint agree. Partial or inconsistent artifacts cause a hard failure and are not overwritten.

## Primary cross-fitted evaluation

For each horizon, evaluation enumerates 48 replacements: six P-only, six D-only, and 36 P+D combinations. Candidate component predictions are five-fold county OOF predictions. For every outer fold, the other four folds select `alpha` from 0.1/0.25/0.5/0.75/1 and `plain` or `protected` blending by RMSE. Protected blending computes the positive v1.12 prediction q95 using only outer-training rows, blends below q95, and leaves predictions at or above q95 unchanged. Ties prefer lower alpha, then protected mode, then lexicographic candidate name.

This is cross-fitted blend-parameter selection, not fully nested candidate-model evaluation. Candidate models used to predict the four outer-training folds were trained in the original five-fold OOF scheme, so some of them include counties from the current outer validation fold. Fully strict nesting would require retraining every component candidate within each outer-training partition. Selecting the lowest-RMSE result from 48 candidates on this same OOF framework also creates best-of-many selection optimism. The reported gains are therefore exploratory; these limitations do not weaken the decision not to promote any candidate.

Outputs include all-candidate cross-fitted predictions, outer-fold blend selections, pooled and fold metrics, county metrics, actual-top-5%-OSI metrics, county trajectory bootstrap intervals, residual Pearson/Spearman correlations, and `primary_decision.json`. The existing `primary_all_candidates_nested_oof.parquet` and `primary_winner_*` filenames are retained for artifact compatibility; the latter tables contain all candidates rather than winners only.

A primary winner passes only if all gates hold: pooled RMSE change ≤ -0.5%, improvement in at least 4/5 folds, county-bootstrap 95% upper bound below zero, actual-top-5% RMSE degradation ≤ 0.25%, and pooled MAE degradation ≤ 1%.

## Primary results

| Horizon | Exploratory winner | RMSE change vs v1.12 | MAE change | Improved folds | Bootstrap 95% upper bound | Actual top-5% RMSE change | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| t+24 | `PD__tweedie_p15__tweedie_p15` | -0.0689% | -2.7289% | 4/5 | +0.0000245 | +0.7568% | Not promoted |
| t+48 | `PD__quantile_q90__tweedie_p13` | -0.9936% | +9.4816% | 5/5 | +0.0001119 | -1.3282% | Not promoted |

At t+24 the pooled RMSE change is negligible, the county-bootstrap interval crosses zero, and severe-outage performance worsens beyond the tail gate. At t+48 the q90 P component reduces a small number of large squared errors and improves the actual high-OSI tail, but broadly raises low-value predictions, worsening MAE by 9.48%; county-level uncertainty also remains too large. Winner residuals remain highly correlated with v1.12 (Pearson 0.9984 at t+24 and 0.9927 at t+48), so the objectives provide little independent error diversity.

Tweedie and especially quantile best iterations vary sharply across county folds; some quantile fits stop within 3–40 rounds while others approach 2000. DART used 600 fixed rounds and is not part of that early-stopping instability. No horizon passed all gates, repeat training was skipped, no final v1.20 models or submission were produced, and v1.12 remains the current internal best.

## Repeat and finalization limitation

Repeat fold files are deterministic state × severity stratified, county-balanced five-fold assignments for seeds 20260921 and 20260922. Repeat training is restricted to component candidates used by promoted primary winners.

The original v1.12 OOF predictions are generated under `balanced_v1`; treating them as independent predictions under the two new repeat folds would be invalid. This first implementation therefore records `blocked_requires_baseline_retraining` whenever a primary winner passes. It does not compute repeat promotion from the original OOF and `finalize` refuses to train final models or create a submission while blocked. A future repeat evaluation must independently retrain the v1.12 baseline on each repeat assignment, then require both repeats to improve pooled RMSE, each to improve at least 3/5 folds, and the combined county-bootstrap upper bound to be below zero.

This blocking rule is a statistical safeguard, not a runtime error to bypass.

## Artifacts

All generated files remain in `models/`, `predictions/`, `folds/`, `results/`, `manifests/`, or `artifacts/` below this directory. Verification checks input hashes, deterministic folds and county isolation, prediction coverage and target NaN masks, sidecar fingerprints, artifact hashes, LightGBM reload predictions, exact v1.8 C1 recomposition, and v1.12 key/column alignment. Use `--model-sample N` for a quicker model reload check; omit it to verify every existing model.
