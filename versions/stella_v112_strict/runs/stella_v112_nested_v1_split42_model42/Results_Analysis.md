# Stella v1.12 strict seed42 experiment report

Run ID: `stella_v112_nested_v1_split42_model42`  
Protocol: `stella_v112_nested_v1`  
Run identity: `0a019474a437758214bb2eb3de8e13fb0a112320f6c69df0fe76b41166788be4`

## Outcome

L2 improved pooled RMSE relative to L0 at both long horizons:

| Horizon | L0 RMSE | L1 RMSE | L2 RMSE | L2 - L0 | Change | County bootstrap 95% CI | P(delta < 0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24h | 0.00849260 | 0.00852948 | 0.00844538 | -0.00004722 | -0.556% | [-0.000145, 0.000037] | 0.8560 |
| 48h | 0.00769135 | 0.00782487 | 0.00766168 | -0.00002967 | -0.386% | [-0.000104, 0.000025] | 0.8505 |

Both point estimates satisfy the handoff's condition for considering confirmation splits. Both confidence intervals cross zero, so seed42 alone is limited evidence rather than a statistically secure improvement.

L1 alone worsened RMSE by 0.434% at 24h and 1.736% at 48h, despite lowering MAE. The q95 gate recovered the high-error tail: L2 beat L1 by 0.986% RMSE at 24h and 2.086% at 48h while changing MAE by only about one millionth.

The fixed short-horizon control was preserved exactly:

| Horizon | L0 RMSE | L1 RMSE | L2 RMSE | Row equality |
|---|---:|---:|---:|---|
| 1h | 0.010991 | 0.010991 | 0.010991 | exact, including NaN positions |
| 6h | 0.010080 | 0.010080 | 0.010080 | exact, including NaN positions |

## Gate and threshold behavior

| Horizon | Gate rows | Gate coverage | Actual top-5% overlap | Approx. precision | Approx. recall |
|---|---:|---:|---:|---:|---:|
| 24h | 625 | 2.179% | 428 | 68.5% | 29.9% |
| 48h | 457 | 1.992% | 255 | 55.8% | 22.3% |

The five outer-fold q95 thresholds ranged from 0.019945 to 0.024143 at 24h and from 0.015771 to 0.017115 at 48h. They were computed only from the four inner predicted folds for each outer context, with the outer fold and the currently predicted inner fold excluded from every contributing fit.

## Stability by fold and county

At 24h, L2 improved folds 0, 2 and 4 but worsened folds 1 and 3. The per-fold RMSE changes versus L0 were -0.000149, +0.000062, -0.000207, +0.000145 and -0.000015.

At 48h, L2 improved folds 0, 1, 2 and 4 and worsened fold 3. The per-fold changes were -0.000058, -0.000010, -0.000051, +0.000035 and -0.000056.

L2 improved county-level squared error for 153 of 239 counties at 24h and 159 of 239 at 48h. County `39157` produced the largest gain at both horizons. The five largest positive county contributions accounted for 32.4% of gross positive improvement at 24h and 36.5% at 48h; therefore the benefit is meaningfully concentrated, although not solely produced by one county. Negative county contributions partially offset those gains, which is why the net point estimate is sensitive to county composition and the bootstrap intervals cross zero.

Six-source error correlations were high: 0.942–0.993 at 24h and 0.964–0.995 at 48h. This confirms limited source diversity and helps explain why unconditional equal averaging (L1) did not improve RMSE.

## Execution and acceptance evidence

- Frozen seed42 data: 34,416 training rows, 239 counties, 163 features.
- Scoreable rows: 34,177 / 32,982 / 28,680 / 22,944 for 1h / 6h / 24h / 48h.
- New fit calls: 820 exactly.
- New saved models: 180 exactly — 50 XGBoost outer, 50 CatBoost outer and 80 deduplicated three-fold LightGBM inner-anchor models.
- Referenced models: 50 strict LightGBM outer models; all were canonicalized from Git CRLF checkout bytes to their frozen LF receipt hashes without modifying the source files.
- Source preflight: PASS. Protocol examples: PASS. Production scope checks: 820. Poisoned-label checks: 640.
- Fresh-process verification: 180 new models and 50 referenced models reloaded from disk. Maximum prediction difference was 0 for both groups under `atol=1e-12, rtol=0`.
- All row-level predictions, thresholds, metrics and bootstrap results were recomputed by the verifier before `CV_COMPLETE` was written.
- No test-set inference, final all-data fit, submission generation or confirmation-split training was performed.

## Decision

The seed42 point estimates support proceeding to the two pre-registered confirmation splits without changing candidates, q, weights, folds, features, losses or early-stopping rules. Because both bootstrap confidence intervals include zero and the benefit is partly concentrated by county/fold, confirmation should be described as testing reproducibility, not as validating an already established gain.
