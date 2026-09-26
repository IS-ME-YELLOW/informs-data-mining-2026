## 4. Results

We report county-level out-of-fold performance using the evaluation protocol in Section 3.5. Each horizon is scored over all valid county-hour observations: 34,177, 32,982, 28,680, and 22,944 rows for 1h, 6h, 24h, and 48h, respectively. Tree-model summaries average the pooled RMSEs from the three fixed county partitions. Spatial refinement is evaluated against the matching complete tree forecast on seed42. Percentage reductions are defined as

$$
\mathrm{RMSE\ reduction}
=100\left(1-\frac{\mathrm{RMSE}_{\mathrm{candidate}}}{\mathrm{RMSE}_{\mathrm{reference}}}\right)\%.
$$

Positive values indicate improvement. For comparisons of three-partition averages, the reduction is calculated from the two mean RMSEs; ranges describe the minimum and maximum reductions across the paired partition-level comparisons.

### 4.1 Overall Forecasting Performance

We compare successive configurations of the forecasting framework. The **initial component baseline** uses the 163 base features to regress each component directly with LightGBM, followed by the short-term target-time aggregation, long-term component reconstruction, and shared postprocessing described in Section 3. The **enhanced component framework** adds the retained initial-state formulations and spatial-context features from Section 3.2. The **complete tree forecast** further incorporates the tail-gated long-term ensemble and fixed forecast composition from Section 3.3. The **spatially refined model** adds the residual correction from Section 3.4.

**Table 1. Overall OSI forecasting performance. Lower RMSE is better. Panel A reports arithmetic means of three partition-level pooled RMSEs. Panel B reports paired results on seed42.**

**Panel A: Tree configurations across three county partitions**

| Configuration | 1h RMSE | 6h RMSE | 24h RMSE | 48h RMSE |
|---|---:|---:|---:|---:|
| Initial component baseline | 0.01191130 | 0.01043621 | 0.00843269 | 0.00765768 |
| Enhanced component framework | 0.01081914 | 0.00984660 | 0.00834415 | 0.00765768 |
| Complete tree forecast | **0.01081914** | **0.00984660** | **0.00830560** | **0.00760743** |

The complete tree forecast reduces mean RMSE relative to the initial component baseline by **9.17% at 1h** and **5.65% at 6h**, with reductions of **1.51% at 24h** and **0.66% at 48h**. The reduction direction is consistent across all three partitions at all four horizons. Most short-term improvement is already present in the enhanced component framework, while the long-term ensemble supplies an additional contribution at 24h and 48h.

**Panel B: Spatial refinement on the seed42 partition**

| Configuration | 1h RMSE | 6h RMSE | 24h RMSE | 48h RMSE |
|---|---:|---:|---:|---:|
| Complete tree forecast | 0.01084069 | 0.00990906 | 0.00841429 | 0.00766153 |
| Spatially refined model | **0.01061279** | **0.00985497** | **0.00821388** | **0.00749934** |
| Additional RMSE reduction | **2.10%** | **0.55%** | **2.38%** | **2.12%** |

Spatial refinement lowers pooled RMSE at all four horizons on the matched partition. The largest relative increment occurs at 24h, with approximately 2% reductions also obtained at 1h and 48h. Panel B contains the final model's validation scores; Panel A summarizes the broader partition-level evaluation of its tree foundation.

### 4.2 Effects of Component Formulations and Spatial Context

Controlled incremental comparisons identify which changes contribute to the enhanced component framework. Table 2 reports reductions in final OSI RMSE, with the reference configuration specified for each experiment. Each comparison holds the remaining pipeline fixed at that stage. The percentage ranges therefore describe staged increments rather than additive contributions to the total reduction in Table 1.

**Table 2. Retained component and feature changes. Each range spans the three county partitions. Source horizons identify the models modified; output horizons identify the final forecasts evaluated.**

| Change to a source model | Reference configuration | Final-output RMSE reduction |
|---|---|---|
| Known-history reconstruction of D1 | Initial component baseline | 1h: 1.24%–1.45%; other horizons unchanged |
| Initial-state reconstruction of P1 | Direct P1 with the retained D reconstruction | 1h: 4.53%–7.44%; 6h: 2.56%–4.01% |
| Neighbor features for P1 | Same P1 reconstruction using the 163 base features | 1h: 0.61%–1.10%; 6h: 0.60%–1.02% |
| Initial-state reconstruction of P6 | Tree configuration with P1 reconstruction, P1 neighbor features, and D reconstruction | 1h: 1.07%–1.72%; 6h: 1.34%–2.13% |
| Neighbor features for P24 | Preceding configuration with direct P24 using the 163 base features | 24h: 0.68%–1.65%; 1h: 0.11%–0.22%; 6h: 0.14%–0.27% |

The P1 initial-state formulation provides the largest individual short-term increment. To distinguish its effect from changing the training objective, we also compared it with a direct P1 model trained with squared-error loss. The structured formulation still reduced final 1h RMSE by 4.52%–7.46% across the partitions. This supports the combined use of branch decomposition, contribution-scale weighting, and reconstruction. Extending the formulation to P6 yielded a further short-term reduction, while the corresponding P24 and P48 replacements were not retained after the initial seed42 comparison.

Known-history reconstruction of D improves the earliest predictions by retaining the observed contribution to the six-hour rolling target. Its application is restricted to positive known contributions within target hours 73–76, leaving the later 1h predictions and the other three final horizons unchanged. Neighbor information provides smaller additional reductions for P1 and P24. The P24 change also improves short-term outputs through common-target-time aggregation, consistent with the information flow described in Section 3.3.

For N and R, the seed42 occurrence–magnitude comparison produced limited overall benefit. Replacing only the N1 source reduced final 1h RMSE by approximately 0.05%, and adding the R1 replacement did not improve on that result. The two-stage formulations reduced source-model errors on zero and small positive targets, but increased errors where the official component value exceeded 0.01. We therefore retained direct N and R regression in the complete framework.

### 4.3 Effects of Tail Gating and Forecast Composition

The long-term ensemble comparison separates the reference component model, the unconditional six-source average, and the tail-gated ensemble. Table 3 summarizes their mean pooled RMSEs over the three county partitions.

**Table 3. Long-term ensemble configurations. Values are mean pooled RMSEs across three partitions; lower values are better.**

| Configuration | 24h RMSE | 48h RMSE |
|---|---:|---:|
| Reference LightGBM component model | 0.00843269 | 0.00765768 |
| Unconditional six-source average | 0.00839580 | 0.00768882 |
| Tail-gated ensemble | **0.00832437** | **0.00760743** |

Unconditional averaging depends on the county partition: it increased RMSE relative to the reference at both long horizons in the first two partitions and reduced it in the third. The gated ensemble improved on both the reference and the unconditional average in every partition at both horizons. Relative to the reference, its RMSE reductions ranged from 0.56% to 2.18% at 24h and from 0.39% to 0.99% at 48h.

The gate retained the reference prediction for 1.99%–2.40% of valid 24h observations and 1.77%–1.99% of valid 48h observations. These proportions are the numbers of outer-fold predictions routed to the reference divided by all valid predictions at the corresponding horizon. Most observations therefore use the average, with a small predicted upper-tail region receiving the reference output.

The fixed equal-weight combination of the enhanced component forecast and gated ensemble reduced 24h RMSE relative to the enhanced component forecast in all three partitions, by 0.075%–1.106%. Relative to the gated ensemble alone, the combination improved two partitions and slightly worsened the third. We retained this fixed combination at 24h and the gated ensemble at 48h, yielding the complete tree forecast in Table 1. The short-term outputs remain those of the enhanced component framework.

### 4.4 Spatial Residual Refinement and Result Stability

The spatial model adds a further correction to the complete tree forecast. Alongside the RMSE reductions in Table 1, MAE decreases by 1.68%, 0.43%, 3.38%, and 1.29% at 1h, 6h, 24h, and 48h, respectively. RMSE improves in three of five outer folds at 1h, 6h, and 48h, and in four of five folds at 24h.

County-level effects are heterogeneous. Forest County, Pennsylvania, provides the largest 1h benefit, accounting for 97.3% of the net SSE reduction across all counties. This share uses the total reduction after improvements and deteriorations in individual counties have been offset. Calhoun and Braxton counties in West Virginia contribute substantial reductions at 24h and 48h, while Clay County remains a source of deterioration at those horizons. These results indicate that the aggregate gains include corrections to several high-error counties, with tradeoffs elsewhere.

All four pooled RMSE point estimates favor spatial refinement, although the county-bootstrap 95% intervals for the differences include zero. We consequently report the observed reductions together with their outer-fold and county-level variation. The comparison evaluates the complete spatial residual module, including its geographic inputs, neighborhood summaries, and graph learner. The concentration and persistence of county-specific errors are examined further in Section 5.

<!--
Editorial source notes; not part of the manuscript.

Draft aligned with essay_2_revised_draft.md and essay_3_revised_draft.md.
All displayed results are CV/OOF results, not hidden-test scores.

Table 1 Panel A:
- versions/xyy/v1.5.8_strict_baseline/d_unknown_contribution/k_positive_switch/summary/primary_metrics.csv: A_rmse is the initial component baseline.
- versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/summary/three_cv_scores.csv: B0 is the enhanced component framework; I3 is the complete tree forecast.
- Compute pooled RMSE within each partition, then arithmetic means across the three partitions. Headline reductions use the ratio of these means.

Table 1 Panel B and Section 4.4:
- ../informs-data-mining-wendyxu/code_phase2_final_evaluate/outputs/runs/i3_m2_nested_cv_seed42/cv_summary.csv
- Same directory: fold_metrics.csv, county_metrics.csv, county_bootstrap.csv.
- Use the baseline from this paired GAT evaluation for Panel B; its float32 input conversion produces negligible differences from the original saved float64 tree scores.
- Forest share = sum(base_sse - gat_sse) for FIPS42053 divided by the same sum over all239 counties, at1h.

Table 2 and Section 4.2 (under versions/xyy/v1.5.8_strict_baseline):
- d_unknown_contribution/k_positive_switch/summary/primary_metrics.csv: G vs A, not G vs the lower-bound projection B.
- p_initial_state_structure/confirmation/summary/primary_scores.csv: E2 vs E0; direct squared-loss control E2 vs E1.
- p_information_increment/confirmation/summary/primary_scores.csv: F2 minus F0.
- p_structure_horizon_transfer/confirmation/summary/primary_scores.csv: AB_P06 vs B0.
- p_neighbor_horizon_increment/confirmation/summary/primary_scores.csv: NB_P24 vs B0.
- nr_two_stage/runs/v18_nr2stage_v1_split42_model42/metrics/comparisons.csv and component_metrics.csv: seed42-only alternative, not retained.

Table 3 and Section 4.3:
- versions/stella_v112_strict/runs/stella_v112_nested_v1_split{42,20260917,20260918}_model42/summary_metrics.csv: L0/L1/L2 correspond to reference/average/gated ensemble, using24h and48h only.
- versions/stella_v112_strict/三分折复现结论.md: gate coverage and cross-partition directions.
- versions/xyy/v1.5.8_strict_baseline/tree_stella_integration/summary/comparisons.csv: I3 minus I1 for24h gains over enhanced T; I3 minus I2 for comparison with S alone.

Compression options for the final six-page report: integrate Table3 into prose; shorten Table2 references after the baseline definitions are established; move detailed county examples to Discussion. Keep the two evaluation scopes and the matched GAT comparison explicit.
-->
