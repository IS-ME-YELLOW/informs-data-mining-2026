## 4. Empirical Evaluation

We evaluate the predictive value of component reconstruction, horizon-specific forecast combination, and spatial residual refinement using the county-level out-of-fold protocol in Section 3.5. Comparisons hold the remaining pipeline fixed within each experiment. The results provide empirical support for the modelling choices; all scores below are validation results on training counties.

The component reference uses LightGBM with the 163 base features to regress all four components directly, followed by short-term target-time aggregation and horizon-specific long-term reconstruction. It provides a common reference for assessing the additional component formulations, spatial features, and ensembles. All configurations share OSI postprocessing.

**Table 1. OSI prediction RMSE. Panel A averages pooled RMSEs over three fixed county partitions. Panel B compares the complete tree forecast and final spatially refined model on the same seed42 partition. Lower values are better.**

**Panel A: Tree framework across three partitions**

| Configuration | 1h | 6h | 24h | 48h |
|---|---:|---:|---:|---:|
| Component reference | 0.01191130 | 0.01043621 | 0.00843269 | 0.00765768 |
| Complete tree forecast | **0.01081914** | **0.00984660** | **0.00830560** | **0.00760743** |

**Panel B: Spatial refinement on seed42**

| Configuration | 1h | 6h | 24h | 48h |
|---|---:|---:|---:|---:|
| Complete tree forecast | 0.01084069 | 0.00990906 | 0.00841429 | 0.00766153 |
| Final spatially refined model | **0.01061279** | **0.00985497** | **0.00821388** | **0.00749934** |
| RMSE reduction | **2.10%** | **0.55%** | **2.38%** | **2.12%** |

The complete tree forecast reduces the three-partition mean RMSE by 9.17%, 5.65%, 1.51%, and 0.66% at 1h, 6h, 24h, and 48h, respectively. Each partition shows improvement at every horizon, with the largest reductions at short horizons.

**Initial-state and component structure.** Before target-time aggregation, a controlled seed42 comparison using the same 163-feature LightGBM setup found lower RMSE at all four horizons when predicting components separately and reconstructing OSI than when directly predicting OSI. Further comparisons support incorporating initial conditions into the component models. Across three partitions, the initial-state formulation for the 1h $P$ source reduced final 1h RMSE by 4.52%–7.46% relative to direct $P$ regression with the same squared-error objective. This supports the combined branch decomposition, contribution weighting, and reconstruction beyond changing the loss alone. Known-history reconstruction of $D$ reduced 1h RMSE by 1.24%–1.45% relative to the component reference, despite acting only near the observation cutoff. Extending the $P$ formulation to the 6h source and adding neighbor features to the 1h and 24h $P$ sources yielded further improvements. These are incremental comparisons against their respective controls, so their reductions are not additive.

A seed42 comparison of occurrence–magnitude models for the N1 and R1 sources produced at most a 0.05% reduction in final 1h RMSE, obtained by replacing N1 alone. The two-stage models reduced source-component errors for zero and small positive targets but increased errors for official component values above 0.01, supporting retention of direct N and R regression.

**Target-time alignment and tail-gated combination.** In the controlled seed42 comparison, common-target-time aggregation reduced RMSE relative to horizon-specific component reconstruction by approximately 1.04% at 1h and 1.83% at 6h, while increasing 48h RMSE by 0.62%. This horizon dependence supports retaining separate long-term sources. For 24h and 48h, unconditional six-source averaging worsened RMSE relative to the reference component model in two of the three partitions. Tail gating improved on both the reference and the unconditional average in all three partitions at both horizons, with reductions over the reference of 0.56%–2.18% at 24h and 0.39%–0.99% at 48h. The gate retained the reference for approximately 2% of observations. At 24h, the fixed average of the enhanced component forecast and gated ensemble further improved on the enhanced component forecast in all three partitions, although it outperformed the gated ensemble alone in only two. These comparisons support the differentiated composition in Section 3.3.

**Spatial residual refinement.** The final spatial model lowers RMSE at every horizon relative to the matching complete tree forecast (Table 1, Panel B). The largest relative gain is 2.38% at 24h. MAE also decreases at all four horizons. RMSE improves in four of five outer folds at 24h and three of five folds at the other horizons. This comparison evaluates the complete spatial residual module, including geographic inputs, neighborhood summaries, and the graph learner.

**Variation and limitations.** All four county-bootstrap 95% intervals for the spatial model's RMSE differences include zero. Gains also vary substantially across counties: Forest County accounts for 97.3% of the net 1h SSE reduction after county-level gains and losses are offset. Thus, the observed pooled improvements coexist with concentrated benefits and deterioration elsewhere. The three tree partitions provide evidence across alternative county assignments within the same event; spatial refinement has been evaluated on one partition. Transfer to other storms remains untested.

<!--
Editorial source notes; not part of the manuscript.

The main results were reorganized from essay_4_draft.md. A brief N/R comparison was subsequently added to support Section 5 and checked against its saved metrics; no models or saved predictions were changed.

- Table 1, aggregate tree reductions, staged component changes, ensemble comparisons, spatial fold results, and county-level limitations are taken from essay_4_draft.md. Its source notes retain the detailed experiment references.
- The squared-error P control is the reported E2-versus-E1 comparison; its 4.52%–7.46% range is a final 1h OSI RMSE reduction, not a source-component metric.
- Direct OSI versus direct component prediction, and the seed42 target-time aggregation comparison, come from versions/xyy/v1.5.8_strict_baseline/Results_2026-09-17.md, Section 3. The reported C3-versus-C1 changes of 1.037%, 1.827%, and +0.623% are rounded to 1.04%, 1.83%, and +0.62% in the manuscript.
- The component reference in Table 1 already applies short-term aggregation. The direct-component comparator used to assess aggregation does not. The two comparisons remain distinct.
- Panel A and Panel B have different evaluation scopes. No percentage combines a three-partition tree mean with a seed42 spatial score.
- Historical staged comparisons are not presented as removing individual modules from the final model. A short N/R comparison supports the Discussion; detailed rejected alternatives and county case studies are omitted from this compact evaluation section.
- The N1/R1 occurrence–magnitude comparison comes from versions/xyy/v1.5.8_strict_baseline/nr_two_stage/Results_2026-09-21.md and its saved comparison/component metrics. The best1h point estimate is N_only, with a0.04963% reduction versus the matching complete enhanced tree forecast. This experiment used seed42 only.
-->
