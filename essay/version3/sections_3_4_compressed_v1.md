## 3. Methodology

### 3.1 Framework Overview

Our framework first constructs a tree-based OSI forecast from definition-guided component models. It then combines forecasts according to horizon: 1h and 6h predictions aggregate component estimates that refer to the same target hour, whereas 24h and 48h use horizon-specific sources and tail-gated ensembles. Finally, a graph attention network (GAT) learns a spatial residual correction to the complete tree forecast. A separate model is fitted for each horizon. Figure 1 summarizes the framework.

<!-- Insert Figure 1 here. -->

### 3.2 Definition-Guided Component Forecasting

LightGBM[^lightgbm] models the four OSI components from the tabular features in Section 2. Direct component models use a Huber objective. For $P$ and $D$, we also exploit information fixed at the observation cutoff.

Let $p_i=P_{i,71}$ be the last observed outage fraction for county $i$. We represent future outages by the retained portion of this initial level and the excess above it:

$$
a_{i,s}=\frac{\min(P_{i,s},p_i)}{p_i},\qquad
b_{i,s}=\frac{\max(P_{i,s}-p_i,0)}{1-p_i},
$$

$$
\widehat P_{i,s}
=p_i\operatorname{clip}(\widehat a_{i,s},0,1)
+(1-p_i)\operatorname{clip}(\widehat b_{i,s},0,1).
$$

Each branch is fitted only where its denominator is nonzero. Its squared-error loss is weighted by the square of its contribution scale, $p_i^2$ or $(1-p_i)^2$. This reconstruction is used for the 1h and 6h $P$ sources.

Because $D$ is a six-hour rolling mean of $P$, its window can partly overlap the observed period. We write

$$
D_{i,s}=K_{i,s}+U_{i,s},\qquad
K_{i,s}=\frac{1}{6}\sum_{u=s-5}^{\min(s,71)}P_{i,u},
$$

where an empty sum is zero, and learn only the unknown contribution $U_{i,s}$. This reconstruction is used for the 1h $D$ source when $K_{i,s}>0$; otherwise, $D$ is predicted directly. All $N$ and $R$ sources also use direct regression.

The final tree configuration adds the 20 neighborhood features from Section 2.3 to the 1h and 24h $P$ sources. Other component models use the 163 base features. Component predictions are clipped to $[0,1]$ before OSI reconstruction.

### 3.3 Horizon-Specific Forecast Combination

For 1h and 6h, several source horizons can predict the same county and target hour from different origins. We average all eligible predictions of each component at that target hour, then reconstruct OSI from the four averaged components. The 24h and 48h component forecasts instead retain their horizon-specific predictions.

For the two longer horizons, LightGBM, XGBoost[^xgboost], and CatBoost[^catboost] each provide a direct OSI forecast and a component-based forecast. Their equal-weight mean is used except in the upper tail, where averaging can dilute large reference predictions. Specifically, the LightGBM component forecast is retained when it exceeds the 95th percentile of its positive cross-fitted predictions. This threshold is estimated within each outer training fold.

The complete tree forecast therefore uses target-time component aggregation at 1h and 6h, the average of the enhanced component forecast and gated ensemble at 24h, and the gated ensemble at 48h. All OSI outputs are clipped to the training range $[0,0.65]$, with values below $0.001$ set to zero. The resulting forecast is also supplied to the spatial model.

### 3.4 Spatial Residual Refinement

An independent GAT[^gat] is trained for each horizon to predict errors remaining after the complete tree forecast. The residual targets use cross-fitted base predictions, so a county's target never enters its own reference prediction.

Each node combines the base forecast with the 163 base features, coordinates, terrain descriptors, and neighborhood summaries described in Section 2.3. Two edge-aware attention layers use the fixed county graph and its distance, bearing, and shared-border attributes. All counties participate in message passing, while the loss uses valid targets from training counties only.

For horizon $h$, the final forecast is

$$
\widehat y_{i,t}^{h}
=H\!\left(\widehat y_{i,t}^{\mathrm{base},h}
+\alpha_h\widehat r_{i,t}^{h}\right),
$$

where $\widehat r_{i,t}^{h}$ is the predicted residual and $H$ applies the OSI postprocessing above. Inner validation selects one correction strength $\alpha_h$ per horizon; $\alpha_h=0$ recovers the complete tree forecast.

### 3.5 Training and Validation

Tree configurations are evaluated on three fixed five-fold county partitions balanced by state and organizer-provided severity tier. Each county's full trajectory remains in one fold, and all candidates use identical assignments. Within each outer training set, inner county splits select boosting rounds before the model is refitted on all outer training counties. Gate thresholds and residual targets are generated from internal cross-fitted predictions, preserving the same county-level separation throughout model selection.

Spatial refinement is evaluated against its matching tree forecast. Inner GAT predictions select the residual correction strength before the final model is fitted on the complete outer training set. RMSE is pooled over all valid county-hour predictions separately by horizon. Comparisons use identical observations, and uncertainty is estimated by paired bootstrap resampling of whole county trajectories. Final models are fitted on all 239 training counties.

[^lightgbm]: Ke, G., et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). *NeurIPS*.

[^xgboost]: Chen, T., and Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *KDD*.

[^catboost]: Prokhorenkova, L., et al. (2018). [CatBoost: Unbiased Boosting with Categorical Features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html). *NeurIPS*.

[^gat]: Veličković, P., et al. (2018). [Graph Attention Networks](https://arxiv.org/abs/1710.10903). *ICLR*.


## 4. Discussion

County-level out-of-fold validation supports the tree configuration more consistently than the spatial refinement. Across three fixed partitions, the complete tree forecast reduced reference RMSE by 9.17%, 5.65%, 1.51%, and 0.66% at 1h, 6h, 24h, and 48h, improving every horizon in every partition. On the seed42 partition, spatial refinement reduced pooled RMSE by 0.55%–2.38%, but all county-bootstrap 95% intervals included zero and the gains were uneven. These comparisons support model selection rather than estimate hidden-test performance.

**Definition-aware modelling was most useful when the target was anchored by observed state.** The last observed outage level scales $P$, while the observed portion of the rolling window contributes directly to $D$. Comparable structure did not materially improve the sparse flow components $N$ and $R$. Component models should therefore follow both the variable definition and their effect on reconstructed OSI rather than treat all four components symmetrically.

**Forecast averaging introduced a horizon-dependent tradeoff.** Common-target-time aggregation combined complementary views at 1h and 6h but did not transfer to longer horizons. At 24h and 48h, unconditional averaging could attenuate large predictions; tail gating preserved these cases while retaining the ensemble elsewhere. Combinations should therefore be selected by final OSI error rather than any single source metric.

**The spatial correction remains promising but not conclusive.** Pooled gains coexisted with deterioration in some folds and counties and were concentrated in a small number of counties. Because the comparison changes geographic inputs, neighborhood summaries, and the graph learner together, it does not isolate message passing. That would require a matched non-graph model with the same inputs.

Finally, validation on held-out counties within one storm does not establish transfer to other events or regions. Operational use would replace the fixed outage cutoff and full-event weather fields with rolling observations and issue-time forecasts, requiring updates to the initial state for $P$, the known contribution to $D$, and the derived features. The evidence is therefore strongest for the tree framework within the competition's information setting.
