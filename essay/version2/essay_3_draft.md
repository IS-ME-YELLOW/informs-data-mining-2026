## 3. Methodology

### 3.1 Framework Overview

Our framework first uses definition-guided component models to construct a tree-based forecast. Forecasts are then combined according to horizon: 1h and 6h predictions aggregate component estimates for a common target hour, while 24h and 48h use horizon-specific sources and tail-gated ensembles. Finally, a graph attention network (GAT) learns residual corrections from the tree forecast and geographic context. A separate model is fitted for each horizon. Figure 1 summarizes the framework.

<!-- Insert Figure 1 here. -->

### 3.2 Definition-Guided Component Forecasting

LightGBM[^lightgbm] models each OSI component from the tabular features in Section 2. Direct component regressors use a Huber objective. We introduce dedicated formulations for $P$ and $D$, whose definitions expose information known at the observation cutoff.

**Initial-state reconstruction of $P$.** Let $p_i=P_{i,71}$ be the last observed outage fraction for county $i$. Two targets describe the portion of future outages within this initial level and the excess above it:

$$
a_{i,s}=\frac{\min(P_{i,s},p_i)}{p_i}\quad(p_i>0),
\qquad
b_{i,s}=\frac{\max(P_{i,s}-p_i,0)}{1-p_i}\quad(p_i<1),
$$

$$
\widehat P_{i,s}
=p_i\operatorname{clip}(\widehat a_{i,s},0,1)
+(1-p_i)\operatorname{clip}(\widehat b_{i,s},0,1).
$$

Each branch is fitted only on its valid support; when $p_i$ is zero or one, only the contributing branch is used. Squared-error losses are weighted by $p_i^2$ for $a$ and $(1-p_i)^2$ for $b$, then normalized within each fitting sample. These weights match each branch's contribution scale in the reconstructed $P$. This formulation is used for the 1h and 6h sources.

**Known-history reconstruction of $D$.** Because $D$ is a six-hour rolling mean of $P$, part of its value is known when its window overlaps the observed period:

$$
D_{i,s}=K_{i,s}+U_{i,s},
\qquad
K_{i,s}=\frac{1}{6}\sum_{u=s-5}^{\min(s,71)}P_{i,u},
$$

where an empty sum is zero. A Huber regressor learns the unknown contribution $U_{i,s}=D_{i,s}-K_{i,s}$ from all valid 1h samples. When $K_{i,s}>0$, prediction adds its nonnegative estimate to $K_{i,s}$ and clips the result to $[0,1]$; otherwise, a direct $D$ estimate is used. The reconstruction applies to target hours 73–76. Other $D$ models, and all $N$ and $R$ models, use direct regression with outputs clipped to $[0,1]$.

The final configuration applies initial-state reconstruction to the 1h and 6h $P$ sources and adds the 20 spatial features from Section 2.3 to the 1h and 24h $P$ sources. Known-history reconstruction is used only for 1h $D$; all remaining component sources use the 163 base features and direct regression.

### 3.3 Horizon-Specific Forecast Combination

All OSI sources use the same fixed postprocessing function $H$: predictions are clipped to the training target range $[0,0.65]$, and values below $0.001$ are set to zero.

**Common-target-time aggregation.** Several source horizons can predict the same county and target hour from different origins. Let $\widehat C_{i,s}^{(h)}$ be the clipped forecast of component $C$ from origin $s-h$. Eligible sources are averaged as

$$
\overline C_{i,s}
=\frac{1}{|\mathcal H_s|}\sum_{h\in\mathcal H_s}\widehat C_{i,s}^{(h)},
\qquad
\mathcal H_s=\{h\in\{1,6,24,48\}:72\le s-h\le215\},
$$

for $C\in\{P,N,D,R\}$ and valid target hours $73\le s\le215$. Applying the OSI formula and $H$ to the averaged components gives the 1h and 6h tree forecasts, $T^1$ and $T^6$. The 24h and 48h forecasts, $T^{24}$ and $T^{48}$, use their own horizon-specific components.

**Long-horizon ensembles and tail gating.** For 24h and 48h, LightGBM, XGBoost[^xgboost], and CatBoost[^catboost] each provide a direct OSI source and a component-based source, giving six forecasts $v_{m,h}$. Their postprocessed equal-weight mean is $E_h$. To avoid diluting large reference predictions, the gate retains the LightGBM component source $B_h$, which directly regresses $P$, $N$, $D$, and $R$, above an upper-tail threshold:

$$
E_h=H\!\left(\frac{1}{6}\sum_{m=1}^{6}v_{m,h}\right),
\qquad
S_h=
\begin{cases}
B_h,&B_h>\theta_{q,h},\\
E_h,&B_h\le\theta_{q,h}.
\end{cases}
$$

The threshold $\theta_{q,h}$ is the 95th percentile of positive, postprocessed reference predictions obtained by cross-fitting within outer-fold $q$'s training counties. The quantile level and gating direction are fixed; only the threshold value is re-estimated for each fold and horizon.

For a prediction issued at origin $t$, the complete tree forecast is

$$
\widehat y_{i,t}^{\mathrm{base},h}=
\begin{cases}
T_{i,t}^{h},&h\in\{1,6\},\\
H\!\left((T_{i,t}^{24}+S_{i,t}^{24})/2\right),&h=24,\\
S_{i,t}^{48},&h=48.
\end{cases}
$$

This composition is fixed across counties, dates, and validation splits. Its output becomes both the reference prediction and a node feature for spatial refinement.

### 3.4 Spatial Residual Refinement

An independent GAT[^gat] is trained for each horizon to predict errors remaining after the complete tree forecast. Training residuals use base predictions cross-fitted within the current training-county set $\mathcal C$, preventing a county's target from entering its own reference prediction. Residuals are divided by $\rho_{\mathcal C,h}$, the larger of 0.003 and their standard deviation.

Each node has 205 inputs: the base forecast, 163 base features, two coordinates, seven terrain descriptors, and 32 neighborhood summaries. Inputs are standardized using the training counties and horizon-valid times, then clipped to $[-5,5]$. Two edge-aware attention layers process each hourly snapshot, followed by a linear skip connection and a multilayer perceptron. All 302 counties participate in message passing, while the loss uses valid targets from training counties only.

For GAT function $f_{\phi_h}$, graph $\mathcal G$, edge attributes $\mathbf Q$, and standardized inputs $\widetilde{\mathbf X}_t^h$, the final prediction is

$$
r_{i,t}^{h}=y_{i,t+h}-\widehat y_{i,t}^{\mathrm{base},h},
\qquad
\widehat r_{i,t}^{h}
=\rho_{\mathcal C,h}\bigl[f_{\phi_h}(\widetilde{\mathbf X}_t^h,\mathcal G,\mathbf Q)\bigr]_i,
\qquad
\widehat y_{i,t}^{h}
=H\!\left(\widehat y_{i,t}^{\mathrm{base},h}+\alpha_{\mathcal C,h}\widehat r_{i,t}^{h}\right).
$$

Inner validation selects one correction strength per horizon from $\{0,0.05,0.10,0.20,0.35,0.50,0.75,1\}$, with ties favoring the smaller value. The coefficient is shared across counties and times within a fitted model; $\alpha=0$ recovers the complete tree forecast.

### 3.5 Training and Validation

Tree configurations are evaluated with three fixed five-fold county partitions balanced by state and organizer-provided severity tier. Each county's complete trajectory belongs to one fold, and all candidates share the same assignments. Spatial refinement is evaluated on the seed42 partition against its matching tree forecast.

Within each outer training set, four inner validation runs select boosting rounds by RMSE before refitting on all outer training counties. Gate thresholds are computed from additional internal cross-fitted reference predictions, so each prediction excludes its county from fitting and early stopping. For residual learning, every supervised county likewise receives a base prediction fitted without that county's fold.

The GAT minimizes normalized residual MSE using AdamW. Checkpointing uses dropout-free MSE on supervised training nodes. Separately, four inner GAT fits generate held-out predictions for selecting $\alpha$; the final GAT is then fitted on all outer training counties and applied to the outer fold.

RMSE pools all valid outer-fold county-hour predictions separately by horizon. Comparisons use identical observations, and uncertainty is estimated with 2,000 paired bootstrap samples of whole county trajectories. Tree summaries average the three partition-level pooled RMSEs; spatial comparisons use the shared seed42 partition. Final models are fitted on all 239 training counties.

[^lightgbm]: Ke, G., et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). *NeurIPS*.

[^xgboost]: Chen, T., and Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *KDD*.

[^catboost]: Prokhorenkova, L., et al. (2018). [CatBoost: Unbiased Boosting with Categorical Features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html). *NeurIPS*.

[^gat]: Veličković, P., et al. (2018). [Graph Attention Networks](https://arxiv.org/abs/1710.10903). *ICLR*.
