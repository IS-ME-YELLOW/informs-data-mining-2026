## 3. Methodology

### 3.1 Framework Overview

Our framework combines three stages. First, component models incorporate the observed initial outage state and the definitions of the OSI components. Second, forecasts are combined according to horizon: short-term predictions aggregate component estimates for a common target hour, while long-term predictions use heterogeneous ensembles with prediction-tail gating. These outputs form a complete tree-based forecast. Third, a graph attention model learns residual corrections from this forecast and the geographic context described in Section 2. Internal validation selects the correction strength separately for each horizon.

### 3.2 Component Forecasting with Initial-State Conditioning

We use LightGBM[^lightgbm] to model each OSI component and horizon from the tabular features in Section 2. Direct component regressors use a Huber objective. We introduce dedicated formulations for $P$ and $D$ to incorporate initial conditions and known historical contributions. The following expressions describe one source horizon at a time, using the county and target-time notation from Section 2.

**Initial-state reconstruction of $P$.** Counties enter the prediction window with different outage levels. We express the future outage fraction relative to the last observed level $p_i=P_{i,71}$ through two training targets:

$$
a_{i,s}=\frac{\min(P_{i,s},p_i)}{p_i}\quad(p_i>0),
\qquad
b_{i,s}=\frac{\max(P_{i,s}-p_i,0)}{1-p_i}\quad(p_i<1).
$$

The branches describe the normalized portion within the initial level and the normalized excess above it. This representation makes the observed initial level explicit in the output, while allowing subsequent severity to decrease or increase. Separate regressors estimate these quantities, which are reconstructed as

$$
\widehat P_{i,s}
=p_i\operatorname{clip}(\widehat a_{i,s},0,1)
+(1-p_i)\operatorname{clip}(\widehat b_{i,s},0,1).
$$

Each branch is fitted on its valid support; when $p_i$ is zero or one, only the contributing branch is used. Both regressors use squared-error loss with weights $w_{a,i}=p_i^2$ and $w_{b,i}=(1-p_i)^2$, normalized to mean one within each fitting sample. These weights reflect each branch's contribution scale in the reconstructed outage fraction. Inner validation evaluates errors on that scale before clipping. This formulation is used for the 1h and 6h source models.

**Known-history reconstruction of $D$.** Its six-hour rolling definition provides a known contribution whenever the averaging window overlaps the observation period:

$$
D_{i,s}=K_{i,s}+U_{i,s},
\qquad
K_{i,s}=\frac{1}{6}\sum_{u=s-5}^{\min(s,71)}P_{i,u},
$$

with an empty sum equal to zero. A Huber regressor learns $U_{i,s}=D_{i,s}-K_{i,s}$ from all valid 1h training samples. At prediction time,

$$
\widehat D_{i,s}=
\begin{cases}
\min\!\left(1,K_{i,s}+\max(0,\widehat U_{i,s})\right),&K_{i,s}>0,\\
\widehat D^{\mathrm{dir}}_{i,s},&K_{i,s}=0,
\end{cases}
$$

where the direct estimate is clipped to $[0,1]$. Reconstruction applies only where the known contribution is positive, within target hours 73–76. Other target hours and longer-horizon $D$ models use direct regression. Separate direct regressors predict $N$ and $R$, with outputs clipped to $[0,1]$.

The final component configuration is summarized below. Horizons identify source models before target-time aggregation; the 20 additional spatial features are defined in Section 2.4.

| Component source | Prediction structure | Inputs |
|---|---|---|
| $P$, 1h | Initial-state reconstruction | 163 base + 20 spatial |
| $P$, 6h | Initial-state reconstruction | 163 base |
| $P$, 24h | Direct regression | 163 base + 20 spatial |
| $P$, 48h | Direct regression | 163 base |
| $D$, 1h | Known-history reconstruction when $K>0$; direct otherwise | 163 base |
| $D$, 6/24/48h; $N/R$, all horizons | Direct regression | 163 base |

### 3.3 Target-Time Alignment and Tail-Gated Ensembles

We use a fixed postprocessing function $H$: clip OSI predictions to $[0,0.65]$, then set values strictly below $0.001$ to zero. The component forecasts and ensemble sources are processed in the order specified below.

**Common-target-time aggregation.** Different source horizons can predict the same county and target hour: for example, hour 120 is predicted by the 1h model at origin 119 and the 6h model at origin 114. All models share the initial outage history and common feature construction, including all four weather-window spans. For a common target, their different origins produce different weather-window and temporal representations.

Let $\widehat C_{i,s}^{(h)}$ denote a component forecast from origin $s-h$, clipped to $[0,1]$. We average the eligible sources:

$$
\overline C_{i,s}
=\frac{1}{|\mathcal H_s|}\sum_{h\in\mathcal H_s}\widehat C_{i,s}^{(h)},
\qquad
\mathcal H_s=\{h\in\{1,6,24,48\}:72\le s-h\le215\},
$$

for $C\in\{P,N,D,R\}$ and valid target hours $73\le s\le215$. Applying the OSI formula and $H$ to these aggregated components yields the 1h and 6h forecasts, denoted $T^1$ and $T^6$. The long-term forecasts $T^{24}$ and $T^{48}$ instead use their respective horizon-specific components, followed by OSI reconstruction and $H$.

**Long-term ensembles and tail gating.** For 24h and 48h, LightGBM, XGBoost[^xgboost], and CatBoost[^catboost] each provide a direct OSI regressor and a component-based OSI source, all using the 163 base features. These component sources regress all four components directly. Training uses Huber or pseudo-Huber objectives, with RMSE for early stopping.

Each direct OSI source is processed by $H$. Each component source clips its four estimates to $[0,1]$, reconstructs OSI, and applies $H$. Denoting these six processed sources by $v_{m,h}$, their equal-weight ensemble is $E_h=H(\frac{1}{6}\sum_{m=1}^{6}v_{m,h})$.

Earlier comparisons motivated retaining a reference forecast in the upper prediction tail, where unconditional averaging could increase large errors. The reference $B_h$ is the LightGBM source that directly regresses all four components using the 163 base features. For outer fold $q$, the gated forecast is

$$
S_h=
\begin{cases}
B_h,&B_h>\theta_{q,h},\\
E_h,&B_h\le\theta_{q,h}.
\end{cases}
$$

The threshold is the 95th percentile of positive, postprocessed reference predictions obtained by cross-fitting within the outer training counties. The quantile level and gating direction are fixed; the numerical threshold is estimated separately for each fold and horizon. The gate therefore operates on predicted severity. Section 3.5 describes its internal fitting procedure.

**Base forecast composition.** Restoring county and origin indices, the complete tree forecast is

$$
\widehat y_{i,t}^{\mathrm{base},h}=
\begin{cases}
T_{i,t}^{h},&h\in\{1,6\},\\
H\!\left((T_{i,t}^{24}+S_{i,t}^{24})/2\right),&h=24,\\
S_{i,t}^{48},&h=48.
\end{cases}
$$

This composition is fixed across counties, dates, and validation splits. Its output supplies both a node feature and the reference prediction for residual learning. The 24h average and subsequent spatial correction can modify predictions retained by the ensemble gate.

### 3.4 Spatial Residual Refinement

An independent graph attention network[^gat] is trained for each horizon to refine the complete tree forecast. The residual formulation makes the tree prediction the starting point and directs the graph model toward its remaining errors, using the local and regional context in the node inputs. Writing $y_{i,s}=\mathrm{OSI}_{i,s}$, its supervision is

$$
r_{i,t}^{h}=y_{i,t+h}-\widehat y_{i,t}^{\mathrm{base},h},
\qquad
\widetilde r_{i,t}^{h}=r_{i,t}^{h}/\rho_{\mathcal C,h},
$$

where training residuals use base predictions cross-fitted within the current training county set $\mathcal C$. The scale $\rho_{\mathcal C,h}$ is the larger of 0.003 and the standard deviation of valid supervised residuals.

Each node has 205 inputs: the base forecast, 163 base features, two coordinates, seven terrain descriptors, and 32 neighborhood summaries. Features are standardized using the training counties and horizon-valid times, then clipped to $[-5,5]$. The graph and edge attributes follow Section 2.4.

Two attention layers process each hourly snapshot: four concatenated 24-dimensional heads followed by one 24-dimensional head, with ELU activations. Attention uses node representations and edge attributes. A linear skip connection projects the standardized node input into the learned representation, and a multilayer perceptron outputs a normalized residual. All 302 counties participate in message passing; the loss uses valid targets from training counties.

For standardized inputs $\widetilde{\mathbf X}_t^h$, graph $\mathcal G$, and edge attributes $\mathbf Q$, the final forecast is

$$
\widehat r_{i,t}^{h}
=\rho_{\mathcal C,h}\bigl[f_{\phi_h}(\widetilde{\mathbf X}_t^h,\mathcal G,\mathbf Q)\bigr]_i,
\qquad
\widehat y_{i,t}^{h}
=H\!\left(\widehat y_{i,t}^{\mathrm{base},h}+\alpha_{\mathcal C,h}\widehat r_{i,t}^{h}\right).
$$

Inner validation selects $\alpha_{\mathcal C,h}$ to minimize postprocessed OSI RMSE over $\{0,0.05,0.10,0.20,0.35,0.50,0.75,1\}$. The selected coefficient is uniform across counties and times for each fitted model; zero recovers the base forecast.

### 3.5 Training and Evaluation Protocol

Tree configurations are evaluated with three fixed five-fold county partitions, balanced by state and organizer-provided severity tier. Each county's complete trajectory belongs to one fold, and candidates share the same assignments. The spatial model is evaluated on the seed42 partition against its matching tree baseline. The architecture and forecast composition above are fixed; boosting rounds, numerical gate thresholds, and residual correction coefficients are estimated within the relevant training counties.

Within each outer training set, four inner validation runs select boosting rounds by RMSE. Their best iteration counts are averaged and rounded down, with a minimum of one, before refitting on all outer training counties. For the $P$ branches, validation uses the contribution scale defined in Section 3.2. Gate thresholds require additional internal cross-fitting: each reference prediction excludes its county's fold from both fitting and early stopping, and all such operations remain within the outer training set.

For residual learning, each supervised county receives a base prediction fitted on the remaining folds within the current training set; held-out and test counties receive predictions fitted on that full set. The GAT minimizes normalized residual MSE using AdamW. Checkpoint selection and stopping use dropout-free MSE on supervised training nodes. Separately, four inner GAT fits generate held-out predictions for selecting $\alpha$, with ties favoring the smaller coefficient. The GAT is then fitted on all outer training counties to predict the outer fold.

RMSE pools all valid outer-fold county-hour predictions separately for each horizon. Comparisons use identical observations, with 95% percentile intervals from 2,000 paired bootstrap samples of entire county trajectories. Tree summaries average the three partition-level pooled RMSEs; spatial comparisons use the shared seed42 partition. Section 4 reports the configuration comparisons and incremental effects. For final test prediction, the selected tree and graph models are fitted using all 239 training counties.

[^lightgbm]: Ke, G., et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). *NeurIPS*.

[^xgboost]: Chen, T., and Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *KDD*.

[^catboost]: Prokhorenkova, L., et al. (2018). [CatBoost: Unbiased Boosting with Categorical Features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html). *NeurIPS*.

[^gat]: Veličković, P., et al. (2018). [Graph Attention Networks](https://arxiv.org/abs/1710.10903). *ICLR*.
