## 3. Methodology

### 3.1 Overview of the Forecasting Framework

Our framework models all four OSI components separately and uses component-specific experiments to determine their prediction structures. We adopt initial-state reconstruction for *P* and incorporate known historical contributions into *D*, while retaining direct regression for *N* and *R*. This configuration follows comparisons of alternative formulations using component errors and overall OSI RMSE. We denote the resulting enhanced component framework’s OSI forecasts by *Tₕ*, where *h* is the forecast horizon.

We extend this component framework with models that predict OSI directly and with additional tree learners. These models provide alternative approximations to the same forecasting target. Their predictions are combined through averaging and a gate that retains the reference component model at high predicted severity, yielding the long-term ensemble forecasts *Sₕ*. The component forecasts and gated ensemble form a base tree prediction for each horizon. A spatial residual model then refines this prediction using learned interactions between neighboring counties. Section 4 reports the comparisons supporting the tree configuration and the additional spatial correction.

### 3.2 Component-Specific Forecasting

We use LightGBM[^lightgbm] as the common learner for the tabular features described in Section 2. Its boosted trees represent nonlinear responses and interactions among outage history, weather, and county characteristics, while supporting efficient repeated fitting for county-level validation. The initial component baseline fits a separate regressor for each component and horizon using the 163 base features and a Huber objective. It applies target-time aggregation for the 1-hour and 6-hour forecasts (Section 3.3), horizon-specific component reconstruction for the longer horizons, and the shared postprocessing in Section 3.5. The following modifications retain this backbone while changing selected component formulations and inputs.

For county *i*, prediction origin *t*, and horizon *h*, the target hour is

$$
s=t+h.
$$

Component formulations are motivated by their definitions and observed behavior, and evaluated through comparisons with direct regression. The following formulas describe one horizon-specific model at a time.

**Initial-state reconstruction for P.** Counties enter the prediction window with substantially different outage levels. This motivates learning normalized responses relative to the observed initial state and then reconstructing the absolute outage fraction. Using the last observed outage fraction at hour 71, with hours indexed from zero, we define the initial level and two training targets:

$$
p_i=P_{i,71},
\qquad
a_{i,s}=\frac{\min(P_{i,s},p_i)}{p_i}
\quad (p_i>0),
\qquad
b_{i,s}=\frac{\max(P_{i,s}-p_i,0)}{1-p_i}
\quad (p_i<1).
$$

The first represents the normalized portion within the initial outage level; the second represents the normalized excess above it. Separate regression models estimate these quantities, and their predictions are reconstructed as

$$
\widehat P_{i,s}
=
p_i\,\operatorname{clip}(\widehat a_{i,s},0,1)
+
(1-p_i)\,\operatorname{clip}(\widehat b_{i,s},0,1).
$$

The observed initial level thus enters the output explicitly, while the learned branches allow outage severity to decrease or increase. The *a* model is fitted on rows with a positive initial outage fraction, and the *b* model on rows with an initial fraction below one. When the initial fraction is zero or one, reconstruction uses only the contributing branch.

Both models use squared-error loss, with branch weights

$$
w_{a,i}=p_i^2,
\qquad
w_{b,i}=(1-p_i)^2.
$$

These weights reflect how each branch’s error scales in the reconstructed *P* and are normalized to mean one within each model’s training support. Inner validation evaluates branch errors on the same contribution scale before clipping. The retained configuration uses this structure for the 1-hour and 6-hour *P* models and direct regression for the 24-hour and 48-hour models.

**Known-history reconstruction for *D*.** The six-hour rolling definition of *D* provides a computable contribution whenever its averaging window overlaps the observation period. We separate this known contribution from the remaining quantity to be predicted:

$$
D_{i,s}=K_{i,s}+U_{i,s},
\qquad
K_{i,s}
=
\frac{1}{6}
\sum_{u=s-5}^{\min(s,71)}P_{i,u},
$$

where an empty sum is zero. A LightGBM model with the baseline Huber objective learns the unknown contribution, obtained by subtracting the known contribution from the official *D* label, using all valid 1-hour training samples. The retained prediction rule is

$$
\widehat D_{i,s}
=
\begin{cases}
\min\!\left(1,K_{i,s}+\max(0,\widehat U_{i,s})\right),
& K_{i,s}>0,\\[2mm]
\widehat D^{\mathrm{dir}}_{i,s},
& K_{i,s}=0,
\end{cases}
$$

Here, the direct prediction is clipped between zero and one. This incorporates known persistence at eligible target hours 73–76 and retains direct regression elsewhere. Longer-horizon *D* models also use direct regression.

**Direct prediction for *N* and *R*.** These components contain many zeros alongside relatively rare large values, motivating a comparison between direct regression and a two-stage occurrence–magnitude formulation. For the 1-hour source models, the latter combines the predicted probability of a positive component value with its estimated conditional magnitude. Its limited overall OSI gains supported retaining separate direct LightGBM regressors for *N* and *R* at all horizons, using the baseline Huber objective and outputs clipped to the interval from zero to one.

The table summarizes the enhanced component framework *T*, retained after controlled comparisons of component formulations and feature sets. Selection was guided by overall OSI RMSE, with the corresponding incremental comparisons and validation results presented in Section 4. Horizons below refer to individual source models, before the aggregation described in Section 3.3.

| Component source | Prediction structure | Input features |
|---|---|---|
| *P*, 1h | Initial-state reconstruction | 163 base + 20 spatial |
| *P*, 6h | Initial-state reconstruction | 163 base |
| *P*, 24h | Direct regression | 163 base + 20 spatial |
| *P*, 48h | Direct regression | 163 base |
| *D*, 1h | Historical reconstruction when the known contribution is positive; direct otherwise | 163 base |
| *D*, 6/24/48h; *N* and *R*, all horizons | Direct regression | 163 base |

### 3.3 Aggregation at a Common Target Time

Predictions from different origins and horizons can refer to the same county and target hour. For example, predictions for hour 120 are available from the 1-hour model at origin 119, the 6-hour model at origin 114, and the 24-hour and 48-hour models at origins 96 and 72. These forecasts share the same initial outage history and supplied weather series, but use different temporal and weather-window representations.

For each component, forecasts from eligible horizon models are aligned to the same target hour. We clip the component predictions between zero and one and compute

$$
\overline C_{i,s}
=
\frac{1}{|\mathcal H_s|}
\sum_{h\in\mathcal H_s}
\widehat C_{i,s}^{(h)},
\qquad C\in\{P,N,D,R\}.
$$

The averaging set contains the horizons whose corresponding origins fall within the prediction window. Thus, the number of contributing sources increases as more horizons become available for a target hour.

For the enhanced component framework, the four aggregated components are combined using the OSI formula and then postprocessed to obtain *T₁* and *T₆*. Its 24-hour and 48-hour forecasts, *T₂₄* and *T₄₈*, use their respective horizon-specific component predictions. The initial component baseline uses the same aggregation and postprocessing rules with its original direct component regressors. This arrangement allows short-term forecasts to draw on multiple representations of the same target while preserving the individual long-term sources for subsequent ensemble composition.

### 3.4 Heterogeneous Ensembles with Prediction-Tail Gating

For the 24-hour and 48-hour horizons, we combine models that predict OSI directly with models that reconstruct it from its components. Preliminary ensemble comparisons showed that lower mean absolute error could coexist with larger errors at high outage severity. This motivated a combination of equal-weight averaging and a gate that retains a reference component model in the upper prediction tail.

We use LightGBM, XGBoost[^xgboost], and CatBoost[^catboost], each trained on the same 163 base features. Each learner provides two prediction sources: a direct OSI regressor and four component regressors whose outputs are combined through the official OSI formula. The component sources here use direct regression for all four targets. Together, these yield six OSI predictions per horizon. The learners use Huber or pseudo-Huber training objectives, with RMSE guiding early stopping; training details are provided in Section 3.7.

Each direct OSI prediction is postprocessed. For each component source, the four predictions are first clipped between zero and one, combined into OSI, and then postprocessed. Indexing the six processed sources by *m* and suppressing county and time indices for brevity, the equal-weight ensemble is

$$
E_h
=
H\!\left(
\frac{1}{6}\sum_{m=1}^{6}v_{m,h}
\right),
$$

where *H* is the output-processing function defined in Section 3.5. Fixed equal weights provide a simple combination while limiting additional weight-selection parameters.

The reference is the LightGBM component prediction, which corresponds to the initial component baseline at these two long-term horizons. For outer validation fold *q*, the gated ensemble is

$$
B_h=v_{\mathrm{LightGBM,component},h},
\qquad
S_h=
\begin{cases}
B_h, & B_h>\theta_{q,h},\\
E_h, & B_h\leq\theta_{q,h}.
\end{cases}
$$

This rule preserves the reference prediction at high predicted severity and uses the ensemble elsewhere. Both the gate and its threshold operate on predicted OSI.

The threshold is the 95th percentile of positive, postprocessed reference predictions generated by inner cross-fitting within the outer training counties:

$$
\theta_{q,h}
=
Q_{0.95}
\left(
\left\{
\widetilde B_{j,h}^{(-q)}
:
\widetilde B_{j,h}^{(-q)}>0
\right\}
\right),
$$

where *j* indexes valid county-hour samples. Each inner prediction is generated by a model whose fitting and early stopping exclude the predicted county’s fold. The quantile level is fixed at 0.95, while its numerical threshold is estimated separately for each outer fold and horizon. Section 3.7 details this nested procedure, and Section 4 compares the reference model, unconditional average, and gated ensemble.

### 3.5 Tree Forecast Composition

The enhanced component framework and gated ensemble form the base prediction for spatial residual learning. For the 1-hour and 6-hour horizons, we use the component forecasts obtained through the target-time aggregation in Section 3.3. For the longer horizons, we combine the enhanced component forecast with the gated ensemble. Restoring the county and prediction-origin indices, the base forecast is

$$
\widehat y_{i,t}^{\mathrm{base},h}
=
\begin{cases}
T_{i,t}^{h}, & h\in\{1,6\},\\
H\!\left(\lambda_h T_{i,t}^{h}+(1-\lambda_h)S_{i,t}^{h}\right),
& h\in\{24,48\},
\end{cases}
\qquad
\lambda_{24}=0.5,\quad\lambda_{48}=0.
$$

Thus, the 24-hour base forecast averages the two tree approaches, while the 48-hour base forecast uses the gated ensemble. These mixing weights are fixed across all counties, dates, and cross-validation splits. This processed base forecast is both an input to the spatial model and the reference for its residual targets.

The shared postprocessing function *H* first clips OSI predictions between 0 and 0.65 and then sets values strictly below 0.001 to zero. It is applied after the long-term tree combination, following the source-level processing in Section 3.4, and again after the spatial correction in Section 3.6. Section 4 presents the comparisons supporting these stages.

### 3.6 Spatial Residual Learning

We train an independent graph attention network[^gat] for each forecast horizon to refine the complete tree-ensemble prediction. Residual supervision uses baseline predictions cross-fitted within the corresponding training county scope:

$$
r_{i,t}^{h}
=
y_{i,t+h}
-
\widehat y_{i,t}^{\mathrm{base},h},
\qquad
\widetilde r_{i,t}^{h}
=
\frac{r_{i,t}^{h}}{\rho_{\mathcal C,h}}.
$$

The scope index identifies the current training counties. The residual scale is the larger of 0.003 and the standard deviation of their valid supervised residuals. Node features are standardized using statistics from the same training counties and horizon-valid times, then clipped between −5 and 5 to improve robustness to extreme inputs. This limits their influence during spatial aggregation.

Each hourly snapshot is processed by two graph attention layers. The first concatenates four attention heads with 24 dimensions per head, and the second produces a 24-dimensional representation using one head. Attention weights depend on the connected counties’ representations and their edge attributes, allowing the model to learn how neighboring conditions contribute to the correction. ELU activations follow the two graph layers. A linear skip connection adds a projection of the standardized and clipped node input to the learned representation, and a multilayer perceptron produces a scalar normalized residual. All county nodes participate in message passing, while the MSE loss uses valid targets from the training counties.

The predicted residual is restored to the original OSI scale and added to the base forecast:

$$
\begin{aligned}
\widehat r_{i,t}^{h}
&=
\rho_{\mathcal C,h}
\left[
f_{\phi_h}
\left(
\widetilde{\mathbf X}_{t}^{h},
\mathcal G,
\mathbf Q
\right)
\right]_i,\\
\widehat y_{i,t}^{h}
&=
H\!\left(
\widehat y_{i,t}^{\mathrm{base},h}
+
\alpha_{\mathcal C,h}\widehat r_{i,t}^{h}
\right).
\end{aligned}
$$

Here, the network receives the standardized and clipped node features, fixed graph, and edge attributes described in Section 2.4. The correction coefficient is selected by inner cross-validation to minimize the postprocessed OSI RMSE over the fixed grid

$$
\mathcal A
=
\{0,\;0.05,\;0.10,\;0.20,\;0.35,\;0.50,\;0.75,\;1.00\}.
$$

For each fitted horizon model, the selected coefficient is applied uniformly across counties and times. The shared postprocessing function is applied after correction. Section 3.7 describes the training and coefficient-selection procedures.

### 3.7 Model Training and Validation

We assess the initial component baseline and retained tree configurations using three fixed repetitions of five-fold cross-validation over the 239 training counties. Folds are balanced by state and organizer-provided severity tier, with each county’s complete trajectory assigned to one fold. All tree prediction sources use identical county assignments within a repetition. Partition seeds are 42, 20260917, and 20260918, while the tree-model seed remains fixed at 42. The spatial extension is evaluated on the seed42 partition against its matching tree-ensemble forecasts.

For each outer fold, tree models are fitted using the remaining four folds. Boosting rounds are selected through four inner validation runs, each fitting on three folds and validating on the fourth. The mean best iteration count is rounded down, with a minimum of one, and the model is refitted on all four training folds using that fixed count. Validation uses RMSE, with the *P* branches evaluated on the contribution scale described in Section 3.2. Branch weights are normalized within each fitting sample. Tree learners use a learning rate of 0.05 and early-stopping patience of 100 rounds. Maximum iteration counts are 2,000 for LightGBM, 4,000 for CatBoost and direct XGBoost models, and 8,000 for XGBoost component models.

The tree gate thresholds require an additional cross-fitting step within each outer training set. One of its four folds is reserved for reference prediction, while the remaining three are used to construct the reference component models. Their boosting rounds are selected internally through two-fold fitting and one-fold validation, followed by refitting on all three folds. Repeating this process produces reference predictions for the four inner folds; their positive values determine the horizon-specific threshold defined in Section 3.4.

For spatial learning, base predictions are generated for each training scope using the complete tree pipeline. A supervised county receives a base prediction fitted on the remaining folds within that scope, while held-out and test nodes receive predictions fitted on the full current training scope. Input statistics and residual scales are estimated from that scope’s valid supervised observations. GAT training minimizes normalized residual MSE using AdamW with learning rate 0.003, weight decay 0.0002, and dropout 0.12. Checkpoint selection and stopping use dropout-free MSE on the supervised training nodes, with a maximum of 220 epochs and patience of 35 epochs. GAT initialization seeds are deterministically derived from the partition seed, training scope, and horizon.

Within each outer training set, four inner GAT fits produce held-out predictions for correction-weight selection. Each fit uses three folds and their corresponding base predictions. The coefficient minimizing pooled, postprocessed inner OSI RMSE is selected, with ties resolved toward the smaller coefficient. A GAT fitted on all four outer training folds then supplies the correction for the outer fold. For final inference, coefficient selection is performed within all five training folds, followed by GAT fitting on all training counties.

Performance is calculated from combined outer-fold predictions, pooling all valid county-hour observations separately for each horizon. Candidate and reference forecasts are compared on identical observations. We estimate 95% percentile intervals for RMSE differences using 2,000 paired bootstrap samples, resampling entire county trajectories and applying the same draws to both forecasts. Tree-only summaries use the arithmetic mean of the three partition-level pooled RMSEs; the spatial model and its tree reference are compared on the shared seed42 partition. Partition-level results, model configurations, and saved predictions are retained in the accompanying experiment records.

[^lightgbm]: Ke, G., et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). *NeurIPS*.

[^xgboost]: Chen, T., and Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *KDD*.

[^catboost]: Prokhorenkova, L., et al. (2018). [CatBoost: Unbiased Boosting with Categorical Features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html). *NeurIPS*.

[^gat]: Veličković, P., et al. (2018). [Graph Attention Networks](https://arxiv.org/abs/1710.10903). *ICLR*.
