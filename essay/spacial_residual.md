\subsection{Spatial Residual Learning 简略版}

For each forecast horizon $h\in\{1,6,24,48\}$, we train an independent
spatial residual learner to correct the frozen I3 prediction. Let
$\widehat{y}_{i,t}^{\mathrm{I3},h}$ denote the I3 prediction for county $i$
at forecast origin $t$. The residual target is defined as
\[
r_{i,t}^{h}
=
y_{i,t+h}^{h}
-
\widehat{y}_{i,t}^{\mathrm{I3},h},
\qquad
\widetilde r_{i,t}^{h}
=
\frac{r_{i,t}^{h}}{\rho_{S,h}},
\]
where $S$ is the training county scope and
\[
\rho_{S,h}
=
\max\left\{
\operatorname{Std}_{(i,t)\in\Omega_{S,h}}
\left(r_{i,t}^{h}\right),\,0.003
\right\}.
\]
Only valid county--time cells within the training scope are used for
supervision.

We construct a graph with all 302 counties as nodes. Its edges combine
symmetric 8-nearest-neighbor relations, shared county borders, and
self-loops. Each edge is associated with normalized distance, sine and
cosine bearing, and a shared-border indicator. The resulting 205-dimensional
node input combines the I3 prediction, 163 frozen features, county
coordinates, seven terrain variables, and 16 prespecified local-history and
weather variables with their neighbor means and neighbor-minus-own
differences.

For the $m2_{\mathrm{robust\_input}}$ specification, feature statistics are
fitted only on the corresponding training scope and valid horizon times:
\[
\widetilde{\mathbf{x}}_{i,t}^{h}
=
\operatorname{clip}_{[-5,5]}
\left(
\frac{\mathbf{x}_{i,t}^{h}-\boldsymbol{\mu}_{S,h}}
{\boldsymbol{\sigma}_{S,h}}
\right).
\]
The standardized graph snapshots are processed by a two-layer edge-aware
GAT with four heads in the first layer, one head in the second layer, hidden
dimension 24, ELU activations, and a raw-input skip path. The GAT output is
mapped to an original-scale residual correction:
\[
\widehat r_{i,t}^{h}
=
\rho_{S,h}
\left[
f_{\theta_h}^{G}
\left(
\widetilde{\mathbf X}_{t}^{h},G,\mathbf Q
\right)
\right]_i,
\]
where $\widetilde{\mathbf X}_{t}^{h}$ contains all county nodes in the
snapshot and $\mathbf Q$ denotes the edge attributes. Training uses MSE
loss and AdamW with learning rate $0.003$, weight decay $2\times10^{-4}$,
dropout $0.12$, a maximum of 220 epochs, and patience of 35 epochs.

The correction weight $\alpha_h$ is selected separately for each horizon by
scope-aware inner cross-validation from
\[
\mathcal{A}
=
\{0,0.05,0.10,0.20,0.35,0.50,0.75,1.00\}.
\]
The final prediction is
\[
\widehat y_{i,t}^{h}
=
\mathcal{P}_{\mathrm{OSI}}
\left(
\widehat y_{i,t}^{\mathrm{I3},h}
+
\alpha_h\widehat r_{i,t}^{h}
\right),
\]
where $\mathcal{P}_{\mathrm{OSI}}$ clips predictions to $[0,0.65]$ and
sets values below $0.001$ to zero.

\subsection{Spatial Residual Learning 详细版}

For each forecast horizon
$h\in\{1,6,24,48\}$, we train an independent spatial residual
learner to correct the frozen I3 prediction. Let
$\widehat{y}_{i,t}^{\mathrm{I3},h}$ denote the I3 prediction for county
$i$ at forecast origin $t$. The residual target is defined as
\begin{equation}
r_{i,t}^{h}
=
y_{i,t+h}
-
\widehat{y}_{i,t}^{\mathrm{I3},h},
\qquad
\widetilde r_{i,t}^{h}
=
\frac{r_{i,t}^{h}}{\rho_{S,h}},
\label{eq:residual_target}
\end{equation}
where $y_{i,t+h}$ is the observed OSI value at the target time. The
residual scale is computed using only valid supervised county--time
cells within the training scope:
\begin{equation}
\rho_{S,h}
=
\max\left\{
\operatorname{Std}_{(i,t)\in\Omega_{S,h}}
\left(r_{i,t}^{h}\right),
0.003
\right\}.
\label{eq:residual_scale}
\end{equation}

We construct a graph with all 302 counties as nodes. The graph combines
symmetric eight-nearest-neighbor relations, shared county borders, and
self-loops. Each directed edge is associated with four attributes:
normalized geographic distance, sine and cosine of the bearing, and a
shared-border indicator. The 205-dimensional node input consists of the
I3 prediction, 163 frozen features, county coordinates, seven terrain
variables, and 16 prespecified local-history and weather variables
together with their neighbor means and neighbor-minus-own differences.

For the $m2_{\mathrm{robust\_input}}$ specification, feature statistics
are fitted only on the corresponding training scope and
horizon-valid training times. The standardized input is
\begin{equation}
\widetilde{\mathbf{x}}_{i,t}^{h}
=
\operatorname{clip}_{[-5,5]}
\left(
\frac{
\mathbf{x}_{i,t}^{h}
-
\boldsymbol{\mu}_{S,h}
}{
\boldsymbol{\sigma}_{S,h}
}
\right).
\label{eq:input_standardization}
\end{equation}

The standardized graph snapshots are processed by a two-layer
edge-aware graph attention network. Let
$\mathbf{h}_{i,t}^{(\ell)}$ denote the representation of county $i$ at
layer $\ell$, and let
$\mathbf{z}_{i,t}^{(\ell,k)}
=
W^{(\ell,k)}\mathbf{h}_{i,t}^{(\ell)}$ denote its linearly transformed
representation for attention head $k$. For an edge $j\rightarrow i$,
the edge-aware attention score is
\begin{equation}
s_{ij,t}^{(\ell,k)}
=
\operatorname{LeakyReLU}_{0.2}
\left(
\left(\mathbf{a}_{\mathrm{src}}^{(\ell,k)}\right)^{\top}
\mathbf{z}_{j,t}^{(\ell,k)}
+
\left(\mathbf{a}_{\mathrm{dst}}^{(\ell,k)}\right)^{\top}
\mathbf{z}_{i,t}^{(\ell,k)}
+
\left(U^{(\ell)}\mathbf{e}_{ji}\right)_{k}
\right),
\label{eq:attention_score}
\end{equation}
where $\mathbf{e}_{ji}$ is the four-dimensional edge-attribute vector.
The normalized attention coefficient is
\begin{equation}
\alpha_{ij,t}^{(\ell,k)}
=
\frac{
\exp\left(s_{ij,t}^{(\ell,k)}\right)
}{
\displaystyle
\sum_{m\in\mathcal{N}(i)}
\exp\left(s_{im,t}^{(\ell,k)}\right)
}.
\label{eq:attention_coefficient}
\end{equation}
The neighborhood message for each attention head is then computed as
\begin{equation}
\mathbf{m}_{i,t}^{(\ell,k)}
=
\sum_{j\in\mathcal{N}(i)}
\alpha_{ij,t}^{(\ell,k)}
W^{(\ell,k)}
\mathbf{h}_{j,t}^{(\ell)}.
\label{eq:message_aggregation}
\end{equation}

The first GAT layer uses four attention heads and a hidden dimension of
24 per head, while the second layer uses one attention head and outputs
a 24-dimensional representation. ELU activations, dropout with rate
0.12, and a raw-input skip path are used. The resulting representation
is mapped to a scalar normalized residual through a
$24\rightarrow24\rightarrow1$ multilayer perceptron. The graph topology
is fixed; the attention mechanism learns the relative contribution of
each neighbor and its edge attributes.

The model is trained using mean squared error on the normalized residual
targets. AdamW is used with learning rate $0.003$, weight decay
$2\times10^{-4}$, a maximum of 220 epochs, and early stopping with
patience 35 epochs. The predicted residual is transformed back to the
original scale as
\begin{equation}
\widehat r_{i,t}^{h}
=
\rho_{S,h}
\left[
f_{\theta_h}^{G}
\left(
\widetilde{\mathbf{X}}_{t}^{h},
\mathcal{G},
\mathbf{Q}
\right)
\right]_{i},
\label{eq:predicted_residual}
\end{equation}
where $\widetilde{\mathbf{X}}_{t}^{h}$ contains all county nodes in the
graph snapshot and $\mathbf{Q}$ denotes the edge-attribute matrix.

The correction weight is selected separately for each horizon by
scope-aware inner cross-validation from
\begin{equation}
\mathcal{A}
=
\{0,0.05,0.10,0.20,0.35,0.50,0.75,1.00\}.
\label{eq:alpha_grid}
\end{equation}
The final prediction is
\begin{equation}
\widehat{y}_{i,t}^{h}
=
\mathcal{P}_{\mathrm{OSI}}
\left(
\widehat{y}_{i,t}^{\mathrm{I3},h}
+
\alpha_h\widehat r_{i,t}^{h}
\right),
\label{eq:final_prediction}
\end{equation}
where $\mathcal{P}_{\mathrm{OSI}}$ clips predictions to
$[0,0.65]$ and sets values below $0.001$ to zero.
