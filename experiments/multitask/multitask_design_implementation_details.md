# Multitask D-Optimal Design Experiment: Implementation Details

This note lists implementation details that can supplement the numerical results section for the multitask D-optimal design experiment. The goal is to record the experimental choices that affect the interpretation of the curves, especially where the Bregman/simplex setting differs from the Euclidean portfolio allocation experiment.

## 1. Data Preprocessing and Design Points

The experiment uses the Ames Housing dataset. The preprocessing script selects numerical housing variables, coerces them to numeric values, imputes missing entries with column medians, and keeps all 2930 rows after cleaning.

The $2930$ homes are then clustered into $d=20$ communities by k-means on standardized selected variables. The saved preprocessing metadata records cluster sizes ranging from 3 to 344 homes. The design points used in the D-optimal design problem are the community-level feature averages, not the original individual homes.

The clustering and preprocessing use seed 7. The resulting artifacts are saved under `preprocessed/`, including:

- `multitask_community_means.csv`;
- `multitask_community_sizes.csv`;
- `multitask_design_matrices.npz`;
- `multitask_preprocessing_metadata.json`;
- `multitask_preprocessing_summary.csv`.

## 2. Task Definitions

The experiment uses $S=4$ related regression tasks:

- `SalePrice`;
- `GrLivArea`;
- `GarageArea`;
- `TotalBsmtSF`.

Each task has a community-level design matrix with 20 design points and 6 columns, consisting of 5 task-specific predictors plus an intercept.

The feature sets are:

| Task | Predictors |
|---|---|
| `SalePrice` | `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt` |
| `GrLivArea` | `TotRmsAbvGrd`, `BedroomAbvGr`, `FullBath`, `FirstFlrSF`, `SecondFlrSF` |
| `GarageArea` | `GarageCars`, `OverallQual`, `YearBuilt`, `LotArea`, `Fireplaces` |
| `TotalBsmtSF` | `LotArea`, `YearBuilt`, `FirstFlrSF`, `OverallQual`, `OverallCond` |

This is different from the portfolio experiment because $\lambda$ does not encode arbitrary covariates or a reference point. Here $\lambda \in \Delta_4$ is a vector of task weights, so changing $\lambda$ directly changes the relative importance of the four D-optimal design objectives.

## 3. Sampling of $\lambda$

Training samples task weights from the uniform distribution on the task simplex:

$$
\lambda \sim \operatorname{Dirichlet}(1,1,1,1).
$$

The fixed-$\gamma$ sweep uses a common pre-generated sequence of $T=1000$ Dirichlet samples across all fixed values of $\gamma$. This makes the fixed-$\gamma$ curves more directly comparable, because they see the same stochastic task-weight sequence.

The adaptive-$\gamma$ run uses its own Dirichlet stream with seed offset `SEED + 2026`.

## 4. Basis Construction and Parameterization

The experiment uses $q=20$ basis values per design point. A shared set of $q$ random directions in $\mathbb{R}^4$ is sampled once and held fixed:

$$
w_1,\ldots,w_q \in \mathbb{R}^4.
$$

For a given $\lambda$, the basis vector is

$$
\phi(\lambda)
=
\operatorname{softmax}
\left(
(w_1^\top \lambda,\ldots,w_q^\top \lambda)
\right).
$$

The coefficient vector $\beta \in \Delta_{qd}$ is stored as $d$ blocks,

$$
\beta = (\beta_1^\top,\ldots,\beta_d^\top)^\top,
\qquad
\beta_i \in \mathbb{R}^q.
$$

The implementation first forms positive community scores

$$
\alpha_i(\lambda,\beta) = \phi(\lambda)^\top \beta_i,
$$

and then normalizes them:

$$
\theta_i(\lambda,\beta)
=
\frac{\alpha_i(\lambda,\beta)}
{\sum_{r=1}^d \alpha_r(\lambda,\beta)}.
$$

Thus $\theta(\lambda,\beta)$ is always in $\Delta_d$. This normalization detail is worth reporting, because the implementation is not merely an unconstrained linear map followed by projection. It builds positivity through the softmax basis and the simplex-constrained $\beta$, then enforces the final survey-budget constraint by normalizing the community scores.

## 5. Information Matrix Stabilization

For each task $s$, the ideal information matrix is

$$
M^{(s)}(\theta)=X^{(s)\top}\operatorname{Diag}(\theta)X^{(s)}.
$$

The numerical implementation uses a small ridge term when evaluating log determinants and gradients:

$$
M^{(s)}_{\mathrm{num}}(\theta)
=
X^{(s)\top}\operatorname{Diag}(\theta)X^{(s)}
+ 10^{-5} I.
$$

This stabilizes the log-determinant calculation and matrix inverse, especially when the allocation becomes concentrated near a lower-dimensional face of the simplex.

## 6. Objective and Reported Quantity

The trained objective is the log-barrier-regularized stochastic objective

$$
\mathbb{E}_{\lambda}
\left[
\sum_{s=1}^4
\lambda_s
\left(
-\log\det M^{(s)}_{\mathrm{num}}(\theta(\lambda,\beta))
\right)
\right]
+ \gamma \psi(\beta),
\qquad
\psi(\beta)=-\sum_j \log \beta_j.
$$

The plotted objective is the unregularized stochastic objective,

$$
\mathbb{E}_{\lambda}
\left[
h(\theta(\lambda,\beta),\lambda)
\right],
$$

estimated over a fixed Monte Carlo evaluation set. This matches the logic used in the portfolio experiment: $\gamma$ is an optimization and continuation device, while the unregularized objective is the common target used to compare all fixed and adaptive runs.

The current training notebook uses `n_eval = 1000` fixed Dirichlet samples for this Monte Carlo estimate. If the final paper uses curves regenerated with a larger evaluation set, this number should be updated in the text.

## 7. Initialization

The coefficient vector is initialized near a simplex vertex rather than at the simplex center:

$$
\beta_j = 10^{-3} \quad \text{for most } j,
\qquad
\beta_1 = 1 - p \cdot 10^{-3},
\qquad
p=qd.
$$

With $q=20$ and $d=20$, this gives $p=400$. This vertex-biased initialization is relevant because the log-barrier geometry strongly affects how the iterates move away from boundary-near regions. It also explains the `_vertex` suffix in the current result files.

Earlier or alternative files with `_random` in the name correspond to a different initialization/basis experiment and should not be mixed with the current figure unless explicitly discussed.

## 8. Bregman Update and Scalar Root Solve

Training uses the stochastic reference-function descent update with the log-barrier reference function. Given the stochastic gradient at time $t$, the update solves

$$
\beta_{t+1}
=
\arg\min_{\beta\in\Delta_{qd}}
\left\{
\eta \nabla f_\gamma(\beta_t,\lambda_t)^\top(\beta-\beta_t)
+D_\psi(\beta,\beta_t)
\right\}.
$$

The KKT conditions reduce the update to

$$
\beta_{t+1,j}=\frac{1}{v_j+\xi},
$$

where $\xi$ is chosen so that $\sum_j \beta_{t+1,j}=1$. The scalar equation

$$
\sum_j \frac{1}{v_j+\xi}=1
$$

is solved by bisection on the interval $\xi>-\min_j v_j$. After the solve, the implementation clips entries below $10^{-15}$ and renormalizes to guard against floating-point roundoff.

## 9. Learning Rate Selection

For both fixed and adaptive runs, the learning rate follows the relative-smoothness scaling

$$
\eta(\gamma)=\frac{0.5}{1+\gamma}.
$$

The factor $1+\gamma$ is tied to the Bregman relative-smoothness calculation for the log-barrier regularized D-optimal design objective. The additional factor $0.5$ is a conservative numerical constant.

## 10. Fixed and Adaptive Gamma Schedules

The fixed-$\gamma$ sweep uses

$$
\gamma \in \{1,\ 0.1,\ 0.01,\ 0.001\}.
$$

Each run lasts $T=1000$ stochastic iterations and records snapshots every 100 iterations.

The adaptive run starts at

$$
\gamma_0=1
$$

and shrinks by a factor of $0.1$ after saturation:

$$
\gamma \leftarrow 0.1\gamma.
$$

The current adaptive run uses the gradient-based saturation detector. It compares adjacent rolling averages of the raw stochastic gradients using a window of 3. A phase is declared saturated when the norm of the difference between adjacent rolling averages falls below

$$
\texttt{threshold\_ref}
\left(
\frac{\gamma}{\gamma_0}
\right)^{\alpha},
$$

with `threshold_ref = 100` and $\alpha=0.5$. After each shrink, the learning rate is reset using $\eta(\gamma)=0.5/(1+\gamma)$.

## 11. Worst-Case Gap Computation

For a saved iterate $\beta_t$, the numerical worst-case gap is

$$
\max_{\lambda\in\Delta_4}
\left[
h(\theta(\lambda,\beta_t),\lambda)
-
h(\theta^*(\lambda),\lambda)
\right].
$$

For each candidate $\lambda$, the benchmark design $\theta^*(\lambda)$ is approximated by projected-gradient descent on the simplex, initialized at the simplex center. The implementation uses up to 250 iterations for this inner solve.

The outer maximization over $\lambda$ uses 32 starts on the task simplex. The starts include the simplex vertices and additional random Dirichlet points. If `cyipopt` is unavailable, the implementation falls back to a multistart projected-gradient method with finite-difference gradients and up to 40 outer iterations. The current saved worst-case-gap files report this fallback method:

`multistart_projected_gradient_finite_difference`.

Because the outer maximization is nonconvex and the benchmark $\theta^*(\lambda)$ is computed numerically, the reported value should be described as a numerical worst-case gap estimate rather than a certified global maximum.

## 12. Result Files Used by the Current Figure

The current vertex-initialized fixed and adaptive experiments are recorded in:

- `gamma_sweep_results_vertex.csv`;
- `adaptive_gamma_results_vertex.csv`;
- `gamma_sweep_beta_snapshots_vertex.npz`;
- `adaptive_gamma_beta_snapshots_vertex.npz`;
- `gamma_sweep_worst_case_gaps_vertex.csv`;
- `adaptive_gamma_worst_case_gaps_vertex.csv`.

The main curve in the paper appears to use the unregularized objective column `objective_estimate`. The regularized objective is also logged as `reg_objective_estimate`, but should not be used for cross-$\gamma$ comparison because it changes the plotted target.

## Suggested Paper Text

The Ames Housing experiment is run after cleaning selected numerical variables, imputing missing values by column medians, and clustering the 2930 homes into $d=20$ community-level design points using k-means with seed 7. The four downstream tasks are `SalePrice`, `GrLivArea`, `GarageArea`, and `TotalBsmtSF`; each task uses five predictors plus an intercept. We sample $\lambda$ from the uniform Dirichlet distribution on $\Delta_4$ and use $q=20$ fixed random softmax basis values shared across communities. The learned vector $\beta\in\Delta_{qd}$ is arranged into community blocks, converted to positive scores $\alpha_i(\lambda,\beta)=\phi(\lambda)^\top\beta_i$, and normalized to obtain $\theta(\lambda,\beta)\in\Delta_d$. Log determinants are evaluated with a ridge term $10^{-5}I$ for numerical stability. All curves report the unregularized objective, estimated on a fixed Monte Carlo set of Dirichlet samples, so that fixed and adaptive values of $\gamma$ are compared on the same target. The Bregman update is solved by reducing the log-barrier proximal step to a scalar bisection problem for the simplex multiplier. Fixed runs use $\gamma\in\{1,0.1,0.01,0.001\}$ and stepsize $\eta=0.5/(1+\gamma)$. The adaptive run starts at $\gamma=1$ and shrinks by a factor of $0.1$ when a rolling-gradient saturation detector triggers, resetting the stepsize after each shrink. Worst-case gaps, when reported, are numerical estimates obtained by multistart projected-gradient maximization over $\lambda\in\Delta_4$, with the fixed-$\lambda$ oracle design approximated by a projected-gradient solve on the design simplex.
