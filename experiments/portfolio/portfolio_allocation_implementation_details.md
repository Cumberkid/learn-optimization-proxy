# Portfolio Allocation Experiment: Implementation Details

This note lists the implementation details we should use to supplement the numerical results section for the portfolio allocation experiment.

## 1. Sampling of $\lambda$

Training samples $\lambda$ uniformly from the boxed parameter domain

$$
\lambda_1 \in [0.2,1],
\qquad
\lambda_2 \in [0,1],
\qquad
\lambda_3,\ldots,\lambda_{12} \in [-2,2].
$$

Each SGD epoch uses one fresh draw of $\lambda$ from this uniform distribution. The same boxed domain is also used when estimating expected objectives and when computing the worst-case gap.

## 2. Basis Dimensions

The portfolio allocation experiment now uses different fixed basis dimensions for the two random feature families:

- random ReLU features use $q=16$ basis vectors;
- random Fourier features use $q=64$ basis vectors.

For both bases, the random features are sampled once before training and then held fixed. The learned decision rule has the form

$$
\hat\theta_\beta(\lambda)=B\phi(\lambda),
$$

where $B$ is the learned coefficient matrix and $\phi(\lambda)$ is the fixed random feature map.

## 3. Optimizer and Initialization

The optimization is performed with `torch.optim.SGD` on the $\gamma$-regularized stochastic objective. The coefficient matrix $B$ is randomly initialized before training.

The training objective is

$$
h(\hat\theta_\beta(\lambda),\lambda)
+ \frac{\gamma}{2}\|B\|_F^2,
$$

while the main reported objective values remove the regularization term so that all $\gamma$ values are compared on the same unregularized target.

## 4. Monte Carlo Estimation of the Expected Objective

The plotted expected objectives are estimated by Monte Carlo with 1000 points. The training and validation curves use fixed, independent Monte Carlo sets:

- the training estimate uses one fixed set of 1000 uniformly sampled $\lambda$ values;
- the validation estimate uses a second fixed set of 1000 uniformly sampled $\lambda$ values.

These sets are held fixed across evaluation times and across runs, so the observed differences between $\gamma$ values are not driven by fresh Monte Carlo noise at every logging time.

The plotted quantity is

$$
\mathbb E_\lambda\left[h(\hat\theta_\beta(\lambda),\lambda)\right],
$$

estimated over the corresponding fixed Monte Carlo set.

## 5. Learning Rate Selection

For each $\gamma$, the learning rate is chosen using the theorem-motivated scaling

$$
\eta = \frac{1}{CL+\gamma}.
$$

Here $L$ is the smoothness scale of the unregularized sample objective and $C$ is the basis-dependent operator bound controlling the feature map. This choice should be referenced back to the Euclidean regularized SGD theorem in the paper, since the theorem is the reason the stepsize is scaled by the combined quantity $CL+\gamma$.

## 6. Adaptive Gamma Rule

In the adaptive gamma experiment, training begins with a larger value of $\gamma$ and moves to smaller values only after the current phase appears to have saturated.

The saturation decision is based on the stochastic gradient: the implementation compares adjacent rolling averages of the raw SGD gradients and shrinks $\gamma$ when the norm of their difference falls below the chosen threshold. After each gamma shrink, the learning rate is reset according to the same formula

$$
\eta(\gamma)=\frac{1}{CL+\gamma}.
$$

This lets the method use large $\gamma$ for fast early progress and smaller $\gamma$ after the large-$\gamma$ phase stops making meaningful gradient progress.

## 7. Worst-Case Gap Computation

For a saved iterate $B_t$, the worst-case gap is computed as

$$
\max_{\lambda \in \Lambda}
\left[
h(\hat\theta_t(\lambda),\lambda)
- h(\theta^*(\lambda),\lambda)
\right],
$$

where $\hat\theta_t(\lambda)=B_t\phi(\lambda)$ and $\theta^*(\lambda)$ is the exact optimizer of the fixed-$\lambda$ quadratic portfolio problem.

The maximization is performed over $\lambda$ as a post-training numerical optimization. The current script first evaluates the gap at 2048 uniformly sampled feasible $\lambda$ values, then runs box-constrained L-BFGS-B local optimization from the best sampled starts. By default, it uses 16 local starts.

The script optionally supports SciPy differential evolution before local polishing, but the standard local optimizer is L-BFGS-B, not IPOPT. Each local solve enforces the same box constraints on $\lambda$, and the reported `worst_case_gap` is the largest value found across the random search and local-start optimizations.

Because the outer maximization over $\lambda$ is nonconvex, the value should be described as a numerical worst-case estimate rather than a certified global maximum.

## Suggested Paper Text

All plotted objectives are unregularized Monte Carlo estimates of $\mathbb E_\lambda[h(\hat\theta_\beta(\lambda),\lambda)]$, computed using fixed training and validation sets of 1000 uniformly sampled $\lambda$ values. The solution path is trained with `torch.optim.SGD` on the $\gamma$-regularized objective from random initialization, using $q=16$ random ReLU features and $q=64$ random Fourier features. For each $\gamma$, the learning rate follows the theorem-motivated scaling $\eta=(CL+\gamma)^{-1}$. In the adaptive experiment, $\gamma$ is decreased when a rolling-gradient saturation detector indicates that the current phase has plateaued. We also report a numerical worst-case gap, computed by maximizing $h(\hat\theta_t(\lambda),\lambda)-h(\theta^*(\lambda),\lambda)$ over $\lambda$ using random feasible starts followed by box-constrained L-BFGS-B local optimization from the best starts.
