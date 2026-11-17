# Savio Worst-Case Gap Jobs

This document describes the second-stage Savio jobs for computing worst-case
solution-path gaps after training has saved beta snapshots.

## Goal

For each trained run and each saved snapshot epoch, compute

$$
\max_{\lambda \in \Lambda}
\left[
h(\Phi(\lambda)\beta_t,\lambda)
-
h(\theta^*(\lambda),\lambda)
\right],
$$

where $\beta_t$ is the saved beta snapshot at training epoch $t$.

This stage is independent of training. It should only read saved training
artifacts and write worst-case gap results.

## Inputs

Each task reads one trained gamma run directory, for example:

```text
results/gamma_sweep_sgd_random_relu_zero_init/gamma_...
```

Required files:

```text
beta_snapshots.pt
summary.json
```

Useful files:

```text
beta_snapshots.csv
history.csv
```

The `summary.json` file contains the basis type, basis dimension, basis seed,
random feature scale, random bias scale, gamma, and other metadata needed to
reconstruct $\Phi(\lambda)$. The `beta_snapshots.pt` file contains records of
the form:

```python
{"epoch_index": epoch, "beta": beta_tensor}
```

## Lambda Domain

The maximization is over the same boxed lambda domain used in training:

$$
\lambda_1 \in [0.2, 1],
\qquad
\lambda_2 \in [0, 1],
\qquad
\lambda_3,\ldots,\lambda_{12} \in [-2, 2].
$$

Equivalently:

```python
lam_min = [0.2, 0.0, -2.0, ..., -2.0]
lam_max = [1.0, 1.0,  2.0, ...,  2.0]
```

The optimizer must enforce these box constraints directly. Candidate
$\lambda$ values should never be projected silently after evaluating outside
the box.

## Objective

For the Savio portfolio experiment,

$$
h(\theta,\lambda)
=
\lambda_1 \theta^\top \Sigma \theta
-
\lambda_2 \mu^\top \theta
+
\|\theta-\lambda_{3:12}\|_2^2.
$$

The learned decision is:

$$
\hat\theta_t(\lambda)=\Phi(\lambda)\beta_t.
$$

The exact optimizer for a fixed $\lambda$ is available in closed form:

$$
\theta^*(\lambda)
=
(I+\lambda_1\Sigma)^{-1}
\left(
\lambda_{3:12}+\frac{\lambda_2}{2}\mu
\right).
$$

Thus each objective evaluation can compute:

```text
gap(lambda) = h(theta_hat(lambda), lambda) - h(theta_star(lambda), lambda)
```

and the second-stage task maximizes `gap(lambda)`.

## Job Granularity

Use one Savio array task per gamma. Each task processes all requested saved
snapshot epochs for that gamma.

For example, if there are 18 gamma values, the worst-case gap array has 18
tasks. Each task processes all requested snapshot epochs for one gamma and
writes per-epoch files plus a combined file for that gamma:

```text
task_id = gamma_index
```

The implementation files are:

```text
scripts/compute_worst_case_gap.py
savio/submit_worst_case_gap_array.sh
```

The current ReLU and Fourier grids each contain 18 gamma rows, so use 18 array
tasks:

```text
#SBATCH --array=0-17
```

Each task should:

1. Read the gamma row from the same gamma grid CSV used by training.
2. Convert the gamma value to the training output directory name.
3. Load `beta_snapshots.pt`.
4. Select each beta record matching the requested snapshot epochs.
5. Reconstruct the basis from `summary.json`.
6. Maximize the gap over the boxed lambda domain for each snapshot.
7. Write per-epoch result files and a combined `worst_case_gap.csv` for this gamma.

## Outputs

Each task should write a small task-local output file to avoid write conflicts:

```text
results/.../gamma_.../worst_case_gap_epoch_<epoch>.csv
results/.../gamma_.../worst_case_gap.csv
```

Suggested columns:

```text
gamma
epoch_index
worst_case_gap
lambda_1
lambda_2
...
lambda_12
optimizer_method
optimizer_status
objective_evaluations
random_seed
elapsed_seconds
```

After the array completes, a merge script can collect these files into:

```text
results/.../worst_case_gap.csv
```

or a cross-gamma summary file:

```text
results/worst_case_gap_<basis>_<weight_init>_init.csv
```

## Optimization Strategy

The objective is generally nonconvex in $\lambda$, so local optimization alone
does not certify the global maximum.

Recommended default:

1. Generate many feasible initial points inside the lambda box.
2. Evaluate the gap at all initial points.
3. Run box-constrained local maximization from the best initial points.
4. Report the best solution found.

For random Fourier features, the objective is smooth, so IPOPT or another
gradient-based box-constrained local optimizer is appropriate.

For random ReLU features, the objective is piecewise smooth and has activation
kinks. IPOPT may still be useful from many starts, but a derivative-free global
search plus local polishing is more robust.

The result should be described as a numerical worst-case estimate unless a
global certification method is added.

## Submission Examples

Random ReLU, using the default snapshot epochs `0,500,...,20000`:

```bash
sbatch --array=0-17 savio/submit_worst_case_gap_array.sh \
  random_relu \
  results/gamma_sweep_sgd_random_relu_zero_init \
  savio/gamma_grid_relu.csv
```

Random Fourier, using the default snapshot epochs `0,50,...,2000`:

```bash
sbatch --array=0-17 savio/submit_worst_case_gap_array.sh \
  random_fourier \
  results/gamma_sweep_sgd_random_fourier_zero_init \
  savio/gamma_grid_fourier.csv
```

The ellipses above are for readability. The submit script generates the full
comma-separated lists automatically when the fourth argument is omitted.

The matching training jobs now default to basis dimension 16. Random ReLU runs
for 20000 epochs and saves snapshots every 500 epochs. Random Fourier runs for
2000 epochs and saves snapshots every 50 epochs.

```bash
sbatch --array=0-17 savio/submit_gamma_array.sh random_relu

sbatch --array=0-17 savio/submit_gamma_array.sh random_fourier
```

If the training output directory includes a prefix such as `256_`, pass that
exact directory as the second argument.

## Failure Handling

Each task should fail clearly if:

- `beta_snapshots.pt` is missing;
- the requested `snapshot_epoch` is not present;
- `summary.json` is missing required basis metadata;
- the gamma output directory cannot be found.

Optimizer failures should still write an output row when possible, with
`optimizer_status` indicating the failure mode and the best feasible sampled
point found so far.
