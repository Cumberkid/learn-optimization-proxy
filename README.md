# Learning Optimization Proxies

Reproducibility code for the numerical experiments in *Learning Optimization Proxies* (INFORMS manuscript).

The submission contains two experiments: portfolio allocation in Euclidean geometry, and multitask D-optimal design in Bregman geometry. The portfolio worst-case-gap experiment is included for the appendix. Historical Legendre, basis-expansion and steepest-descent experiments are outside this submission.

On Windows, long result-directory names may require cloning with `git -c core.longpaths=true clone https://github.com/Cumberkid/learn-optimization-proxy.git`.

## Environment

For full reproduction, including the IPOPT-based multitask worst-case-gap search:

```sh
conda env create -f environment.yml
conda activate optimization-proxies
```

For plotting stored results and running portfolio experiments, Python 3.11+ with `pip install -r requirements.txt` is sufficient. Install IPOPT/cyipopt through conda-forge for the full multitask evaluation. The notebook has a fallback when cyipopt is unavailable; that fallback is not equivalent to the paper's IPOPT search. The conda specification targets PyTorch 2.3.1 used by the original SAVIO jobs. `requirements.txt` is a compatibility list, not an exact historical lockfile.

## Recreate plots from included results

Run from the repository root:

```sh
python tools/run_notebook.py portfolio-plots --basis random_fourier
python tools/run_notebook.py portfolio-plots --basis random_relu
python tools/run_notebook.py multitask-plots
```

Figures are saved under `figures/`. The runner sets each notebook's working directory and exports every displayed figure in cell order. It retains the original plotting calculations and styles. See `REPRODUCIBILITY.md` for the paper-to-code mapping and configuration choices. The Fourier adaptive threshold reference is **7**, confirmed against the manuscript figure; see [the figure audit](docs/FOURIER_THRESHOLD.md).

## Rerun experiments

```sh
python tools/run_notebook.py multitask-preprocess
python tools/run_notebook.py multitask-train
python tools/run_notebook.py multitask-plots
```

The full multitask experiment includes expensive multistart worst-case-gap searches. Included preprocessed artifacts allow training without repeating clustering. Run notebooks from their experiment directory when using Jupyter interactively.

Portfolio training has a script interface:

```sh
cd experiments/portfolio
python scripts/run_gamma_experiment.py --help
python scripts/compute_worst_case_gap.py --help
```

To regenerate commands using the settings recorded in the included training summaries:

```sh
python tools/replay_portfolio.py --basis random_fourier
python tools/replay_portfolio.py --basis random_relu
```

This prints commands by default; add `--execute` to run them. New training results go to `validation-output/replayed/`, preserving the supplied reference data. The recorded summaries and histories are the source of truth for seed, basis dimension, initialization, learning rate and adaptive-gamma parameters. Evaluation uses `compute_worst_case_gap.py` and saved beta snapshots; details and Slurm wrappers are in `experiments/portfolio/savio/`. Submit cluster wrappers from `experiments/portfolio` and supply your own allocation with `sbatch --account=YOUR_ACCOUNT ...`.

## Data and layout

- `experiments/portfolio/`: training and plotting notebooks, local library, CLI scripts, gamma grids, market statistics and reference results. Source returns are Fama-French 10 Industry Portfolios, August 2014–July 2024: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- `experiments/multitask/`: preprocessing, training and plotting notebooks, Ames Housing input, preprocessed matrices and reference results. Dataset: https://www.openml.org/d/42165
- `provenance.json`: source-relative file names and SHA-256 hashes of the original files before notebook output stripping and portability edits.
- `VALIDATION.md`: checks actually performed and their limits.

Training can overwrite CSVs in its working directory. Keep a fresh checkout of the reference results if comparing a new run. Numerical worst-case gaps are estimates from nonconvex searches, not certified global maxima. Floating-point results and clustering can vary across library versions/platforms. See LICENSE for the code license; source data retain their original terms.

