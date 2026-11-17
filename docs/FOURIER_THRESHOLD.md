# Fourier adaptive-gamma threshold: confirmed as 7

The final manuscript (`6_numerics.tex`, `fig:portfolio_fourier_expected`) includes `Figs/unreg_objective_portfolio_val_fourier.png`. An unmodified copy of that exact image is stored in [reference_figures](reference_figures/unreg_objective_portfolio_val_fourier.png); its SHA-256 is recorded in [the audit data](fourier_threshold_audit.json).

The gray vertical dotted lines mark adaptive-gamma shrink events. We extracted their pixel-column centers from the original 2190 x 1790 image and mapped the linear x-axis (pixels 490 to 2107, gradient calls 0 to 500) back to epochs. They match the threshold-7 event CSV one-for-one, within 0.5 gradient calls of rasterization error.

| Candidate reference threshold | Events in the plotted range | First event | Match to final image |
|---|---:|---:|---|
| 7 | 17 | 4 | All 17 markers match |
| 5 | 12 | 14 | Excluded by event count and positions |

The confirmed sequence is `4, 8, 12, 16, 22, 26, 30, 34, 50, 66, 71, 79, 150, 165, 178, 189, 497`. In particular, the markers at 4, 8, 12, 71 and 189 distinguish this run from threshold 5. This identifies the run independently of the notebook's saved selector or its stale comment.

Here **7 is the reference threshold**, not a constant threshold at every gamma:

`threshold(gamma) = 7 * clip((gamma / 89.25)^0.1, 0.01, 100)`.

Other settings of the confirmed Fourier run: start gamma 89.25, shrink factor 0.5, rolling window 3, maximum phase length 500, C=3, L=476, seed 8675309, feature dimension 64, random initialization scale 0.025. Training records cover 1000 epochs; the final figure displays 0 to 500.

The plotting notebook and runner use reference threshold 7. The old “5 if fourier” comment was incorrect and has been replaced. The default adaptive Slurm threshold grid now contains the confirmed reference value 7; for all recorded publication settings use `tools/replay_portfolio.py --basis random_fourier` (add `--execute` to train). The generic Slurm wrapper still exposes other experiment parameters explicitly.

Run `python tools/verify_fourier_threshold.py` to check the reference image hash, marker alignment, included event CSV and summary, and plotting selectors. This audit establishes which threshold generated the manuscript figure; it does not assert pixel-identical rendering across plotting-library versions.
