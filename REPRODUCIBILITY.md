# Paper-to-code mapping

Scope was checked against `6_numerics.tex` and the portfolio worst-case-gap figures discussed in the manuscript appendix. The two canonical experiment directories replace the incomplete earlier submission notebooks.

| Paper result | Code / input |
|---|---|
| Portfolio expected objective, random Fourier | `experiments/portfolio/04 Portfolio SGD Gamma Plots.ipynb`, fixed grid rows 0,7,9,12; adaptive start 89.25 |
| Portfolio expected objective, random ReLU | Same notebook, grid rows 0,4,6,9; adaptive start 368.15625 |
| Portfolio appendix worst-case gap | Same notebook's worst-case-gap cells; `results/*/worst_case_gap.csv`; evaluator `scripts/compute_worst_case_gap.py` |
| Multitask expected objective and worst-case gap | `experiments/multitask/07 Multitask D_optimal Design Plots.ipynb`, vertex-initialized fixed/adaptive CSVs |
| Multitask preprocessing and training | `05 Multitask D_optimal Design Preprocessing.ipynb`, then `06 Multitask D_optimal Design.ipynb` |

Portfolio configurations retain random Fourier dimension 64, random ReLU dimension 32, seed 8675309, initialization scale 0.025 / 0.05 respectively. The plotting runner follows the saved notebook selector: gradient saturation threshold reference 7, exponent 0.1, forced maximum phase lengths 500 / 5000, and plotting horizons 500 / 10000. The source notebook comment also mentions threshold 5 for Fourier; both runs existed, but the executable selector is 7. This choice must be checked against the final author-approved figure before claiming pixel-identical publication reproduction.

Fixed-gamma grids and histories are included along with beta snapshots and per-epoch worst-case-gap evaluations. Only the adaptive runs selected above are included. Other experimental variants, preliminary notebooks, caches, local environment directories and cluster credentials were excluded.

The existing published LICENSE is preserved. Original notebooks are copied with outputs and execution counts cleared. Preprocessing references and cluster account handling were adjusted; mathematical training code was copied unchanged. `provenance.json` refers to original source bytes, so edited notebook hashes deliberately differ.

For a quick training sanity check run `python tools/run_notebook.py multitask-smoke`. It evaluates a two-step reference optimization without overwriting the reference CSVs. This is not a full reproduction of the paper.

