# Validation status

## Completed locally

- Parsed all 28 research Python files, 4 cleared notebooks and repository tools.
- Read 1,899 CSV headers; checked 38 training summaries have adjacent histories.
- Verified all 2,093 provenance entries exist.
- Original exploration notebook contained literal PowerShell newline escapes in a plotting cell and was omitted; the production CLI training script is provided instead.

## Runtime validation

Local environment installation was attempted using official PyPI and an alternate mirror. SSL failures/timeouts prevented installing the scientific stack. Full training and numerical agreement with the publication have therefore not been verified locally.

GitHub Actions passed on commit `cec8c6a`: both portfolio plotting configurations, multitask plotting, a short multitask reference optimization, a two-epoch portfolio training check, and bundle validation. Run: https://github.com/Cumberkid/learn-optimization-proxy/actions/runs/44086601788 . Generated figures are available as the `reproduction-figures` workflow artifact. These checks do not certify the long training runs or global optimality of worst-case searches.

## Figure audit and remaining limits

Fourier adaptive threshold **7** is confirmed against the manuscript figure by all 17 transition markers, with a maximum positional discrepancy below 0.5 gradient calls. Threshold 5 is excluded (12 markers). See [the audit](docs/FOURIER_THRESHOLD.md). The environment file is a historical version target, not a recovered exact package lock. Full IPOPT evaluations and all long SAVIO runs remain to be rerun in a suitable environment.

