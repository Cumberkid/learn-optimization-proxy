# Validation status

## Completed locally

- Parsed all 28 research Python files, 4 cleared notebooks and repository tools.
- Read 1,899 CSV headers; checked 38 training summaries have adjacent histories.
- Verified all 2,093 provenance entries exist.
- Original exploration notebook contained literal PowerShell newline escapes in a plotting cell and was omitted; the production CLI training script is provided instead.

## Runtime validation

Local environment installation was attempted using official PyPI and an alternate mirror. SSL failures/timeouts prevented installing the scientific stack. Full training and numerical agreement with the publication have therefore not been verified locally.

GitHub Actions passed on commit `cec8c6a`: both portfolio plotting configurations, multitask plotting, a short multitask reference optimization, a two-epoch portfolio training check, and bundle validation. Run: https://github.com/Cumberkid/learn-optimization-proxy/actions/runs/44086601788 . Generated figures are available as the `reproduction-figures` workflow artifact. These checks do not certify the long training runs or global optimality of worst-case searches.

## Remaining author checks

Confirm Fourier adaptive threshold 7 versus the comment suggesting 5 against the final manuscript figure. The prepared bundle follows the executable notebook selector. The environment file is a historical version target, not a recovered exact package lock. Full IPOPT evaluations and all long SAVIO runs remain to be rerun in a suitable environment.

