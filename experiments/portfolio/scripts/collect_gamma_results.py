#!/usr/bin/env python
"""Collect per-gamma SAVIO results into one CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/gamma_sweep_sgd"))
    parser.add_argument("--out", type=Path, default=Path("results/gamma_sweep_sgd_summary.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for summary_path in sorted(args.results_dir.glob("gamma_*/summary.json")):
        with summary_path.open() as f:
            summary = json.load(f)
        row = {
            "gamma": summary["gamma"],
            "learning_rate": summary["learning_rate"],
            "epochs": summary["epochs"],
            "seed": summary["seed"],
            "basis_dim": summary["basis_dim"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        row.update({f"final_{k}": v for k, v in summary["final"].items()})
        diagnostics_summary = summary_path.with_name("diagnostics_summary.csv")
        row["has_diagnostics"] = diagnostics_summary.exists()
        rows.append(row)

    if not rows:
        raise SystemExit(f"No summary.json files found under {args.results_dir}")

    df = pd.DataFrame(rows).sort_values("gamma")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} rows")


if __name__ == "__main__":
    main()
