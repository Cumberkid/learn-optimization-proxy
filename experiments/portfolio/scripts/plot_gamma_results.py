#!/usr/bin/env python
"""Plot SAVIO gamma-sweep histories in the style of the notebook plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/gamma_sweep_sgd"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/gamma_sweep_sgd_plots"),
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="*",
        default=None,
        help="Optional subset of gamma values to plot, e.g. --gammas 1 2 5 10",
    )
    return parser.parse_args()


def load_histories(results_dir: Path):
    runs = []
    for summary_path in sorted(results_dir.glob("gamma_*/summary.json")):
        with summary_path.open() as f:
            summary = json.load(f)
        gamma = float(summary["gamma"])
        history_path = summary_path.with_name("history.csv")
        history = pd.read_csv(history_path)
        runs.append((gamma, history))
    return sorted(runs, key=lambda item: item[0])


def plot_metric(runs, metric: str, ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(18, 18))
    font_size = 28

    for gamma, history in runs:
        plt.plot(
            history["epoch_index"],
            history[metric],
            "--o",
            linewidth=4,
            markersize=9,
            alpha=0.8,
            label=fr"$\gamma$={gamma:g}",
        )

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel("# SGD epochs", fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.legend(fontsize=font_size * 0.7)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    runs = load_histories(args.results_dir)

    if args.gammas is not None:
        requested = set(args.gammas)
        runs = [(gamma, history) for gamma, history in runs if gamma in requested]

    if not runs:
        raise SystemExit(f"No histories found in {args.results_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "all" if args.gammas is None else "_".join(str(gamma).rstrip("0").rstrip(".") for gamma in args.gammas)

    plot_metric(
        runs,
        metric="loss_train",
        ylabel="Average training loss",
        title="Training Loss by Gamma",
        out_path=args.out_dir / f"train_loss_{suffix}.png",
    )
    plot_metric(
        runs,
        metric="loss_val",
        ylabel="Average validation loss",
        title="Validation Loss by Gamma",
        out_path=args.out_dir / f"val_loss_{suffix}.png",
    )

    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
