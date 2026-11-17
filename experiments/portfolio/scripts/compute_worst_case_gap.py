#!/usr/bin/env python
"""Compute worst-case gaps for saved beta snapshots from one gamma run.

One invocation handles one gamma row and all requested snapshot epochs. It is
intended to be called from a Savio Slurm array after training has produced
beta_snapshots.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution, minimize

from scripts.run_gamma_experiment import make_basis, make_lambda_box


def gamma_dir_name(gamma: float) -> str:
    return f"gamma_{gamma:.6g}".replace(".", "p")


def read_grid(grid_csv: Path):
    with grid_csv.open(newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "gamma": float(row["gamma"]),
                    "basis_dim": int(row["basis_dim"]) if row.get("basis_dim") else None,
                }
            )
        return rows


def run_dir_name(sweep_axis: str, gamma: float, basis_dim: int | None):
    if sweep_axis == "gamma":
        return gamma_dir_name(gamma)
    if sweep_axis == "basis_dim":
        if basis_dim is None:
            raise ValueError("basis_dim sweep requires a basis_dim column in the grid")
        return f"basis_dim_{basis_dim}"
    raise ValueError(f"Unsupported sweep axis: {sweep_axis}")


def parse_epoch_list(epoch_list: str):
    if not epoch_list:
        return []
    return [int(item.strip()) for item in epoch_list.split(",") if item.strip()]


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_snapshot(snapshots, epoch: int):
    for item in snapshots:
        if int(item["epoch_index"]) == epoch:
            return item["beta"].detach().cpu().float()
    available = [int(item["epoch_index"]) for item in snapshots]
    raise ValueError(f"snapshot epoch {epoch} not found; available epochs: {available}")


def make_basis_from_summary(summary: dict, dim_lambda: int, lam_min, lam_max):
    basis = summary["basis"]
    basis_dim = int(summary["basis_dim"])
    args = SimpleNamespace(
        basis=basis,
        basis_dim=basis_dim if basis != "taylor" else None,
        basis_dim_pool=summary.get("basis_dim_pool"),
        basis_scale_mode=summary.get("basis_scale_mode", "basis_dim") or "basis_dim",
        basis_seed=int(summary["basis_seed"]),
        random_feature_scale=float(summary["random_feature_scale"]),
        random_bias_scale=float(summary["random_bias_scale"]),
        taylor_order=int(summary["taylor_order"]),
    )
    rebuilt_basis_dim, phi_lam = make_basis(args, dim_lambda, lam_min, lam_max)
    if rebuilt_basis_dim != basis_dim:
        raise ValueError(f"rebuilt basis_dim {rebuilt_basis_dim} does not match summary basis_dim {basis_dim}")
    return phi_lam


def portfolio_objective(theta, lam, sigma, mean):
    risk = sigma @ theta
    target = lam[2:]
    return lam[0] * float(risk @ risk) - lam[1] * float(mean @ theta) + float(np.sum((theta - target) ** 2))


def theta_star(lam, sigma, mean):
    system = np.eye(mean.shape[0]) + lam[0] * sigma
    rhs = lam[2:] + 0.5 * lam[1] * mean
    return np.linalg.solve(system, rhs)


def make_gap_function(beta, phi_lam, sigma, mean, basis_dim):
    def gap(lam):
        lam = np.asarray(lam, dtype=np.float64)
        with torch.no_grad():
            phi = phi_lam(lam, basis_dim, "cpu").detach().cpu().float()
            theta_hat = (beta @ phi).detach().cpu().numpy().astype(np.float64)
        theta_opt = theta_star(lam, sigma, mean)
        return portfolio_objective(theta_hat, lam, sigma, mean) - portfolio_objective(theta_opt, lam, sigma, mean)

    return gap


def maximize_gap(gap, bounds, args):
    start = perf_counter()
    rng = np.random.default_rng(args.seed)
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)

    random_lams = rng.uniform(lower, upper, size=(args.num_random_starts, len(bounds)))
    random_values = np.asarray([gap(lam) for lam in random_lams])
    best_idx = int(np.argmax(random_values))
    best_lam = random_lams[best_idx]
    best_gap = float(random_values[best_idx])
    objective_evaluations = int(len(random_lams))
    statuses = [f"random_search_best={best_gap:.12g}"]

    if args.use_differential_evolution:
        de_result = differential_evolution(
            lambda lam: -gap(lam),
            bounds,
            seed=args.seed,
            maxiter=args.de_maxiter,
            popsize=args.de_popsize,
            polish=False,
            updating="immediate",
            workers=1,
            tol=args.de_tol,
        )
        objective_evaluations += int(de_result.nfev)
        de_gap = -float(de_result.fun)
        statuses.append(f"differential_evolution_success={bool(de_result.success)}")
        if de_gap > best_gap:
            best_gap = de_gap
            best_lam = np.asarray(de_result.x, dtype=np.float64)

    local_starts = [best_lam]
    if args.num_local_starts > 1:
        top_count = min(args.num_local_starts - 1, len(random_lams))
        top_indices = np.argsort(random_values)[-top_count:]
        local_starts.extend(random_lams[idx] for idx in top_indices)

    for start_lam in local_starts:
        local_result = minimize(
            lambda lam: -gap(lam),
            np.asarray(start_lam, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": args.local_maxiter, "ftol": args.local_ftol},
        )
        objective_evaluations += int(local_result.nfev)
        local_gap = -float(local_result.fun)
        statuses.append(f"local_success={bool(local_result.success)}")
        if local_gap > best_gap:
            best_gap = local_gap
            best_lam = np.asarray(local_result.x, dtype=np.float64)

    return {
        "worst_case_gap": best_gap,
        "lambda": best_lam,
        "optimizer_status": ";".join(statuses),
        "objective_evaluations": objective_evaluations,
        "elapsed_seconds": perf_counter() - start,
    }


def run(args):
    grid_rows = read_grid(args.grid_csv)
    snapshot_epochs = parse_epoch_list(args.snapshot_epochs)
    if not snapshot_epochs:
        raise ValueError("at least one snapshot epoch is required")

    task_count = len(grid_rows)
    if args.task_id < 0 or args.task_id >= task_count:
        raise IndexError(f"task id {args.task_id} outside [0, {task_count - 1}]")

    grid_index = args.task_id
    grid_row = grid_rows[grid_index]
    gamma = grid_row["gamma"]
    grid_basis_dim = grid_row["basis_dim"]

    run_dir = args.out_dir / run_dir_name(args.sweep_axis, gamma, grid_basis_dim)
    summary_path = run_dir / "summary.json"
    snapshots_path = run_dir / "beta_snapshots.pt"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary file: {summary_path}")
    if not snapshots_path.exists():
        raise FileNotFoundError(f"missing beta snapshots file: {snapshots_path}")

    with summary_path.open() as f:
        summary = json.load(f)
    snapshots = load_torch(snapshots_path)
    summary_basis_dim = int(summary["basis_dim"])
    if grid_basis_dim is not None and grid_basis_dim != summary_basis_dim:
        raise ValueError(
            f"grid basis_dim {grid_basis_dim} does not match summary basis_dim {summary_basis_dim}"
        )

    decomp_cov = np.asarray(pd.read_csv(args.decomp_cov_csv), dtype=np.float64)
    mean = np.asarray(pd.read_csv(args.mean_csv), dtype=np.float64).squeeze()
    sigma = decomp_cov.T @ decomp_cov

    lam_min, lam_max = make_lambda_box(args.dim_lambda)
    bounds = list(zip(lam_min.tolist(), lam_max.tolist()))
    phi_lam = make_basis_from_summary(summary, args.dim_lambda, lam_min, lam_max)
    rows = []
    for snapshot_index, snapshot_epoch in enumerate(snapshot_epochs):
        beta = find_snapshot(snapshots, snapshot_epoch)
        gap = make_gap_function(beta, phi_lam, sigma, mean, int(summary["basis_dim"]))
        result = maximize_gap(gap, bounds, args)

        row = {
            "gamma": gamma,
            "gamma_index": grid_index,
            "grid_index": grid_index,
            "basis_dim": summary_basis_dim,
            "epoch_index": snapshot_epoch,
            "snapshot_index": snapshot_index,
            "worst_case_gap": result["worst_case_gap"],
            "optimizer_method": args.optimizer_method_label,
            "optimizer_status": result["optimizer_status"],
            "objective_evaluations": result["objective_evaluations"],
            "random_seed": args.seed,
            "elapsed_seconds": result["elapsed_seconds"],
            "run_dir": str(run_dir),
        }
        for idx, value in enumerate(result["lambda"], start=1):
            row[f"lambda_{idx}"] = float(value)
        rows.append(row)

        epoch_out_path = run_dir / f"worst_case_gap_epoch_{snapshot_epoch}.csv"
        pd.DataFrame([row]).to_csv(epoch_out_path, index=False)
        print(f"wrote {epoch_out_path}")

    out_path = run_dir / "worst_case_gap.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-csv", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--sweep-axis",
        choices=["gamma", "basis_dim"],
        default="gamma",
        help="Grid dimension used for locating run directories.",
    )
    parser.add_argument("--snapshot-epochs", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dim-lambda", type=int, default=12)
    parser.add_argument("--decomp-cov-csv", type=Path, default=PROJECT_ROOT / "decomp_cov.csv")
    parser.add_argument("--mean-csv", type=Path, default=PROJECT_ROOT / "mean.csv")
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--num-random-starts", type=int, default=2048)
    parser.add_argument("--num-local-starts", type=int, default=16)
    parser.add_argument("--local-maxiter", type=int, default=500)
    parser.add_argument("--local-ftol", type=float, default=1e-10)
    parser.add_argument("--use-differential-evolution", action="store_true")
    parser.add_argument("--de-maxiter", type=int, default=80)
    parser.add_argument("--de-popsize", type=int, default=10)
    parser.add_argument("--de-tol", type=float, default=1e-6)
    parser.add_argument("--optimizer-method-label", default="random_search_plus_lbfgsb")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
