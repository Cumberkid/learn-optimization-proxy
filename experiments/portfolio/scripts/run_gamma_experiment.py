#!/usr/bin/env python
"""Run one regularized SGD gamma experiment.

This is the cluster-friendly version of the repeated gamma blocks from
``03 Regularized SGD (high demension ERM).ipynb``.  It is designed for one
process per gamma, e.g. through a Slurm job array on SAVIO.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from lib.fast_tensor_data_loader import FastTensorDataLoader
from lib.lsp.learn_solution_path import learn_solution_path_high_dim


def taylor(lam, k: int):
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    taylor_terms = np.append(np.ones(1), lam)
    for i in range(k):
        taylor_terms = np.append(taylor_terms, lam * (lam[0] ** (i + 1)))
    return taylor_terms


def make_taylor_expansion(k: int):
    def taylor_expansion(lam, basis_dim, device="cpu"):
        del basis_dim
        vec = torch.tensor(taylor(lam, k), dtype=torch.float32)
        return vec.to(device)

    return taylor_expansion


def default_taylor_basis_dim(dim_lambda: int, taylor_order: int) -> int:
    return (taylor_order + 1) * dim_lambda + 1


def make_lambda_box(dim_lambda: int):
    lam_max = np.ones(dim_lambda, dtype=float)
    lam_min = np.zeros(dim_lambda, dtype=float)
    lam_min[2:] = -2.0
    lam_min[0] = 0.2
    lam_max[2:] = 2.0
    return lam_min, lam_max


def compute_basis_outer_eigmax_bound(args, dim_lambda: int, basis_dim: int, lam_min, lam_max):
    if args.basis == "taylor":
        lam0_max = float(lam_max[0])
        squared_sum_max = float((lam_max[0] ** 2) + (lam_max[1] ** 2) + np.sum(lam_max[2:] ** 2))
        taylor_block_sum = sum(lam0_max ** (2 * r) for r in range(args.taylor_order + 1))
        return {
            "uniform_outer_eigmax_bound": 1.0 + squared_sum_max * taylor_block_sum,
            "bound_type": "exact_closed_form",
        }

    if args.basis == "random_fourier":
        feature_dim = basis_dim - 1
        if feature_dim < 1:
            raise ValueError("random_fourier requires basis_dim >= 2")
        pool_feature_dim = random_feature_pool_dim(args, basis_dim) - 1
        scale_bound = 1.0 + 2.0 * feature_dim / pool_feature_dim if args.basis_scale_mode == "pool" else 3.0
        return {
            "uniform_outer_eigmax_bound": scale_bound,
            "bound_type": "exact_closed_form",
            "feature_dim": feature_dim,
            "pool_feature_dim": pool_feature_dim,
            "basis_scale_mode": args.basis_scale_mode,
        }

    if args.basis == "random_relu":
        feature_dim = basis_dim - 1
        if feature_dim < 1:
            raise ValueError("random_relu requires basis_dim >= 2")
        pool_feature_dim = random_feature_pool_dim(args, basis_dim) - 1

        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.basis_seed)
        pooled_random_weight = torch.randn(
            pool_feature_dim,
            dim_lambda,
            generator=generator,
            dtype=torch.float32,
        ) * args.random_feature_scale
        pooled_random_bias = torch.randn(
            pool_feature_dim,
            generator=generator,
            dtype=torch.float32,
        ) * args.random_bias_scale
        random_weight = pooled_random_weight[:feature_dim]
        random_bias = pooled_random_bias[:feature_dim]

        scale_feature_dim = pool_feature_dim if args.basis_scale_mode == "pool" else feature_dim
        feature_scale_sq = 2.0 / scale_feature_dim
        lam_min_t = torch.tensor(lam_min, dtype=torch.float32)
        lam_max_t = torch.tensor(lam_max, dtype=torch.float32)

        endpoint_choice = torch.where(random_weight >= 0, lam_max_t, lam_min_t)
        affine_sup = (random_weight * endpoint_choice).sum(dim=1) + random_bias
        unit_sups = torch.relu(affine_sup)
        analytic_upper_bound = float(1.0 + feature_scale_sq * torch.sum(unit_sups.square()).item())

        corners = np.array(list(itertools.product(*zip(lam_min, lam_max))), dtype=np.float32)
        corner_tensor = torch.tensor(corners, dtype=torch.float32)
        hidden = torch.relu(corner_tensor @ random_weight.T + random_bias)
        corner_values = 1.0 + feature_scale_sq * torch.sum(hidden.square(), dim=1)
        exact_corner_upper_bound = float(torch.max(corner_values).item())

        return {
            "uniform_outer_eigmax_bound": exact_corner_upper_bound,
            "bound_type": "exact_corner_enumeration",
            "analytic_upper_bound": analytic_upper_bound,
            "feature_dim": feature_dim,
            "pool_feature_dim": pool_feature_dim,
            "basis_scale_mode": args.basis_scale_mode,
            "corner_count": int(corners.shape[0]),
        }

    raise ValueError(f"Unsupported basis for bound computation: {args.basis}")


def random_feature_pool_dim(args, basis_dim: int) -> int:
    if args.basis_dim_pool is None:
        return basis_dim
    if args.basis_dim_pool < basis_dim:
        raise ValueError(
            f"--basis-dim-pool must be >= --basis-dim; got pool={args.basis_dim_pool}, basis_dim={basis_dim}"
        )
    return args.basis_dim_pool


def make_random_relu_features(
    dim_lambda: int,
    basis_dim: int,
    seed: int,
    weight_scale: float,
    bias_scale: float,
    pool_basis_dim: int | None = None,
    basis_scale_mode: str = "basis_dim",
):
    if basis_dim < 2:
        raise ValueError("random_relu requires basis_dim >= 2")

    feature_dim = basis_dim - 1
    pool_basis_dim = basis_dim if pool_basis_dim is None else pool_basis_dim
    if pool_basis_dim < basis_dim:
        raise ValueError(f"pool_basis_dim must be >= basis_dim; got {pool_basis_dim} < {basis_dim}")
    pool_feature_dim = pool_basis_dim - 1
    if basis_scale_mode not in {"basis_dim", "pool"}:
        raise ValueError(f"Unsupported basis_scale_mode: {basis_scale_mode}")
    scale_feature_dim = pool_feature_dim if basis_scale_mode == "pool" else feature_dim

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pooled_random_weight = torch.randn(
        pool_feature_dim,
        dim_lambda,
        generator=generator,
        dtype=torch.float32,
    ) * weight_scale
    pooled_random_bias = torch.randn(
        pool_feature_dim,
        generator=generator,
        dtype=torch.float32,
    ) * bias_scale
    random_weight = pooled_random_weight[:feature_dim]
    random_bias = pooled_random_bias[:feature_dim]
    feature_scale = math.sqrt(2.0 / scale_feature_dim)

    def random_relu_features(lam, basis_dim_arg, device="cpu"):
        del basis_dim_arg
        lam_vec = torch.as_tensor(lam, dtype=torch.float32, device=device).reshape(-1)
        if lam_vec.numel() != dim_lambda:
            raise ValueError(f"expected lambda dimension {dim_lambda}, got {lam_vec.numel()}")
        weight = random_weight.to(device)
        bias = random_bias.to(device)
        hidden = torch.relu(weight @ lam_vec + bias)
        intercept = torch.ones(1, dtype=torch.float32, device=device)
        return torch.cat((intercept, feature_scale * hidden))

    return random_relu_features


def make_random_fourier_features(
    dim_lambda: int,
    basis_dim: int,
    seed: int,
    frequency_scale: float,
    lam_min,
    lam_max,
    pool_basis_dim: int | None = None,
    basis_scale_mode: str = "basis_dim",
):
    if basis_dim < 2:
        raise ValueError("random_fourier requires basis_dim >= 2")

    feature_dim = basis_dim - 1
    pool_basis_dim = basis_dim if pool_basis_dim is None else pool_basis_dim
    if pool_basis_dim < basis_dim:
        raise ValueError(f"pool_basis_dim must be >= basis_dim; got {pool_basis_dim} < {basis_dim}")
    pool_feature_dim = pool_basis_dim - 1
    if basis_scale_mode not in {"basis_dim", "pool"}:
        raise ValueError(f"Unsupported basis_scale_mode: {basis_scale_mode}")
    scale_feature_dim = pool_feature_dim if basis_scale_mode == "pool" else feature_dim

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pooled_random_frequency = torch.randn(
        pool_feature_dim,
        dim_lambda,
        generator=generator,
        dtype=torch.float32,
    ) * frequency_scale
    pooled_random_phase = 2.0 * math.pi * torch.rand(
        pool_feature_dim,
        generator=generator,
        dtype=torch.float32,
    )
    random_frequency = pooled_random_frequency[:feature_dim]
    random_phase = pooled_random_phase[:feature_dim]
    feature_scale = math.sqrt(2.0 / scale_feature_dim)
    lam_min_cpu = torch.tensor(lam_min, dtype=torch.float32)
    lam_range_cpu = torch.tensor(np.asarray(lam_max) - np.asarray(lam_min), dtype=torch.float32)
    if torch.any(lam_range_cpu <= 0):
        raise ValueError("Each lambda coordinate must have lam_max > lam_min for Fourier normalization")

    def random_fourier_features(lam, basis_dim_arg, device="cpu"):
        del basis_dim_arg
        lam_vec = torch.as_tensor(lam, dtype=torch.float32, device=device).reshape(-1)
        if lam_vec.numel() != dim_lambda:
            raise ValueError(f"expected lambda dimension {dim_lambda}, got {lam_vec.numel()}")
        frequency = random_frequency.to(device)
        phase = random_phase.to(device)
        lam_min_t = lam_min_cpu.to(device)
        lam_range_t = lam_range_cpu.to(device)
        lam_normalized = 2.0 * (lam_vec - lam_min_t) / lam_range_t - 1.0
        features = torch.cos(frequency @ lam_normalized + phase)
        intercept = torch.ones(1, dtype=torch.float32, device=device)
        return torch.cat((intercept, feature_scale * features))

    return random_fourier_features


def make_basis(args, dim_lambda: int, lam_min, lam_max):
    default_basis_dim = default_taylor_basis_dim(dim_lambda, args.taylor_order)
    basis_dim = args.basis_dim if args.basis_dim is not None else default_basis_dim

    if args.basis == "taylor":
        if args.basis_dim is not None and args.basis_dim != default_basis_dim:
            raise ValueError(
                "Taylor basis_dim is determined by dim_lambda and taylor_order; "
                f"expected {default_basis_dim}, got {args.basis_dim}"
            )
        return default_basis_dim, make_taylor_expansion(args.taylor_order)
    if args.basis == "random_relu":
        return basis_dim, make_random_relu_features(
            dim_lambda,
            basis_dim,
            args.basis_seed,
            args.random_feature_scale,
            args.random_bias_scale,
            random_feature_pool_dim(args, basis_dim),
            args.basis_scale_mode,
        )
    if args.basis == "random_fourier":
        return basis_dim, make_random_fourier_features(
            dim_lambda,
            basis_dim,
            args.basis_seed,
            args.random_feature_scale,
            lam_min,
            lam_max,
            random_feature_pool_dim(args, basis_dim),
            args.basis_scale_mode,
        )
    raise ValueError(f"Unsupported basis: {args.basis}")


def make_initial_weight(args, input_dim: int, basis_dim: int):
    if args.weight_init == "zero":
        return None
    if args.weight_init == "random":
        init_basis_dim = basis_dim
        if args.init_basis_dim_pool is not None:
            if args.init_basis_dim_pool < basis_dim:
                raise ValueError(
                    f"--init-basis-dim-pool must be >= --basis-dim; got "
                    f"pool={args.init_basis_dim_pool}, basis_dim={basis_dim}"
                )
            init_basis_dim = args.init_basis_dim_pool
        init_generator = torch.Generator(device="cpu")
        init_generator.manual_seed(args.init_seed)
        init_weight = torch.randn(
            input_dim,
            init_basis_dim,
            generator=init_generator,
            dtype=torch.float32,
        ) * args.init_scale
        return init_weight[:, :basis_dim]
    raise ValueError(f"Unsupported weight initialization: {args.weight_init}")


def thresh_basis(basis_dim):
    del basis_dim
    return -1


def allocation_cost_reg(gamma, hyper_params, decomp_cov, mean, model, device="cpu"):
    theta = model(hyper_params, device)
    risk = torch.mm(decomp_cov, theta.view(-1, 1))
    exp_rtrn = torch.mm(mean.view(-1, 1).T, theta.view(-1, 1))
    target = torch.tensor(
        hyper_params[2:],
        dtype=torch.float32,
        device=decomp_cov.device,
    )
    cost = (theta - target).norm(p=2) ** 2
    return (
        hyper_params[0] * risk.norm(p=2) ** 2
        - hyper_params[1] * exp_rtrn
        + cost
        + gamma * 0.5 * ((model.linear.weight).norm(p=2) ** 2)
    )


def allocation_cost_val(hyper_params, decomp_cov, mean, model, device="cpu"):
    theta = model(hyper_params, device)
    risk = torch.mm(decomp_cov, theta.view(-1, 1))
    exp_rtrn = torch.mm(mean.view(-1, 1).T, theta.view(-1, 1))
    target = torch.tensor(
        hyper_params[2:],
        dtype=torch.float32,
        device=decomp_cov.device,
    )
    cost = (theta - target).norm(p=2) ** 2
    return hyper_params[0] * risk.norm(p=2) ** 2 - hyper_params[1] * exp_rtrn + cost


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log_linear_lr(gamma: float) -> float:
    """Interpolate between (gamma=1, lr=1e-5) and (gamma=10, lr=1e-4)."""
    if gamma <= 0:
        raise ValueError("gamma must be positive for log-linear learning-rate interpolation")
    log_lr = math.log10(1e-5) + (math.log10(gamma) - math.log10(1.0)) * (
        (math.log10(1e-4) - math.log10(1e-5))
        / (math.log10(10.0) - math.log10(1.0))
    )
    return 10**log_lr


def adaptive_lr_c(args, basis_dim: int, basis_outer_eigmax_bound: dict) -> float:
    if args.adaptive_lr_c is not None:
        return args.adaptive_lr_c
    if args.basis == "random_fourier":
        return 3.0
    if args.basis == "random_relu":
        relu_constants = {16: 133.0, 32: 104.0, 64: 99.0}
        if basis_dim not in relu_constants:
            raise ValueError(
                "adaptive gamma for random_relu needs --adaptive-lr-c when "
                f"basis_dim is not one of {sorted(relu_constants)}; got {basis_dim}"
            )
        return relu_constants[basis_dim]
    return float(basis_outer_eigmax_bound["uniform_outer_eigmax_bound"])


def gamma_lipschitz_lr(gamma: float, c_value: float, l_value: float) -> float:
    return 1.0 / (c_value * l_value + gamma)


def read_grid_row(grid_csv: Path, task_id: int):
    with grid_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if task_id < 0 or task_id >= len(rows):
        raise IndexError(f"task id {task_id} outside grid with {len(rows)} rows")
    row = rows[task_id]
    gamma = float(row["gamma"])
    lr = float(row["lr"]) if row.get("lr") else log_linear_lr(gamma)
    basis_dim = int(row["basis_dim"]) if row.get("basis_dim") else None
    return {
        "gamma": gamma,
        "lr": lr,
        "basis_dim": basis_dim,
    }


def run_dir_name(args, gamma: float, basis_dim: int):
    if args.sweep_axis == "gamma":
        return f"gamma_{gamma:.6g}".replace(".", "p")
    if args.sweep_axis == "basis_dim":
        return f"basis_dim_{basis_dim}"
    raise ValueError(f"Unsupported sweep axis: {args.sweep_axis}")


def make_diagnostic_epochs(args, gamma: float):
    if args.diagnostic_start_epoch is None or args.diagnostic_end_epoch is None:
        return []
    if args.diagnostic_gamma is not None and gamma != args.diagnostic_gamma:
        return []
    start_epoch = args.diagnostic_start_epoch
    end_epoch = args.diagnostic_end_epoch
    if end_epoch < start_epoch:
        raise ValueError("diagnostic_end_epoch must be >= diagnostic_start_epoch")
    return list(range(start_epoch, end_epoch + 1))


def parse_epoch_list(epoch_list: str):
    if not epoch_list:
        return []
    epochs = []
    for raw_item in epoch_list.split(","):
        item = raw_item.strip()
        if not item:
            continue
        epoch = int(item)
        if epoch < 0:
            raise ValueError("beta snapshot epochs must be nonnegative")
        epochs.append(epoch)
    return sorted(set(epochs))


def run(args):
    if args.random_feature_scale is None:
        args.random_feature_scale = 0.25 if args.basis == "random_fourier" else 1.0
    if args.adaptive_gamma:
        if args.adaptive_gamma_shrink <= 0:
            raise ValueError("--adaptive-gamma-shrink must be positive")
        if args.adaptive_gamma_window < 1:
            raise ValueError("--adaptive-gamma-window must be a positive integer")
        if args.adaptive_l <= 0:
            raise ValueError("--adaptive-l must be positive")
        if args.adaptive_gamma_saturation_threshold < 0:
            raise ValueError("--adaptive-gamma-saturation-threshold must be nonnegative")
        if args.adaptive_gamma_threshold_alpha < 0:
            raise ValueError("--adaptive-gamma-threshold-alpha must be nonnegative")
        if args.adaptive_gamma_threshold_scale_min <= 0:
            raise ValueError("--adaptive-gamma-threshold-scale-min must be positive")
        if args.adaptive_gamma_threshold_scale_max < args.adaptive_gamma_threshold_scale_min:
            raise ValueError("--adaptive-gamma-threshold-scale-max must be >= --adaptive-gamma-threshold-scale-min")
        if (
            args.adaptive_gamma_max_epochs_per_value is not None
            and args.adaptive_gamma_max_epochs_per_value < 1
        ):
            raise ValueError("--adaptive-gamma-max-epochs-per-value must be a positive integer")

    if args.grid_csv is not None:
        grid_row = read_grid_row(args.grid_csv, args.task_id)
        gamma = grid_row["gamma"]
        lr = grid_row["lr"]
        if grid_row["basis_dim"] is not None:
            args.basis_dim = grid_row["basis_dim"]
    else:
        gamma = args.gamma
        lr = args.lr if args.lr is not None else log_linear_lr(args.gamma)
    if args.adaptive_gamma and args.adaptive_gamma_start is not None:
        gamma = args.adaptive_gamma_start

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    decomp_cov_np = np.array(pd.read_csv(args.decomp_cov_csv))
    mean_np = np.array(pd.read_csv(args.mean_csv)).squeeze()
    decomp_cov = torch.tensor(decomp_cov_np, dtype=torch.float32, device=device)
    mean = torch.tensor(mean_np, dtype=torch.float32, device=device)

    GD_data_loader = FastTensorDataLoader(
        decomp_cov,
        mean,
        batch_size=len(decomp_cov),
        shuffle=False,
    )
    test_data_loader = FastTensorDataLoader(
        decomp_cov,
        mean,
        batch_size=len(decomp_cov),
        shuffle=False,
    )

    dim_lambda = args.dim_lambda
    lam_min, lam_max = make_lambda_box(dim_lambda)

    input_dim = decomp_cov.shape[1]
    basis_dim, phi_lam = make_basis(args, dim_lambda, lam_min, lam_max)
    basis_outer_eigmax_bound = compute_basis_outer_eigmax_bound(
        args,
        dim_lambda,
        basis_dim,
        lam_min,
        lam_max,
    )
    adaptive_c_value = None
    adaptive_lr_constant = None
    if args.adaptive_gamma:
        adaptive_c_value = adaptive_lr_c(args, basis_dim, basis_outer_eigmax_bound)
        adaptive_lr_constant = adaptive_c_value * args.adaptive_l
        lr = gamma_lipschitz_lr(gamma, adaptive_c_value, args.adaptive_l)

    init_weight = make_initial_weight(args, input_dim, basis_dim)
    gamma_state = {"gamma": gamma, "gamma_ref": gamma}

    def train_loss_fn(hyper_params, X, y, model, device="cpu"):
        return allocation_cost_reg(gamma_state["gamma"], hyper_params, X, y, model, device)

    val_loss_fn = allocation_cost_val
    start = perf_counter()

    diagnostic_epochs = make_diagnostic_epochs(args, gamma)
    beta_snapshot_epochs = parse_epoch_list(args.beta_snapshot_epochs)
    beta_snapshot_epochs = [epoch for epoch in beta_snapshot_epochs if epoch <= args.epochs]

    (
        eval_epochs,
        loss_train,
        loss_val,
        loss_train_reg,
        loss_val_reg,
        weight,
        diagnostics,
        beta_snapshots,
        adaptive_records,
    ) = learn_solution_path_high_dim(
        input_dim,
        basis_dim,
        phi_lam,
        args.epochs,
        GD_data_loader,
        test_data_loader,
        train_loss_fn,
        val_loss_fn,
        lam_min,
        lam_max,
        init_lr=lr,
        diminish=False,
        init_weight=init_weight,
        intercept=False,
        weighted_avg=False,
        thresh_basis=thresh_basis,
        record_frequency=args.record_frequency,
        distribution=args.distribution,
        opt_method="SGD",
        device=device,
        trace_frequency=args.trace_frequency,
        return_eval_epochs=True,
        diagnostic_epochs=diagnostic_epochs,
        return_diagnostics=True,
        beta_snapshot_epochs=beta_snapshot_epochs,
        return_beta_snapshots=True,
        adaptive_gamma_state=gamma_state,
        adaptive_gamma_saturation_metric=(
            args.adaptive_gamma_saturation_metric if args.adaptive_gamma else "none"
        ),
        adaptive_gamma_saturation_threshold=args.adaptive_gamma_saturation_threshold,
        adaptive_gamma_window=args.adaptive_gamma_window,
        adaptive_gamma_shrink=args.adaptive_gamma_shrink,
        adaptive_gamma_lr_constant=adaptive_lr_constant,
        adaptive_gamma_threshold_alpha=args.adaptive_gamma_threshold_alpha,
        adaptive_gamma_threshold_scale_min=args.adaptive_gamma_threshold_scale_min,
        adaptive_gamma_threshold_scale_max=args.adaptive_gamma_threshold_scale_max,
        adaptive_gamma_max_epochs_per_value=args.adaptive_gamma_max_epochs_per_value,
        return_adaptive_records=True,
        return_val_regularized=True,
    )

    if args.adaptive_gamma:
        out_dir_name = (
            f"adaptive_gamma_start_{gamma:.6g}_"
            f"{args.adaptive_gamma_saturation_metric}_threshold_"
            f"{args.adaptive_gamma_saturation_threshold:.6g}_"
            f"alpha_{args.adaptive_gamma_threshold_alpha:.6g}_"
            f"scale_{args.adaptive_gamma_threshold_scale_min:.6g}_"
            f"{args.adaptive_gamma_threshold_scale_max:.6g}_"
            f"max_epochs_{args.adaptive_gamma_max_epochs_per_value or 'none'}"
        )
        out_dir = args.out_dir / out_dir_name.replace(".", "p")
    else:
        out_dir = args.out_dir / run_dir_name(args, gamma, basis_dim)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(
        {
            "epoch_index": eval_epochs,
            "objective_train_unregularized": loss_train,
            "objective_val_unregularized": loss_val,
            "gamma": adaptive_records["gamma_history"],
            "learning_rate": adaptive_records["learning_rate_history"],
        }
    )
    if len(loss_train_reg) == len(loss_train):
        history["objective_train_gamma_regularized"] = loss_train_reg
    if len(loss_val_reg) == len(loss_val):
        history["objective_val_gamma_regularized"] = loss_val_reg
    # Backward-compatible aliases for older plotting code.
    history["loss_train"] = history["objective_train_unregularized"]
    history["loss_val"] = history["objective_val_unregularized"]
    if "objective_train_gamma_regularized" in history.columns:
        history["loss_train_reg"] = history["objective_train_gamma_regularized"]
    if "objective_val_gamma_regularized" in history.columns:
        history["loss_val_reg"] = history["objective_val_gamma_regularized"]
    history.to_csv(out_dir / "history.csv", index=False)
    np.savez(
        out_dir / "losses.npz",
        eval_epochs=np.asarray(eval_epochs),
        objective_train_unregularized=np.asarray(loss_train),
        objective_val_unregularized=np.asarray(loss_val),
        objective_train_gamma_regularized=np.asarray(loss_train_reg),
        objective_val_gamma_regularized=np.asarray(loss_val_reg),
        gamma=np.asarray(adaptive_records["gamma_history"], dtype=float),
        learning_rate=np.asarray(adaptive_records["learning_rate_history"], dtype=float),
        loss_train=np.asarray(loss_train),
        loss_val=np.asarray(loss_val),
        loss_train_reg=np.asarray(loss_train_reg),
        loss_val_reg=np.asarray(loss_val_reg),
    )
    torch.save(weight.detach().cpu(), out_dir / "weight.pt")
    if beta_snapshots:
        torch.save(beta_snapshots, out_dir / "beta_snapshots.pt")
        pd.DataFrame(
            {
                "epoch_index": [item["epoch_index"] for item in beta_snapshots],
                "shape": [list(item["beta"].shape) for item in beta_snapshots],
            }
        ).to_csv(out_dir / "beta_snapshots.csv", index=False)
    if diagnostics:
        torch.save(diagnostics, out_dir / "diagnostics.pt")
        diagnostic_rows = []
        for item in diagnostics:
            diagnostic_rows.append(
                {
                    "epoch_index": item["epoch_index"],
                    "learning_rate": item["learning_rate"],
                    "loss_pre_step": item["loss_pre_step"],
                    "phi_norm": item["phi_norm"],
                    "phi_max_abs": item["phi_max_abs"],
                    "phi_outer_eigmax": item["phi_outer_eigmax"],
                    "beta_update_norm": item["beta_update_norm"],
                    "beta_max_abs_after": item["beta_max_abs_after"],
                    "theta_update_norm": item["theta_update_norm"],
                    "theta_max_abs_after": item["theta_max_abs_after"],
                    "grad_norm": item["grad_norm"],
                    "grad_max_abs": item["grad_max_abs"],
                }
            )
        pd.DataFrame(diagnostic_rows).to_csv(out_dir / "diagnostics_summary.csv", index=False)
    if args.adaptive_gamma:
        adaptive_event_history = pd.DataFrame(adaptive_records["events"])
        adaptive_event_history.to_csv(out_dir / "adaptive_gamma_events.csv", index=False)
        if not adaptive_event_history.empty:
            adaptive_event_history[
                ["epoch_index", "detector", "old_gamma", "new_gamma", "new_learning_rate"]
            ].to_csv(out_dir / "adaptive_gamma_shrink_epochs.csv", index=False)
        else:
            pd.DataFrame(
                columns=["epoch_index", "detector", "old_gamma", "new_gamma", "new_learning_rate"]
            ).to_csv(out_dir / "adaptive_gamma_shrink_epochs.csv", index=False)

    final = {
        "epoch_index": int(eval_epochs[-1]) if eval_epochs else None,
        "objective_train_unregularized": float(loss_train[-1]) if loss_train else None,
        "objective_val_unregularized": float(loss_val[-1]) if loss_val else None,
        "objective_train_gamma_regularized": float(loss_train_reg[-1]) if loss_train_reg else None,
        "objective_val_gamma_regularized": float(loss_val_reg[-1]) if loss_val_reg else None,
        "gamma": float(adaptive_records["gamma_history"][-1]) if adaptive_records["gamma_history"] else gamma,
        "learning_rate": (
            float(adaptive_records["learning_rate_history"][-1])
            if adaptive_records["learning_rate_history"]
            else lr
        ),
        "loss_train": float(loss_train[-1]) if loss_train else None,
        "loss_val": float(loss_val[-1]) if loss_val else None,
        "loss_train_reg": float(loss_train_reg[-1]) if loss_train_reg else None,
        "loss_val_reg": float(loss_val_reg[-1]) if loss_val_reg else None,
    }

    summary = {
        "gamma": gamma,
        "learning_rate": lr,
        "final_gamma": final["gamma"],
        "final_learning_rate": final["learning_rate"],
        "epochs": args.epochs,
        "seed": args.seed,
        "init_seed": args.init_seed,
        "init_scale": args.init_scale,
        "init_basis_dim_pool": args.init_basis_dim_pool,
        "weight_init": args.weight_init,
        "device": device,
        "basis": args.basis,
        "basis_dim": basis_dim,
        "basis_dim_pool": random_feature_pool_dim(args, basis_dim) if args.basis in {"random_relu", "random_fourier"} else None,
        "basis_scale_mode": args.basis_scale_mode if args.basis in {"random_relu", "random_fourier"} else None,
        "sweep_axis": args.sweep_axis,
        "basis_seed": args.basis_seed,
        "random_feature_scale": args.random_feature_scale,
        "random_bias_scale": args.random_bias_scale,
        "random_fourier_input_normalization": (
            "coordinatewise [-1, 1] using lam_min and lam_max"
            if args.basis == "random_fourier"
            else None
        ),
        "basis_outer_eigmax_bound": basis_outer_eigmax_bound,
        "adaptive_gamma": {
            "enabled": args.adaptive_gamma,
            "start_gamma": gamma if args.adaptive_gamma else None,
            "shrink": args.adaptive_gamma_shrink if args.adaptive_gamma else None,
            "saturation_metric": (
                args.adaptive_gamma_saturation_metric if args.adaptive_gamma else None
            ),
            "saturation_threshold": (
                args.adaptive_gamma_saturation_threshold if args.adaptive_gamma else None
            ),
            "saturation_threshold_rule": (
                "threshold_ref * clip((gamma / gamma_ref)^alpha, scale_min, scale_max)"
                if args.adaptive_gamma
                else None
            ),
            "saturation_threshold_ref": (
                args.adaptive_gamma_saturation_threshold if args.adaptive_gamma else None
            ),
            "saturation_threshold_alpha": (
                args.adaptive_gamma_threshold_alpha if args.adaptive_gamma else None
            ),
            "saturation_threshold_scale_min": (
                args.adaptive_gamma_threshold_scale_min if args.adaptive_gamma else None
            ),
            "saturation_threshold_scale_max": (
                args.adaptive_gamma_threshold_scale_max if args.adaptive_gamma else None
            ),
            "gamma_ref": gamma if args.adaptive_gamma else None,
            "max_epochs_per_value": (
                args.adaptive_gamma_max_epochs_per_value if args.adaptive_gamma else None
            ),
            "window": args.adaptive_gamma_window if args.adaptive_gamma else None,
            "lipschitz_L": args.adaptive_l if args.adaptive_gamma else None,
            "lipschitz_C": adaptive_c_value,
            "learning_rate_formula": "1 / (C * L + gamma)" if args.adaptive_gamma else None,
            "num_shrinks": len(adaptive_records["events"]) if args.adaptive_gamma else 0,
        },
        "taylor_order": args.taylor_order,
        "beta_snapshot_epochs_requested": beta_snapshot_epochs,
        "beta_snapshot_epochs_saved": [int(item["epoch_index"]) for item in beta_snapshots],
        "elapsed_seconds": perf_counter() - start,
        "final": final,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--adaptive-gamma",
        action="store_true",
        help="Shrink gamma during training when the selected rolling saturation detector fires.",
    )
    parser.add_argument(
        "--adaptive-gamma-start",
        type=float,
        default=None,
        help="Starting gamma for adaptive runs. Defaults to --gamma or the selected grid row.",
    )
    parser.add_argument("--adaptive-gamma-shrink", type=float, default=0.5)
    parser.add_argument(
        "--adaptive-gamma-saturation-metric",
        choices=["gradient", "objective"],
        default="gradient",
    )
    parser.add_argument("--adaptive-gamma-saturation-threshold", type=float, default=1e-4)
    parser.add_argument("--adaptive-gamma-window", type=int, default=5)
    parser.add_argument(
        "--adaptive-gamma-threshold-alpha",
        type=float,
        default=0.5,
        help="Power-law exponent for threshold(gamma) = threshold_ref * (gamma/gamma_ref)^alpha.",
    )
    parser.add_argument("--adaptive-gamma-threshold-scale-min", type=float, default=0.01)
    parser.add_argument("--adaptive-gamma-threshold-scale-max", type=float, default=100.0)
    parser.add_argument(
        "--adaptive-gamma-max-epochs-per-value",
        type=int,
        default=None,
        help="Force gamma to shrink after this many training epochs at one gamma value.",
    )
    parser.add_argument("--adaptive-l", type=float, default=476.0)
    parser.add_argument(
        "--adaptive-lr-c",
        type=float,
        default=None,
        help="Override C in the adaptive learning rate 1/(C*L+gamma).",
    )
    parser.add_argument("--grid-csv", type=Path, default=None)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--sweep-axis",
        choices=["gamma", "basis_dim"],
        default="gamma",
        help="Grid dimension used for naming run directories.",
    )
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--record-frequency", type=int, default=10)
    parser.add_argument("--trace-frequency", type=int, default=100)
    parser.add_argument("--diagnostic-gamma", type=float, default=10.0)
    parser.add_argument("--diagnostic-start-epoch", type=int, default=12450)
    parser.add_argument("--diagnostic-end-epoch", type=int, default=12550)
    parser.add_argument(
        "--beta-snapshot-epochs",
        default="",
        help="Comma-separated training epochs at which to save beta snapshots, e.g. 0,2500,5000.",
    )
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--init-seed", type=int, default=8675309)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument(
        "--init-basis-dim-pool",
        type=int,
        default=None,
        help=(
            "Draw random initialization at this pooled basis dimension and use the prefix. "
            "Leave unset to draw directly at --basis-dim."
        ),
    )
    parser.add_argument("--weight-init", choices=["random", "zero"], default="random")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--distribution", default="uniform")
    parser.add_argument("--dim-lambda", type=int, default=12)
    parser.add_argument(
        "--basis",
        choices=["taylor", "random_relu", "random_fourier"],
        default="taylor",
    )
    parser.add_argument(
        "--basis-dim",
        type=int,
        default=None,
        help="Number of basis functions for random bases. Taylor dimension is determined by --taylor-order.",
    )
    parser.add_argument(
        "--basis-dim-pool",
        type=int,
        default=None,
        help=(
            "Generate random features from this larger basis dimension and take the prefix. "
            "Use with --basis-scale-mode pool for strictly nested basis-dimension sweeps."
        ),
    )
    parser.add_argument(
        "--basis-scale-mode",
        choices=["basis_dim", "pool"],
        default="basis_dim",
        help=(
            "Use sqrt(2/(basis_dim-1)) scaling, or sqrt(2/(basis_dim_pool-1)) scaling "
            "when drawing from a larger pooled random basis."
        ),
    )
    parser.add_argument("--basis-seed", type=int, default=8675309)
    parser.add_argument(
        "--random-feature-scale",
        type=float,
        default=None,
        help="Random feature/frequency scale. Defaults to 0.25 for random_fourier and 1.0 otherwise.",
    )
    parser.add_argument("--random-bias-scale", type=float, default=1.0)
    parser.add_argument("--taylor-order", type=int, default=7)
    parser.add_argument("--decomp-cov-csv", type=Path, default=PROJECT_ROOT / "decomp_cov.csv")
    parser.add_argument("--mean-csv", type=Path, default=PROJECT_ROOT / "mean.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results/gamma_sweep_sgd")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
