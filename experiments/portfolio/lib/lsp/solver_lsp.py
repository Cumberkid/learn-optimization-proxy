import torch
import numpy as np
from scipy.stats import semicircular, uniform, expon, beta
from lib.utils import gauss_legendre_integral, gauss_legendre_integral_2d, monte_carlo

# itr: input is number of iterations run before current epoch, returns number of iterations run after current epoch
# avg_weight: keeps track of and updates weighted average iterates including before current epoch according to Lacoste-Julien et al.
# step_size function take in 2 parameters: current iteration number, and a self-defined constant const
# step_size function returns the learning rate for current iteration
# distribution: takes parameters 'uniform', 'exponential', 'semicircle', 'beta'
# alpha_beta holds the parameters for the beta distribution. Default case [-.5, -.5] yields Chebyshev
def train_lsp(itr, init_weight, dataloader, model, loss_fn, optimizer, lam_min=[0], lam_max=[1], weighted_avg=True, 
              step_size=None, const=None, distribution='uniform', alpha_beta=[-.5, -.5], device='cpu',
              capture_diagnostics=False):
    model.train()
    avg_weight = init_weight
    diagnostic_payload = None
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        # hyper_params = np.zeros(len(lam_max))
        # if distribution == 'uniform':
        #     samples = uniform.rvs(loc=0, scale=1, size=len(lam_min))
        #     hyper_params = samples * (lam_max - lam_min) + lam_min
        
        lam_min_arr = np.asarray(lam_min)
        lam_max_arr = np.asarray(lam_max)
        lam_range = lam_max_arr - lam_min_arr

        # SGD picks random regulation parameter lambda
        if distribution == 'uniform':
            hyper_params = uniform.rvs(loc=lam_min_arr, scale=lam_range, size=len(lam_min_arr))
        elif distribution == 'exponential':
            hyper_params = expon.rvs(scale=lam_range / 2, size=len(lam_min_arr))
            mask = hyper_params > lam_range
            while np.any(mask):
                hyper_params[mask] /= 2
                mask = hyper_params > lam_range
        elif distribution == 'semicircle':
            hyper_params = semicircular.rvs(loc=lam_range / 2, scale=lam_range / 2, size=len(lam_min_arr))
        elif distribution == 'beta':
            # Sample from the standard Beta distribution on [0, 1] with flipped alpha, beta
            # Then transform to the interval [a, b]
            hyper_params = beta.rvs(
                alpha_beta[1] + 1,
                alpha_beta[0] + 1,
                loc=lam_min_arr,
                scale=lam_range,
                size=len(lam_min_arr),
            )
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        hyper_params = hyper_params.tolist()

        if len(hyper_params) == 1:
            hyper_params = hyper_params[0]
            # print(hyper_params)

        beta_before = model.linear.weight.clone().detach()
        phi = model.basis_fn(hyper_params, model.basis_dim, device)
        theta_before = model(hyper_params, device).clone().detach()
        loss = loss_fn(hyper_params, X_train, y_train, model, device)
        loss_value = loss.item()

        if step_size is not None:
            # shrink learning rate as customized
            for param_group in optimizer.param_groups:
                param_group['lr'] = step_size(itr, const)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # record raw gradient
        grad = model.linear.weight.grad.clone().detach()
        optimizer.step()
        beta_after = model.linear.weight.clone().detach()
        theta_after = model(hyper_params, device).clone().detach()

        if capture_diagnostics:
            phi_norm = torch.norm(phi).item()
            diagnostic_payload = {
                "hyper_params": hyper_params if isinstance(hyper_params, list) else [hyper_params],
                "loss_pre_step": loss_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "phi": phi.clone().detach().cpu(),
                "phi_norm": phi_norm,
                "phi_max_abs": phi.abs().max().item(),
                # phi is a vector, so the largest eigenvalue of phi phi^T is ||phi||_2^2.
                "phi_outer_eigmax": phi_norm ** 2,
                "beta_before": beta_before.cpu(),
                "beta_after": beta_after.cpu(),
                "beta_update_norm": (beta_after - beta_before).norm().item(),
                "beta_max_abs_after": beta_after.abs().max().item(),
                "theta_before": theta_before.cpu(),
                "theta_after": theta_after.cpu(),
                "theta_update_norm": (theta_after - theta_before).norm().item(),
                "theta_max_abs_after": theta_after.abs().max().item(),
                "grad": grad.cpu(),
                "grad_norm": grad.norm().item(),
                "grad_max_abs": grad.abs().max().item(),
            }
        
        rho = 2 / (itr+3)
        itr += 1
        if weighted_avg and itr > 50:
            # update weighted average iterates
            avg_weight = (1-rho) * avg_weight + rho * model.linear.weight.clone().detach()
        else:
            avg_weight = model.linear.weight.clone().detach()

    if capture_diagnostics:
        return grad, avg_weight, itr, diagnostic_payload
    return grad, avg_weight, itr

def train_lsp_bfgs(itr, init_weight, dataloader, model, loss_fn, optimizer, lam_min=None, lam_max=None, weighted_avg=False, 
              step_size=None, const=None, distribution='uniform', device='cpu'):
    model.train()
    avg_weight = init_weight

    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        num_quadpts = 10 # for quadrature
        num_pts = 2000 # for monte-carlo
        gradient_evaluations=0
        def closure():
            nonlocal gradient_evaluations  # Use the nonlocal keyword to modify the counter
            gradient_evaluations += 1

            # compute expectation directly with quadrature and print output
            def func(hyper_params):
                return loss_fn(hyper_params, X_train, y_train, model, device) # current implementation for uniform distribution only
            if len(lam_min) == 1:
                loss = gauss_legendre_integral(func, lam_min[0], lam_max[0], num_quadpts)
            elif len(lam_min) == 2:
                loss = gauss_legendre_integral_2d(func, lam_min, lam_max, num_quadpts)
            elif len(lam_min) > 2:
                loss = monte_carlo(func, lam_min, lam_max, num_pts, distribution, 42)
                # print(f"before lbfgs step average loss: {loss}")

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            return loss
    
        optimizer.step(closure)
        itr += gradient_evaluations

    # record raw gradient
    grad = model.linear.weight.grad.clone().detach()
    # rho = 2 / (itr+3)
    # if weighted_avg and itr > 50:
    #     # update weighted average iterates
    #     avg_weight = (1-rho) * avg_weight + rho * model.linear.weight.clone().detach()
    # else:
    avg_weight = model.linear.weight.clone().detach()

    return grad, avg_weight, itr
    
# Test function computes loss for a fixed input hyperparameter lam
def test_lsp(dataloader, model, loss_fn, hyper_params, device='cpu'):
    model.eval() #important
    with torch.no_grad():  #makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
          X_test, y_test = X_test.to(device), y_test.to(device)
          
          oos = loss_fn(hyper_params, X_test, y_test, model, device)
          
    return oos.item()
