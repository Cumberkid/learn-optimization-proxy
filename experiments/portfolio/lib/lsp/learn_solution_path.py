import torch
import math
from collections import deque
from lib.lsp.basis_tf_module import Basis_TF_SGD
from lib.lsp.solver_lsp import train_lsp, train_lsp_bfgs, test_lsp
from lib.lsp.utils_lsp import get_sup_error_lsp
from lib.utils import gauss_legendre_integral, gauss_legendre_integral_2d, monte_carlo
import itertools
from scipy.special import beta as beta_func


def distance(init_weight, prev_weight, curr_weight, n, k, thresh=0.6, q=1.5, k_0=5):
    if n == int(q**k):
        # print(n)
        if k >= k_0:
            s = (torch.log((init_weight - curr_weight).norm(p=2)**2) - torch.log((init_weight - prev_weight).norm(p=2)**2)) / (math.log(n) - math.log(n/q))
            diag = (s < thresh)
        else:
            diag = False
        while n == int(q**k):
            k += 1
        prev_weight = curr_weight
    else:
        diag = False

    return diag, k, prev_weight

# Default for diminishing stepsize is to use the 'distance diagnostic' described in this paper:
# Returns the list of epochs with corresponding solution path errors, final weight of the trained model for the modified stochastic
# optimization problem, final learning rate if use distance diagnostic, and total number of iterations
# lam_min and lam_max must be list objects
def learn_solution_path(input_dim, basis_dim, phi_lam, max_epochs, trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max,
                        true_losses=None, init_lr=1e-3, diminish=False, gamma=0.97, q=1.3, k_0=5, thresh_lr=0.6,
                        step_size=None, const=None, init_weight=None, intercept=True, weighted_avg=False, itr=0, thresh_basis=None,
                        record_frequency=10, record_fctn=get_sup_error_lsp, opt_method='SGD', distribution='uniform', 
                        alpha_beta=[-.5, -.5], device='cpu', trace_frequency=-1):
    # build the model
    model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)
    lr=init_lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if opt_method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_method == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # memorize the previous stochastic objective value to decide if need to add more basis function
    stochastic_obj_list = deque(maxlen=2)

    # # memorize the last 10 gradient info to decide if need to add more basis function
    # norm_grad_list = deque(maxlen=20)

    # initialize weighted averaging sum
    if weighted_avg:
        avg_model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)


    sup_err_history = []
    num_pass_history = []

    if diminish:
        # initilize for distance diagnostic
        init_weight = model.linear.weight.clone().detach() # record inital weight theta_0
        prev_weight = model.linear.weight.clone().detach() # memorize theta_n/q for every n=q^k until max_epochs/q
        k = 0

    if thresh_basis is None:
        thresh_basis = lambda x: 1e-4

    itr = itr
    weight = model.linear.weight.clone().detach()
    
    for t in range(max_epochs):
        # run one pass of dataset
        if (opt_method == 'SGD') or (opt_method == 'Adam'):
            new_grad, weight, itr = train_lsp(itr, weight, trainDataLoader, model, loss_fn, optimizer, 
                                            lam_min=lam_min, lam_max=lam_max, 
                                            weighted_avg=weighted_avg, step_size=step_size, 
                                            const=const, distribution=distribution, alpha_beta=alpha_beta, device=device)
        elif opt_method == 'lbfgs':
            new_grad, weight, itr = train_lsp_bfgs(itr, weight, trainDataLoader, model, loss_fn, optimizer, 
                                            lam_min=lam_min, lam_max=lam_max, 
                                            step_size=step_size, 
                                            const=const, distribution=distribution, device=device)

        if diminish: # diminish according to distance diagnostic
            # run distance diagnostic
            curr_weight = model.linear.weight.clone().detach()
            distance_diag, k, prev_weight = distance(init_weight, prev_weight, curr_weight, t+1, k, thresh_lr, q, k_0)
            if distance_diag:
                lr = gamma * lr
                # print(f"diminish at iteration #{t+1}, new lr = {lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        # record iteration result
        if (record_frequency > 0) & ((t+1) % record_frequency == 0):
            if (opt_method == 'SGD') or (opt_method == 'Adam'):
                num_pass_history.append(t+1)
            elif opt_method == 'lbfgs':
                num_pass_history.append(itr)
            if weighted_avg:
                with torch.no_grad():
                    avg_model.linear.weight.copy_(weight)
                sup_err = record_fctn(lam_min, lam_max, true_losses,
                                            avg_model, testDataLoader, loss_fn, device)
            else:
                sup_err = record_fctn(lam_min, lam_max, true_losses,
                                            model, testDataLoader, loss_fn, device)
            sup_err_history.append(sup_err)
            if (trace_frequency > 0) & ((t+1) % trace_frequency == 0):
                if (opt_method == 'SGD') or (opt_method == 'Adam'):
                    print(f"--------approximate solution path for # itr = {t+1} complete--------")
                    
                elif opt_method == 'lbfgs':
                    print(f"--------approximate solution path for # itr = {itr} complete--------")
                
                print(f"# epoch: {t+1}\t sup error: {sup_err}")

        # when the change in the stochastic objective is small enough,
        # stop to add more basis functions
        def func(hyper_params):
            return test_lsp(testDataLoader, model, loss_fn, hyper_params, device)
        stochastic_obj = 0.0
        num_quadpts = 10
        num_pts = 1000
        if len(lam_min) == 1:
            stochastic_obj = gauss_legendre_integral(func, lam_min[0], lam_max[0], num_quadpts)
        elif len(lam_min) == 2:
            stochastic_obj = gauss_legendre_integral_2d(func, lam_min, lam_max, num_quadpts)
        elif len(lam_min) > 2:
            stochastic_obj = monte_carlo(func, lam_min, lam_max, num_pts, distribution, 88)
            print(f"--------approximate solution path for # function evaluation = {itr} complete--------")
            print(f"# gradient call: {t+1}\t average loss: {stochastic_obj}")
        stochastic_obj_list.append(stochastic_obj)
        if (len(stochastic_obj_list) >= 2):
            if abs(stochastic_obj_list[1] - stochastic_obj_list[0]) < thresh_basis(basis_dim):
                break

        # # when the change in second moment of gradient is small enough,
        # # stop to add more basis functions
        # norm_grad_list.append(new_grad.norm(p=2)**2)
        # if (len(norm_grad_list) >= 20) and abs(sum(list(norm_grad_list)[:10]) - sum(list(norm_grad_list)[-10:])) < thresh_basis(basis_dim):
        #     break
        # if opt_method == 'lbfgs':
        #     if (len(norm_grad_list) >= 2) and abs(norm_grad_list[len(norm_grad_list) - 1] - norm_grad_list[len(norm_grad_list) - 2]) < thresh_basis(basis_dim):
        #         break

    return num_pass_history, sup_err_history, weight, lr, itr
    

def adaptive_lsp(input_dim, start_basis_dim, end_basis_dim, phi_lam, max_epochs,
                 trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max, true_losses=None,
                 init_lr=1e-3, diminish=True, gamma=0.97, q=1.3, k_0=5, thresh_lr=0.6,
                 step_size=None, const=None, init_weight=None, intercept=True, weighted_avg=False,
                 avg_upon_prev=False, thresh_basis=None, record_frequency=10, record_fctn=get_sup_error_lsp, 
                 opt_method='SGD', distribution='uniform', alpha_beta=[-.5, -.5], 
                 device='cpu', basis_list=None, lr_list=None, trace_frequency=-1):
    if step_size is not None:
        diminish = False

    num_pass_history = []
    sup_err_history = []
    breaks = []
    weight = init_weight
    lr = init_lr
    break_itr = 0
    const = const
    itr = 0

    if opt_method=='lbfgs':
        break_grads = []
        
    if basis_list is not None and lr_list is not None:
        iter1, iter2 = itertools.tee(basis_list)
        next(iter2)  # Advance the second iterator by 1 to access the next element

        for current_dim, next_dim, lr in zip(iter1, iter2, lr_list):
            print(f"********** now running lsp with #basis dimension = {current_dim} ***********")
            print(f'lr = {lr}')
            num_pass_current, sup_err_current, weight, lr, itr = learn_solution_path(input_dim, current_dim, phi_lam, 
                                max_epochs, trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max, true_losses, 
                                init_lr=lr, diminish=diminish, gamma=gamma, q=q, k_0=k_0, thresh_lr=thresh_lr,
                                step_size=step_size, const=const, init_weight=weight, intercept=intercept, 
                                weighted_avg=weighted_avg, itr=itr, thresh_basis=thresh_basis, 
                                record_frequency=record_frequency, record_fctn=record_fctn, opt_method=opt_method, 
                                distribution=distribution, alpha_beta=alpha_beta, device=device, 
                                trace_frequency=trace_frequency)

            num_pass_history += [(x + break_itr) for x in num_pass_current]
            sup_err_history += sup_err_current
            break_itr = num_pass_history[len(num_pass_history)-1]
            breaks.append(break_itr)
            if opt_method=='lbfgs':
                break_grads.append(len(num_pass_history))
            if const is not None:
                const = const/(2**0.5)
            if not avg_upon_prev:
                itr = 0
            # increase basis dimension by 1
            weight = torch.cat((weight, torch.zeros(weight.size(dim=0), next_dim - current_dim)), 1)

    else:
        for basis_dim in range(start_basis_dim, end_basis_dim):
            print(f"********** now running lsp with #basis dimension = {basis_dim} ***********")
            print(f'lr = {lr}')
            num_pass_current, sup_err_current, weight, lr, itr = learn_solution_path(input_dim, basis_dim, phi_lam, 
                                max_epochs, trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max, true_losses, 
                                init_lr=lr, diminish=diminish, gamma=gamma, q=q, k_0=k_0, thresh_lr=thresh_lr,
                                step_size=step_size, const=const, init_weight=weight, intercept=intercept, 
                                weighted_avg=weighted_avg, itr=itr, thresh_basis=thresh_basis, 
                                record_frequency=record_frequency, record_fctn=record_fctn, opt_method=opt_method, 
                                distribution=distribution, alpha_beta=alpha_beta, device=device, 
                                trace_frequency=trace_frequency)

            num_pass_history += [(x + break_itr) for x in num_pass_current]
            sup_err_history += sup_err_current
            break_itr = num_pass_history[len(num_pass_history)-1]
            breaks.append(break_itr)
            if const is not None:
                const = const/(2**0.5)
            if not avg_upon_prev:
                itr = 0
            # increase basis dimension by 1
            weight = torch.cat((weight, torch.zeros(weight.size(dim=0), 1)), 1)

    if opt_method=='lbfgs':
        return num_pass_history, sup_err_history, breaks, break_grads

    return num_pass_history, sup_err_history, breaks







def learn_solution_path_high_dim(input_dim, basis_dim, phi_lam, max_epochs, trainDataLoader, testDataLoader, loss_fn_train, loss_fn_val, lam_min, lam_max,
                        true_losses=None, init_lr=1e-3, diminish=False, gamma=0.97, q=1.3, k_0=5, thresh_lr=0.6,
                        step_size=None, const=None, init_weight=None, intercept=True, weighted_avg=False, itr=0, thresh_basis=None,
                        record_frequency=10, record_fctn=get_sup_error_lsp, opt_method='SGD', distribution='uniform', 
                        alpha_beta=[-.5, -.5], device='cpu', trace_frequency=-1, return_eval_epochs=False,
                        diagnostic_epochs=None, return_diagnostics=False, beta_snapshot_epochs=None,
                        return_beta_snapshots=False, adaptive_gamma_state=None,
                        adaptive_gamma_saturation_metric='none', adaptive_gamma_saturation_threshold=None,
                        adaptive_gamma_window=5, adaptive_gamma_shrink=1.0,
                        adaptive_gamma_lr_constant=None, adaptive_gamma_threshold_alpha=0.5,
                        adaptive_gamma_threshold_scale_min=0.01,
                        adaptive_gamma_threshold_scale_max=100.0,
                        adaptive_gamma_max_epochs_per_value=None,
                        return_adaptive_records=False,
                        return_val_regularized=False):
    # build the model
    model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)
    lr=init_lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if opt_method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_method == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # memorize the previous stochastic objective value to decide if need to add more basis function
    stochastic_obj_list = deque(maxlen=2)

    # # memorize the last 10 gradient info to decide if need to add more basis function
    # norm_grad_list = deque(maxlen=20)

    # initialize weighted averaging sum
    if weighted_avg:
        avg_model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)


    sup_err_history = []
    num_pass_history = []

    if diminish:
        # initilize for distance diagnostic
        init_weight = model.linear.weight.clone().detach() # record inital weight theta_0
        prev_weight = model.linear.weight.clone().detach() # memorize theta_n/q for every n=q^k until max_epochs/q
        k = 0

    if thresh_basis is None:
        thresh_basis = lambda x: 1e-4

    itr = itr
    weight = model.linear.weight.clone().detach()

    def func_val(hyper_params):
        return test_lsp(testDataLoader, model, loss_fn_val, hyper_params, device)
    
    def func_train_unreg(hyper_params):
        return test_lsp(trainDataLoader, model, loss_fn_val, hyper_params, device)

    def func_train_reg(hyper_params):
        return test_lsp(trainDataLoader, model, loss_fn_train, hyper_params, device)

    def func_val_reg(hyper_params):
        return test_lsp(testDataLoader, model, loss_fn_train, hyper_params, device)
    
    if record_frequency <= 0:
        raise ValueError("record_frequency must be a positive integer")

    loss_val = []
    loss_train = []
    loss_train_reg = []
    loss_val_reg = []
    eval_epochs = []
    gamma_history = []
    learning_rate_history = []
    diagnostics = []
    epochs = 0
    num_quadpts = 10
    num_pts = 1000
    diagnostic_epoch_set = set(diagnostic_epochs or [])
    beta_snapshot_epoch_set = set(beta_snapshot_epochs or [])
    beta_snapshots = []
    adaptive_events = []
    gradient_window = deque(maxlen=adaptive_gamma_window + 1)
    objective_window = deque(maxlen=adaptive_gamma_window + 1)
    adaptive_gamma_enabled = adaptive_gamma_saturation_metric != 'none'
    if adaptive_gamma_enabled:
        if opt_method not in ('SGD', 'Adam'):
            raise ValueError("adaptive gamma is only supported for SGD/Adam training")
        if adaptive_gamma_state is None or 'gamma' not in adaptive_gamma_state:
            raise ValueError("adaptive_gamma_state must contain a 'gamma' entry")
        if adaptive_gamma_saturation_metric not in ('gradient', 'objective'):
            raise ValueError("adaptive_gamma_saturation_metric must be one of: none, gradient, objective")
        if adaptive_gamma_saturation_threshold is None:
            raise ValueError("adaptive_gamma_saturation_threshold is required when adaptive gamma is enabled")
        if adaptive_gamma_window < 1:
            raise ValueError("adaptive_gamma_window must be a positive integer")
        if adaptive_gamma_shrink <= 0:
            raise ValueError("adaptive_gamma_shrink must be positive")
        if adaptive_gamma_lr_constant is None or adaptive_gamma_lr_constant <= 0:
            raise ValueError("adaptive_gamma_lr_constant must be positive when adaptive gamma is enabled")
        if adaptive_gamma_threshold_alpha < 0:
            raise ValueError("adaptive_gamma_threshold_alpha must be nonnegative")
        if adaptive_gamma_threshold_scale_min <= 0:
            raise ValueError("adaptive_gamma_threshold_scale_min must be positive")
        if adaptive_gamma_threshold_scale_max < adaptive_gamma_threshold_scale_min:
            raise ValueError("adaptive_gamma_threshold_scale_max must be >= adaptive_gamma_threshold_scale_min")
        if adaptive_gamma_max_epochs_per_value is not None and adaptive_gamma_max_epochs_per_value < 1:
            raise ValueError("adaptive_gamma_max_epochs_per_value must be a positive integer")
        if 'gamma_ref' not in adaptive_gamma_state:
            adaptive_gamma_state['gamma_ref'] = adaptive_gamma_state['gamma']

    def current_gamma():
        if adaptive_gamma_state is None:
            return None
        return float(adaptive_gamma_state['gamma'])

    def current_learning_rate():
        return float(optimizer.param_groups[0]['lr'])

    def rolling_average_difference(values):
        stacked = torch.stack(list(values))
        prev_avg = stacked[:-1].mean(dim=0)
        curr_avg = stacked[1:].mean(dim=0)
        return (curr_avg - prev_avg).norm().item()

    def current_saturation_threshold():
        gamma_ref = float(adaptive_gamma_state['gamma_ref'])
        if gamma_ref <= 0:
            raise ValueError("adaptive gamma reference value must be positive")
        scale = (current_gamma() / gamma_ref) ** adaptive_gamma_threshold_alpha
        scale = min(max(scale, adaptive_gamma_threshold_scale_min), adaptive_gamma_threshold_scale_max)
        return adaptive_gamma_saturation_threshold * scale

    epochs_since_gamma_change = 0

    def shrink_gamma(epoch_idx, detector_value, detector_name, threshold=None):
        nonlocal epochs_since_gamma_change
        if threshold is None:
            threshold = current_saturation_threshold()
        if detector_value is not None and detector_value > threshold:
            return
        old_gamma = current_gamma()
        new_gamma = old_gamma * adaptive_gamma_shrink
        adaptive_gamma_state['gamma'] = new_gamma
        new_lr = 1.0 / (adaptive_gamma_lr_constant + new_gamma)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        adaptive_events.append(
            {
                "epoch_index": epoch_idx,
                "detector": detector_name,
                "detector_value": detector_value,
                "threshold": threshold,
                "threshold_ref": adaptive_gamma_saturation_threshold,
                "threshold_alpha": adaptive_gamma_threshold_alpha,
                "gamma_ref": float(adaptive_gamma_state['gamma_ref']),
                "old_gamma": old_gamma,
                "new_gamma": new_gamma,
                "new_learning_rate": new_lr,
            }
        )
        epochs_since_gamma_change = 0
        gradient_window.clear()
        objective_window.clear()

    def record_beta_snapshot(epoch_idx):
        if epoch_idx in beta_snapshot_epoch_set:
            beta_snapshots.append(
                {
                    "epoch_index": epoch_idx,
                    "beta": model.linear.weight.clone().detach().cpu(),
                }
            )

    def record_losses(epoch_idx):
        stochastic_obj_val = 0.0
        stochastic_obj_train = 0.0
        stochastic_obj_train_reg = None
        stochastic_obj_val_reg = None

        if len(lam_min) == 1:
            stochastic_obj_val = gauss_legendre_integral(func_val, lam_min[0], lam_max[0], num_quadpts)
            stochastic_obj_train = gauss_legendre_integral(func_train_unreg, lam_min[0], lam_max[0], num_quadpts)
            stochastic_obj_train_reg = gauss_legendre_integral(func_train_reg, lam_min[0], lam_max[0], num_quadpts)
            stochastic_obj_val_reg = gauss_legendre_integral(func_val_reg, lam_min[0], lam_max[0], num_quadpts)
        elif len(lam_min) == 2:
            stochastic_obj_val = gauss_legendre_integral_2d(func_val, lam_min, lam_max, num_quadpts)
            stochastic_obj_train = gauss_legendre_integral_2d(func_train_unreg, lam_min, lam_max, num_quadpts)
            stochastic_obj_train_reg = gauss_legendre_integral_2d(func_train_reg, lam_min, lam_max, num_quadpts)
            stochastic_obj_val_reg = gauss_legendre_integral_2d(func_val_reg, lam_min, lam_max, num_quadpts)
        elif len(lam_min) > 2:
            stochastic_obj_train = monte_carlo(func_train_unreg, lam_min, lam_max, num_pts, distribution, 42)
            stochastic_obj_val = monte_carlo(func_val, lam_min, lam_max, num_pts, distribution, 88)
            stochastic_obj_train_reg = monte_carlo(func_train_reg, lam_min, lam_max, num_pts, distribution, 42)
            stochastic_obj_val_reg = monte_carlo(func_val_reg, lam_min, lam_max, num_pts, distribution, 88)

        eval_epochs.append(epoch_idx)
        loss_val.append(stochastic_obj_val)
        loss_train.append(stochastic_obj_train)
        loss_train_reg.append(stochastic_obj_train_reg)
        loss_val_reg.append(stochastic_obj_val_reg)
        gamma_history.append(current_gamma())
        learning_rate_history.append(current_learning_rate())

        should_trace = epoch_idx == 0
        if trace_frequency > 0 and epoch_idx > 0 and (epoch_idx % trace_frequency == 0):
            should_trace = True
        if should_trace:
            print(f"--------approximate solution path for # gradient calls = {epoch_idx} complete--------")
            print(f"# gradient call: {epoch_idx}\t average loss on training: {stochastic_obj_train}")
            print(f"# gradient call: {epoch_idx}\t average loss on validation: {stochastic_obj_val}")

    # Record the initial model before any training updates.
    record_beta_snapshot(0)
    record_losses(0)

    for t in range(max_epochs):
        epochs += 1
        current_epoch = t + 1
        epochs_since_gamma_change += 1
        # run one pass of dataset
        if (opt_method == 'SGD') or (opt_method == 'Adam'):
            capture_diagnostics = current_epoch in diagnostic_epoch_set
            train_result = train_lsp(itr, weight, trainDataLoader, model, loss_fn_train, optimizer, 
                                            lam_min=lam_min, lam_max=lam_max, 
                                            weighted_avg=weighted_avg, step_size=step_size, 
                                            const=const, distribution=distribution, alpha_beta=alpha_beta, device=device,
                                            capture_diagnostics=capture_diagnostics)
            if capture_diagnostics:
                new_grad, weight, itr, diagnostic_payload = train_result
                diagnostic_payload["epoch_index"] = current_epoch
                diagnostics.append(diagnostic_payload)
            else:
                new_grad, weight, itr = train_result
        elif opt_method == 'lbfgs':
            new_grad, weight, itr = train_lsp_bfgs(itr, weight, trainDataLoader, model, loss_fn_train, optimizer, 
                                            lam_min=lam_min, lam_max=lam_max, 
                                            step_size=step_size, 
                                            const=const, distribution=distribution, device=device)

        if adaptive_gamma_enabled:
            if adaptive_gamma_saturation_metric == 'gradient':
                gradient_window.append(new_grad.detach().cpu().float().reshape(-1))
                if len(gradient_window) == adaptive_gamma_window + 1:
                    diff_norm = rolling_average_difference(gradient_window)
                    shrink_gamma(current_epoch, diff_norm, 'gradient')
            elif adaptive_gamma_saturation_metric == 'objective':
                stochastic_obj_train = 0.0
                if len(lam_min) == 1:
                    stochastic_obj_train = gauss_legendre_integral(func_train_unreg, lam_min[0], lam_max[0], num_quadpts)
                elif len(lam_min) == 2:
                    stochastic_obj_train = gauss_legendre_integral_2d(func_train_unreg, lam_min, lam_max, num_quadpts)
                elif len(lam_min) > 2:
                    stochastic_obj_train = monte_carlo(func_train_unreg, lam_min, lam_max, num_pts, distribution, 42)
                objective_window.append(torch.tensor([float(stochastic_obj_train)], dtype=torch.float32))
                if len(objective_window) == adaptive_gamma_window + 1:
                    diff_norm = rolling_average_difference(objective_window)
                    shrink_gamma(current_epoch, diff_norm, 'objective')
            if (
                adaptive_gamma_max_epochs_per_value is not None
                and epochs_since_gamma_change >= adaptive_gamma_max_epochs_per_value
            ):
                shrink_gamma(
                    current_epoch,
                    None,
                    'max_epochs_per_gamma',
                    threshold=adaptive_gamma_max_epochs_per_value,
                )

        if (current_epoch % record_frequency == 0) or (current_epoch == max_epochs):
            record_losses(current_epoch)

        record_beta_snapshot(current_epoch)

        # # use this for adaptive run
        # if len(loss_train) > 1 and (loss_train[-2] - loss_train[-1]) < 1e-1:
        #     break

    # use this for non-adaptive run
    adaptive_records = {
        "gamma_history": gamma_history,
        "learning_rate_history": learning_rate_history,
        "events": adaptive_events,
    }

    result = []
    if return_eval_epochs:
        result.append(eval_epochs)
    result += [loss_train, loss_val, loss_train_reg]
    if return_val_regularized:
        result.append(loss_val_reg)
    result.append(weight)
    if return_diagnostics:
        result.append(diagnostics)
    if return_beta_snapshots:
        result.append(beta_snapshots)
    if return_adaptive_records:
        result.append(adaptive_records)
    return tuple(result)

    # # use this for adaptive run
    # return loss_train, loss_val, loss_train_reg, weight, epochs
