import numpy as np
import torch
from lib.ngs.solver import train_bfgs, test
from lib.ngs.portfolio_alloc_module import Portfolio_Allocation
from lib.ngs.utils_ngs_2d import get_first_last, get_errs_linear, get_errs_linear_2d
from lib.ngs.utils_ngs import find_closest, make_unique, make_adjustment
from bisect import bisect_left
from scipy.interpolate import RegularGridInterpolator


'''Calibrated Uniform Discretization'''

# running gradient descent with fixed learning rate on a single grid point, i.e. for one specified lambda
def GD_on_a_grid(lam, epochs, loss_fn, model, test_model, optimizer, trainDataLoader,
                 step_size=None, const=None, weighted_avg=False, testDataLoader=None, 
                 oracle=False, true_loss=None, 
                 stopping_criterion=None, check_frequency=1, device="cpu"):
    
    weight = model.linear.weight.clone().detach().squeeze() # weighted averaging sum initialized
    intercept = 0
    if model.linear.bias is not None:
        intercept = model.linear.bias.clone().detach().squeeze()

    early_stop = False
    itr = 0
    grad_calls = 0
    error = 0
                   
    for t in range(epochs):

        itr, weight, intercept = train_bfgs(itr, weight, intercept, trainDataLoader, model, loss_fn, 
                                       optimizer, weighted_avg, step_size, const, device)
      
        if true_loss is not None:
            if (t+1) % check_frequency == 0:
                # do an accuracy check
                test_model.hyper_param = lam
                with torch.no_grad():
                    test_model.linear.weight.copy_(weight)
                    if model.linear.bias is not None:
                        test_model.linear.bias.copy_(intercept)
                approx_loss = test(testDataLoader, test_model, loss_fn, lam, device)
                    
                error = approx_loss - true_loss
                
                if oracle: # performs early-stop if we are within stopping criterion
                  if error <= stopping_criterion:
                      grad_calls += (t+1)
                      early_stop = True
                      break  
            
    if not early_stop:
        grad_calls += epochs
        
    return itr, grad_calls, error, weight, intercept

"""Uniform discretization starts from $\lambda = 1$ and decreases $\lambda$ by $\Delta\lambda = \frac{\lambda_\text{max} - \lambda_\text{min}}{\text{# of grid}}$. The model trained on each grid point $(\lambda - \Delta\lambda)$ initializes weight with the linear weight of the model trained on the previous grid point $\lambda$."""

# Solve the objective at a list of given hyperparameters from lam_min to lam_max
# Warm-started with given weight at each corresponding hyperparameter
# returns a list of trained models
def discretization(fixed_lam, epochs, loss_fn, trainDataLoader, prev_params, prev_intercepts, prev_weights, 
                    data_input_dim, lr=1e-3, step_size=None, const=None, weighted_avg=False, 
                    testDataLoader=None, oracle=False, true_loss_list=None, 
                    true_loss_params=None, stopping_criterion=None, 
                    check_frequency=1, device="cpu"):

    hyper_params_1d = []
    weights = []
    intercepts = []
    total_evals = 0
    total_grad_calls = 0
    grid_pass_error = 0
    true_loss = None
    
    # first weight is initialized at 0
    weight = torch.zeros(data_input_dim)
    intercept = 0
    model = Portfolio_Allocation(data_input_dim, 1, [fixed_lam, prev_params[0]], weight).to(device)
    test_model = Portfolio_Allocation(data_input_dim, 1, [fixed_lam, prev_params[0]], weight).to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    optimizer.zero_grad()


    for idx in range(len(prev_params)):
        # print(f"Running model on lambda = {lam}")
        model.hyper_param = [fixed_lam, prev_params[idx]] 
        with torch.no_grad():    
            model.linear.weight.copy_(torch.tensor(prev_weights[idx]).clone().detach().requires_grad_(True))
            if model.linear.bias is not None:
                model.linear.bias.copy_(prev_intercepts[idx])

        if true_loss_list is not None: # align with true loss' param
            true_loss_idx, true_loss_param = find_closest(prev_params[idx], true_loss_params)
            model.hyper_param = [fixed_lam, true_loss_param]
            true_loss = true_loss_list[true_loss_idx]
        
        evals, grad_calls, grid_error, weight, intercept = GD_on_a_grid(model.hyper_param, epochs, loss_fn, 
                                                            model, test_model, optimizer,
                                                            trainDataLoader=trainDataLoader,
                                                            step_size=step_size, const=const, 
                                                            weighted_avg=weighted_avg,
                                                            testDataLoader=testDataLoader,
                                                            oracle=oracle,
                                                            true_loss=true_loss,
                                                            stopping_criterion=stopping_criterion,
                                                            check_frequency=check_frequency,
                                                            device=device)
        weights.append(weight)
        intercepts.append(intercept)               

        hyper_params_1d.append(model.hyper_param[1])
        total_evals += evals
        total_grad_calls += grad_calls
        grid_pass_error = max([grid_pass_error, grid_error])

    return total_evals, total_grad_calls, hyper_params_1d, intercepts, weights, grid_pass_error


# Fix lam in first dimension. 
# In the second dimension, starting with lam_min and lam_max, adaptively build finer grid points 
# in between to achieve higher accuracy accross 1-d solution path
def calibrated_discretization(fixed_lam, lam_min, lam_max, epochs, loss_fn, trainDataLoader, data_input_dim, 
                                prev_params=None, prev_intercepts=None, prev_weights=None, 
                                lr=1e-3, step_size=None, const=None, weighted_avg=False, 
                                testDataLoader=None, oracle=False, true_loss_list=None, 
                                true_loss_params=None, stopping_criterion=None, 
                                check_frequency=1, device="cpu"):
    hyper_params_1d = [lam_max, lam_min]
    weights = [torch.zeros(data_input_dim), torch.zeros(data_input_dim)]
    intercepts = [0, 0]
    grid_pass_error = 0

    if prev_params is not None:
        hyper_params_1d = np.array(prev_params)[:, 1]
        weights = prev_weights
        intercepts = prev_intercepts

    total_evals, total_grad_calls, hyper_params_1d, intercepts, weights, grid_pass_error = discretization(
                                                                                            fixed_lam, epochs, loss_fn, trainDataLoader,
                                                                                            hyper_params_1d, intercepts, weights, 
                                                                                            data_input_dim, lr=lr, step_size=step_size, 
                                                                                            const=const, weighted_avg=weighted_avg, 
                                                                                            testDataLoader=testDataLoader, oracle=oracle, 
                                                                                            true_loss_list=true_loss_list, 
                                                                                            true_loss_params=true_loss_params[:, 1],
                                                                                            stopping_criterion=stopping_criterion, 
                                                                                            check_frequency=check_frequency, device=device)

    insufficient = True
    while insufficient: # refine grid if current one is insufficient
        # Compute midpoints using slicing
        coords = np.array(hyper_params_1d)
        # print(coords)
        new_params_1d = (coords[:-1] + coords[1:]) / 2
        # new_params_1d_filtered = []

        # if true_loss_list is not None: # filter out the params that don't need to further optimize
        #     selected_true_losses = []
        #     for idx in range(len(new_params_1d)):
        #         true_loss_idx, true_loss_param = find_closest(new_params_1d[idx], true_loss_params[:, 1])
        #         if (new_params_1d_filtered[-1] != true_loss_param) and (true_loss_param not in coords):
        #             new_params_1d_filtered.append(true_loss_param)
        #             selected_true_losses.append(true_loss_list[true_loss_idx])
        #     new_params_1d = np.array(new_params_1d_filtered)

        #     # Transform to 2-d parameters
        #     hyper_params_2d = [[fixed_lam, value] for value in hyper_params_1d]
        #     new_params_2d = [[fixed_lam, value] for value in new_params_1d]
        #     print(new_params_2d)
        #     errs = get_errs_linear(new_params_2d, selected_true_losses, intercepts, weights, hyper_params_2d, testDataLoader, loss_fn, device)
        #     new_params_1d = [new_params_1d[i] for i in range(len(new_params_1d)) if errs[i] > (2 * stopping_criterion)]




        if true_loss_list is not None:
            # print(new_params_1d)
            new_params_1d, selected_true_losses = make_adjustment(new_params_1d, true_loss_params[:, 1], true_loss_list)
            # print(new_params_1d)
            # Transform to 2-d parameters
            hyper_params_2d = [[fixed_lam, value] for value in hyper_params_1d]
            new_params_2d = [[fixed_lam, value] for value in new_params_1d]
            # print(new_params_2d)
            errs = get_errs_linear(new_params_2d, selected_true_losses, intercepts, weights, hyper_params_2d, testDataLoader, loss_fn, device)
            # print(errs)
            new_params_1d = [new_params_1d[i] for i in range(len(new_params_1d)) if errs[i] > (2 * stopping_criterion)]

        # print(coords)
        # print(new_params_1d)
        new_params_1d = [new_params_1d[i] for i in range(len(new_params_1d)) if (new_params_1d[i] not in coords)]


        if len(new_params_1d) == 0: # stop when no more new grid needs to be added
            insufficient = False
            break
        
        weight_interp = RegularGridInterpolator((hyper_params_1d,), weights)
        intercept_interp = RegularGridInterpolator((hyper_params_1d,), intercepts)
        new_weights = weight_interp(new_params_1d)
        new_intercepts = intercept_interp(new_params_1d)

        new_evals, new_grad_calls, new_params_1d, new_intercepts, new_weights, new_grid_error = discretization(
                                                                                                fixed_lam, epochs, loss_fn, trainDataLoader,
                                                                                                new_params_1d, new_intercepts, new_weights, 
                                                                                                data_input_dim, lr=lr, step_size=step_size, 
                                                                                                const=const, weighted_avg=weighted_avg, 
                                                                                                testDataLoader=testDataLoader, oracle=oracle, 
                                                                                                true_loss_list=true_loss_list, 
                                                                                                true_loss_params=true_loss_params[:, 1],
                                                                                                stopping_criterion=stopping_criterion, 
                                                                                                check_frequency=check_frequency, device=device)

        total_evals += new_evals
        total_grad_calls += new_grad_calls
        grid_pass_error = max([grid_pass_error, new_grid_error])    
        for idx in range(len(new_params_1d)):
            insert_idx = bisect_left([-lam for lam in hyper_params_1d], -new_params_1d[idx])

            # Insert at the found index
            hyper_params_1d.insert(insert_idx, new_params_1d[idx])
            weights.insert(insert_idx, new_weights[idx])
            intercepts.insert(insert_idx, new_intercepts[idx])

    hyper_params_2d = [[fixed_lam, value] for value in hyper_params_1d]

    return total_evals, total_grad_calls, hyper_params_2d, intercepts, weights, grid_pass_error



def calibrated_discretization_2d(lam_min_2d, lam_max_2d, epochs, loss_fn, trainDataLoader, data_input_dim, 
                                 prev_params_2d=None, prev_intercepts_2d=None, prev_weights_2d=None, 
                                 lr=1e-3, step_size=None, const=None, weighted_avg=False, 
                                 testDataLoader=None, oracle=False, true_loss_list_2d=None, 
                                 true_loss_params_2d=None, stopping_criterion=None, 
                                 check_frequency=1, device="cpu"):
    
    total_evals_2d = 0
    total_grad_calls_2d = 0
    grid_pass_error_2d = 0
    hyper_params_2d = []
    weights_2d = []
    intercepts_2d = []
    true_loss_list = None
    true_loss_params = None

    if prev_params_2d is None:
        hyper_params_2d.append([[lam_max_2d[0], lam_max_2d[1]], [lam_max_2d[0], lam_min_2d[1]]])
        hyper_params_2d.append([[lam_min_2d[0], lam_max_2d[1]], [lam_min_2d[0], lam_min_2d[1]]])
        weights_2d = [[torch.zeros(data_input_dim), torch.zeros(data_input_dim)], [torch.zeros(data_input_dim), torch.zeros(data_input_dim)]]
        intercepts_2d = [[0, 0], [0, 0]]

    else:
        hyper_params_2d = prev_params_2d
        weights_2d = prev_weights_2d
        intercepts_2d = prev_intercepts_2d

    for idx in range(len(hyper_params_2d)):
        fixed_lam = hyper_params_2d[idx][0][0]
        lam_min = lam_min_2d[1]
        lam_max = lam_max_2d[1]

        if true_loss_list_2d is not None: # find corresponding true losses
            true_loss_idx, true_loss_fixed_lam = find_closest(fixed_lam, true_loss_params_2d[:, 0, 0])
            fixed_lam = true_loss_fixed_lam
            true_loss_list = true_loss_list_2d[true_loss_idx]
            true_loss_params = true_loss_params_2d[true_loss_idx]

        total_evals, total_grad_calls, hyper_params_2d[idx], intercepts_2d[idx], weights_2d[idx], grid_pass_error = calibrated_discretization(
                                                                                                                    fixed_lam, lam_min, lam_max, 
                                                                                                                    epochs, loss_fn, 
                                                                                                                    trainDataLoader, data_input_dim, 
                                                                                                                    prev_params=hyper_params_2d[idx], 
                                                                                                                    prev_intercepts=intercepts_2d[idx], 
                                                                                                                    prev_weights=weights_2d[idx], 
                                                                                                                    lr=lr, step_size=step_size, 
                                                                                                                    const=const, weighted_avg=weighted_avg, 
                                                                                                                    testDataLoader=testDataLoader, 
                                                                                                                    oracle=oracle, true_loss_list=true_loss_list, 
                                                                                                                    true_loss_params=true_loss_params, 
                                                                                                                    stopping_criterion=stopping_criterion, 
                                                                                                                    check_frequency=check_frequency, device=device)
        total_evals_2d += total_evals
        total_grad_calls_2d += total_grad_calls
        grid_pass_error_2d = max([grid_pass_error, grid_pass_error_2d])
        # print(grid_pass_error)

    insufficient = True
    while insufficient: # refine grid if current one is insufficient
        # Compute midpoints using slicing
        coords = get_first_last(hyper_params_2d)
        coords_1d = coords[:, 0, 0]
        new_params_1d = (coords_1d[:-1] + coords_1d[1:]) / 2 ##########check this process#################

        # new_params_filtered = np.array([new_params[0]])

        # if true_loss_list_2d is not None: # filter out the params in first dimension that don't need to further optimize
            
        #     selected_true_losses = []
        #     for idx in range(len(new_params)):
        #         true_loss_idx, true_loss_param = find_closest(new_params[idx, 0, 0], true_loss_params_2d[:, 0, 0])
        #         new_params[idx, 0, 0] = new_params[idx, -1, 0] = true_loss_param
        #         if idx==0:
        #             selected_true_losses.append(true_loss_list_2d[true_loss_idx, [0, -1]])
        #         elif (true_loss_param not in new_params_filtered[:, 0, 0]) and (true_loss_param not in coords):
        #             new_params_filtered = np.append(new_params_filtered, [new_params[idx]], axis=0)
        #             selected_true_losses.append(true_loss_list_2d[true_loss_idx, [0, -1]])
            
        #     coords_intercepts = get_first_last(intercepts_2d)
        #     coords_weights = get_first_last(weights_2d)
        #     errs = get_errs_linear_2d(new_params_filtered, selected_true_losses, coords_intercepts, coords_weights, coords, testDataLoader, loss_fn, device)
        #     new_params = [new_params_filtered[i] for i in range(len(new_params_filtered)) if (max(errs[i]) > (2 * stopping_criterion)) 
        #                   and (new_params_filtered[i, 0, 0] not in np.array(hyper_params_2d)[:, 0, 0])]


        if true_loss_list_2d is not None:
            new_params_1d, selected_true_losses_left = make_adjustment(new_params_1d, true_loss_params_2d[:, 0, 0], true_loss_list_2d[:, 0])
            new_params_1d, selected_true_losses_right = make_adjustment(new_params_1d, true_loss_params_2d[:, -1, 0], true_loss_list_2d[:, -1])
            selected_true_losses = np.column_stack([selected_true_losses_left, selected_true_losses_right])
            # Transform to 2-d parameters
            new_params = [[[value, hyper_params_2d[0][0][1]], [value, hyper_params_2d[0][1][1]]] for value in new_params_1d]
            # print(new_params)
            coords_intercepts = get_first_last(intercepts_2d)
            coords_weights = get_first_last(weights_2d)
            # print(f"true losses: {selected_true_losses}")
            errs = get_errs_linear_2d(new_params, selected_true_losses, coords_intercepts, coords_weights, coords, testDataLoader, loss_fn, device)
            # print(f"errors: {errs}")
            new_params = [new_params[i] for i in range(len(new_params)) if (max(errs[i]) > (2 * stopping_criterion))]
            # print(new_params)
            
        new_params = [new_params[i] for i in range(len(new_params)) if (new_params[i][0][0] not in coords[:, 0, 0])]
        # print(f"added: {new_params}")

        if len(new_params) == 0: # stop when no more new grid needs to be added
            insufficient = False
            break
          
        for fixed_lam in np.array(new_params)[:, 0, 0]:
            # print(f"new: {fixed_lam}")

            if true_loss_list_2d is not None: # find corresponding true losses
                true_loss_idx, true_loss_fixed_lam = find_closest(fixed_lam, true_loss_params_2d[:, 0, 0])
                fixed_lam = true_loss_fixed_lam
                true_loss_list = true_loss_list_2d[true_loss_idx]
                true_loss_params = true_loss_params_2d[true_loss_idx]


            new_evals, new_grad_calls, new_params_1d, new_intercepts, new_weights, new_grid_error = calibrated_discretization(
                                                                                                    fixed_lam, lam_min, lam_max, 
                                                                                                    epochs, loss_fn, 
                                                                                                    trainDataLoader, data_input_dim, 
                                                                                                    prev_params=None, 
                                                                                                    prev_intercepts=None, 
                                                                                                    prev_weights=None, 
                                                                                                    lr=lr, step_size=step_size, 
                                                                                                    const=const, weighted_avg=weighted_avg, 
                                                                                                    testDataLoader=testDataLoader, 
                                                                                                    oracle=oracle, true_loss_list=true_loss_list, 
                                                                                                    true_loss_params=true_loss_params, 
                                                                                                    stopping_criterion=stopping_criterion, 
                                                                                                    check_frequency=check_frequency, device=device)

            total_evals_2d += new_evals
            total_grad_calls_2d += new_grad_calls
            grid_pass_error_2d = max([grid_pass_error_2d, new_grid_error])    
            # print(new_grid_error, new_evals)
            
            insert_idx = bisect_left([-lam for lam in get_first_last(hyper_params_2d)[:, 0, 0]], -fixed_lam)

            # Insert at the found index
            hyper_params_2d.insert(insert_idx, new_params_1d)
            weights_2d.insert(insert_idx, new_weights)
            intercepts_2d.insert(insert_idx, new_intercepts)
        # print(np.array(hyper_params_2d)[:, 0, 0])
    return total_evals_2d, total_grad_calls_2d, hyper_params_2d, intercepts_2d, weights_2d, grid_pass_error_2d