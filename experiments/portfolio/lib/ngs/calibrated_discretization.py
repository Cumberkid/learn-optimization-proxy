import numpy as np
import torch
from lib.ngs.solver import train, test
from lib.ngs.log_reg_module import Logistic_Regression
from lib.ngs.utils_ngs import find_closest, make_unique, make_adjustment, get_errs_linear
from bisect import bisect_left
from scipy.interpolate import RegularGridInterpolator


'''Calibrated Uniform Discretization'''

# running gradient descent with fixed learning rate on a single grid point, i.e. for one specified lambda
def GD_on_a_grid(lam, epochs, loss_fn, model, test_model, optimizer, trainDataLoader,
                 step_size=None, const=None, weighted_avg=False, testDataLoader=None, 
                 oracle=False, true_loss=None, 
                 stopping_criterion=None, check_frequency=1, device="cpu"):
    
    weight = model.linear.weight.clone().detach().squeeze() # weighted averaging sum initialized
    if model.linear.bias is not None:
        intercept = model.linear.bias.clone().detach().squeeze()

    early_stop = False
    itr = 0
    passes = 0
    error = 0
                   
    for t in range(epochs):

        itr, weight, intercept = train(itr, weight, intercept, trainDataLoader, model, loss_fn, 
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
                      passes += (t+1)
                      early_stop = True
                      break  
            
    if not early_stop:
        passes += epochs
        
    return passes, error, weight, intercept

"""Uniform discretization starts from $\lambda = 1$ and decreases $\lambda$ by $\Delta\lambda = \frac{\lambda_\text{max} - \lambda_\text{min}}{\text{# of grid}}$. The model trained on each grid point $(\lambda - \Delta\lambda)$ initializes weight with the linear weight of the model trained on the previous grid point $\lambda$."""

# Solve the objective at a list of given hyperparameters from lam_min to lam_max
# Warm-started with given weight at each corresponding hyperparameter
# returns a list of trained models
def discretization(epochs, loss_fn, trainDataLoader, prev_params, prev_intercepts, prev_weights, 
                    data_input_dim, lr=1e-3, step_size=None, const=None, weighted_avg=False, 
                    testDataLoader=None, oracle=False, true_loss_list=None, 
                    true_loss_params=None, stopping_criterion=None, 
                    check_frequency=5, device="cpu"):

    hyper_params = []
    weights = []
    intercepts = []
    total_passes = 0
    grid_pass_error = 0
    true_loss = None
    
    # first weight is initialized at 0
    weight = torch.zeros(data_input_dim)
    intercept = 0
    model = Logistic_Regression(data_input_dim, 1, prev_params[0], weight, intercept).to(device)
    test_model = Logistic_Regression(data_input_dim, 1, prev_params[0], weight, intercept).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()


    for idx in range(len(prev_params)):
        # print(f"Running model on lambda = {lam}")
        model.hyper_param = prev_params[idx] 
        with torch.no_grad():    
            model.linear.weight.copy_(torch.tensor(prev_weights[idx]))
            if model.linear.bias is not None:
                model.linear.bias.copy_(prev_intercepts[idx])

        if true_loss_list is not None: # align with true loss' param
            true_loss_idx, true_loss_param = find_closest(prev_params[idx], true_loss_params)
            model.hyper_param = true_loss_param
            true_loss = true_loss_list[true_loss_idx]
        
        passes, grid_error, weight, intercept = GD_on_a_grid(model.hyper_param, epochs, loss_fn, 
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

        hyper_params.append(model.hyper_param)
        total_passes += passes
        grid_pass_error = max([grid_pass_error, grid_error])

    return total_passes, hyper_params, intercepts, weights, grid_pass_error


# Starting with lam_min and lam_max, adaptively build finer grid points 
# in between to achieve higher accuracy accross solution path
def calibrated_discretization(lam_min, lam_max, epochs, loss_fn, trainDataLoader, data_input_dim, 
                                prev_params=None, prev_intercepts=None, prev_weights=None, 
                                lr=1e-3, step_size=None, const=None, weighted_avg=False, 
                                testDataLoader=None, oracle=False, true_loss_list=None, 
                                true_loss_params=None, stopping_criterion=None, 
                                check_frequency=5, device="cpu"):
    hyper_params = [lam_max, lam_min]
    weights = [torch.zeros(data_input_dim), torch.zeros(data_input_dim)]
    intercepts = [0, 0]

    if prev_params is not None:
        hyper_params = prev_params
        weights = prev_weights
        intercepts = prev_intercepts

    total_passes, hyper_params, intercepts, weights, grid_pass_error = discretization(epochs, loss_fn, trainDataLoader,
                                                                                    hyper_params, intercepts, weights, 
                                                                                    data_input_dim, lr=lr, step_size=step_size, 
                                                                                    const=const, weighted_avg=weighted_avg, 
                                                                                    testDataLoader=testDataLoader, oracle=oracle, 
                                                                                    true_loss_list=true_loss_list, 
                                                                                    true_loss_params=true_loss_params,
                                                                                    stopping_criterion=stopping_criterion, 
                                                                                    check_frequency=check_frequency, device=device)

    insufficient = True
    while insufficient: # refine grid if current one is insufficient
        # Compute midpoints using slicing
        coords = np.array(hyper_params)
        new_params = (coords[:-1] + coords[1:]) / 2

        # if true_loss_list is not None: # filter out the params that already exist and don't need to further optimize
        #     selected_true_losses = []
        #     for idx in range(len(new_params)):
        #         true_loss_idx, true_loss_param = find_closest(new_params[idx], true_loss_params)
        #         if (new_params_filtered[-1] != true_loss_param) and (true_loss_param not in coords):
        #             new_params_filtered.append(true_loss_param)
        #             selected_true_losses.append(true_loss_list[true_loss_idx])
        #     new_params = np.array(new_params_filtered)
        #     errs = get_errs_linear(new_params, selected_true_losses, intercepts, weights, hyper_params, testDataLoader, loss_fn, device)
        #     new_params = [new_params[i] for i in range(len(new_params)) if errs[i] > (2 * stopping_criterion)]

        if true_loss_list is not None:
            new_params, selected_true_losses = make_adjustment(new_params, true_loss_params, true_loss_list)
            errs = get_errs_linear(new_params, selected_true_losses, intercepts, weights, hyper_params, testDataLoader, loss_fn, device)
            new_params = [new_params[i] for i in range(len(new_params)) if errs[i] > (2 * stopping_criterion)]

        new_params = [new_params[i] for i in range(len(new_params)) if (new_params[i] not in coords)]
        
        if len(new_params) == 0: # stop when no more new grid needs to be added
            insufficient = False
            break

        weight_interp = RegularGridInterpolator((hyper_params,), weights)
        intercept_interp = RegularGridInterpolator((hyper_params,), intercepts)
        new_weights = weight_interp(new_params)
        new_intercepts = intercept_interp(new_params)

        new_passes, new_params, new_intercepts, new_weights, new_grid_error = discretization(epochs, loss_fn, trainDataLoader,
                                                                                        new_params, new_intercepts, new_weights, 
                                                                                        data_input_dim, lr=lr, step_size=step_size, 
                                                                                        const=const, weighted_avg=weighted_avg, 
                                                                                        testDataLoader=testDataLoader, oracle=oracle, 
                                                                                        true_loss_list=true_loss_list, 
                                                                                        true_loss_params=true_loss_params,
                                                                                        stopping_criterion=stopping_criterion, 
                                                                                        check_frequency=check_frequency, device=device)

        total_passes += new_passes
        grid_pass_error = max([grid_pass_error, new_grid_error])    
        for idx in range(len(new_params)):
            insert_idx = bisect_left([-lam for lam in hyper_params], -new_params[idx])

            # Insert at the found index
            hyper_params.insert(insert_idx, new_params[idx])
            weights.insert(insert_idx, new_weights[idx])
            intercepts.insert(insert_idx, new_intercepts[idx])

    return total_passes, hyper_params, intercepts, weights, grid_pass_error

