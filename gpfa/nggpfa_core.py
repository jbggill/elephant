# -*- coding: utf-8 -*-
"""
GPFA core functionality.

:copyright: Copyright 2014-2023 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import time
import warnings

import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.sparse as sparse
from sklearn.decomposition import FactorAnalysis
from tqdm import trange
import torch

from . import gpfa_util
import sys
from IPython.display import clear_output
from tqdm.notebook import trange


import matplotlib.pyplot as plt





def fit(seqs_train, x_dim=3, bin_width=20.0, min_var_frac=0.01, em_tol=1.0E-8,
        em_max_iters=1000, tau_init=100.0, eps_init=1.0E-3, freq_ll=5,
        verbose=False, cnf = None, cnf_lr=.1,device='cpu',convergence=True,save_dir=None,reverse=False):
    """
    Fit the GPFA model with the given training data.

    Parameters
    ----------
    seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields
        T : int
            number of bins
        y : (#units, T) np.ndarray
            neural data
    x_dim : int, optional
        state dimensionality
        Default: 3
    bin_width : float, optional
        spike bin width in msec
        Default: 20.0
    min_var_frac : float, optional
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
        Default: 0.01
        (See Martin & McDonald, Psychometrika, Dec 1975.)
    em_tol : float, optional
        stopping criterion for EM
        Default: 1e-8
    em_max_iters : int, optional
        number of EM iterations to run
        Default: 500
    tau_init : float, optional
        GP timescale initialization in msec
        Default: 100
    eps_init : float, optional
        GP noise variance initialization
        Default: 1e-3
    freq_ll : int, optional
        data likelihood is computed at every freq_ll EM iterations. freq_ll = 1
        means that data likelihood is computed at every iteration.
        Default: 5
    verbose : bool, optional
        specifies whether to display status messages
        Default: False
    cnf: cnf, required
        the cnf which will be trained during fitting
    cnf_lr : float, option  
        the learning rate for the cnf


    Returns
    -------
    parameter_estimates : dict
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
            covType: {'rbf', 'tri', 'logexp'}
                type of GP covariance
            gamma: np.ndarray of shape (1, #latent_vars)
                related to GP timescales by 'bin_width / sqrt(gamma)'
            eps: np.ndarray of shape (1, #latent_vars)
                GP noise variances
            d: np.ndarray of shape (#units, 1)
                observation mean
            C: np.ndarray of shape (#units, #latent_vars)
                mapping between the neuronal data space and the latent variable
                space
            R: np.ndarray of shape (#units, #latent_vars)
                observation noise covariance

    fit_info : dict
        Information of the fitting process and the parameters used there
        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
    
    cnf : cnf
        the trained CNF
    """
    # For compute efficiency, train on equal-length segments of trials
    seqs_train_cut = gpfa_util.cut_trials(seqs_train)
    if len(seqs_train_cut) == 0:
        warnings.warn('No segments extracted for training. Defaulting to '
                      'segLength=Inf.')
        seqs_train_cut = gpfa_util.cut_trials(seqs_train, seg_length=np.inf)

    # ==================================
    # Initialize state model parameters
    # ==================================
    params_init = dict()
    params_init['covType'] = 'rbf'
    # GP timescale
    # Assume binWidth is the time step size.
    params_init['gamma'] = (bin_width / tau_init) ** 2 * np.ones(x_dim)
    # GP noise variance
    params_init['eps'] = eps_init * np.ones(x_dim)

    # ========================================
    # Initialize observation model parameters
    # ========================================
    print('Initializing parameters using factor analysis...')

    y_all = np.hstack(seqs_train_cut['y'])
    fa = FactorAnalysis(n_components=x_dim, copy=True,
                        noise_variance_init=np.diag(np.cov(y_all, bias=True)))
    fa.fit(y_all.T)
    params_init['d'] = y_all.mean(axis=1)
    params_init['C'] = fa.components_.T
    params_init['R'] = np.diag(fa.noise_variance_)

    # Define parameter constraints
    params_init['notes'] = {
        'learnKernelParams': True,
        'learnGPNoise': False,
        'RforceDiagonal': True,
    }

    # =====================
    # Fit model parameters
    # =====================
    print('\nFitting GPFA model with CNF...')
    params_est, seqs_train_cut, ll_cut, iter_time, cnf = em(
    params_init, seqs_train_cut, device,min_var_frac=min_var_frac,
    max_iters=em_max_iters, tol=em_tol, freq_ll=freq_ll, verbose=verbose, cnf=cnf, cnf_lr=cnf_lr,convergence=convergence,save_dir=save_dir,reverse=reverse)

    fit_info = {'iteration_time': iter_time, 'log_likelihoods': ll_cut}

    return params_est, fit_info, cnf


def em(params_init, seqs_train, device,max_iters=500, tol=1.0E-8, min_var_frac=0.01,
       freq_ll=5, verbose=False, cnf = None, cnf_lr=.1,convergence=False,save_dir=None,reverse=False):
    """
    Fits GPFA model parameters using expectation-maximization (EM) algorithm.

    Parameters
    ----------
    params_init : dict
        GPFA model parameters at which EM algorithm is initialized
        covType : {'rbf', 'tri', 'logexp'}
            type of GP covariance
        gamma : np.ndarray of shape (1, #latent_vars)
            related to GP timescales by
            'bin_width / sqrt(gamma)'
        eps : np.ndarray of shape (1, #latent_vars)
            GP noise variances
        d : np.ndarray of shape (#units, 1)
            observation mean
        C : np.ndarray of shape (#units, #latent_vars)
            mapping between the neuronal data space and the
            latent variable space
        R : np.ndarray of shape (#units, #latent_vars)
            observation noise covariance
    seqs_train : np.recarray
        training data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
        T : int
            number of bins
        y : np.ndarray (yDim x T)
            neural data
    max_iters : int, optional
        number of EM iterations to run
        Default: 500
    tol : float, optional
        stopping criterion for EM
        Default: 1e-8
    min_var_frac : float, optional
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
        Default: 0.01
        (See Martin & McDonald, Psychometrika, Dec 1975.)
    freq_ll : int, optional
        data likelihood is computed at every freq_ll EM iterations.
        freq_ll = 1 means that data likelihood is computed at every
        iteration.
        Default: 5
    verbose : bool, optional
        specifies whether to display status messages
        Default: False
    cnf: cnf, required
        the cnf which will be trained during fitting

    Returns
    -------
    params_est : dict
        GPFA model parameter estimates, returned by EM algorithm (same
        format as params_init)
    seqs_latent : np.recarray
        a copy of the training data structure, augmented with the new
        fields:
        latent_variable : np.ndarray of shape (#latent_vars x #bins)
            posterior mean of latent variables at each time bin
        Vsm : np.ndarray of shape (#latent_vars, #latent_vars, #bins)
            posterior covariance between latent variables at each
            timepoint
        VsmGP : np.ndarray of shape (#bins, #bins, #latent_vars)
            posterior covariance over time for each latent
            variable
    ll : list
        list of log likelihoods after each EM iteration
    iter_time : list
        lisf of computation times (in seconds) for each EM iteration
    cnf : cnf
        the trained CNF
    """
    
    params = params_init
    t = seqs_train['T']
    y_dim, x_dim = params['C'].shape
    lls = []
    ll_old = ll_base = ll = 0.0
    iter_time = []
    var_floor = min_var_frac * np.diag(np.cov(np.hstack(seqs_train['y'])))
    seqs_latent = None

    # CNF optimizer using ADAM (this can be changed)
    cnf_optimizer = torch.optim.Adam(cnf.parameters(), lr=cnf_lr)

    # Loop once for each iteration of EM algorithm
    for iter_id in trange(1, max_iters + 1, desc='EM iteration',
                          disable=not verbose):
       
        if verbose:
            print()
        tic = time.time()

        # ==== E STEP =====
        if not np.isnan(ll):
            ll_old = ll

        cnf_optimizer.zero_grad()
        seqs_latent, ll, cnf, delta_log_p = exact_inference_with_ll(cnf, seqs_train, params, device,reverse=reverse)
        lls.append(ll.item())  # Assuming ll is a tensor, use .item() to extract its scalar value for logging
        loss = torch.tensor(-ll, requires_grad=True) +  delta_log_p

        clip_value = 1.0
        # Apply gradients

        loss.backward()
        cnf_optimizer.step()
        if save_dir:
            save_gpfa_confidence_intervals(save_dir,seqs_latent, iter_id)
            plot_cnf_loss(save_dir,lls,iter_id)
            transform_and_plot_linear_gaussian(save_dir,cnf,iter_id)
#            plot_and_save_vector_field(cnf)


        # ==== M STEP ====
        sum_p_auto = np.zeros((x_dim, x_dim))
        for seq_latent in seqs_latent:
            sum_p_auto += seq_latent['Vsm'].sum(axis=2) \
                + seq_latent['latent_variable'].dot(
                seq_latent['latent_variable'].T)
        y = np.hstack(seqs_train['y'])
        latent_variable = np.hstack(seqs_latent['latent_variable'])
        sum_yxtrans = y.dot(latent_variable.T)
        sum_xall = latent_variable.sum(axis=1)[:, np.newaxis]
        sum_yall = y.sum(axis=1)[:, np.newaxis]

        # term is (xDim+1) x (xDim+1)
        term = np.vstack([np.hstack([sum_p_auto, sum_xall]),
                          np.hstack([sum_xall.T, t.sum().reshape((1, 1))])])
        # yDim x (xDim+1)
        cd = gpfa_util.rdiv(np.hstack([sum_yxtrans, sum_yall]), term)

        params['C'] = cd[:, :x_dim]
        params['d'] = cd[:, -1]

        # yCent must be based on the new d
        # yCent = bsxfun(@minus, [seq.y], currentParams.d);
        # R = (yCent * yCent' - (yCent * [seq.latent_variable]') * \
        #     currentParams.C') / sum(T);
        c = params['C']
        d = params['d'][:, np.newaxis]
        if params['notes']['RforceDiagonal']:
            sum_yytrans = (y * y).sum(axis=1)[:, np.newaxis]
            yd = sum_yall * d
            term = ((sum_yxtrans - d.dot(sum_xall.T)) * c).sum(axis=1)
            term = term[:, np.newaxis]
            r = d ** 2 + (sum_yytrans - 2 * yd - term) / t.sum()

            # Set minimum private variance
            r = np.maximum(var_floor, r)
            params['R'] = np.diag(r[:, 0])
        else:
            sum_yytrans = y.dot(y.T)
            yd = sum_yall.dot(d.T)
            term = (sum_yxtrans - d.dot(sum_xall.T)).dot(c.T)
            r = d.dot(d.T) + (sum_yytrans - yd - yd.T - term) / t.sum()

            params['R'] = (r + r.T) / 2  # ensure symmetry

        if params['notes']['learnKernelParams']:
            res = learn_gp_params(seqs_latent, params, verbose=verbose)
            params['gamma'] = res['gamma']

        t_end = time.time() - tic
        iter_time.append(t_end)

        # Verify that likelihood is growing monotonically
        if iter_id % freq_ll:
            print('Iter ID: ',iter_id, 'Log Likelihood: ', ll)
        if iter_id <= 10:
            ll_base = ll
        elif verbose and ll < ll_old:
            print('\nError: Data likelihood has decreased ',
                  'from {0} to {1}'.format(ll_old, ll))
        elif (ll - ll_base) < (1 + tol) * (ll_old - ll_base) and convergence:
            break


    if len(lls) < max_iters:
        print('Fitting has converged after {0} EM iterations.)'.format(
            len(lls)))

    if np.any(np.diag(params['R']) == var_floor):
        warnings.warn('Private variance floor used for one or more observed '
                      'dimensions in GPFA.')

    return params, seqs_latent, lls, iter_time, cnf



def exact_inference_with_ll(cnf, seqs, params, device,reverse=False):
    y_dim, x_dim = params['C'].shape
    dtype_out = [(x, seqs[x].dtype) for x in seqs.dtype.names] + \
                [('latent_variable', object), ('Vsm', object), ('VsmGP', object)]
    seqs_latent = np.empty(len(seqs), dtype=dtype_out)
    for dtype_name in seqs.dtype.names:
        seqs_latent[dtype_name] = seqs[dtype_name]

    if params['notes']['RforceDiagonal']:
        rinv = np.diag(1.0 / np.diag(params['R']))
        logdet_r = np.sum(np.log(np.diag(params['R'])))
    else:
        rinv = linalg.inv(params['R'])
        rinv = (rinv + rinv.T) / 2
        logdet_r = np.linalg.slogdet(params['R'])[1]

    c_rinv = params['C'].T.dot(rinv)
    c_rinv_c = c_rinv.dot(params['C'])
    t_all = seqs['T']
    t_uniq = np.unique(t_all)
    ll = 0.

    for t in t_uniq:
        k_big, k_big_inv, logdet_k_big = gpfa_util.make_k_big(params, t)
        k_big = sparse.csr_matrix(k_big)
        blah = [c_rinv_c for _ in range(t)]
        c_rinv_c_big = linalg.block_diag(*blah)
        minv, logdet_m = gpfa_util.inv_persymm(k_big_inv + c_rinv_c_big, x_dim)
        vsm = np.full((x_dim, x_dim, t), np.nan)
        idx = np.arange(0, x_dim * t + 1, x_dim)
        for i in range(t):
            vsm[:, :, i] = minv[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]
        vsm_gp = np.full((t, t, x_dim), np.nan)
        for i in range(x_dim):
            vsm_gp[:, :, i] = minv[i::x_dim, i::x_dim]

        n_list = np.where(t_all == t)[0]
        dif = np.hstack(seqs[n_list]['y']) - params['d'][:, np.newaxis]
        term1_mat = c_rinv.dot(dif).reshape((x_dim * t, -1), order='F')

        t_half = int(np.ceil(t / 2.0))
        blk_prod = np.zeros((x_dim * t_half, x_dim * t))
        idx = range(0, x_dim * t_half + 1, x_dim)
        for i in range(t_half):
            blk_prod[idx[i]:idx[i + 1], :] = c_rinv_c.dot(minv[idx[i]:idx[i + 1], :])
        blk_prod = k_big[:x_dim * t_half, :].dot(gpfa_util.fill_persymm(np.eye(x_dim * t_half, x_dim * t) - blk_prod, x_dim, t))
        latent_variable_mat = gpfa_util.fill_persymm(blk_prod, x_dim, t).dot(term1_mat)
        delta_log_py_list = []
        for i, n in enumerate(n_list):
            latent_var_np = latent_variable_mat[:, i].reshape((x_dim, t), order='F')
            latent_var_tensor = torch.tensor(latent_var_np, dtype=torch.float32).to(device)
            delta_log_p, _, transformed_latent_var_tensor = apply_flow(cnf, latent_var_tensor.T, reverse=reverse)
            seqs_latent[n]['latent_variable'] = transformed_latent_var_tensor.detach().cpu().numpy().T
            seqs_latent[n]['Vsm'] = vsm
            seqs_latent[n]['VsmGP'] = vsm_gp
            delta_log_py_list.append(delta_log_p)


            transformed_dif = seqs_latent[n]['y'] - (params['C'] @ seqs_latent[n]['latent_variable'] + params['d'][:, np.newaxis])
            
            # Update LL with transformed 'dif'
            val = -t * logdet_r - logdet_k_big - logdet_m - y_dim * t * np.log(2 * np.pi)
            ll += len(n_list) * val - (rinv.dot(transformed_dif) * transformed_dif).sum() + (term1_mat.T.dot(minv) * term1_mat.T).sum()
        delta_log_py = torch.stack(delta_log_py_list).mean()
        #val = -t * logdet_r - logdet_k_big - logdet_m - y_dim * t * np.log(2 * np.pi)
 

       # ll += len(n_list) * val - (rinv.dot(dif) * dif).sum() + (term1_mat.T.dot(minv) * term1_mat.T).sum()

    return seqs_latent, ll, cnf, delta_log_py




def learn_gp_params(seqs_latent, params, verbose=False):
    """Updates parameters of GP state model, given neural trajectories.

    Parameters
    ----------
    seqs_latent : np.recarray
        data structure containing neural trajectories;
    params : dict
        current GP state model parameters, which gives starting point
        for gradient optimization;
    verbose : bool, optional
        specifies whether to display status messages (default: False)

    Returns
    -------
    param_opt : np.ndarray
        updated GP state model parameter

    Raises
    ------
    ValueError
        If `params['covType'] != 'rbf'`.
        If `params['notes']['learnGPNoise']` set to True.

    """
    if params['covType'] != 'rbf':
        raise ValueError("Only 'rbf' GP covariance type is supported.")
    if params['notes']['learnGPNoise']:
        raise ValueError("learnGPNoise is not supported.")
    param_name = 'gamma'

    param_init = params[param_name]
    param_opt = {param_name: np.empty_like(param_init)}

    x_dim = param_init.shape[-1]
    precomp = gpfa_util.make_precomp(seqs_latent, x_dim)

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        const = {'eps': params['eps'][i]}
        initp = np.log(param_init[i])
        res_opt = optimize.minimize(gpfa_util.grad_betgam, initp,
                                    args=(precomp[i], const),
                                    method='L-BFGS-B', jac=True)
        param_opt['gamma'][i] = np.exp(res_opt.x.item())

        if verbose:
            print('\n Converged p; xDim:{}, p:{}'.format(i, res_opt.x))

    return param_opt

def apply_flow(cnf, labels,reverse=False):
    cnf.float()
    y, delta_log_py = cnf(labels, torch.zeros(labels.size(0), 1).to(labels),reverse=reverse )
    y = y.squeeze()
    return delta_log_py, labels, y
    

def orthonormalize(params_est, seqs):
    """
    Orthonormalize the columns of the loading matrix C and apply the
    corresponding linear transform to the latent variables.

    Parameters
    ----------
    params_est : dict
        First return value of extract_trajectory() on the training data set.
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
        covType : {'rbf', 'tri', 'logexp'}
            type of GP covariance
            Currently, only 'rbf' is supported.
        gamma : np.ndarray of shape (1, #latent_vars)
            related to GP timescales by 'bin_width / sqrt(gamma)'
        eps : np.ndarray of shape (1, #latent_vars)
            GP noise variances
        d : np.ndarray of shape (#units, 1)
            observation mean
        C : np.ndarray of shape (#units, #latent_vars)
            mapping between the neuronal data space and the latent variable
            space
        R : np.ndarray of shape (#units, #latent_vars)
            observation noise covariance

    seqs : np.recarray
        Contains the embedding of the training data into the latent variable
        space.
        Data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
        T : int
          number of timesteps
        y : np.ndarray of shape (#units, #bins)
          neural data
        latent_variable : np.ndarray of shape (#latent_vars, #bins)
          posterior mean of latent variables at each time bin
        Vsm : np.ndarray of shape (#latent_vars, #latent_vars, #bins)
          posterior covariance between latent variables at each
          timepoint
        VsmGP : np.ndarray of shape (#bins, #bins, #latent_vars)
          posterior covariance over time for each latent variable

    Returns
    -------
    params_est : dict
        Estimated model parameters, including `Corth`, obtained by
        orthonormalizing the columns of C.
    seqs : np.recarray
        Training data structure that contains the new field
        `latent_variable_orth`, the orthonormalized neural trajectories.
    """
    C = params_est['C']
    X = np.hstack(seqs['latent_variable'])
    latent_variable_orth, Corth, _ = gpfa_util.orthonormalize(X, C)
    seqs = gpfa_util.segment_by_trial(
        seqs, latent_variable_orth, 'latent_variable_orth')

    params_est['Corth'] = Corth

    return Corth, seqs


def setup_flow(device, params, context_dim):
    if params.use_conditional:
        cnf = build_conditional_cnf(params, params.num_tasks, context_dim).to(device)
    else:
        regularization_fns, regularization_coeffs = create_regularization_fns(params)
        cnf = build_model_tabular(params, params.num_tasks, regularization_fns).to(device)
    if params.spectral_norm:
        add_spectral_norm(cnf)
    set_cnf_options(params, cnf)
    return cnf

def save_gpfa_confidence_intervals(save_dir,seqs_latent, iteration):
    """
    Save confidence interval plots for each latent variable at a specified iteration.

    Parameters:
    - seqs_latent: The output from the GPFA fitting process, containing latent variables and their variances.
    - iteration: The current iteration number of the fitting process.
    - save_dir: The directory where plots will be saved.
    """
    save_dir +='/confidence'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Determine number of latent variables and time points (assuming consistent across trials)
    num_latent_vars = seqs_latent[0]['latent_variable'].shape[0]
    num_time_points = seqs_latent[0]['latent_variable'].shape[1]

    for latent_variable_idx in range(num_latent_vars):
        aggregated_means = np.zeros(num_time_points)
        aggregated_variances = np.zeros(num_time_points)
        
        # Aggregate data across trials for the current latent variable
        for trial_data in seqs_latent:
            means = trial_data['latent_variable'][latent_variable_idx, :]
            variances = trial_data['Vsm'][latent_variable_idx, latent_variable_idx, :]
            aggregated_means += means
            aggregated_variances += variances
        
        # Averaging across trials
        aggregated_means /= len(seqs_latent)
        aggregated_variances /= len(seqs_latent)
        std_devs = np.sqrt(aggregated_variances)
        
        # Calculate confidence intervals
        lower_bounds = aggregated_means - 1.96 * std_devs
        upper_bounds = aggregated_means + 1.96 * std_devs
        time_points = np.arange(num_time_points)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.fill_between(time_points, lower_bounds, upper_bounds, color='skyblue', alpha=0.4)
        plt.plot(time_points, aggregated_means, label=f'Latent Variable {latent_variable_idx+1}')
        plt.xlabel('Time Points')
        plt.ylabel('Latent Variable Value')
        plt.title(f'95% Confidence Interval for Latent Variable {latent_variable_idx+1} (Iteration {iteration})')
        plt.legend()
        
        # Save the plot
        file_name = f'{latent_variable_idx+1}_Iteration_{iteration}_Latent.png'
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

def plot_cnf_loss(save_dir,cnf_loss_history,iteration):
    save_dir += '/loss'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(8, 5))
    plt.plot([i for i in range(len(cnf_loss_history))],cnf_loss_history, label='CNF Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('CNF Training Loss Over Iterations')
    plt.legend()
    file_name = f'Iteration_{iteration}_Loss.png'

    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()



def transform_and_plot_linear_gaussian(save_dir, cnf, iteration):
    """
    Generates Linear Gaussian data, transforms it using a CNF, and saves a plot
    of the original and the two distinct transformed data sets.

    Parameters:
    - cnf: The Continuous Normalizing Flow model.
    - iteration: Iteration number to include in the filename.
    - save_dir (str): Directory where the plot will be saved.
    """
    save_dir += '/transformation'  # Append transformation to the save directory
    filename = f'linear_cnf_transformation_iter{iteration}.png'
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Generate Linear Gaussian Data
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X + np.random.normal(0, 1, X.shape)
    
    # Stack the data to match CNF input shape [(100, 2) in this example]
    data = np.hstack([X, y])

    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Assuming the CNF model has been properly defined and accepts the data tensor format correctly
    # Transform the Data Using the CNF
    with torch.no_grad():
        transformed_data, _ = cnf(data_tensor, torch.zeros(data_tensor.size(0), 1).to(data_tensor))#.detach().numpy()  # Updated to include CNF usage as per previous context

    # Assuming the output is structured as two trajectories concatenated
    transformed_data_1 = transformed_data[:len(transformed_data)//2]
    transformed_data_2 = transformed_data[len(transformed_data)//2:]

    # Plot Original vs Transformed Data
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.scatter(data[:, 0], data[:, 1], label='Original Data', alpha=0.6)
    plt.title('Original Linear Gaussian Data')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1, 3, 2)
    plt.scatter(transformed_data_1[:, 0], transformed_data_1[:, 1], label=f'Transformed Data 1 iter {iteration}', alpha=0.6, color='r')
    plt.title('First CNF Transformation')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1, 3, 3)
    plt.scatter(transformed_data_2[:, 0], transformed_data_2[:, 1], label=f'Transformed Data 2 iter {iteration}', alpha=0.6, color='g')
    plt.title('Second CNF Transformation')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))

    # Save the plot
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()  # Close the plot to free up memory



def plot_latent_trajectories(seqs_latent, save_dir, iteration, trial_to_plot=0):
    """
    Plots and saves latent trajectories for a specified trial.

    Parameters:
    - seqs_latent: np.recarray containing latent variables and their variances for each trial.
    - save_dir: Directory where the plots will be saved.
    - iteration: Iteration number, used for naming the saved plot file.
    - trial_to_plot: Index of the trial to plot.
    """
    save_dir += '/latent_trajectories'  # Subdirectory for latent trajectories plots
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Prepare the plot
    plt.figure(figsize=(10, 6))
    num_latent_vars = seqs_latent[trial_to_plot]['latent_variable'].shape[0]
    
    for i in range(num_latent_vars):
        plt.plot(seqs_latent[trial_to_plot]['latent_variable'][i, :], label=f'Dimension {i+1}')

    plt.title(f'Latent Trajectories for Trial {trial_to_plot}')
    plt.xlabel('Time Bins')
    plt.ylabel('Latent Variable Value')
    plt.legend()

    # Save the plot
    filename = f'latent_trajectories_trial_{trial_to_plot}_iter_{iteration}.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()  # Close the plot to free up memory
