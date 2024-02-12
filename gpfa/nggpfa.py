"""
https://elephant.readthedocs.io/en/latest/_modules/elephant/gpfa/gpfa.html
"""

from __future__ import division, print_function, unicode_literals
from argparse import Namespace
import neo
import numpy as np
import quantities as pq
import sklearn
import sys
sys.path.append('/Users/jessegill/Desktop/nggp/nggp_lib')
#from nggp_lib.methods.regression_methods import NGGP, get_transforms
from nggp_lib.run_regression import *
from nggp_lib.training.configs import Config as config
from nggp_lib.models import backbone

from nggp_lib.training.utils import build_model_tabular, build_conditional_cnf





from elephant.gpfa import gpfa_core, gpfa_util, nggpfa_core


__all__ = [
    "GPFA"
]


class NGGPFA(sklearn.base.BaseEstimator):
    r"""
    Apply Gaussian process factor analysis (GPFA) to spike train data

    There are two principle scenarios of using the GPFA analysis, both of which
    can be performed in an instance of the GPFA() class.

    In the first scenario, only one single dataset is used to fit the model and
    to extract the neural trajectories. The parameters that describe the
    transformation are first extracted from the data using the `fit()` method
    of the GPFA class. Then the same data is projected into the orthonormal
    basis using the method `transform()`. The `fit_transform()` method can be
    used to perform these two steps at once.

    In the second scenario, a single dataset is split into training and test
    datasets. Here, the parameters are estimated from the training data. Then
    the test data is projected into the low-dimensional space previously
    obtained from the training data. This analysis is performed by executing
    first the `fit()` method on the training data, followed by the
    `transform()` method on the test dataset.

    The GPFA class is compatible to the cross-validation functions of
    `sklearn.model_selection`, such that users can perform cross-validation to
    search for a set of parameters yielding best performance using these
    functions.

    Parameters
    ----------
    x_dim : int, optional
        state dimensionality
        Default: 3
    bin_size : float, optional
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
    cnf_lr : float, option
        specifies the learning rate for the cng
        Default: .01

    Attributes
    ----------
    valid_data_names : tuple of str
        Names of the data contained in the resultant data structure, used to
        check the validity of users' request
    has_spikes_bool : np.ndarray of bool
        Indicates if a neuron has any spikes across trials of the training
        data.
    params_estimated : dict
        Estimated model parameters. Updated at each run of the fit() method.

        covType : str
            type of GP covariance, either 'rbf', 'tri', or 'logexp'.
            Currently, only 'rbf' is supported.
        gamma : (1, #latent_vars) np.ndarray
            related to GP timescales of latent variables before
            orthonormalization by :math:`\frac{bin\_size}{\sqrt{gamma}}`
        eps : (1, #latent_vars) np.ndarray
            GP noise variances
        d : (#units, 1) np.ndarray
            observation mean
        C : (#units, #latent_vars) np.ndarray
            loading matrix, representing the mapping between the neuronal data
            space and the latent variable space
        R : (#units, #latent_vars) np.ndarray
            observation noise covariance
    fit_info : dict
        Information of the fitting process. Updated at each run of the fit()
        method.

        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
        log_likelihoods : list
            log likelihoods after each EM iteration.
    transform_info : dict
        Information of the transforming process. Updated at each run of the
        transform() method.

        log_likelihood : float
            maximized likelihood of the transformed data
        num_bins : nd.array
            number of bins in each trial
        Corth : (#units, #latent_vars) np.ndarray
            mapping between the neuronal data space and the orthonormal
            latent variable space

    Raises
    ------
    ValueError
        If `bin_size` or `tau_init` is not a `pq.Quantity`.

    Examples
    --------
    In the following example, we calculate the neural trajectories of 20
    independent Poisson spike trains recorded in 50 trials with randomized
    rates up to 100 Hz.

    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.gpfa import GPFA
    >>> from elephant.spike_train_generation import StationaryPoissonProcess
    >>> data = []
    >>> for trial in range(50):  # noqa
    ...     n_channels = 20
    ...     firing_rates = np.random.randint(low=1, high=100,
    ...                                      size=n_channels) * pq.Hz
    >>> spike_times = [StationaryPoissonProcess(rate
    ...                ).generate_spiketrain() for rate in firing_rates]
    >>> data.append((trial, spike_times))
    ...
    >>> gpfa = GPFA(bin_size=20*pq.ms, x_dim=8)
    >>> gpfa.fit(data)  # doctest: +SKIP
    >>> results = gpfa.transform(data, returned_data=['latent_variable_orth',
    ...                                               'latent_variable'])  # doctest: +SKIP
    >>> latent_variable_orth = results['latent_variable_orth']  # doctest: +SKIP
    >>> latent_variable = results['latent_variable']  # doctest: +SKIP

    or simply

    >>> results = GPFA(bin_size=20*pq.ms, x_dim=8).fit_transform(data,  # doctest: +SKIP
    ...                returned_data=['latent_variable_orth',
    ...                               'latent_variable'])
    """

    def __init__(self, bin_size=20 * pq.ms, x_dim=3, min_var_frac=0.01,
                 tau_init=100.0 * pq.ms, eps_init=1.0E-3, em_tol=1.0E-8,
                 em_max_iters=500, freq_ll=5, verbose=False, cnf_lr = .01,device='cpu'):
        # Initialize object
        self.bin_size = bin_size
        self.x_dim = x_dim
        self.min_var_frac = min_var_frac
        self.tau_init = tau_init
        self.eps_init = eps_init
        self.em_tol = em_tol
        self.em_max_iters = em_max_iters
        self.freq_ll = freq_ll
        self.valid_data_names = (
            'latent_variable_orth',
            'latent_variable',
            'Vsm',
            'VsmGP',
            'y')
        self.verbose = verbose
        self.cnf_lr = cnf_lr
        self.device=device


        if not isinstance(self.bin_size, pq.Quantity):
            raise ValueError("'bin_size' must be of type pq.Quantity")
        if not isinstance(self.tau_init, pq.Quantity):
            raise ValueError("'tau_init' must be of type pq.Quantity")

        self.params_estimated = dict()
        self.fit_info = dict()
        self.transform_info = dict()
        self.save_path_th = '/Users/jessegill/Desktop/nggp/nggp_lib/save/nggp_rbf_5e3/checkpoints/neural/MLP2_NGGP_model.th'

        self.model_params = Namespace(seed=1, model='MLP2', method='NGGP', dataset='neural', update_batch_size=5, meta_batch_size=5, output_dim=40, multidimensional_amp=False, multidimensional_phase=False, noise='gaussian', kernel_type='rbf', save_dir='./save/nggp_rbf_5e3', num_tasks=self.x_dim, multi_type=3, method_lr=cnf_lr, feature_extractor_lr=0.001, cnf_lr=0.001, all_lr=0.005, neptune=False, use_conditional=False, context_type='backbone', layer_type='concatsquash', dims='32-32-32', num_blocks=2, time_length=0.5, train_T=False, add_noise=False, divergence_fn='brute_force', nonlinearity='tanh', solver='dopri5', atol=1e-05, rtol=1e-05, step_size=None, test_solver=None, test_atol=None, test_rtol=None, residual=False, rademacher=False, spectral_norm=False, batch_norm=False, bn_lag=0, l1int=None, l2int=None, dl2int=None, JFrobint=None, JdiagFrobint=None, JoffdiagFrobint=None, start_epoch=0, stop_epoch=100, test=False, n_support=5, n_test_epochs=10, out_of_range=False, device=device)
        setup_seed(self.model_params)
        config = Config(self.model_params)
        checkpoint_dir, save_path = setup_checkpoint_dir(self.model_params)

        results_logger = ResultsLogger(self.model_params)


        device = self.model_params.device
        logging.info('Device: {}'.format(device))

        self.bb = setup_backbone(device, self.model_params)
        self.nggp_model = setup_model(self.bb, config, device, self.model_params)
        optimizer = setup_optimizer(self.nggp_model, self.model_params)
        setup_checkpoint_dir(self.model_params)
        self.cnf = setup_flow(device, self.model_params, self.x_dim)
        if device=='mps':
            torch.set_default_dtype(torch.float32)


        print("*"*4, "NGGP Model Initiated", "*"*4)
        
    def setup_flow(self):
        if self.model_params.use_conditional:
            cnf = build_conditional_cnf(self.model_params, self.model_params.num_tasks, self.x_dim).to(self.device)
        else:
            regularization_fns, regularization_coeffs = create_regularization_fns(self.model_params)
            cnf = build_model_tabular(self.model_params, self.model_params.num_tasks, regularization_fns).to(self.device)
            #cnf = build_model_tabular(params, 5, regularization_fns).to(device)

        if self.model_params.spectral_norm:
            add_spectral_norm(cnf)
        set_cnf_options(self.model_params, cnf)
        ##
        ##
        return cnf

    def fit(self, spiketrains):
        """
        Fit the model with the given training data.

        Parameters
        ----------
        spiketrains : list of list of neo.SpikeTrain
            Spike train data to be fit to latent variables.
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that
            `spiketrains[l][n]` is the spike train of neuron `n` in trial `l`.
            Note that the number and order of `neo.SpikeTrain` objects per
            trial must be fixed such that `spiketrains[l][n]` and
            `spiketrains[k][n]` refer to spike trains of the same neuron
            for any choices of `l`, `k`, and `n`.

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError
            If `spiketrains` is an empty list.

            If `spiketrains[0][0]` is not a `neo.SpikeTrain`.

            If covariance matrix of input spike data is rank deficient.
        """
        self._check_training_data(spiketrains)
        seqs_train = self._format_training_data(spiketrains)
        # Check if training data covariance is full rank
        y_all = np.hstack(seqs_train['y'])
        y_dim = y_all.shape[0]

        if np.linalg.matrix_rank(np.cov(y_all)) < y_dim:
            errmesg = 'Observation covariance matrix is rank deficient.\n' \
                      'Possible causes: ' \
                      'repeated units, not enough observations.'
            raise ValueError(errmesg)

        if self.verbose:
            print('Number of training trials: {}'.format(len(seqs_train)))
            print('Latent space dimensionality: {}'.format(self.x_dim))
            print('Observation dimensionality: {}'.format(
                self.has_spikes_bool.sum()))

        self.params_estimated, self.fit_info, self.cnf = nggpfa_core.fit(
            seqs_train=seqs_train,
            x_dim=self.x_dim,
            bin_width=self.bin_size.rescale('ms').magnitude,
            min_var_frac=self.min_var_frac,
            em_max_iters=self.em_max_iters,
            em_tol=self.em_tol,
            tau_init=self.tau_init.rescale('ms').magnitude,
            eps_init=self.eps_init,
            freq_ll=self.freq_ll,
            verbose=self.verbose,
            cnf = self.cnf,
            cnf_lr = self.cnf_lr, 
            device=self.device
            )
 

        return self



    @staticmethod
    def _check_training_data(spiketrains):
        if len(spiketrains) == 0:
            raise ValueError("Input spiketrains cannot be empty")
        """
        if not isinstance(spiketrains[0][0], neo.SpikeTrain):
            raise ValueError("structure of the spiketrains is not correct: "
                             "0-axis should be trials, 1-axis neo.SpikeTrain"
                             "and 2-axis spike times")
        """

    def _format_training_data(self, spiketrains):
        seqs = gpfa_util.get_seqs(spiketrains, self.bin_size)
        # Remove inactive units based on training set
        self.has_spikes_bool = np.hstack(seqs['y']).any(axis=1)
        for seq in seqs:
            seq['y'] = seq['y'][self.has_spikes_bool, :]
        return seqs


    def transform(self, spiketrains, returned_data=['latent_variable_orth']):
        """
        Obtain trajectories of neural activity in a low-dimensional latent
        variable space by inferring the posterior mean of the obtained GPFA
        model and applying an orthonormalization on the latent variable space.

        Parameters
        ----------
        spiketrains : list of list of neo.SpikeTrain
            Spike train data to be transformed to latent variables.
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that
            `spiketrains[l][n]` is the spike train of neuron `n` in trial `l`.
            Note that the number and order of `neo.SpikeTrain` objects per
            trial must be fixed such that `spiketrains[l][n]` and
            `spiketrains[k][n]` refer to spike trains of the same neuron
            for any choices of `l`, `k`, and `n`.
        returned_data : list of str
            The dimensionality reduction transform generates the following
            resultant data:

               'latent_variable_orth': orthonormalized posterior mean of latent
               variable

               'latent_variable': posterior mean of latent variable before
               orthonormalization

               'Vsm': posterior covariance between latent variables

               'VsmGP': posterior covariance over time for each latent variable

               'y': neural data used to estimate the GPFA model parameters

            `returned_data` specifies the keys by which the data dict is
            returned.

            Default is ['latent_variable_orth'].

        Returns
        -------
        np.ndarray or dict
            When the length of `returned_data` is one, a single np.ndarray,
            containing the requested data (the first entry in `returned_data`
            keys list), is returned. Otherwise, a dict of multiple np.ndarrays
            with the keys identical to the data names in `returned_data` is
            returned.

            N-th entry of each np.ndarray is a np.ndarray of the following
            shape, specific to each data type, containing the corresponding
            data for the n-th trial:

                `latent_variable_orth`: (#latent_vars, #bins) np.ndarray

                `latent_variable`:  (#latent_vars, #bins) np.ndarray

                `y`:  (#units, #bins) np.ndarray

                `Vsm`:  (#latent_vars, #latent_vars, #bins) np.ndarray

                `VsmGP`:  (#bins, #bins, #latent_vars) np.ndarray

            Note that the num. of bins (#bins) can vary across trials,
            reflecting the trial durations in the given `spiketrains` data.

        Raises
        ------
        ValueError
            If the number of neurons in `spiketrains` is different from that
            in the training spiketrain data.

            If `returned_data` contains keys different from the ones in
            `self.valid_data_names`.
        """
        if len(spiketrains[0]) != len(self.has_spikes_bool):
            raise ValueError("'spiketrains' must contain the same number of "
                             "neurons as the training spiketrain data")
        invalid_keys = set(returned_data).difference(self.valid_data_names)
        if len(invalid_keys) > 0:
            raise ValueError("'returned_data' can only have the following "
                             "entries: {}".format(self.valid_data_names))
        seqs = gpfa_util.get_seqs(spiketrains, self.bin_size)
        print('seqs: ',np.shape(seqs))
        for seq in seqs:
            seq['y'] = seq['y'][self.has_spikes_bool, :]
        print('seqs2: ', np.shape(seqs))
        seqs, ll, cnf, _ = nggpfa_core.exact_inference_with_ll(self.cnf,seqs,
                                                     self.params_estimated,device=self.device)
        self.transform_info['log_likelihood'] = ll
        self.transform_info['num_bins'] = seqs['T']
        Corth, seqs = gpfa_core.orthonormalize(self.params_estimated, seqs)
        self.transform_info['Corth'] = Corth
        if len(returned_data) == 1:
            return seqs[returned_data[0]]
        return {x: seqs[x] for x in returned_data}


    def fit_transform(self, spiketrains, returned_data=[
                      'latent_variable_orth']):
        """
        Fit the model with `spiketrains` data and apply the dimensionality
        reduction on `spiketrains`.

        Parameters
        ----------
        spiketrains : list of list of neo.SpikeTrain
            Refer to the :func:`GPFA.fit` docstring.

        returned_data : list of str
            Refer to the :func:`GPFA.transform` docstring.

        Returns
        -------
        np.ndarray or dict
            Refer to the :func:`GPFA.transform` docstring.

        Raises
        ------
        ValueError
             Refer to :func:`GPFA.fit` and :func:`GPFA.transform`.

        See Also
        --------
        GPFA.fit : fit the model with `spiketrains`
        GPFA.transform : transform `spiketrains` into trajectories

        """
        self.fit(spiketrains)
        return self.transform(spiketrains, returned_data=returned_data)

    def score(self, spiketrains):
        """
        Returns the log-likelihood of the given data under the fitted model

        Parameters
        ----------
        spiketrains : list of list of neo.SpikeTrain
            Spike train data to be scored.
            The outer list corresponds to trials and the inner list corresponds
            to the neurons recorded in that trial, such that
            `spiketrains[l][n]` is the spike train of neuron `n` in trial `l`.
            Note that the number and order of `neo.SpikeTrain` objects per
            trial must be fixed such that `spiketrains[l][n]` and
            `spiketrains[k][n]` refer to spike trains of the same neuron
            for any choice of `l`, `k`, and `n`.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the given spiketrains under the fitted model.
        """
        self.transform(spiketrains)
        return self.transform_info['log_likelihood']