import copy
def em(params_init, seqs_train, device,max_iters=500, tol=1.0E-8, min_var_frac=0.01,
       freq_ll=5, verbose=False, cnf = None, cnf_lr=.1,convergence=True,reverse=False,save_dir=None): 
    params = params_init
    t = seqs_train['T']
    y_dim, x_dim = params['C'].shape
    lls = []
    ll_old = ll_base = ll = 0.0
    iter_time = []
    var_floor = min_var_frac * np.diag(np.cov(np.hstack(seqs_train['y'])))
    seqs_latent = None


    # Loop once for each iteration of EM algorithm
    for iter_id in trange(1, max_iters + 1, desc='EM iteration',
                          disable=not verbose):
        if verbose:
            print()
        tic = time.time()


        
        # ==== E STEP =====
        if isinstance(ll, torch.Tensor):
            ll_numpy = ll.detach().cpu().numpy()
        else:
            ll_numpy = ll
        if not np.isnan(ll_numpy):
            ll_old = ll
        #seqs_laxtent, ll = exact_inference_with_ll(seqs_train, params)
        seqs_latent, ll = exact_inference_with_ll_torch(seqs_train, params)
        lls.append(ll) 


        # ==== Convert back to numpy ====
        # Convert PyTorch tensors to NumPy arrays after the E-step
        for i,seq_latent in enumerate(seqs_latent):
            if seq_latent['Vsm'].requires_grad:
                # Detach and convert to numpy if the tensor requires gradients
                seqs_latent[i]['Vsm'] = seq_latent['Vsm'].detach().cpu().numpy()
            else:
                # Directly convert to numpy if it does not require gradients
                seqs_latent[i]['Vsm'] = seq_latent['Vsm'].cpu().numpy()

            if seq_latent['latent_variable'].requires_grad:
                seqs_latent[i]['latent_variable'] = seq_latent['latent_variable'].detach().cpu().numpy()
            else:
                seqs_latent[i]['latent_variable'] = seq_latent['latent_variable'].cpu().numpy()
            if seq_latent['VsmGP'].requires_grad:
                # Detach and convert to numpy if the tensor requires gradients
                seqs_latent[i]['VsmGP'] = seq_latent['VsmGP'].detach().cpu().numpy()
            else:
                # Directly convert to numpy if it does not require gradients
                seqs_latent[i]['VsmGP'] = seq_latent['VsmGP'].cpu().numpy()
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
        print('Iter ID: ',iter_id, 'EM Log Likelihood: ',ll)

        if iter_id <= 2:
            ll_base = ll
        elif verbose and ll < ll_old:
            print('\nError: Data likelihood has decreased ',
                  'from {0} to {1}'.format(ll_old, ll))
        elif (ll - ll_base) < (1 + tol) * (ll_old - ll_base):
            break


    if len(lls) < max_iters:
        print('Fitting has converged after {0} EM iterations.)'.format(
            len(lls)))
    else:
        print(f'Fitting never converged and ran for {max_iters} EM iterations')

    if np.any(np.diag(params['R']) == var_floor):
        warnings.warn('Private variance floor used for one or more observed '
                      'dimensions in GPFA.')
    
    return params, seqs_latent, lls, iter_time,cnf

        
        
        
def exact_inference_with_ll_torch(seqs, params, get_ll=True,device='cpu'):
    """
    Extracts latent trajectories from neural data, given GPFA model parameters,
    adapted for PyTorch tensors.
    
    Parameters are similar to the original function, with adjustments for PyTorch.
    """
    y_dim, x_dim = params['C'].shape

    # copy the contents of the input data structure to output structure
    dtype_out = [(x, seqs[x].dtype) for x in seqs.dtype.names]
    dtype_out.extend([('latent_variable', object), ('Vsm', object),('VsmGP', object)])
    seqs_latent = np.empty(len(seqs), dtype=dtype_out)
    for dtype_name in seqs.dtype.names:
        seqs_latent[dtype_name] = seqs[dtype_name]


    # Assuming `seqs` is a list of dictionaries, each containing 'y' and 'T'
    # and params are converted to tensors.

    params['R'] = torch.tensor(params['R'], dtype=torch.float32, device=device)  # Add `.to(device)` if working with GPUs
    params['C'] = torch.tensor(params['C'], dtype=torch.float32, device=device)  # Add `.to(device)` if working with GPUs
    # Precomputations
    if params['notes']['RforceDiagonal']:
        rinv = torch.diag(1.0 / torch.diag(params['R']))
        logdet_r = torch.log(torch.diag(params['R'])).sum()
    else:
        rinv = torch.inverse(params['R'])
        rinv = (rinv + rinv.T) / 2  # ensure symmetry
        # PyTorch does not have a direct logdet function for tensors in older versions, 
        # but it does in newer ones. Otherwise, use a workaround.
        logdet_r = torch.logdet(params['R'])  

    c_rinv =   gpfa_util_torch.flexible_dot(params['C'].T,  rinv)
    #c_rinv_c =     gpfa_util_torch.flexible_dot(c_rinv , params['C'])
    c_rinv_c = torch.matmul(c_rinv, params['C'])
    t_all = torch.tensor(seqs_latent['T'])
    t_uniq = t_all.unique()
    ll = 0.
    for t in t_uniq:
        # You need to adapt or implement the make_k_big function for PyTorch
        k_big, k_big_inv, logdet_k_big = gpfa_util_torch.make_k_big_torch(params, t.item())
        k_big_csr = sparse.csr_matrix(k_big)

        # Convert SciPy CSR matrix to PyTorch sparse CSR tensor
        row_indices = torch.tensor(k_big_csr.indptr, dtype=torch.int64)
        col_indices = torch.tensor(k_big_csr.indices, dtype=torch.int64)
        values = torch.tensor(k_big_csr.data, dtype=torch.float32)
        size = torch.Size(k_big_csr.shape)

        k_big_sparse = torch.sparse_csr_tensor(row_indices, col_indices, values, size)
        
        blah = [c_rinv_c for _ in range(t)]
        c_rinv_c_big = torch.block_diag(*blah)
        
        # Adapt inv_persymm for PyTorch or implement equivalent functionality
        minv, logdet_m = gpfa_util_torch.inv_persymm_torch(k_big_inv + c_rinv_c_big, x_dim)

        vsm = torch.full((x_dim, x_dim, t), float('nan'))
        idx = np.arange(0, x_dim * t + 1, x_dim)
        for i in range(t):
            vsm[:, :, i] = minv[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]
        vsm_gp = torch.full((t, t, x_dim), float('nan'))
        for i in range(x_dim):
            vsm_gp[:, :, i] = minv[i::x_dim, i::x_dim]

        #n_list = (t_all == t).nonzero(as_tuple=True)[0]
        n_list = torch.where(t_all == t)[0]

        params['d'] = torch.tensor(params['d'], dtype=torch.float32, device=device)
        dif_tensors = [torch.tensor(item, dtype=torch.float32, device=device) for item in seqs_latent[n_list]['y']]

        dif = torch.hstack(dif_tensors) - params['d'][:, None]     
        term1 = torch.matmul(c_rinv, dif)

        # Now use the custom reshape function to reshape `term1` in a way that mimics Fortran-style ordering
        #        term1_mat = c_rinv.dot(dif).reshape((x_dim * t, -1), order='F')

        term1_mat = reshape_fortran(term1, (x_dim * t, -1))        


        # Compute block product (blk_prod) efficiently
        t_half = int(torch.ceil(torch.tensor(t / 2.0)))
        blk_prod = torch.zeros((x_dim * t_half, x_dim * t), dtype=torch.float32)
        idx = range(0, x_dim * t_half + 1, x_dim)

        for i in range(t_half):
            blk_prod[idx[i]:idx[i + 1], :] =     gpfa_util_torch.flexible_dot(c_rinv_c , minv[idx[i]:idx[i + 1], :])

        # Assuming you've implemented a PyTorch-compatible version of `fill_persymm`
        blk_prod =     gpfa_util_torch.flexible_dot(k_big.to_dense()[:x_dim * t_half, :] , gpfa_util_torch.fill_persymm_torch(torch.eye(x_dim * t_half, x_dim * t, dtype=torch.float32) - blk_prod, x_dim, t))
        # Note: The above step might need to adapt `fill_persymm` logic directly into the calculation if not available.

        # You need a similar adaptation for the next line, where `fill_persymm` is used again.
        latent_variable_mat = gpfa_util_torch.flexible_dot(gpfa_util_torch.fill_persymm_torch(blk_prod, x_dim, t) , term1_mat)  # Placeholder for actual fill_persymm function
        print(latent_variable_mat)
        # Update the sequences with latent variables and covariances
        for i, n in enumerate(n_list):
            n = n.item()
           # seqs_latent[n]['latent_variable'] = latent_variable_mat[:, i].view(x_dim, t)
            seqs_latent[n]['latent_variable'] = reshape_fortran(latent_variable_mat[:, i], (x_dim, t))

            seqs_latent[n]['Vsm'] = vsm  # Assuming vsm is already calculated and is a tensor
            seqs_latent[n]['VsmGP'] = vsm_gp  
        rinv = rinv.float()  # Convert to float32
        dif = dif.float()  # Convert to float32
        term1_mat = term1_mat.float()  # Convert to float32
        minv = minv.float()  
        
        # Compute data likelihood
        #val = -t * logdet_r - logdet_k_big - logdet_m - y_dim * t * np.log(2 * np.pi)
        #ll = ll + len(n_list) * val - (rinv.dot(dif) * dif).sum() + (term1_mat.T.dot(minv) * term1_mat.T).sum()
        # Compute data likelihood (adapted for PyTorch)
        val = -t * logdet_r - logdet_k_big - logdet_m - y_dim * t * np.log(2 * np.pi)
        ll += len(n_list) * val - (gpfa_util_torch.flexible_dot(rinv , dif) * dif).sum() + (gpfa_util_torch.flexible_dot(term1_mat.T , minv) * term1_mat.T).sum()
        
    ll /= 2
    return seqs_latent, ll
