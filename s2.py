def exact_inference_with_ll(cnf, seqs, params, device):
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
            delta_log_p, _, transformed_latent_var_tensor = apply_flow(cnf, latent_var_tensor.T)
            seqs_latent[n]['latent_variable'] = transformed_latent_var_tensor.detach().cpu().numpy().T
            seqs_latent[n]['Vsm'] = vsm
            seqs_latent[n]['VsmGP'] = vsm_gp
            delta_log_py_list.append(delta_log_p)
        delta_log_py = torch.stack(delta_log_py_list).mean()
        val = -t * logdet_r - logdet_k_big - logdet_m - y_dim * t * np.log(2 * np.pi)
        ll += len(n_list) * val - (rinv.dot(dif) * dif).sum() + (term1_mat.T.dot(minv) * term1_mat.T).sum()

    return seqs_latent, ll, cnf, delta_log_p
