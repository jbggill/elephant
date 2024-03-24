import torch
import numpy as np
import quantities as pq
import warnings
from elephant.conversion import BinnedSpikeTrain



def logdet(A):
    """
    Computes log(det(A)) where A is positive-definite.
    Adapted for PyTorch tensors.
    """
    U = torch.linalg.cholesky(A)
    return 2 * torch.log(torch.diag(U)).sum()

def flexible_dot(tensor_a, tensor_b):
    """
    Computes the dot product of two tensors if they are vectors (1D),
    or performs matrix multiplication if one or both of the tensors have more dimensions.
    
    Parameters:
    - tensor_a (torch.Tensor): The first tensor.
    - tensor_b (torch.Tensor): The second tensor.

    Returns:
    - torch.Tensor: The result of the dot product or matrix multiplication.
    """

    if tensor_a.dim() == 1 and tensor_b.dim() == 1:
        # Vector-vector product (dot product)
        return torch.dot(tensor_a, tensor_b)
    elif tensor_a.dim() <= 2 and tensor_b.dim() == 1:
        # Matrix-vector product
        return torch.mv(tensor_a, tensor_b)
    else:
        # Matrix-matrix product or batched matrix product
        return torch.matmul(tensor_a, tensor_b)



def rdiv_torch(a, b):
    """
    Returns the solution to x b = a using PyTorch. Equivalent to MATLAB right matrix
    division: a / b
    
    Parameters
    ----------
    a : torch.Tensor
        The right-hand side matrix.
    b : torch.Tensor
        The matrix to be divided by.
        
    Returns
    -------
    torch.Tensor
        The result of the division.
    """
    # Transpose both `a` and `b`, solve for `x` in the equation `b.T x = a.T`
    result = torch.linalg.solve(b.T, a.T)
    # Transpose the result to get the final output
    return result.T

def inv_persymm_torch(M, blk_size):
    T = int(M.shape[0] / blk_size)
    Thalf = int(np.ceil(T / 2.0))
    mkr = blk_size * Thalf

    invA11 = torch.inverse(M[:mkr, :mkr])
    invA11 = (invA11 + invA11.T) / 2

    A12 = M[:mkr, mkr:]
    term = flexible_dot(invA11,  A12)
    F22 = M[mkr:, mkr:] - flexible_dot(A12.T, term)

    res12 = rdiv_torch(-term, F22)
    res11 = invA11 - flexible_dot(res12,term.T)
    res11 = (res11 + res11.T) / 2

    invM = fill_persymm_torch(torch.cat([res11, res12], dim=1), blk_size, T)

    logdet_M = -torch.logdet(invA11) + torch.logdet(F22)

    return invM, logdet_M


def fill_persymm_torch(p_in, blk_size, n_blocks, blk_size_vert=None):
    if blk_size_vert is None:
        blk_size_vert = blk_size

    Nh = blk_size * n_blocks
    Nv = blk_size_vert * n_blocks
    Thalf = int(np.floor(n_blocks / 2.0))
    THalf = int(np.ceil(n_blocks / 2.0))

    Pout = torch.empty((blk_size_vert * n_blocks, blk_size * n_blocks), dtype=p_in.dtype, device=p_in.device)
    Pout[:blk_size_vert * THalf, :] = p_in
    for i in range(Thalf):
        for j in range(n_blocks):
            Pout[Nv - (i + 1) * blk_size_vert:Nv - i * blk_size_vert,
                 Nh - (j + 1) * blk_size:Nh - j * blk_size] \
                = p_in[i * blk_size_vert:(i + 1) * blk_size_vert,
                       j * blk_size:(j + 1) * blk_size]

    return Pout
def make_k_big_torch(params, n_timesteps):
    """
    Constructs full GP covariance matrix across all state dimensions and
    timesteps using PyTorch.

    Parameters
    ----------
    params : dict
        GPFA model parameters
    n_timesteps : int
        number of timesteps

    Returns
    -------
    K_big : torch.Tensor
        GP covariance matrix with dimensions (xDim * T) x (xDim * T).
    K_big_inv : torch.Tensor
        Inverse of K_big
    logdet_K_big : float
        Log determinant of K_big
    """
    if params['covType'] != 'rbf':
        raise ValueError("Only 'rbf' GP covariance type is supported.")

    xDim = params['C'].shape[1]

    K_big = torch.zeros(xDim * n_timesteps, xDim * n_timesteps)
    K_big_inv = torch.zeros(xDim * n_timesteps, xDim * n_timesteps)
    Tdif = torch.tile(torch.arange(0, n_timesteps), (n_timesteps, 1)).T \
        - torch.tile(torch.arange(0, n_timesteps), (n_timesteps, 1))
    logdet_K_big = 0

    for i in range(xDim):
        K = (1 - params['eps'][i]) * torch.exp(-params['gamma'][i] / 2 * Tdif.float() ** 2) \
            + params['eps'][i] * torch.eye(n_timesteps)
        K_big[i::xDim, i::xDim] = K
        K_inv = torch.inverse(K)
        K_big_inv[i::xDim, i::xDim] = K_inv
        logdet_K = torch.logdet(K)
        logdet_K_big += logdet_K

    return K_big, K_big_inv, logdet_K_big.item()



# Note: Further adaptations may be necessary depending on specific usage within your code.
