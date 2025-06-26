import torch
import numpy as np
import ot
import statsmodels.api as sm
from scipy.stats import binned_statistic

def swd_stable(X, Y, n_runs=10, n_proj=256):
    X = X.numpy()
    Y = Y.numpy()
    swd_all = []
    for _ in range(n_runs):
        swd_temp = swd(X=X, Y=Y, n_proj=n_proj)
        swd_all.append(swd_temp)
    return np.mean(swd_all)
    

def swd(X, Y, n_proj=256, seed=None):
    """
    Sliced Wasser
    X : [N, n_chan, n_seq]
    Y : [N, n_chan, n_seq]
    """
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    Yf = Y.reshape(N, -1)
    a = b = np.full(N, 1. / N)
    return ot.sliced.sliced_wasserstein_distance(
        Xf, Yf, a=a, b=b, n_projections=n_proj, p=2, seed=seed
    )

def unbiased_mmd2_torch(X, Y,
                        gamma: float = None, device='cpu') -> torch.Tensor:
    """
    Compute the unbiased squared MMD between samples X and Y using an RBF kernel in PyTorch.

    Parameters
    ----------
    X : [N, n_chan, n_seq]
    Y : [N, n_chan, n_seq]
    gamma : float or None
        RBF bandwidth parameter. 

    Returns
    -------
    Tensor
        Scalar Tensor giving the unbiased MMD^2 estimate (clamped ≥ 0 for stability).
    """
    # Flatten samples to shape (m, D) and (n, D), m=n=N
    X = torch.Tensor(X).to(device)
    Y = torch.Tensor(Y).to(device)
    
    m = X.size(0)
    n = Y.size(0)
    X_flat = X.reshape(m, -1)
    Y_flat = Y.reshape(n, -1)
    D = X_flat.size(1) # D is the discretization of the domain

    if gamma is None:
        gamma = 1.0 

    # Pairwise squared distances
    #   ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    xx = (X_flat * X_flat).sum(dim=1, keepdim=True)  # (m,1)
    yy = (Y_flat * Y_flat).sum(dim=1, keepdim=True)  # (n,1)
    # compute gram matrices of squared distances
    # shape (m, n)
    dist_xy = xx + yy.t() - 2.0 * (X_flat @ Y_flat.t())
    dist_xx = xx + xx.t() - 2.0 * (X_flat @ X_flat.t())
    dist_yy = yy + yy.t() - 2.0 * (Y_flat @ Y_flat.t())

    # Kernel matrices
    Kxy = torch.exp(-gamma * dist_xy / D) 
    Kxx = torch.exp(-gamma * dist_xx / D)
    Kyy = torch.exp(-gamma * dist_yy / D)

    # Unbiased: zero out diagonals of Kxx and Kyy
    Kxx.fill_diagonal_(0)
    Kyy.fill_diagonal_(0)

    # Compute the three terms
    term_x = Kxx.sum() / (m * (m - 1))
    term_y = Kyy.sum() / (n * (n - 1))
    term_xy = 2.0 * Kxy.sum() / (m * n)

    mmd2 = term_x + term_y - term_xy

    # numerical stability: clamp to zero if slightly negative
    return torch.sqrt(torch.clamp(mmd2, min=0.0)).cpu().numpy() # return mmd 


## for fluid related
def spectrum_2d(signal, n_observations, normalize=True):
    """
    from neuralop library
    """

    T = signal.shape[0]
    signal = signal.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal, norm="ortho")
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), norm="backward"
        )

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]

    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].sum(dim=1)).abs() ** 2

    spectrum = spectrum.mean(dim=0)
    return spectrum

def compute_acovf(z, nlag=50):
    # z shape : [n, ndim, ndim] (only works for 2D)
    res = z.shape[-1]
    z_hat = torch.fft.rfft2(z)
    acf = torch.fft.irfft2(torch.conj(z_hat) * z_hat)
    acf = torch.fft.fftshift(acf).mean(dim=0) / z[0].numel() # ndim*ndim
    acf_r = acf.view(-1).cpu().detach().numpy()
    lags_x, lags_y = torch.meshgrid(torch.arange(res) - res//2, torch.arange(res) - res//2)
    lags_r = torch.sqrt(lags_x**2 + lags_y**2).view(-1).cpu().detach().numpy()

    idx = np.argsort(lags_r)
    lags_r = lags_r[idx]
    acf_r = acf_r[idx]

    bin_means, bin_edges, binnumber = binned_statistic(lags_r, acf_r, 'mean', bins=np.linspace(0.0, res, nlag))
    return bin_edges[:-1], bin_means