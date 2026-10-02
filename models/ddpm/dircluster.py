import numpy as np
from scipy.stats import gamma
from numpy import linalg as LA
from utils import mvnrvs

def sample_mu_lam(xs0, ns, ks, k, _mu0, _ka0, _a0, _b0):
    """
    Sample mu and lambda of the k-th cluster (from normal-gamma distribution)
    xs0: data matrix, mat of N X D
    ns: ns[k] is the size of the k-th cluster, ns[k]==|C_k|, vec of K
    ks: ks[i] is the cluster index of the i-th data, ks[i] == k <-> xs0[i] in C_k, vec of N
    k: sample from the k-th cluster
    _mu0, _ka0, _a0, _b0: hyper priors
    """
    n = ns[k]
    D = xs0.shape[1]
    xx = xs0[ks == k]
    nn = len(xx)
    mzk = np.mean(xx, axis=0)
    mu_n = (_ka0 * _mu0 + n * mzk) / (_ka0 + n)
    ka_n = _ka0 + n
    a_n = _a0 + n * D / 2
    b_n = _b0
    b_n += np.sum((xx - np.tile(mzk, (nn, 1))) ** 2)
    b_n += _ka0 * n * LA.norm(mzk - _mu0) ** 2 / (2 * (_ka0 + n))
    lam_k = gamma.rvs(a_n, scale=1 / b_n)
    mu_k = mvnrvs(mu_n, 1 / (ka_n * lam_k))
    return (lam_k, mu_k)
