from typing import Tuple

import numpy as np
from scipy import stats
from fbm import utils

def covariance_lrt_test(expected_covariance_inv: np.ndarray, 
        paths:np.ndarray, alpha:float):
    n, p = paths.shape

    sample_cov = np.zeros((p,p))
    for path in paths:
        sample_cov += np.outer(path,path)
    sample_cov /= n

    exp_det = np.linalg.det(expected_covariance_inv)
    sample_det = np.linalg.det(sample_cov)

    w = n*np.einsum('ij,ji->', expected_covariance_inv, sample_cov)
    w -= n * (np.log(exp_det) + np.log(sample_det))
    w -= n * p

    critical_val:float = stats.chi2.ppf(alpha, p*(p+1) // 2).item()
    accept = w <= critical_val

    return accept, w, critical_val

def fGn_lrt_test(Xs:np.ndarray, H:float, alpha:float):
    N = Xs.shape[1]
    
    cov_inv = np.linalg.inv(utils.cov(N, H))

    return covariance_lrt_test(cov_inv, Xs, alpha)

def fBm_lrt_test(X: np.ndarray, H: float, alpha: float) \
        -> Tuple[bool, float, float]:
    fGns:np.ndarray = np.diff(X, axis=1)

    a, s, c = fGn_lrt_test(fGns, H, alpha)

    return a, s, c

if __name__ == '__main__':
    from fbm.sim import NaiveFBmGenerator

    naive = NaiveFBmGenerator()
    n = 3000
    size = 100
    Xs = np.ndarray((n, size))

    H = 0.9
    alpha = 0.95
    for i in range(n):
        Xs[i] = naive.generate_norm_fGn(H, size=size)
    print(fGn_lrt_test(Xs, H, alpha))


