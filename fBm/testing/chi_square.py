from logging import critical
from typing import Tuple

import numpy as np
from scipy import stats
import scipy
from scipy.linalg import cholesky, solve_triangular

from fBm import utils

def cov_matrix_chi_square_test(X: np.ndarray, cov: np.ndarray, alpha: float) \
        -> Tuple[bool, float, float]:
    """
    General Chi Square Test for gaussian process

    H0: Covariance = cov
    H1: Covariance != cov

    Test statistics:
        C_N(H) = |Z|^2,

        Z = L^{-1}X
    
    N degrees of freedom -> chi^2_N distribution

    Parameters
    ----------
    X: `(len(N))` ndarray
        fGn. (N > 0)
    cov: `(len(N), len(N))` ndarray
        Covariance matrix in the hypothesis test
    alpha:
        Significance level. (0 <= alpha <= 1)

    Return
    ------
    accept: bool
        Result of test.
    stat: float
        Test statistics.
    critical_val: float
        Critical value to accept H0
    """
    assert 0 <= alpha <= 1
    assert len(cov) == len(X)

    N = X.size
    L = cholesky(cov, lower=True)

    Z:np.ndarray = solve_triangular(L, X, lower=True)
    stat:float = Z.dot(Z).item()

    critical_val:float = stats.chi2.ppf(alpha, N).item()
    accept = stat <= critical_val

    return accept, stat, critical_val


def fGn_chi_square_test(X: np.ndarray, H: float, alpha: float) \
        -> Tuple[bool, float, float]:
    """
    Chi Square Test for fGn with spacing 1

    H0: Hurst Parameter = H
    H1: Hurst Parameter != H

    Test statistics:
        C_N(H) = |Z|^2,

        Z = L^{-1}X
    
    N degrees of freedom -> chi^2_N distribution

    Parameters
    ----------
    X: `(len(N))` ndarray
        fGn. (N > 0)
    H: float
        Hurst parameter in the hypothesis test. (0 < H < 1)
    alpha:
        Significance level. (0 <= alpha <= 1)

    Return
    ------
    accept: bool
        Result of test.
    stat: float
        Test statistics.
    critical_val: float
        Critical value to accept H0

    Notes
    -----
    Method from 'Simulation of FBm' master thesis section 3.2.2
    """
    assert X.ndim == 1
    assert 0 <= alpha <= 1

    N = X.size
    cov = utils.cov(N, H)
    return cov_matrix_chi_square_test(X, cov, alpha)

def fBm_chi_square_test(X: np.ndarray, H: float, alpha: float) \
        -> Tuple[bool, float, float]:
    """
    Chi Square Test for fBm with spacing 1

    H0: Hurst Parameter = H
    H1: Hurst Parameter != H

    Test statistics:
        C_N(H) = |Z|^2,

        Z = L^{-1}X',

        X' = fGn from X
    
    N degrees of freedom -> chi^2_N distribution

    Parameters
    ----------
    X: `(len(N))` ndarray
        fBm. (N > 1)
    H: float
        Hurst parameter in the hypothesis test. (0 < H < 1)
    alpha:
        Significance level. (0 <= alpha <= 1)

    Return
    ------
    accept: bool
        Result of test.
    stat: float
        Test statistics.
    critical_val: float
        Critical value to accept H0

    """
    fGn:np.ndarray = np.diff(X)

    a, s, c = fGn_chi_square_test(fGn, H, alpha)

    return a, s, c

def bfBm_chi_square_test(X: np.ndarray, H1: float, H2: float, rho: float,
        alpha: float) -> Tuple[bool, float, float]:
    """
    Chi Square Test for bfBm (Bivariate fBm) with spacing 1

    H0: bfBm is following the right covariance structure
    H1: bfBm is following the right covariance structure

    Parameters
    ----------
    X: `(len(N))` ndarray
        fBm. (N > 1)
    H1: float
        Hurst parameter in the hypothesis test. (0 < H1 < 1)
    H2: float
        Hurst parameter in the hypothesis test. (0 < H2 < 1)
    rho: float
        Correlation parameter in the hypothesis test. (0 <= rho < 1)
    alpha:
        Significance level. (0 <= alpha <= 1)

    Return
    ------
    accept: bool
        Result of test.
    stat: float
        Test statistics.
    critical_val: float
        Critical value to accept H0

    """
    fGn:np.ndarray = np.diff(X)
    fGn = fGn.flatten()

    cov = utils.bivariate_fGn_cov_structure(
        fGn.size // 2, H1, H2, rho
    )

    a, s, c = cov_matrix_chi_square_test(fGn, cov, alpha)

    return a, s, c


if __name__ == '__main__':
    from fBm.sim.naive import NaiveFBmGenerator
    import matplotlib.pyplot as plt

    size = 100
    H = 0.5
    sim_num = 100
    alpha = 0.95
    plot_graph = True

    fBm_generator = NaiveFBmGenerator()
    fBm_generator.seed(42)

    print(f"""
        Generating {sim_num} simulations of fBm with Hurst {H}, size {size}.
    """)
    
    fBm_ts = [fBm_generator.generate_fBm(H, size) for _ in range(sim_num)]
    
    accept_count = 0
    for ts in fBm_ts:
        plt.plot(np.arange(size), ts)
        a, s, c = fBm_chi_square_test(ts, H, alpha)

        accept_count = accept_count + (1 if a else 0)

    print(f"""
        {accept_count}/{sim_num} chi square tests have been passed.
    """)
    
    if plot_graph:
        plt.show()
