import numpy as np
from scipy.linalg import toeplitz

def rho(n: int, H: float) -> float:
    """
    Autocovariance function of fGn

    Parameters
    ----------
    n: int
        lag parameter.

    H: float
        Hurst parameter. Should be in range `(0, 1)`.

    Returns
    -------
    rho: float
    """
    assert 0 < H < 1
    assert n >= 0

    H2 = 2 * H
    if n == 0:
        return 1
    else:
        return ((n+1)**H2 + (n-1)**H2 - 2*(n**H2)) / 2
    
def cov(n: int, H: float) -> np.ndarray:
    """
    Covariance matrix of fGn

    Parameters
    ----------
    n: int
        lag parameter.

    H: float
        Hurst parameter. Should be in range `(0, 1)`.

    Returns
    -------
    rho: `(n, n)` ndarray
    """
    rho_vec = [rho(i, H) for i in range(n)]
    cov = toeplitz(rho_vec)

    return cov