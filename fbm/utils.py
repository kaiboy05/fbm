import time
import pickle
import os
import math

import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import cholesky
from functools import lru_cache

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

@lru_cache(maxsize=16)
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

@lru_cache(maxsize=16)
def cov_chol(n: int, H:float) -> np.ndarray:
    """
    Return cholesky decomposition of covariance matrix of fGn

    Same arguments as function cov
    """
    return cholesky(cov(n, H), lower=True)


def bivariate_fGn_cross_cov(n:int, H1:float, H2:float, rho:float, 
        sigma1:float=1, sigma2:float=1) -> float:
    """
    Cross-covariance of the corresponding fGns of
    bivariate fBm (W^{H1}_t, W_{H2}_t), where
    corr(W^{H1}_{1}, W^{H2}_{1}) = rho, and
    var1 and var2 are var(W^{H1}_{1}) and var(W^{H2}_{1})
    respectively.

    rho_1,2(n) = var1 * var2 / 2 * (w(n-1) -2w(n) + w(n+1))
    where
    w(h) = rho * abs(h)^{H1 + H2}

    Parameters
    ----------
    n: int
        lag parameter.

    H1: float
        Hurst parameter. Should be in range `(0, 1)`.
    
    H2: float
        Hurst parameter. Should be in range `(0, 1)`.
    
    rho: float
        Correlation of the two fBms. Should be in range `[0, 1]`.
        corr(W^{H1}_{1}, W^{H2}_{1}) = rho
    
    sigma1: float
        Std deviation of W^{H1}_{1}
    
    sigma2: float
        Std deviation of W^{H2}_{1}

    Returns
    -------
    cross_cov: float

    Notes
    -------
    Equation from 'BASIC PROPERTIES OF THE MULTIVARIATE FRACTIONAL BROWNIAN MOTION' section 2.2
    By PIERRE-OLIVIER et al
    """
    def w(h:int)->float:
        h = abs(h)
        return rho * h**(H1 + H2)

    result = w(n-1) - 2*w(n) + w(n+1)
    result = sigma1 * sigma2 / 2 * result

    return result

def bivariate_fGn_cov(n:int, H1:float, H2:float, correlated_rho:float, 
        sigma1:float=1, sigma2:float=1) -> np.ndarray:
    """
    Covariance matrix of the corresponding fGns of
    bivariate fBm (W^{H1}_t, W_{H2}_t), where
    corr(W^{H1}_{1}, W^{H2}_{1}) = rho
    respectively.

    [[rho_1,1(n), rho_1,2(n)], [rho_2,1(n), rho_2,2(n)]]

    Parameters
    ----------
    n: int
        lag parameter.

    H1: float
        Hurst parameter. Should be in range `(0, 1)`.
    
    H2: float
        Hurst parameter. Should be in range `(0, 1)`.
    
    rho: float
        Correlation of the two fBms. Should be in range `[0, 1]`.
        corr(W^{H1}_{1}, W^{H2}_{1}) = rho
    
    sigma1: float
        Std deviation of W^{H1}_{1}
    
    sigma2: float
        Std deviation of W^{H2}_{1}

    Returns
    -------
    cov: `(2, 2)` ndarray
    """

    result = np.ndarray((2,2))
    result[0][0] = rho(n, H1) * sigma1
    result[1][1] = rho(n, H2) * sigma2
    result[0][1] = result[1][0] = bivariate_fGn_cross_cov(n, H1, H2, 
        correlated_rho, sigma1=sigma1, sigma2=sigma2)
    
    return result

@lru_cache(maxsize=8)
def bivariate_fGn_cov_structure(size:int, H1:float, H2:float, 
        correlated_rho:float, sigma1:float=1, sigma2:float=1) -> np.ndarray:
    """
    Covariance structure of the corresponding fGns of
    bivariate fBm (W^{H1}_t, W_{H2}_t), where
    corr(W^{H1}_{1}, W^{H2}_{1}) = rho
    respectively.

    Parameters
    ----------
    size: int
        lag parameter.

    H1: float
        Hurst parameter. Should be in range `(0, 1)`.
    
    H2: float
        Hurst parameter. Should be in range `(0, 1)`.
    
    rho: float
        Correlation of the two fBms. Should be in range `[0, 1]`.
        corr(W^{H1}_{1}, W^{H2}_{1}) = rho
    
    sigma1: float
        Std deviation of W^{H1}_{1}
    
    sigma2: float
        Std deviation of W^{H2}_{1}

    Returns
    -------
    cov: `(2*size, 2*size)` ndarray
    """

    result = np.ndarray((2*size, 2*size))
    result[:size, :size] = cov(size, H1)
    result[-size:, -size:] = cov(size, H2)
    for i in range(size):
        for j in range(size):
            result[i, size+j] = result[size+j, i] = \
                bivariate_fGn_cross_cov(i-j, H1, H2, 
                    correlated_rho, sigma1, sigma2)
    return result

@lru_cache(maxsize=8)
def bivariate_fGn_cov_structure_chol(size:int, H1:float, H2:float, 
        correlated_rho:float, sigma1:float=1, sigma2:float=1) -> np.ndarray:
    """
    Return cholesky decomposition of covariance structure of the corresponding 
    fGns of bivariate fBm (W^{H1}_t, W_{H2}_t), where
    corr(W^{H1}_{1}, W^{H2}_{1}) = rho respectively.

    Same argument as function bivariate_fGn_cov_structure
    """
    return cholesky(
        bivariate_fGn_cov_structure(size, H1, H2, correlated_rho, sigma1, sigma2)
        ,lower=True
    )

class RhoTooLargeError(Exception):
    """Rho is too large for given H1 and H2"""
    pass

@lru_cache(maxsize=128)
def bfBm_max_rho(H1:float, H2:float) -> float:
    result = math.gamma(2*H1+1) * math.gamma(2*H2+1) / math.gamma(H1+H2+1)**2
    result *= math.sin(math.pi*H1)*math.sin(math.pi*H2)
    result /= (math.sin(math.pi*(H1+H2)/2))**2

    assert result >= 0

    return math.sqrt(result)

class BackupHelper:
    def __init__(self, dir_name, file_name=None, suffix=None, save_versions=False):
        self.dir_name = dir_name
        if file_name is None:
            file_name = str(int(time.time()))
        if suffix is None:
            self.file_name = file_name
        else:
            self.file_name = f"{file_name}_{suffix}"

        self.backup_version = 0
        self.save_versions = save_versions

        os.makedirs(os.path.dirname(self.__get_backup_path()), exist_ok=True)
    
    def __get_backup_path(self):
        if self.save_versions:
            return f"{self.dir_name}/{self.file_name}_{self.backup_version}.pickle"
        else:
            return f"{self.dir_name}/{self.file_name}.pickle"

    def dump(self, obj):
        self.file = open(self.__get_backup_path(), 'wb+')
        pickle.dump(obj, self.file)
        self.file.close()
        self.backup_version += 1
    
    def dump_final(self, obj):
        pickle.dump(obj, self.file)
        self.file.close()


class FunctionTimer:
    def __init__(self):
        self.elapsed_counter = 0
        self.__start = time.perf_counter()
        self.__state = 0
    
    def pause(self):
        if self.__state != 1:
            self.elapsed_counter += time.perf_counter() - self.__start
        self.__state = 1
    
    def cont(self):
        if self.__state == 1:
            self.__start = time.perf_counter()
        self.__state = 0
    
    def stop(self):
        if self.__state == 0:
            self.elapsed_counter += time.perf_counter() - self.__start
        self.__state = 2
        return self.elapsed_counter