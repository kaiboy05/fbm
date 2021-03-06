from cProfile import label
from matplotlib.pyplot import legend
from .fbm_generator import FBmGeneratorInterface
from .fbm_generator import BiFBmGeneratorInterface
import numpy as np

from fbm import utils

class NaiveFBmGenerator(FBmGeneratorInterface):
    """
    Naive generator that constructs the covariance matrix of
    the fGn, and generate the time series given the required
    size and Hurst parameter
    """

    def __init__(self) -> None:
        self.__cached_H:float = -1
        self.__cov:np.ndarray = np.array([[1]])

    def generate_norm_fGn(self, H: float, size: int) -> np.ndarray:
        """
        Generate time series of fractional gaussian noise (fGn), with spacing 1.

        Parameters
        ----------
        H: int
            Hurst parameter. Should be in range `(0, 1)`.

        size: int
            Size of time series to generate. Should be larger than 0.

        Returns
        -------
        ts: `(len(size))` ndarray
            Time series of fBm, with spacing 1.
        """
        if H != self.__cached_H or len(self.__cov) != size:
            self.__cov = utils.cov(size, H)
            self.__cached_H = H
        
        ts = np.random.multivariate_normal(np.zeros(size),self.__cov)

        return ts

class NaiveBiFBmGenerator(BiFBmGeneratorInterface):
    """
    Generator Interface for generating bivariate fractional brownian motion (bfBm).

    Notes
    -----
    Method from 'BASIC PROPERTIES OF THE MULTIVARIATE FRACTIONAL BROWNIAN MOTION' section 5
    By PIERRE-OLIVIER et al
    """
    def __init__(self) -> None:
        self.__cached_H1:float = -1
        self.__cached_H2:float = -1
        self.__cov:np.ndarray = np.array([[1]])

    def generate_norm_bifGn(self, H1: float, H2: float, rho:float, size: int) -> np.ndarray:
        """
        Generate time series of fGns of bivariate fBm, with spacing 1,
        two correlated by rho.

        Parameters
        ----------
        H1: float
            Hurst parameter. Should be in range `(0, 1)`.
        
        H2: float
            Hurst parameter. Should be in range `(0, 1)`.
        
        rho: float
            Correlation coefficient. Should be in range `[0, 1]`.

        size: int
            Size of time series to generate. Should be larger than 1.

        Returns
        -------
        ts: `(2, len(size))` ndarray
            Time series of fGns of bivariate fBm, with spacing 1.

        """
        if rho > utils.bfBm_max_rho(H1, H2):
            raise utils.RhoTooLargeError

        if H1 != self.__cached_H1 or H2 != self.__cached_H2 or \
            len(self.__cov) != 2*size:
            self.__cov = utils.bivariate_fGn_cov_structure(size, H1, H2, rho)
            self.__cached_H1 = H1
            self.__cached_H2 = H2
        
        result = np.random.multivariate_normal(np.zeros(2*size), self.__cov)
        ts = np.ndarray((2, size))
        ts[0] = result[0:size]
        ts[1] = result[size:]

        return ts


if __name__ == '__main__':
    from .generator_test_utils import fBm_generator_chi_square_test
    from .generator_test_utils import bfBm_generator_chi_square_test

    Hs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for H in Hs:
      fBm_generator_chi_square_test(NaiveFBmGenerator(), H=H)

    bfBm_generator_chi_square_test(NaiveBiFBmGenerator(), 
        H1=0.5, H2=0.5, rho=0.6, size=100
    )
    bfBm_generator_chi_square_test(NaiveBiFBmGenerator(), 
        H1=0.1, H2=0.3, rho=0.4, size=100
    )
    bfBm_generator_chi_square_test(NaiveBiFBmGenerator(), 
        H1=0.2, H2=0.2, rho=0.3, size=100
    )
    bfBm_generator_chi_square_test(NaiveBiFBmGenerator(), 
        H1=0.4, H2=0.25, rho=0.2, size=100
    )
    
    
