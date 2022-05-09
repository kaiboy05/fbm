from ctypes import util
from .fbm_generator import FBmGeneratorInterface
from .fbm_generator import BiFBmGeneratorInterface
import numpy as np

from fbm import utils

def closest_pow2(n:int) -> int:
    """
    The closest power of 2 that is larger than or equal to n
    """

    assert n >= 0

    result = 1
    while result < n:
        result <<= 1
    
    return result

class DaviesHarteFBmGenerator(FBmGeneratorInterface):
    """
    Davies and Harte Method generator that makes use of circulant covariance
    matrix's eigendecomposition property, and the fast fourier transform, to
    generate fractional gaussian noise

    Notes
    -----
    Method from 'Simulation of FBm' master thesis section 2.1.3
    """

    def __init__(self) -> None:
        self.__cached_H:float = -1
        self.__processed_eigs:np.ndarray = np.array([1])

    def seed(self, s: int) -> None:
        """
        Set the seed of the generator.

        Parameters
        ----------
        seed : int
        """
        np.random.seed(s)

    def generate_fBm(self, H: float, size: int) -> np.ndarray:
        """
        Generate time series of fBm, with spacing 1,
        and the the first element must be 0.

        Parameters
        ----------
        H: int
            Hurst parameter. Should be in range `(0, 1)`.

        size: int
            Size of time series to generate. Should be larger than 1.

        Returns
        -------
        ts: `(len(size))` ndarray
            Time series of fBm, with spacing 1.
        """
        assert size > 1

        fGn = self.generate_fGn(H, size - 1)
        ts = np.cumsum(np.insert(fGn, 0, 0))

        return ts

    def generate_fGn(self, H: float, size: int) -> np.ndarray:
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
        N = closest_pow2(size)
        if self.__cached_H != H or len(self.__processed_eigs) < 2*size:
            circulant_row1 = np.ndarray(N << 1)

            circulant_row1[:N] = np.array(
                [utils.rho(i, H) for i in range(N)]
            )
            circulant_row1[N] = 0
            circulant_row1[-N+1:] = circulant_row1[N-1:0:-1]

            self.__processed_eigs = np.fft.fft(circulant_row1)
            # Eigenvalues of circulant matrix might not be always positive
            # But thesis stated that it must be positive
            self.__processed_eigs = np.abs(self.__processed_eigs)
            self.__processed_eigs /= (4 * N)
            self.__processed_eigs[0] *= 2
            self.__processed_eigs[N] *= 2
            self.__processed_eigs = np.sqrt(self.__processed_eigs)

        v1 = np.random.standard_normal(N-1)
        v2 = np.random.standard_normal(N-1)

        w = np.ndarray(2*N, dtype=complex)
        w[1:N] = (v1 + 1j*v2) / np.sqrt(2)
        w[-N+1:] = np.conjugate(w[N-1:0:-1])
        w[0] = np.random.standard_normal(1).item()
        w[N] = np.random.standard_normal(1).item()

        # print(self.__processed_eigs)

        ts = np.fft.fft(self.__processed_eigs * w)

        return np.real(ts[:size])
    
class DaviesHarteBiFBmGenerator(BiFBmGeneratorInterface):
    """
    Davies and Harte Method generator that makes use of circulant covariance
    matrix's eigendecomposition property, and the fast fourier transform, to
    generate fractional gaussian noise

    Notes
    -----
    Method from 'BASIC PROPERTIES OF THE MULTIVARIATE FRACTIONAL BROWNIAN MOTION' section 5
    By PIERRE-OLIVIER et al
    """

    def __init__(self) -> None:
        self.__cached_Hs:tuple[float, float] = (0.5, 0.5)
        self.__cached_transformation:np.ndarray = np.ndarray((1,2,2))

    def seed(self, s: int) -> None:
        """
        Set the seed of the generator.

        Parameters
        ----------
        seed : int
        """
        np.random.seed(s)

    def generate_bifBm(self, H1: float, H2: float, rho:float, size: int) -> np.ndarray:
        """
        Generate time series of bivariate fBm, with spacing 1,
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
            Time series of bivariate fBm, with spacing 1.

        """
        assert size > 1

        fGns = self.generate_bifGn(H1, H2, rho, size - 1)
        ts = np.cumsum(np.insert(fGns, 0, 0, axis=1), axis=1)

        return ts

    def generate_bifGn(self, H1: float, H2: float, rho:float, size: int) -> np.ndarray:
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
        N = closest_pow2(size)
        if  self.__cached_Hs != (H1, H2) or \
            len(self.__cached_transformation) < 2*size:

            circulant_row1 = np.ndarray((N << 1, 2, 2))

            circulant_row1[:N] = np.array(
                [utils.bivariate_fGn_cov(i, H1, H2, rho) for i in range(N)]
            )

            circulant_row1[N] = utils.bivariate_fGn_cov(N, H1, H2, rho)
            circulant_row1[-N+1:] = circulant_row1[N-1:0:-1]


            B = np.ndarray((N << 1, 2, 2), dtype=complex)
            B[:,0,0] = np.fft.fft(circulant_row1[:,0,0])
            B[:,0,1] = np.fft.fft(circulant_row1[:,0,1])
            B[:,1,0] = np.conjugate(B[:,0,1])
            B[:,1,1] = np.fft.fft(circulant_row1[:,1,1])

            self.__cached_transformation = np.ndarray((N << 1, 2, 2), dtype=complex)
            for i in range(len(self.__cached_transformation)):
                e, L = np.linalg.eig(B[i])
                assert np.all(e >= 0)
                e = np.diag(np.sqrt(e))
                self.__cached_transformation[i] = L @ e @ np.conjugate(L.T)

        v1 = np.random.standard_normal((2, N-1))
        v2 = np.random.standard_normal((2, N-1))
        w = np.ndarray((2, 2*N), dtype=complex)
        w[:,0] = np.random.standard_normal(2) / np.sqrt(2*N)
        w[:,N] = np.random.standard_normal(2) / np.sqrt(2*N)
        w[:,1:N] = (v1 + 1j*v2) / np.sqrt(4*N)
        w[:,-N+1:] = np.conjugate(w[:,N-1:0:-1])
        for i in range(len(w[0])):
            w[:,i] = self.__cached_transformation[i] @ w[:,i]

        ts = np.ndarray((2, size))
        ts0 = np.fft.fft(w[0])
        ts1 = np.fft.fft(w[1])

        assert(np.isreal(np.all(ts0)))
        assert(np.isreal(np.all(ts1)))

        ts[0] = np.real(ts0[:size])
        ts[1] = np.real(ts1[:size])

        return ts

if __name__ == '__main__':
    from .generator_test_utils import fBm_generator_chi_square_test
    from .generator_test_utils import bfBm_generator_chi_square_test

    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.1)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.25)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.5)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.75)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.9)

    bfBm_generator_chi_square_test(DaviesHarteBiFBmGenerator(), 
        H1=0.1, H2=0.3, rho=0.4
    )
    bfBm_generator_chi_square_test(DaviesHarteBiFBmGenerator(), 
        H1=0.2, H2=0.2, rho=0.3
    )
    bfBm_generator_chi_square_test(DaviesHarteBiFBmGenerator(), 
        H1=0.4, H2=0.25, rho=0.2
    )
    bfBm_generator_chi_square_test(DaviesHarteBiFBmGenerator(), 
        H1=0.5, H2=0.1, rho=0.6
    )

