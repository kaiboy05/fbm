from ctypes import util
from .fbm_generator import FBmGeneratorInterface
import numpy as np

from fBm import utils

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

    @staticmethod
    def __closest_pow2(n:int) -> int:
        assert n >= 0

        result = 1
        while result < n:
            result <<= 1
        
        return result


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
        N = self.__closest_pow2(size)
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

if __name__ == '__main__':
    from .generator_test_utils import fBm_generator_chi_square_test
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.1)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.25)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.5)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.75)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.9)
