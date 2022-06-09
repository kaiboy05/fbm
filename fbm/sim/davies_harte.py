from .fbm_generator import FBmGeneratorInterface
from .fbm_generator import BiFBmGeneratorInterface
from .fbm_generator import MFBmGeneratorInterface
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
        self.__mfBm_gen = DaviesHarteMFBmGenerator()

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
        rhos = np.zeros((2,2))
        rhos[1,0] = rhos[0,1] = rho
        return self.__mfBm_gen.generate_mfGn(np.array([H1,H2]), rhos, size)


class DaviesHarteMFBmGenerator(MFBmGeneratorInterface):
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
        self.__cached_Hs:np.ndarray = np.array([0.5])
        self.__cached_rho:np.ndarray = np.ones(1)
        self.__cached_transformation:np.ndarray = np.ndarray((1,1,1))
        self.__uni_gen:DaviesHarteFBmGenerator = DaviesHarteFBmGenerator()

    def generate_norm_mfGn(self, Hs: np.ndarray, rho:np.ndarray, size: int) -> np.ndarray:
        """
        Generate time series of fGns of bivariate fBm, with spacing 1,
        two correlated by rho.

        Parameters
        ----------
        Hs: np.ndarray
            Hurst parameters. Should be in range `(0, 1)**p`.
        
        rho: np.ndarray
            Correlation coefficients. Should be in range `[0, 1]**(p*p)`.

        size: int
            Size of time series to generate. Should be larger than 1.

        Returns
        -------
        ts: `(2, len(size))` ndarray
            Time series of fGns of bivariate fBm, with spacing 1.

        """
        N = closest_pow2(size)
        p = Hs.size
        if p == 1:
            return self.__uni_gen.generate_fGn(Hs[0], size)

        if  not np.array_equal(self.__cached_Hs, Hs) or \
            not np.array_equal(self.__cached_rho, rho) or \
            len(self.__cached_transformation) < 2*size:

            circulant_row1 = np.ndarray((N << 1, p, p))

            circulant_row1[:N] = np.array(
                [utils.multivariate_fGn_cov(i, Hs, rho) for i in range(N)]
            )

            circulant_row1[N] = utils.multivariate_fGn_cov(N, Hs, rho)
            circulant_row1[-N+1:] = circulant_row1[N-1:0:-1]

            B = np.ndarray((N << 1, p, p), dtype=complex)
            for i in range(p):
                for j in range(i+1):
                    B[:,i,j] = np.fft.fft(circulant_row1[:,i,j])
                    if i != j:
                        B[:,j,i] = np.conjugate(B[:,i,j])

            self.__cached_transformation = np.ndarray((N << 1, p, p), 
                dtype=complex)
            for i in range(len(self.__cached_transformation)):
                e, L = np.linalg.eig(B[i])
                e[e < 0] = 0
                e = np.diag(np.sqrt(e))
                self.__cached_transformation[i] = L @ e @ np.conjugate(L.T)

            self.__cached_Hs = Hs
            self.__cached_rho = rho

        v1 = np.random.standard_normal((p, N-1))
        v2 = np.random.standard_normal((p, N-1))
        w = np.ndarray((p, 2*N), dtype=complex)
        w[:,0] = np.random.standard_normal(p) / np.sqrt(2*N)
        w[:,N] = np.random.standard_normal(p) / np.sqrt(2*N)
        w[:,1:N] = (v1 + 1j*v2) / np.sqrt(4*N)
        w[:,-N+1:] = np.conjugate(w[:,N-1:0:-1])
        w = np.einsum('...ij,j...->i...', self.__cached_transformation, w, 
                optimize='optimal')

        ts = np.ndarray((p, size))
        for i in range(p):
            w[i] = np.fft.fft(w[i])
        ts = np.real(w[:, :size])

        return ts

if __name__ == '__main__':
    from .generator_test_utils import fBm_generator_chi_square_test
    from .generator_test_utils import bfBm_generator_chi_square_test

    import matplotlib.pyplot as plt

    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.1)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.25)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.5)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.75)
    fBm_generator_chi_square_test(DaviesHarteFBmGenerator(), H=0.9)

    bfBm_generator_chi_square_test(DaviesHarteBiFBmGenerator(), 
        H1=0.1, H2=0.1, rho=0
    )
    bfBm_generator_chi_square_test(DaviesHarteBiFBmGenerator(), 
        H1=0.2, H2=0.2, rho=0
    )
    bfBm_generator_chi_square_test(DaviesHarteBiFBmGenerator(), 
        H1=0.4, H2=0.25, rho=0
    )
    bfBm_generator_chi_square_test(DaviesHarteBiFBmGenerator(), 
        H1=0.5, H2=0.1, rho=0
    )
    # mfbm = DaviesHarteMFBmGenerator()
    # rhos=np.zeros((3,3))
    # ts = mfbm.generate_mfBm(Hs=np.array([0.1,0.1,0.1]), rho=rhos, size=1000)
    # plt.plot(ts[0], label="H1=0.5")
    # plt.plot(ts[1], label="H1=0.3")
    # plt.plot(ts[2], label="H1=0.1")
    # plt.legend()
    # plt.show()

