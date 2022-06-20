from ctypes import util
from math import gamma
from .fbm_generator import FBmGeneratorInterface
import numpy as np

from fbm import utils

class HoskingFBmGenerator(FBmGeneratorInterface):
    """
    Hosking method generator of fBm that compute the mean and
    variance of the next fractional gaussian noise recursively.

    Notes
    -----
    Method from 'Simulation of FBm' master thesis section 2.1.1
    """

    def __init__(self) -> None:
        self.cached_H = -1
        self.__autocov = np.asarray([-1])
        self.ds:list[np.ndarray] = []
        self.sig2 = np.asarray([-1.0])

    def __cache_autocov(self, H:float, size:int=100) -> None:
        """
        Compute the autocovariance function given the Hurst parameter,
        and the size needed, and store it in self.__autocov
        """
        if self.cached_H != H:
            self.__autocov = np.asarray(
                [utils.rho(i, H) for i in range(1, size+1)]
            )
        elif self.__autocov.size < size:
            current_size = self.__autocov.size
            self.__autocov:np.ndarray = np.append( self.__autocov,
                [utils.rho(i, H) for i in range(current_size, size+1)]
            )

    def __cache_d_sig2(self, H:float, size:int=100) -> None:
        """
        Compute the vector d and vairance given the Hurst parameter,
        and the size needed, and store it in self.__d and self.__sig2
        recursively.

        c(n) = [rho(1), rho(2), ..., rho(n+1)] (size: n+1)

        d(0) = [rho(1)] (size: 1)
        sig2(0) = 1 - rho(1)^2

        tau(n) = c(n).d(n)[::-1]
        phi(n) = (rho(n+2) - tau(n)) / sig2(n)

        sig2(n+1) = sig2(n) - (rho(n+2)^2 - tau(n))^2 / sig2(n)
        d(n+1) = [(d(n) - phi(n) * d(n)[::-1]), phi(n)] (size: n+2)
        """
        curr_size = size
        self.__cache_autocov(H, size+1)
        if self.cached_H != H:
            self.ds:list[np.ndarray] = [self.__autocov[:1]]
            self.sig2.resize(size)
            self.sig2[0] = 1 - self.__autocov[0]**2
            curr_size = 1

        elif len(self.ds) < size:
            curr_size = len(self.ds)
            self.sig2.resize(size)
        
        while curr_size < size:
            n = curr_size - 1
            d = self.ds[n]
            sig2:float = self.sig2[n]
            c:np.ndarray = self.__autocov[:n+1]

            tau:float = d.dot(c[::-1]).item()
            rho_n_plus_2 = utils.rho(n+2, H)
            phi = (rho_n_plus_2 - tau) / sig2

            new_d:np.ndarray = np.append(d - phi * d[::-1], phi)
            new_sig2 = sig2 - (rho_n_plus_2 - tau)**2 / sig2

            self.ds.append(new_d)
            self.sig2[curr_size] = new_sig2
            curr_size += 1

    def __generate_Xi(self, i, ts) -> float:
        """
        Generate X_i given ts ([X_0, ..., X_i-1]).
        Note that X_0 is a standard normal
        """
        if i == 0:
            return np.random.standard_normal()
        else:
            d = self.ds[i-1]
            sig2 = self.sig2[i-1]
            mu = ts[:i][::-1].dot(d)

            return np.random.normal(mu, np.sqrt(sig2))
            

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
        self.__cache_d_sig2(H, size - 1)

        ts = np.ndarray(size)
        for i in range(size):
            ts[i] = self.__generate_Xi(i, ts)

        return ts

if __name__ == '__main__':
    from .generator_test_utils import fBm_generator_chi_square_test
    from .naive import NaiveFBmGenerator

    fBm_generator_chi_square_test(HoskingFBmGenerator(), H=0.1, sim_num=1, size=5, plot_graph=True)
    fBm_generator_chi_square_test(HoskingFBmGenerator(), H=0.1)
    fBm_generator_chi_square_test(HoskingFBmGenerator(), H=0.25)
    fBm_generator_chi_square_test(HoskingFBmGenerator(), H=0.5)
    fBm_generator_chi_square_test(HoskingFBmGenerator(), H=0.75)
    fBm_generator_chi_square_test(HoskingFBmGenerator(), H=0.9)
