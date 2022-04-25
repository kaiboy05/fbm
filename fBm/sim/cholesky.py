from .fbm_generator import FBmGeneratorInterface
import numpy as np

from fBm import utils

class CholeskyFBmGenerator(FBmGeneratorInterface):
    """
    Cholesky Method generator that constructs the covariance matrix of
    the fGn, compute its Cholesky decomposition,
    and generate the time series given the required
    size and Hurst parameter

    Notes
    -----
    Method from 'Simulation of FBm' master thesis section 2.1.2
    """
    def __init__(self) -> None:
        self.__cached_H:float = -1
        self.__L:list[np.ndarray] = [np.array([1])]

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

    def get_Li(self, i) -> np.ndarray:
        while i >= len(self.__L):
            curr_size = len(self.__L)
            new_l = np.ndarray(curr_size + 1)

            new_l[0] = utils.rho(curr_size, self.__cached_H)
            for j in range(1, curr_size):
                new_l[j] = utils.rho(curr_size - j, self.__cached_H)
                for k in range(j):
                    new_l[j] -= new_l[k] * self.__L[j][k]
                new_l[j] /= self.__L[j][j]
            new_l[curr_size] = 0
            new_l[curr_size] = np.sqrt(1 - new_l.dot(new_l).item())

            self.__L.append(new_l)

        return self.__L[i]

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
        if self.__cached_H != H:
            self.__cached_H = H
            self.__L:list[np.ndarray] = [np.array([1])]

        Z = np.random.standard_normal(size)

        ts = np.zeros(size)
        for i in range(size):
            ts[i] = self.get_Li(i).dot(Z[:i+1])

        return ts

if __name__ == '__main__':
    from .generator_test_utils import fBm_generator_chi_square_test

    fBm_generator_chi_square_test(CholeskyFBmGenerator(), H=0.1)
    fBm_generator_chi_square_test(CholeskyFBmGenerator(), H=0.25)
    fBm_generator_chi_square_test(CholeskyFBmGenerator(), H=0.5)
    fBm_generator_chi_square_test(CholeskyFBmGenerator(), H=0.75)
    fBm_generator_chi_square_test(CholeskyFBmGenerator(), H=0.9)
