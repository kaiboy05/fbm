from torch import norm
from .fbm_generator import FBmGeneratorInterface
import numpy as np

from fBm import utils

class NaiveFBmGenerator(FBmGeneratorInterface):
    """
    Naive generator that constructs the covariance matrix of
    the fGn, and generate the time series given the required
    size and Hurst parameter
    """

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
        cov = utils.cov(size, H)
        ts = np.random.multivariate_normal(np.zeros(size),cov)

        return ts

if __name__ == '__main__':
    from .generator_test_utils import fBm_generator_chi_square_test

    fBm_generator_chi_square_test(
        NaiveFBmGenerator(), H=0.25, plot_graph=True
    )