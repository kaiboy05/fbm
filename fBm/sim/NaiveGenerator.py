import imp
from .FBmGeneratorInterface import FBmGeneratorInterface
import numpy as np
from scipy.linalg import toeplitz

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
    import matplotlib.pyplot as plt

    fBm_generator = NaiveFBmGenerator()
    fBm_generator.seed(42)

    size = 100
    H = 0.5
    sim_num = 5

    print(f"""
        Generating {sim_num} simulations of fBm with Hurst {H}, size {size}.
    """)
    
    fBm_ts = [fBm_generator.generate_fBm(H, size) for _ in range(sim_num)]
    
    for ts in fBm_ts:
        plt.plot(np.arange(size), ts)
    
    plt.show()

