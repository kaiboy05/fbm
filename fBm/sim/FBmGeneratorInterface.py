import numpy as np

class FBmGeneratorInterface:
    """
    Generator Interface for generating fractional brownian motion (fBm).
    """
    def seed(self, s: int) -> None:
        """
        Set the seed of the generator.

        Parameters
        ----------
        seed : int
        """
        raise NotImplementedError

    def generate_fBm(self, H: float, size: int) -> np.ndarray:
        """
        Generate time series of fBm, with spacing 1,
        and the the first element must be 0.

        Parameters
        ----------
        H: float
            Hurst parameter. Should be in range `(0, 1)`.

        size: int
            Size of time series to generate. Should be larger than 1.

        Returns
        -------
        ts: `(len(size))` ndarray
            Time series of fBm, with spacing 1.

        """
        raise NotImplementedError

    def generate_fGn(self, H: float, size: int) -> np.ndarray:
        """
        Generate time series of fractional gaussian noise (fGn), with spacing 1.

        Parameters
        ----------
        H: float
            Hurst parameter. Should be in range `(0, 1)`.

        size: int
            Size of time series to generate. Should be larger than 0.

        Returns
        -------
        ts: `(len(size))` ndarray
            Time series of fBm, with spacing 1.
        """
        raise NotImplementedError

