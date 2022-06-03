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
        np.random.seed(s)

    def generate_norm_fBm(self, H: float, size: int) -> np.ndarray:
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
        assert size > 1
        fGn = self.generate_norm_fGn(H, size - 1)
        ts = np.cumsum(np.insert(fGn, 0, 0))
        return ts

    def generate_norm_fGn(self, H: float, size: int) -> np.ndarray:
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
        assert size > 1
        fGn = self.generate_norm_fBm(H, size + 1)
        ts = np.diff(fGn)
        return ts
    
    def generate_fBm(self, H: float, size: int, T:float=0) -> np.ndarray:
        """
        Generate time series of fBm in interval [0, T], with spacing T/size,
        and the the first element must be 0.

        Parameters
        ----------
        H: float
            Hurst parameter. Should be in range `(0, 1)`.

        size: int
            Size of time series to generate. Should be larger than 1.
        
        T: float
            T in the interval. Should be larger than 0.

        Returns
        -------
        ts: `(len(size))` ndarray
            Time series of fBm, with spacing T/size.

        """
        if T <= 0:
            T = size
        spacing = T / size
        return self.generate_norm_fBm(H, size) * spacing**H

    def generate_fGn(self, H: float, size: int, T:float=0) -> np.ndarray:
        """
        Generate time series of fractional gaussian noise (fGn) in 
        interval [0,T], with spacing T/size.

        Parameters
        ----------
        H: float
            Hurst parameter. Should be in range `(0, 1)`.

        size: int
            Size of time series to generate. Should be larger than 0.
        
        T: float
            T in the interval. Should be larger than 0.

        Returns
        -------
        ts: `(len(size))` ndarray
            Time series of fBm, with spacing T/size.
        """
        if T <= 0:
            T = size
        spacing = T / size
        return self.generate_norm_fGn(H, size) * spacing**H

class BiFBmGeneratorInterface:
    """
    Generator Interface for generating bivariate fractional brownian motion (bfBm).
    """
    def seed(self, s: int) -> None:
        """
        Set the seed of the generator.

        Parameters
        ----------
        seed : int
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError


