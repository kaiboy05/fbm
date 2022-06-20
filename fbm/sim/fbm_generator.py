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
        return self.generate_norm_fGn(H, size) * (spacing**H)

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
        np.random.seed(s)

    def generate_norm_bifBm(self, H1: float, H2: float, rho:float, size: int) -> np.ndarray:
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

        fGns = self.generate_norm_bifGn(H1, H2, rho, size - 1)
        ts = np.cumsum(np.insert(fGns, 0, 0, axis=1), axis=1)

        return ts

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
        assert size > 1
        fGn = self.generate_norm_bifBm(H1, H2, rho, size + 1)
        ts = np.diff(fGn, axis=1)
        return ts
    
    def generate_bifBm(self, H1: float, H2: float, rho:float, size: int, T:float=0) -> np.ndarray:
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
        if T <= 0:
            T = size
        spacing = (T / size)**np.array([H1, H2])
        return self.generate_norm_bifBm(H1, H2, rho, size) * spacing[:,None]

    def generate_bifGn(self, H1: float, H2: float, rho:float, size: int, T:float=0) -> np.ndarray:
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
        if T <= 0:
            T = size
        spacing = (T / size)**np.array([H1, H2])
        return self.generate_norm_bifGn(H1, H2, rho, size) * spacing[:,None]

class MFBmGeneratorInterface:
    """
    Generator Interface for generating multivaraite fractional brownian motion (bfBm).
    """
    def seed(self, s: int) -> None:
        """
        Set the seed of the generator.

        Parameters
        ----------
        seed : int
        """
        np.random.seed(s)

    def generate_norm_mfBm(self, Hs: np.ndarray, rho:np.ndarray, size: int) -> np.ndarray:
        """
        Generate time series of multivaraite fBm, with spacing 1,
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
            Time series of multivaraite fBm, with spacing 1.

        """
        assert size > 1

        fGns = self.generate_norm_mfGn(Hs, rho, size - 1)
        ts = np.cumsum(np.insert(fGns, 0, 0, axis=1), axis=1)

        return ts

    def generate_norm_mfGn(self, Hs: np.ndarray, rho:np.ndarray, size: int) -> np.ndarray:
        """
        Generate time series of fGns of multivaraite fBm, with spacing 1,
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
            Time series of fGns of multivaraite fBm, with spacing 1.

        """
        assert size > 1
        fGn = self.generate_norm_mfBm(Hs, rho, size + 1)
        ts = np.diff(fGn, axis=1)
        return ts
    
    def generate_mfBm(self, Hs: np.ndarray, rho:np.ndarray, size: int, T:float=0) -> np.ndarray:
        """
        Generate time series of multivaraite fBm, with spacing 1,
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
            Time series of multivaraite fBm, with spacing 1.

        """
        if T <= 0:
            T = size
        spacing = (T / size)**Hs
        return self.generate_norm_mfBm(Hs, rho, size) * spacing[:,None]

    def generate_mfGn(self, Hs: np.ndarray, rho:np.ndarray, size: int, T:float=0) -> np.ndarray:
        """
        Generate time series of fGns of multivaraite fBm, with spacing 1,
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
            Time series of fGns of multivaraite fBm, with spacing 1.

        """
        if T <= 0:
            T = size
        spacing = (T / size)**Hs
        return self.generate_norm_mfGn(Hs, rho, size) * spacing[:,None]


