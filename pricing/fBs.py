from .interface import ModelInterface
from fbm.sim import DaviesHarteFBmGenerator
import numpy as np
from scipy.stats import norm

N = norm.cdf

import matplotlib.pyplot as plt

class FractionalBlackScholes(ModelInterface):
    def __init__(self):
        self.p = {
            'h': 0.5,
            'mu': 0.05,
            'sigma': 0.15,
        }

    def simulate_dws(self, T:float, size:int, sim_num:int,
            generator=None, seed=None, verbose=True) -> np.ndarray:
        if generator is None:
            generator = DaviesHarteFBmGenerator()
        if seed is not None:
            generator.seed(seed)

        h = self.p['h']

        dws = np.ndarray((1, sim_num, size - 1))
        for i in range(sim_num):
            if verbose:
                print(f"fBS: Simulating {(i)/sim_num: .2%} dwh", end="\r")
            dws[0, i] = generator.generate_fGn(h, size-1, T)
        if verbose:
            print()

        return dws

    def _simulate_with_dws(self, S0:float, p:int, size:int, dt:np.float64,
            dws:np.ndarray) -> np.ndarray:
        mu = self.p['mu']
        sigma = self.p['sigma']
        h = self.p['h']

        dwh = dws[0]

        s_path = np.ndarray((p, size))
        s_path[:,0] = S = S0

        for t in range(size-1):
            S += S*mu*dt + S*sigma*dwh[:,t]
            if h != 0.5:
                S += 0.5 * S* sigma**2 * dt**(2*h)
            s_path[:,t+1] = S
        
        return s_path

def fbs_call(S, K, T, r, h, sigma):
    d1 = (np.log(S/K) + r*T + sigma**2/2*(T**(2*h))) / (sigma*np.sqrt(T**(2*h)))
    d2 = d1 - sigma * np.sqrt(T**(2*h))
    return S*N(d1) - K*np.exp(-r*T)*N(d2)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from .price_engine import MontelCarloOptionPricingEngine

    fBS = FractionalBlackScholes()
    r = 0.05
    params = {
        'h': 0.5,
        'mu': 0.05,
        'sigma': 0.15,
    }
    fBS.set_parameters(**params)

    T = 1
    size = 1001
    S0 = 300

    # Simulate a path of Heston
    path = fBS.simulate(S0, T, size, return_path=True)
    t = np.linspace(0, T, size)

    plt.plot(t, path)
    plt.show()

    # Compute european call option price by MonteCarlo
    strike = 300
    batch_sim_num = 10000
    pricer = MontelCarloOptionPricingEngine(fBS)

    price = pricer.european_call_option_price(S0, np.array([strike]), r=r,
        T=T, size=size, batch_sim_num=batch_sim_num, batch_num=10,
        **params)
    print(price)