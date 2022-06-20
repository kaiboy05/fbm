from .interface import ModelInterface
import numpy as np

class Heston(ModelInterface):
    def __init__(self):
        self.p = {
            'mu': 0.05,
            'corr': 0.02,

            'v0': 0.0225,
            'v_mr': 2,
            'v_mu': 0.0225,
            'v_vol': 0.25
        }

    def simulate_dws(self, T:float, size:int, sim_num:int,
            generator=None, seed=None, verbose=True) -> np.ndarray:
        dt = T / (size-1)
        if seed is not None:
            np.random.seed(seed)
        corr = self.p['corr']

        if verbose:
            print(f"Heston: Simulating {sim_num} dzs", end="\r")
        dws = np.random.standard_normal((2, sim_num, size-1)) * np.sqrt(dt)
        if verbose:
            print(f"Heston: Simulating {sim_num} dws", end='\r')
        dws[1] = corr * dws[0] + np.sqrt(1 - corr**2)*dws[1]
        if verbose:
            print()

        return dws

    def _simulate_with_dws(self, S0:float, p:int, size:int, dt:float,
            dws:np.ndarray) -> np.ndarray:
        mu = self.p['mu']
        v0 = self.p['v0']
        v_mr = self.p['v_mr']
        v_mu = self.p['v_mu']
        v_vol = self.p['v_vol']

        dw1 = dws[0]
        dw2 = dws[1]

        s_path = np.ndarray((p, size))
        v_path = np.ndarray((p, size))
        s_path[:,0] = S = S0
        v_path[:,0] = V = v0

        print(dws)

        for t in range(size-1):
            S += S*mu*dt + S*np.sqrt(V)*dw1[:,t]
            V += v_mr*(v_mu - V)*dt + v_vol*np.sqrt(V)*dw2[:,t]
            V = np.maximum(V, 0)

            s_path[:,t+1] = S
            v_path[:,t+1] = V
        
        for i in range(p):
            plt.plot(v_path[i])
            plt.show()

        return s_path

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from .price_engine import MontelCarloOptionPricingEngine

    heston = Heston()
    r = 0
    params = {
        'mu': r,
        'corr': -0.681,

        'v0': 0.392,
        'v_mr': 0.1,
        'v_mu': 0.03156,
        'v_vol': 0.331
    }
    heston.set_parameters(**params)

    T = 2
    size = 1001
    dt = T / size
    S0 = 100

    # Simulate a path of Heston
    path = heston.simulate(S0, T, size, return_path=True, seed=42)
    t = np.linspace(0, T, size)

    plt.plot(t, path)
    plt.show()

    # Compute european call option price by MonteCarlo
    strike = 100
    batch_sim_num = 10000
    pricer = MontelCarloOptionPricingEngine(heston)

    price = pricer.european_call_option_price(S0, np.array([strike]), r=r,
        T=T, size=size, batch_sim_num=batch_sim_num, batch_num=10,
        **params, seed=42)
    print(price)