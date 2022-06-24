from .interface import ModelInterface
import numpy as np

class BlackScholes(ModelInterface):
    def __init__(self):
        self.p = {
            'mu': 0.05,
            'vol': 0.15
        }
    
    def _simulate_with_dws(self, S0:float, p:int, size:int, dt:float,
            dws:np.ndarray) -> np.ndarray:
        mu = self.p['mu']
        vol = self.p['vol']
        dw = dws[0]

        paths = np.ndarray((p, size))
        paths[:,0] = S = S0

        for t in range(size-1):
            S += S*mu*dt + S*vol*dw[:,t]
            paths[:,t+1] = S

        return paths

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from .price_engine import MontelCarloOptionPricingEngine

    bs = BlackScholes()
    bs.set_parameters(mu=0.03, vol=0.15)

    T = 1
    size = 1001
    dt = T / size
    S0 = 300

    # Simulate a path of Black Scholes
    # path = bs.simulate(S0, T, size, return_path=True)
    # t = np.linspace(0, T, size)

    # plt.plot(t, path)
    # plt.show()

    # Compute european call option price by MonteCarlo
    strike = 300

    batch_sim_num = 10000
    pricer = MontelCarloOptionPricingEngine(bs)

    r = 0.03
    vol = 0.15

    price = pricer.european_call_option_price(S0, np.array([strike]), r=r,
        T=T, size=size, batch_sim_num=batch_sim_num, batch_num=10,
        mu=r, vol=vol)
    print(price)








    
