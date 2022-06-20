from matplotlib.pyplot import title

from fbm.sim.davies_harte import DaviesHarteBiFBmGenerator, DaviesHarteFBmGenerator

from .interface import ModelInterface

import numpy as np

class UncorrelatedRFSV(ModelInterface):
    def __init__(self):
        self.p = {
            'mu': 0.05,
            'h2': 0.1,

            'x0': np.log(0.15),
            'x_mr': 2,
            'x_mu': np.log(0.15),
            'x_vol': 0.25
        }
    
    def simulate_dws(self, T:float, size:int, sim_num:int,
            generator=None, seed=None, verbose=True) -> np.ndarray:
        dt = T / (size-1)
        if generator is None:
            generator = DaviesHarteFBmGenerator()
        if seed is not None:
            generator.seed(seed)

        h2 = self.p['h2']
        dws = np.ndarray((2, sim_num, size-1))
        if verbose:
            print(f"UncorrRFSV: Simulating {sim_num} dz", end="\r")
        dws[0] = np.random.standard_normal((sim_num, size-1)) * np.sqrt(dt)
        for i in range(sim_num):
            if verbose:
                print(f"UncorrRFSV: Simulating {(i)/sim_num: .2%} dwh", end="\r")
            dws[1, i] = generator.generate_fGn(H=h2, size=size-1, T=T)
        if verbose:
            print()

        return dws

    def _simulate_with_dws(self, S0:float, p:int, size:int, dt:float,
            dws:np.ndarray) -> np.ndarray:
        mu = self.p['mu']
        x0 = self.p['x0']
        x_mr = self.p['x_mr']
        x_mu = self.p['x_mu']
        x_vol = self.p['x_vol']

        dz = dws[0]
        dwh = dws[1]

        s_path = np.ndarray((p, size))
        x_path = np.ndarray((p, size))
        s_path[:,0] = S = S0
        x_path[:,0] = X = x0

        for t in range(size-1):
            S += S*mu*dt + S*np.exp(X)*dz[:,t]
            X += x_mr*(x_mu - X)*dt + x_vol*dwh[:,t]

            s_path[:,t+1] = S
            x_path[:,t+1] = X

        return s_path

class RFSV(ModelInterface):
    def __init__(self):
        self.p = {
            'mu': 0.05,
            'h2': 0.1,
            'corr': 0.02,

            'x0': np.log(0.15),
            'x_mr': 2,
            'x_mu': np.log(0.15),
            'x_vol': 0.25
        }
    
    def simulate_dws(self, T:float, size:int, sim_num:int,
            generator=None, seed=None, verbose=True) -> np.ndarray:
        dt = T / (size-1)
        if generator is None:
            generator = DaviesHarteBiFBmGenerator()
        if seed is not None:
            generator.seed(seed)

        h2 = self.p['h2']
        corr = self.p['corr']
        dws = np.ndarray((2, sim_num, size-1))
        if verbose:
            print(f"RFSV: Simulating {sim_num} dz", end="\r")
        for i in range(sim_num):
            if verbose:
                print(f"RFSV: Simulating {(i)/sim_num: .2%} dwh", end="\r")
            dws[:,i] = generator.generate_bifGn(H1=0.5, H2=h2, rho=corr, size=size-1, T=T)
        if verbose:
            print()

        return dws

    def _simulate_with_dws(self, S0:float, p:int, size:int, dt:float,
            dws:np.ndarray) -> np.ndarray:
        mu = self.p['mu']
        x0 = self.p['x0']
        x_mr = self.p['x_mr']
        x_mu = self.p['x_mu']
        x_vol = self.p['x_vol']

        dz = dws[0]
        dwh = dws[1]

        s_path = np.ndarray((p, size))
        x_path = np.ndarray((p, size))
        s_path[:,0] = S = S0
        x_path[:,0] = X = x0

        for t in range(size-1):
            S += S*mu*dt + S*np.exp(X)*dz[:,t]
            X += x_mr*(x_mu - X)*dt + x_vol*dwh[:,t]

            s_path[:,t+1] = S
            x_path[:,t+1] = X

        return s_path

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from .price_engine import MontelCarloOptionPricingEngine

    # rfsv = UncorrelatedRFSV()
    # r = 0.05
    # params = {
    #     'mu': r,
    #     'h2': 0.1,

    #     'x0': np.log(0.15),
    #     'x_mr': 2,
    #     'x_mu': np.log(0.15),
    #     'x_vol': 0.25
    # }
    # rfsv.set_parameters(**params)

    # T = 1
    # size = 1001
    # dt = T / size
    # S0 = 300

    # Simulate a path of Heston
    # path = rfsv.simulate(S0, T, size, return_path=True)
    # t = np.linspace(0, T, size)

    # plt.plot(t, path)
    # plt.show()

    # Compute european call option price by MonteCarlo

    # strike = 300
    # batch_sim_num = 10000
    # pricer = MontelCarloOptionPricingEngine(rfsv)

    # price = pricer.european_call_option_price(S0, strike, r=r,
    #     T=T, size=size, batch_sim_num=batch_sim_num, batch_num=10,
    #     **params)
    # print(price)

    rfsv = RFSV()
    r = 0.05
    params = {
        'mu': r,
        'h2': 0.1,
        'corr': 0.02,

        'x0': np.log(0.15),
        'x_mr': 2,
        'x_mu': np.log(0.15),
        'x_vol': 0.25
    }
    rfsv.set_parameters(**params)

    T = 1
    size = 1001
    dt = T / size
    S0 = 300

    # Simulate a path of Heston
    paths = rfsv.simulate(S0, T, size, sim_num=10, return_path=True)
    t = np.linspace(0, T, size)

    for p in paths:
        plt.plot(t, p)
    plt.show()

    # Compute european call option price by MonteCarlo

    strike = 300
    batch_sim_num = 10000
    pricer = MontelCarloOptionPricingEngine(rfsv)

    price = pricer.european_call_option_price(S0, np.array([strike]), r=r,
        T=T, size=size, batch_sim_num=batch_sim_num, batch_num=10,
        **params)
    print(price)