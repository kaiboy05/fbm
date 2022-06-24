from turtle import color
from .interface import ModelInterface
from fbm.sim import DaviesHarteBiFBmGenerator
import numpy as np
from numpy.polynomial.legendre import leggauss
# from numpy.polynomial.laguerre import laggauss
from scipy.special import roots_genlaguerre as laggauss
from scipy.special import gamma


class RoughHeston(ModelInterface):
    def __init__(self):
        self.p = {
            'mu': 0.05,
            'corr': 0.02,
            'h2': 0.18,

            'v0': 0.0225,
            'v_mr': 2,
            'v_mu': 0.0225,
            'v_vol': 0.25
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
        h2 = self.p['h2']
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

        f_dt = dt**(h2+0.5) / gamma(h2 + 1.5)

        for t in range(size-1):
            S += S*mu*dt + S*np.sqrt(V)*dw1[:,t]
            V += v_mr*(v_mu - V)*f_dt + v_vol*np.sqrt(V)*dw2[:,t]
            V = np.maximum(V, 0)


            s_path[:,t+1] = S
            v_path[:,t+1] = V

        return s_path

class RoughHestonFast(ModelInterface):
    def __init__(self):
        self.p = {
            'mu': 0.05,
            'corr': 0.02,
            'h2': 0.4999,

            'v0': 0.0225,
            'v_mr': 2,
            'v_mu': 0.0225,
            'v_vol': 0.25,
            'tol': 0.000001
        }
        self.last_v_path:np.ndarray

    def simulate_dws(self, T:float, size:int, sim_num:int,
            generator=None, seed=None, verbose=True) -> np.ndarray:
        dt = np.float64(T) / (size-1)
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
    
    @staticmethod
    def _get_gauss_quadrature(start, end, deg):
        (s, w) = leggauss(deg)
        s = (s + 1)*(end - start)/2 + start
        w = w * (end - start) / 2

        return s, w
    
    @staticmethod
    def _construct_x_w(h, size):
        # No, Ns, Nl, M, M_prime, 
        # xs = list()
        # ws = list()
        # (x_no, w_no) = RoughHestonFast._get_gauss_quadrature(0, 2.**(-M), No)

        # xs.append(x_no)
        # ws.append(w_no*x_no**(-(h+0.5)))
        # for i in range(-M, 0):
        #     (s, w) = RoughHestonFast._get_gauss_quadrature(2.**i, 2.**(i+1), Ns)
        #     xs.append(s)
        #     ws.append(s**(-(h+0.5)) * w)
        # for i in range(0, M_prime+1):
        #     (s, w) = RoughHestonFast._get_gauss_quadrature(2.**i, 2.**(i+1), Nl)
        #     xs.append(s)
        #     ws.append(s**(-(h+0.5)) * w)
        
        # xs = np.concatenate(xs).ravel()
        # ws = np.concatenate(ws).ravel() / math.gamma(0.5 - h)
        # return xs, ws
        x, w = laggauss(size, -0.5-h)
        w = w * np.exp(x)

        return x, w

    @staticmethod
    def _construct_x_w_legendre(h, No, Ns, Nl, M, M_prime): 
        xs = list()
        ws = list()
        (x_no, w_no) = RoughHestonFast._get_gauss_quadrature(0, 2.**(-M), No)

        xs.append(x_no)
        ws.append(w_no)
        for i in range(-M, 0):
            (s, w) = RoughHestonFast._get_gauss_quadrature(2.**i, 2.**(i+1), Ns)
            xs.append(s)
            ws.append(s**(-(h+0.5)) * w)
        for i in range(0, M_prime+1):
            (s, w) = RoughHestonFast._get_gauss_quadrature(2.**i, 2.**(i+1), Nl)
            xs.append(s)
            ws.append(s**(-(h+0.5)) * w)
        
        xs = np.concatenate(xs).ravel()
        ws = np.concatenate(ws).ravel()
        return xs, ws


    def _simulate_with_dws(self, S0:float, p:int, size:int, dt:np.float64,
            dws:np.ndarray) -> np.ndarray:
        mu = self.p['mu']
        h2 = self.p['h2']
        v0 = self.p['v0']
        v_mr = self.p['v_mr']
        v_mu = self.p['v_mu']
        v_vol = self.p['v_vol']
        tol = self.p['tol']

        dw1 = dws[0]
        dw2 = dws[1]

        T = dt * size
        M = np.floor(np.log2(T)).astype(int)
        M_prime = np.floor(np.log2(-np.log2(tol) - np.log2(dt))).astype(int)
        No = Ns  = np.floor(-np.log2(tol)).astype(int)
        Nl = np.floor(-np.log2(tol) - np.log2(dt)).astype(int)
        N_prime = No + M*Ns + (M_prime + 1)*Nl

        # Laguerre
        N_prime = np.log(size).astype(int)+20
        x, w = self._construct_x_w(h2, N_prime)
        # x, w = self._construct_x_w_legendre(h2, No, Ns, Nl, M, M_prime)

        s_path = np.ndarray((p, size), dtype=np.float64)
        v_path = np.ndarray((p, size), dtype=np.float64)
        Hs = np.zeros((p, N_prime), dtype=np.float64)
        Js = np.zeros((p, N_prime), dtype=np.float64)

        def f(x_t):
            return v_mr * (v_mu - x_t)
        def g(x_t):
            return v_vol * np.sqrt(x_t)

        S = s_path[:,0] = S0
        v_path[:,0] = v0
        V = v_path[:,0]

        e_x_dt = np.exp(-x*dt)
        for t in range(size-1):
            dv = dt**(h2 + 0.5)*f(V) / gamma(h2 + 0.5)
            dv += (Hs+Js).dot(w * e_x_dt) / gamma(0.5 - h2)
            dv += dt**(h2 - 0.5)*g(V) * dw2[:,t]
            dv /= gamma(h2 + 0.5)

            Hs = np.multiply.outer(f(V), (1 - e_x_dt)/x) + \
                    Hs * e_x_dt
            Js = np.multiply.outer(g(V), e_x_dt) * dw2[:,t][:, None] + \
                    Js * e_x_dt

            S += mu*S*dt + np.sqrt(V)*S*dw1[:,t]
            V = v0 + dv
            V = np.maximum(V, 0)
            v_path[:,t+1] = V
            s_path[:,t+1] = S
        
        self.last_v_path = v_path[-1]

        return s_path

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from .price_engine import MontelCarloOptionPricingEngine

    heston = RoughHestonFast()
    r = 0
    params = {
        'mu': r,
        'corr': -0.681,
        'h2': 0.12,

        'v0': 0.0392,
        'v_mr': 0.1,
        'v_mu': 0.3156,
        'v_vol': 0.0331
    }
    heston.set_parameters(**params)

    T = 1
    size = 250
    dt = T / size
    S0 = 100

    # Simulate a path of Heston
    path = heston.simulate(S0, T, size, sim_num=1, return_path=True)
    # t = np.linspace(0, T, size)
    # for p in path:
    #     plt.plot(t, p)
    #     plt.show()

    # Compute european call option price by MonteCarlo
    strike = 100
    batch_sim_num = 10000
    pricer = MontelCarloOptionPricingEngine(heston)

    price = pricer.european_call_option_price(S0, np.array([strike]), r=r,
        T=T, size=size, batch_sim_num=batch_sim_num, batch_num=10,
        **params)
    print(price)