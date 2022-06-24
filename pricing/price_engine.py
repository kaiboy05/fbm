from .interface import ModelInterface

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
from scipy import optimize

from py_vollib.black_scholes.implied_volatility import implied_volatility

N_prime = norm.pdf
N = norm.cdf

def blackscholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S*N(d1) - K*np.exp(-r*T)*N(d2)

def blackscholes_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (sigma**2/2)*T) / (sigma*np.sqrt(T))
    vega = S*N_prime(d1)*np.sqrt(T)
    return vega

def find_implied_vol(S, K, T, r, P):
    # f = lambda s: blackscholes_call(S, K, T, r, s) - P
    # v = lambda s: blackscholes_vega(S, K, T, r, s)

    # try:
    #     result = optimize.newton(f, 0.2, fprime=v)
    # except:
    #     result = np.nan

    # return result
    try:
        result = implied_volatility(P, S, K, T, r, 'c')
    except:
        result = 100

    return result


def find_implied_vol_curve(S, T, r, Ks, Ps):
    result = map(lambda k,p: find_implied_vol(S,k,T,r,p), Ks, Ps)
    return np.fromiter(result, dtype=np.float64)


class MontelCarloOptionPricingEngine:
    def __init__(self, model:ModelInterface, generator=None):
        self.model:ModelInterface = model
        self.generator = generator

    def european_call_find_implied_vols(self, 
            S0:float, Ks:np.ndarray, r:float, T:float, 
            size:int, batch_sim_num=10000, batch_num=1, verbose=False,
            **model_args):

        prices = self.european_call_option_price(S0, Ks, r, T, size, 
            batch_sim_num, batch_num, verbose, **model_args
        )
        curve = find_implied_vol_curve(S0, T, r, Ks, prices)

        return curve
    
    def european_call_option_price(self, 
            S0:float, Ks:np.ndarray, r:float, T:float, 
            size:int, batch_sim_num=10000, batch_num=1, verbose=False, quiet=False,
            **model_args):

        self.model.set_parameters(model_args)

        prices = np.zeros(Ks.shape)

        if not quiet:
            print(f'Finsihed {0: .2%}', end="\r")
        for i in range(batch_num):
            if verbose and not quiet:
                print(f'Running batch {(i+1)}/{batch_num}')
            S = self.model.simulate(S0, T, size, sim_num=batch_sim_num, 
                    generator=self.generator, verbose=verbose)

            call_H = np.maximum(np.subtract.outer(S, Ks), 0)  # type: ignore
            call_price_sample = np.exp(-r*T) * call_H
            call_price_sample = np.average(call_price_sample, axis=0)
            # print(call_price_sample)

            prices += call_price_sample
            if not verbose and not quiet:
                print(f'Finsihed {(i+1)/batch_num: .2%}', end="\r")
        if not verbose and not quiet:
            print()
        
        prices /= batch_num

        return np.squeeze(prices)
    
    def european_call_option_price_sim(self, 
            S0:float, Ks:np.ndarray, r:float, T:float, 
            size:int, batch_sim_num=10000, batch_num=1, verbose=False, quiet=False,
            **model_args):

        self.model.set_parameters(model_args)

        prices = np.zeros((batch_num*batch_sim_num, Ks.shape[0]))
        count = 0
        if not quiet:
            print(f'Finsihed {0: .2%}', end="\r")
        for i in range(batch_num):
            if verbose and not quiet:
                print(f'Running batch {(i+1)}/{batch_num}')
            S = self.model.simulate(S0, T, size, sim_num=batch_sim_num, 
                    generator=self.generator, verbose=verbose)

            call_H = np.maximum(np.subtract.outer(S, Ks), 0)  # type: ignore
            call_price_sample = np.exp(-r*T) * call_H
            prices[count*batch_sim_num:(count + 1)*batch_sim_num] = call_price_sample
            count += 1

            if not verbose and not quiet:
                print(f'Finsihed {(i+1)/batch_num: .2%}', end="\r")
        if not verbose and not quiet:
            print()

        return prices


if __name__ == '__main__':
    S = 300
    K = 250
    T = 1
    r = 0.03
    vol = 0.15

    V_market = blackscholes_call(S, K, T, r, vol)
    print ('Market price = %.2f' % V_market)
    vol = find_implied_vol(S, K, T, r, V_market)

    print ('Implied vol: %.2f%%' % (vol * 100))
    print ('Model price = %.2f' % blackscholes_call(S, K, T, r, vol))