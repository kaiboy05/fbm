from .interface import ModelInterface

import numpy as np

class MontelCarloOptionPricingEngine:
    def __init__(self, model:ModelInterface, generator=None):
        self.model:ModelInterface = model
        self.generator = generator
    
    def european_call_option_price(self, 
            S0:float, strike:float, r:float, T:float, 
            size:int, batch_sim_num=10000, batch_num=1, **model_args):

        self.model.set_parameters(model_args)

        call_price = 0

        for i in range(batch_num):
            print(f'Running batch {(i+1)}/{batch_num}')
            S = self.model.simulate(S0, T, size, sim_num=batch_sim_num, 
                    generator=self.generator)

            call_H = np.maximum(S - strike, 0)
            call_price_sample = np.exp(-r*T) * call_H
            call_price_sample = np.average(call_price_sample)

            call_price += call_price_sample
        
        call_price /= batch_num

        return call_price