import numpy as np

class ModelInterface:
    def __init__(self):
        self.p = dict()
    
    def set_parameters(self, model_args_dict:dict={}, **model_args) -> None:
        for k in model_args.keys():
            if k in self.p:
                self.p[k] = model_args[k]
        
        for k in model_args_dict.keys():
            if k in self.p:
                self.p[k] = model_args_dict[k]

    def simulate_dws(self, T:float, size:int, sim_num:int,
            generator=None, seed=None, verbose=True) -> np.ndarray:
        dt = T / (size-1)
        if seed is not None:
            np.random.seed(seed)
        if verbose:
            print(f"Simulating {sim_num} dws", end='\r')
        dw = np.random.standard_normal((sim_num, size-1)) * np.sqrt(dt)
        if verbose:
            print()

        return dw[None,:]
    
    def _simulate_with_dws(self, S0:float, p:int, size:int, dt:np.float64,
            dws:np.ndarray) -> np.ndarray:
        return np.zeros(1)

    def simulate_with_dws(self, S0:float, dt:np.float64, 
            dws:np.ndarray, return_paths=False) -> np.ndarray:
        p = dws.shape[1]
        size = dws.shape[2] + 1
        paths = self._simulate_with_dws(S0, p, size, dt, dws)
        if return_paths:
            return np.squeeze(paths)
        else:
            return np.squeeze(paths[:,-1:])

    def simulate(self, 
            S0:float, T:float, size:int, sim_num:int=1,
            generator=None, seed=None, return_path=False, 
            verbose=True) -> np.ndarray:
        dt = np.float64(T) / (size-1)
        dws = self.simulate_dws(T, size, sim_num, generator, seed, verbose=verbose)
        paths = self.simulate_with_dws(S0, dt, dws, return_path)
        return paths

        

