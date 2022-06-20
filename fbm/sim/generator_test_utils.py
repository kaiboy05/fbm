from .fbm_generator import FBmGeneratorInterface
from .fbm_generator import BiFBmGeneratorInterface
from numpy import arange
from numpy import zeros
from fbm.testing.chi_square import fBm_chi_square_test
from fbm.testing.lrt import fBm_lrt_test
from fbm.testing.chi_square import bfBm_chi_square_test
from fbm import utils
import matplotlib.pyplot as plt
import numpy as np

def fBm_generator_chi_square_test(
        fBm_generator: FBmGeneratorInterface, 
        size:int=100,
        H:float=0.5,
        alpha:float=0.95,
        sim_num:int=100,
        seed:int=-1,
        plot_graph: bool=False
    ) -> None:
    """
    Test function for testing implementation of generator.
    Use chi square test to test.
    """
    if seed != -1:
        fBm_generator.seed(seed)

    print(
        f"Generating {sim_num} simulations of fBm with Hurst {H}, size {size}."
    )
    
    fBm_ts = [zeros(1) for _ in range(sim_num)]
    for ind in range(sim_num):
        if ind + 1 == sim_num:
            print(f'Generating {ind+1}/{sim_num}')
        else:
            print(f'Generating {ind+1}/{sim_num}', end='\r')
        fBm_ts[ind] = fBm_generator.generate_norm_fBm(H, size)
    
    accept_count = 0
    for ind, ts in enumerate(fBm_ts):
        if plot_graph:
            plt.plot(arange(size), ts)
        if ind + 1 == sim_num:
            print(f'Testing {ind+1}/{sim_num}')
        else:
            print(f'Testing {ind+1}/{sim_num}', end='\r')
        a, s, c = fBm_chi_square_test(ts, H, alpha)

        accept_count = accept_count + (1 if a else 0)

    print(
        f"{accept_count}/{sim_num} chi square tests have been passed."
    )
    
    print()
    if plot_graph:
        plt.show()

def fBm_generator_lrt_test(
        fBm_generator: FBmGeneratorInterface, 
        size:int=100,
        H:float=0.5,
        alpha:float=0.95,
        sim_num:int=3000,
        seed:int=-1,
        plot_graph: bool=False,
        quiet=False
    ) -> bool:
    """
    Test function for testing implementation of generator.
    Use LRT test to test.
    """

    if seed != -1:
        fBm_generator.seed(seed)

    if not quiet:
        print(
            f"Generating {sim_num} simulations of fBm with Hurst {H}, size {size}."
        )
    
    fBm_ts = np.ndarray((sim_num, size))
    for ind in range(sim_num):
        if not quiet:
            if ind + 1 == sim_num:
                print(f'Generating {ind+1}/{sim_num}')
            else:
                print(f'Generating {ind+1}/{sim_num}', end='\r')
        ts = fBm_generator.generate_norm_fBm(H, size)
        if plot_graph:
            plt.plot(arange(size), ts)
        fBm_ts[ind] = ts
    
    a, s, c = fBm_lrt_test(fBm_ts, H, alpha)

    if not quiet:
        print(
            f"{'Passed' if a else 'Failed'} LRT with statistics {s}, critical value {c}."
        )
        print()
    
    if plot_graph:
        plt.show()
    
    return a

def bfBm_generator_chi_square_test(
        fBm_generator: BiFBmGeneratorInterface, 
        size:int=100,
        H1:float=0.5,
        H2:float=0.5,
        rho:float=0,
        alpha:float=0.95,
        sim_num:int=100,
        seed:int=-1,
        plot_graph:bool=False,
        separate_compare:bool=False
    ) -> None:
    """
    Test function for testing implementation of generator.
    Use chi square test to test.
    """

    max_rho = utils.bfBm_max_rho(H1, H2) 
    if(rho > max_rho):
        print(f"bfBm with H1({H1}), H2({H2}) has max rho({max_rho})")
        return

    if seed != -1:
        fBm_generator.seed(seed)

    print(
        f"Generating {sim_num} simulations of bfBm with H1({H1}), H2({H2}), rho({rho}), size {size}."
    )
    
    fBm_ts = [zeros(1) for _ in range(sim_num)]
    for ind in range(sim_num):
        if ind + 1 == sim_num:
            print(f'Generating {ind+1}/{sim_num}')
        else:
            print(f'Generating {ind+1}/{sim_num}', end='\r')
        fBm_ts[ind] = fBm_generator.generate_norm_bifBm(H1, H2, rho, size)
    
    accept_count = 0
    ac1 = 0
    ac2 = 0
    for ind, ts in enumerate(fBm_ts):
        plt.plot(arange(size), ts[0])
        plt.plot(arange(size), ts[1])
        if ind + 1 == sim_num:
            print(f'Testing {ind+1}/{sim_num}')
        else:
            print(f'Testing {ind+1}/{sim_num}', end='\r')
        a, s, c = bfBm_chi_square_test(ts, H1, H2, rho, alpha)
        accept_count = accept_count + (1 if a else 0)
        if separate_compare:
            a1, _, _ = fBm_chi_square_test(ts[0], H1, alpha)
            a2, _, _ = fBm_chi_square_test(ts[0], H2, alpha)
            ac1 = ac1 + (1 if a1 else 0)
            ac2 = ac2 + (1 if a2 else 0)


    print(
        f"{accept_count}/{sim_num} chi square tests have been passed."
    )
    if separate_compare:
        print(
            f"{ac1}/{sim_num} ts[0] passed chi square tests of fBm(H1)."
        )
        print(
            f"{ac2}/{sim_num} ts[1] passed chi square tests of fBm(H2)."
        )
    
    print()
    if plot_graph:
        plt.show()