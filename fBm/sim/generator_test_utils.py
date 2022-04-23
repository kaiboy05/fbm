from .fbm_generator import FBmGeneratorInterface
from numpy import arange
from fBm.testing.ChiSquareTest import fBm_chi_square_test
import matplotlib.pyplot as plt

def fBm_generator_chi_square_test(
        fBm_generator: FBmGeneratorInterface, 
        size:int=100,
        H:float=0.5,
        alpha:float=0.95,
        sim_num:int=100,
        seed:int=42,
        plot_graph: bool=False
    ) -> None:
    """
    Test function for testing implementation of generator.
    Use chi square test to test.
    """

    fBm_generator.seed(seed)

    print(f"""
        Generating {sim_num} simulations of fBm with Hurst {H}, size {size}.
    """)
    
    fBm_ts = [fBm_generator.generate_fBm(H, size) for _ in range(sim_num)]
    
    accept_count = 0
    for ts in fBm_ts:
        plt.plot(arange(size), ts)
        a, s, c = fBm_chi_square_test(ts, H, alpha)

        accept_count = accept_count + (1 if a else 0)

    print(f"""
        {accept_count}/{sim_num} chi square tests have been passed.
    """)
    
    if plot_graph:
        plt.show()