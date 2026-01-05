from SALib.analyze import sobol
from .problem import PROBLEM

def sobol_analysis(Y):
    return sobol.analyze(
        PROBLEM,
        Y,
        calc_second_order=False
    )
