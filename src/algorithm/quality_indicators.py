import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from algorithm.config import load_range_from_metadata

def hv(pareto_front):

    MINS, MAXS = load_range_from_metadata()

    front = np.array([ind.fitness.values for ind in pareto_front])

    front[:, 1] *= -1           # negación de SM en los individuos del frente

    ref_point = np.array([
        MAXS[0],
        -MINS[1],
        MAXS[2],
        MAXS[3]
    ]) * 1.05

    hv = HV(ref_point=ref_point)
    hv_value = hv(front)

    return hv_value


