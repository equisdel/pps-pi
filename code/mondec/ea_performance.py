import numpy as np
from pymoo.indicators.hv import HV
from mondec.config_instance import load_range_from_metadata

def normalize_fitness(obj_id, obj_value, MINS, MAXS):
    return (obj_value-MINS[obj_id])/(MAXS[obj_id]-MINS[obj_id])

def normalized(fitness,MINS,MAXS):
    return [normalize_fitness(i,f,MINS,MAXS) for i,f in enumerate(fitness)]

def hv(pareto_front):
    MINS, MAXS = load_range_from_metadata()
    normalized_front = np.array([normalized(ind.fitness.values,MINS,MAXS) for ind in pareto_front])

    weights = (-1.0, +1.0, -1.0, -1.0)
    front_min = normalized_front.copy()

    for i, w in enumerate(weights):
        if w > 0:            # SM es un objetivo originalmente maximizado
            front_min[:, i] = -front_min[:, i]

    ref_point = np.max(front_min, axis=0) * 1.1  
    hv = HV(ref_point=ref_point)
    hv_value = hv(front_min)

    return hv_value