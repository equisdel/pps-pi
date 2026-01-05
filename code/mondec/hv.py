import numpy as np
from pymoo.indicators.hv import HV
from initialization import MINS, MAXS

def normalize_fitness(obj_id, obj_value):
    n_obj_value = obj_value-MINS[obj_id]/MAXS[obj_id]-MINS[obj_id]
    #print("obj_id,min,max,value,normalized_value:  ",obj_id,MINS[obj_id],MAXS[obj_id],obj_value,n_obj_value)
    return (obj_value-MINS[obj_id])/(MAXS[obj_id]-MINS[obj_id])

def normalized(fitness):
    return [normalize_fitness(i,f) for i,f in enumerate(fitness)]

def calculate_hv(pareto_front):
    
    normalized_front = np.array([normalized(ind.fitness.values) for ind in pareto_front])

    weights = (-1.0, +1.0, -1.0, -1.0)
    front_min = normalized_front.copy()

    for i, w in enumerate(weights):
        if w > 0:            # SM es un objetivo originalmente maximizado
            front_min[:, i] = -front_min[:, i]

    ref_point = np.max(front_min, axis=0) * 1.1  
    hv = HV(ref_point=ref_point)
    hv_value = hv(front_min)

    return hv_value