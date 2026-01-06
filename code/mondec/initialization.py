import random
import numpy as np
import time
import sys
import json
from deap import creator, base

from mondec.config_ea import DEFAULT, POP_SIZE
from mondec.config_instance import N_CLASSES, METADATA, INSTANCE, modify_metadata, load_range_from_metadata
from mondec.representations import init_individual, evaluate

M = 100000  # MonteCarlo samples
SEED = 42

best_pop_size = []
weights = (-1,1,-1,-1)

def light_evaluate(fitness,weights):
    return np.dot(np.array(fitness),np.array(weights))

def progress_bar(i, total, width=30):
    progress = (i + 1) / total
    filled = int(width * progress)
    bar = "█" * filled + "-" * (width - filled)
    percent = int(progress * 100)
    sys.stdout.write(f"\r|{bar}| {percent}% ({i+1}/{total})")
    sys.stdout.flush()

def run(creator):

    BEST = []
    if DEFAULT["preheat_with_MC"]:
        M = DEFAULT["MC_samples"]
        MINS, MAXS, raw_best = run_montecarlo(M)
        for ind, obj, _ in raw_best:
            deap_ind = creator.Individual(ind.blocks)
            deap_ind.fitness.values = obj
            BEST.append(deap_ind) 

    else:
        MINS, MAXS = load_range_from_metadata()
        
    return MINS, MAXS, BEST


def run_montecarlo(m: int = M):

    actual_mins, actual_maxs = load_range_from_metadata()
    
    print(f"\nRunning Monte Carlo with M={m} samples for instance {INSTANCE} (N={N_CLASSES})\n")
    
    # Generate random individuals
    individuals = [init_individual(N_CLASSES, SEED) for _ in range(m)]
    
    objs = []
    t0 = time.time()
    for i, ind in enumerate(individuals):
        objs.append(evaluate(ind))
        if i % max(1, m // 100) == 0:
            progress_bar(i, m)
    tf = time.time()
    print(f"\nMonte Carlo elapsed time: {tf - t0:.2f} seconds")
    
    objs = np.array(objs)
    new_mins = np.min(objs, axis=0)
    new_maxs = np.max(objs, axis=0)
   
    update_needed = False
    for idx, (am, nm, aM, nM) in enumerate(zip(actual_mins, new_mins, actual_maxs, new_maxs)):
        print(f"Obj {idx}: actual_min={am}, new_min={nm}, actual_max={aM}, new_max={nM}")
        if nm < am or nM > aM:
            update_needed = True

    if update_needed:
        actual_mins = np.minimum(actual_mins, new_mins)
        actual_maxs = np.maximum(actual_maxs, new_maxs)
        print("\nUpdated bounds detected. Writing new MINS/MAXS to metadata.")
        print("New mins:", actual_mins)
        print("New maxs:", actual_maxs)
        
        objective_names = ["NED", "SM", "ICP", "IN"]  # Keep consistent with metadata
        new_range = {
            "RANGE": {
                "MIN": {name: val for name, val in zip(objective_names, actual_mins)},
                "MAX": {name: val for name, val in zip(objective_names, actual_maxs)}
            }
        }
        modify_metadata(new_range, "w")
    else:
        print("\nNo update needed. MINS/MAXS remain the same.")
    
    # update en metadata solo si amplia el rango en algún objetivo
    norm_objs = (objs - actual_mins) / (actual_maxs - actual_mins + 1e-12)

    light_fitness = [light_evaluate(np.array(i),np.array(weights)) for i in norm_objs]

    #print(light_fitness)

    individuals_sorted = sorted(
        zip(individuals, objs, light_fitness),
        key= lambda x: x[2],
        reverse=True
    )

    print("PEOR: ",individuals_sorted[-1:])
    best_pop_size = individuals_sorted[0:POP_SIZE]

    from mondec.representations import Individual as IndividualClass
    
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, +1.0, -1.0, -1.0))
    creator.create("Individual", IndividualClass, fitness=creator.FitnessMulti)


    return actual_mins, actual_maxs, best_pop_size

#MINS, MAXS, INIT_POP = run(creator)