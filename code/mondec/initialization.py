import random
import numpy as np
import time
import sys
import json
from deap import creator, base

from mondec.config_ea import DEFAULT, POP_SIZE
from mondec.config_instance import N_CLASSES, METADATA, INSTANCE, modify_metadata, load_range_from_metadata
from mondec.representations import init_individual, evaluate

M = 10000000  # MonteCarlo samples
SEED = 42

best_pop_size = []
weights = (-1,1,-1,-1)

def progress_bar(i, total, width=30):
    progress = (i + 1) / total
    filled = int(width * progress)
    bar = "█" * filled + "-" * (width - filled)
    percent = int(progress * 100)
    sys.stdout.write(f"\r|{bar}| {percent}% ({i+1}/{total})")
    sys.stdout.flush()

def run():
    MINS, MAXS = run_montecarlo(M)
    return MINS, MAXS

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
   
    with open("montecarlo.txt", "a") as f:
        f.write(f"time:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
        f.write(f"\M (samples):   {m}")
        f.write(f"\nactual mins:  {actual_mins}")
        f.write(f"\nnew maxs:     {actual_maxs}")
        f.write(f"\nnew mins:     {new_mins}")
        f.write(f"\nnew maxs:     {new_maxs}\n\n")

    return new_mins, new_maxs


if __name__=="__main__":
    run()