import random
import numpy as np
import time
import sys

from config import DEFAULT, N_OBJECTIVES
from instance import N_CLASSES, METADATA, INSTANCE

if DEFAULT["new_representation"]:
    from representations.canonical import init_individual, individual_to_microservices, evaluate
else:
    from representations.original import init_individual, individual_to_microservices, evaluate

M = 1000000  # MonteCarlo samples
SEED = 42

MINS = [0., 0., 0., 0.]  
MAXS = [1., 0.057, 1., 10.]  

def progress_bar(i, total, width=30):
    progress = (i + 1) / total
    filled = int(width * progress)
    bar = "█" * filled + "-" * (width - filled)
    percent = int(progress * 100)
    sys.stdout.write(f"\r|{bar}| {percent}% ({i+1}/{total})")
    sys.stdout.flush()


def run_montecarlo(m: int = M):
    random.seed(SEED) 

    # Generate random individuals
    individuals = [init_individual(N_CLASSES) for _ in range(m)]
    objs = []

    t0 = time.time()
    for i, ind in enumerate(individuals):
        objs.append(evaluate(ind))
        if i % max(1, m // 100) == 0:
            progress_bar(i, m)
    tf = time.time()

    print()
    objs = np.array(objs)  # shape: (M, N_OBJECTIVES)
    new_mins = np.min(objs, axis=0)
    new_maxs = np.max(objs, axis=0)

    return new_mins, new_maxs, tf - t0


if __name__ == "__main__":
    import json

    # Load old metadata
    with open(METADATA, "r") as f:
        metadata = json.load(f)

    order = ["NED", "SM", "ICP", "IN"]
    old_mins = [metadata["RANGE"]["MIN"][k] for k in order]
    old_maxs = [metadata["RANGE"]["MAX"][k] for k in order]

    print("Old mins:", old_mins)
    print("Old maxs:", old_maxs)

    m = 10000  # for testing
    print(f"\nRunning Monte Carlo with M={m} samples for instance {INSTANCE} (N={N_CLASSES})\n")

    new_mins, new_maxs, elapsed = run_montecarlo(m)

    # Update global MINS and MAXS
    update_needed = False
    for i in range(N_OBJECTIVES):
        if new_mins[i] < old_mins[i]:
            old_mins[i] = new_mins[i]
            update_needed = True
        if new_maxs[i] > old_maxs[i]:
            old_maxs[i] = new_maxs[i]
            update_needed = True

    if update_needed:
        print("Updated bounds needed. MINS/MAXS changed.")
    else:
        print("No update needed. MINS/MAXS remain the same.")

    print("New mins:", old_mins)
    print("New maxs:", old_maxs)
    print("Monte Carlo elapsed time:", elapsed, "seconds")

    # Now update your global MINS and MAXS
    MINS[:] = old_mins
    MAXS[:] = old_maxs
