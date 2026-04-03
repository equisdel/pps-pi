import numpy as np
import random
import sys
import os

from algorithm.config import N_CLASSES, INSTANCE, load_range_from_metadata
from algorithm.representations import init_individual, evaluate

M = 10000000  # MonteCarlo samples
DEFAULT_SEED = 42

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

def run_montecarlo(
    m: int = M,
    seed: int | random.Random | None = DEFAULT_SEED,
    log_path: str | None = None,
):

    actual_mins, actual_maxs = load_range_from_metadata()
    
    print(f"\nRunning Monte Carlo with M={m} samples for instance {INSTANCE} (N={N_CLASSES})\n")


    objs = []
    if isinstance(seed, random.Random):
        rng = seed
    else:
        rng = random.Random(seed)

    for i in range(m):

        ind = init_individual(N_CLASSES, rng)
        obj = evaluate(ind)
        objs.append(obj)

        if i % max(1, m // 100) == 0:
            progress_bar(i, m)

    objs = np.array(objs)
    new_mins = np.min(objs, axis=0)
    new_maxs = np.max(objs, axis=0)

    if log_path is not None:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"M (samples):   {m}\n")
            f.write(f"actual mins:   {actual_mins}\n")
            f.write(f"actual maxs:   {actual_maxs}\n")
            f.write(f"new mins:      {new_mins}\n")
            f.write(f"new maxs:      {new_maxs}\n\n")

    return new_mins, new_maxs


if __name__=="__main__":
    run()