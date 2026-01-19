import numpy as np
import time
import sys

from mondec.config_instance import N_CLASSES, INSTANCE, load_range_from_metadata
from mondec.representations import init_individual, evaluate

M = 10000000  # MonteCarlo samples
SEED = 42

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

    t0 = time.time()
    last_hb = t0

    objs = []

    for i in range(m):

        ind = init_individual(N_CLASSES, SEED)
        obj = evaluate(ind)
        objs.append(obj)

        if i % max(1, m // 100) == 0:
            progress_bar(i, m)

        now = time.time()
        if now - last_hb > 30:
            elapsed = now - t0
            rate = (i + 1) / elapsed
            eta = (m - i - 1) / rate if rate > 0 else float("inf")

            print(
                f"\n[HB] {i+1}/{m} | "
                f"{elapsed/60:.1f} min elapsed | "
                f"ETA {eta/60:.1f} min"
            )
            last_hb = now

    tf = time.time()
    print(f"\nMonte Carlo elapsed time: {tf - t0:.2f} seconds")

    objs = np.array(objs)
    new_mins = np.min(objs, axis=0)
    new_maxs = np.max(objs, axis=0)

    with open("montecarlo.txt", "a") as f:
        f.write(f"time:          {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"M (samples):   {m}\n")
        f.write(f"actual mins:   {actual_mins}\n")
        f.write(f"actual maxs:   {actual_maxs}\n")
        f.write(f"new mins:      {new_mins}\n")
        f.write(f"new maxs:      {new_maxs}\n\n")

    return new_mins, new_maxs


if __name__=="__main__":
    run()