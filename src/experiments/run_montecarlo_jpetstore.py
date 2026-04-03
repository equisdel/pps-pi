from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algorithm.config import configure_runtime
from experiments.montecarlo.run import run

BASE_PARAMETERS = {
    "pop_size": 200,
    "num_generations": 300,
    "hof_size": 10,
    "mu": 200,
    "lambda": 300,
    "mut_prob": 0.0,
    "cx_prob": 1.0,
    "proportional_NED": True,
    "new_representation": False,
    "MC_samples": 10000000,
    "seed": 42,
}

configure_runtime(instance="jpetstore", default_overrides=BASE_PARAMETERS)

if __name__ == "__main__":
    result = run(
        m=BASE_PARAMETERS["MC_samples"],
        instance="jpetstore",
        default_overrides=BASE_PARAMETERS,
        seed=BASE_PARAMETERS.get("seed", None),
        output_path=None,
    )
    print("Monte Carlo analysis complete for JPetStore")
    print(result)
