from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algorithm.config import configure_runtime

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

from experiments.sensitivity.run import PROBLEM, run

DATA_DIR = Path(__file__).resolve().parent / "sensitivity" / "data"

if __name__ == "__main__":
    run(
        problem=PROBLEM,
        base_parameters=BASE_PARAMETERS,
        seed=42,
        x_input_path=str(DATA_DIR / "X_jpetstore.txt"),
        y_output_path=str(DATA_DIR / "Y_jpetstore.csv"),
        pf_output_path=str(DATA_DIR / "PF_jpetstore.json"),
        hv_gen_output_path=str(DATA_DIR / "HV_by_generation_jpetstore.csv"),
    )
