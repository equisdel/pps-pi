from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algorithm.config import configure_runtime
from experiments.redundancy.run import process_output

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

DATA_DIR = Path(__file__).resolve().parent / "sensitivity" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "redundancy" / "data"

if __name__ == "__main__":
    output_path = OUTPUT_DIR / "redundancy_jpetstore.csv"
    orig_path = DATA_DIR / "PF_old.json"

    result = process_output(
        orig_path=str(orig_path),
        output_path=str(output_path),
    )

    print(f"Análisis de redundancia completo para '{orig_path.name}'")
