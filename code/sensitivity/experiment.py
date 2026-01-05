import numpy as np
from SALib.sample import saltelli
from mondec.ea import run_ea, DEFAULT
from .problem import PROBLEM, decode_params

def run_sensitivity_experiment(
    n_base_samples=1024,
    seed=42,
):
    np.random.seed(seed)

    X = saltelli.sample(
        PROBLEM,
        n_base_samples,
        calc_second_order=False
    )

    Y = np.zeros(len(X))
    failures = np.zeros(len(X), dtype=bool)

    for i, x in enumerate(X):
        params = decode_params(x, DEFAULT)

        try:
            _, _, _, _, hv = run_ea(seed, params)
            Y[i] = hv
        except ValueError:
            Y[i] = 0.0
            failures[i] = True

    return {
        "X": X,
        "Y": Y,
        "failures": failures,
        "problem": PROBLEM,
    }
