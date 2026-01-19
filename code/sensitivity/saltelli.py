import numpy as np
from SALib.sample import sobol
from sensitivity.paths import X_INPUT_PATH

PROBLEM = {     # for Saltelli sampling
    'num_vars': 3,
    'names': ['mu', 'lambda_prop', 'prob'],
    'bounds': [
        [50, 1000],   # mu
        [1.0, 3.0],   # lambda / mu
        [0.0, 1.0],   # mutation probability
    ]
}

def sample(n_base_samples=1024, seed=42, calc_second_order=False):

    np.random.seed(seed)

    X = sobol.sample(
        PROBLEM,
        n_base_samples,
        calc_second_order=calc_second_order
    )

    np.savetxt(X_INPUT_PATH, X)

    return X

def decode_params(X, default):
    
    mu = int(X[0])
    prop_lambda = float(X[1])
    prob = float(X[2])

    params = default.copy()
    params['mu'] = mu
    params['lambda'] = int(mu * prop_lambda)
    params['mut_prob'] = prob
    params['cx_prob'] = 1 - prob

    return params


if __name__=="__main__":
    sample()

"""
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
"""