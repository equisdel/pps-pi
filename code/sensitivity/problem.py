PROBLEM = {
    'num_vars': 3,
    'names': ['mu', 'lambda_prop', 'prob'],
    'bounds': [
        [50, 1000],   # mu
        [0.5, 3.0],   # lambda / mu
        [0.0, 1.0],   # mutation probability
    ]
}

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
