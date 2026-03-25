from algorithm.config import DEFAULT
from algorithm.ea import run_ea
from sensitivity.paths import X_INPUT_PATH, Y_OUTPUT_PATH, PF_OUTPUT_PATH, HV_GEN_OUTPUT_PATH
from sensitivity.sobol_report import print_sobol_report

from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

import os
import numpy as np

SEED = 42       # if changed, it alters Saltelli matrix
PROBLEM = {     # for Saltelli sampling
    'num_vars': 3,
    'names': ['mu', 'lambda_prop', 'prob'],
    'bounds': [
        [50, 1000],   # mu
        [1.0, 3.0],   # lambda / mu
        [0.0, 1.0],   # mutation probability
    ],
    'base_samples': 1024,
}

def sample(n_base_samples=PROBLEM['base_samples'], seed=42, calc_second_order=False):

    np.random.seed(seed)

    X = sobol_sample.sample(
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

def run():
    
    has_X = os.path.exists(X_INPUT_PATH)
    has_Y = os.path.exists(Y_OUTPUT_PATH)

 
    # SAMPLING
 
    if not has_X:
        print("Sampling...")
        X = sample(
            n_base_samples=PROBLEM['base_samples'],
            seed=SEED,
            calc_second_order=False
        )   # saved in X.txt
    else:
        print("Loading X...")
        X = np.loadtxt(X_INPUT_PATH)


    # RUNNING

    if not has_Y:

        import csv, json
        from algorithm.initialization import progress_bar

        print("Running model...\n")
        
        runs = len(X)
        pf_dict = {}
        Y = np.zeros(runs)
        default = DEFAULT
        with open(Y_OUTPUT_PATH, "w", newline="") as f, open(HV_GEN_OUTPUT_PATH, "w", newline="") as hv_f:
            
            writer = csv.writer(f)
            writer.writerow(["id","mu","lambda","mut_prob","cx_prob","HV"])
            hv_writer = csv.writer(hv_f)
            hv_writer.writerow(["id","generation","HV","mu","lambda","mut_prob","cx_prob"])

            for i, x in enumerate(X):
                    
                progress_bar(i,runs)
                params = decode_params(x, default)

                try:
                    # Use different seed for each sample to ensure independent runs
                    run_seed = SEED + i
                    _, logbook, _, pf, hv = run_ea(run_seed, params)
                    writer.writerow([i, params["mu"], params["lambda"], params["mut_prob"], params["cx_prob"], round(hv,4)])
                    pf_dict[str(i)] = [ind.__str__() for ind in pf] 
                    Y[i] = round(hv,4)
                    for gen, hv_gen in zip(logbook.select("gen"), logbook.select("hv")):
                        hv_writer.writerow([
                            i, gen, round(hv_gen, 6),
                            params["mu"], params["lambda"], params["mut_prob"], params["cx_prob"]
                        ])
                except ValueError:
                    writer.writerow([i, params["mu"], params["lambda"], params["mut_prob"], params["cx_prob"], -1.0])
                    pf_dict[str(i)] = []
                    
            with open(PF_OUTPUT_PATH,"w") as f:      # writes once at the end
                json.dump(pf_dict,f,indent=2)

            
    else:
        print("Loading Y...")
        Y = np.loadtxt(Y_OUTPUT_PATH, delimiter=",", skiprows=1, usecols=-1)
        print(Y)

    # EVALUATING

    print("\nAnalyzing...")
    Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=False)
    print(Si)
    print_sobol_report(PROBLEM, Si)

    return Si

if __name__=="__main__":
    run()
