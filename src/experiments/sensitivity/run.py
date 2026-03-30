from algorithm.ea import run_ea
from experiments.sensitivity.paths import X_INPUT_PATH, Y_OUTPUT_PATH, PF_OUTPUT_PATH, HV_GEN_OUTPUT_PATH
from experiments.sensitivity.sobol_report import print_sobol_report

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

DEFAULT_RUN_PARAMETERS = {
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


def _has_nonempty_data(path):
    if not os.path.exists(path):
        return False

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return False

    if path.lower().endswith(".csv"):
        return len(lines) > 1

    return True


def sample(problem=PROBLEM, n_base_samples=None, seed=42, calc_second_order=False, output_path=X_INPUT_PATH):

    if n_base_samples is None:
        n_base_samples = problem["base_samples"]

    np.random.seed(seed)

    X = sobol_sample.sample(
        problem,
        n_base_samples,
        calc_second_order=calc_second_order
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savetxt(output_path, X)

    return X

def decode_params(X, base_parameters):
    
    mu = int(X[0])
    prop_lambda = float(X[1])
    prob = float(X[2])

    params = base_parameters.copy()
    params['mu'] = mu
    params['lambda'] = int(mu * prop_lambda)
    params['mut_prob'] = prob
    params['cx_prob'] = 1 - prob

    return params

def run(
    *,
    problem=PROBLEM,
    base_parameters=None,
    seed=SEED,
    calc_second_order=False,
    x_input_path=X_INPUT_PATH,
    y_output_path=Y_OUTPUT_PATH,
    pf_output_path=PF_OUTPUT_PATH,
    hv_gen_output_path=HV_GEN_OUTPUT_PATH,
):
    if base_parameters is None:
        base_parameters = DEFAULT_RUN_PARAMETERS.copy()
    
    has_X = _has_nonempty_data(x_input_path)
    has_Y = _has_nonempty_data(y_output_path)

 
    # SAMPLING
 
    if not has_X:
        print("Sampling...")
        X = sample(
            problem=problem,
            n_base_samples=problem['base_samples'],
            seed=seed,
            calc_second_order=calc_second_order,
            output_path=x_input_path,
        )   # saved in X.txt
    else:
        print("Loading X...")
        X = np.loadtxt(x_input_path)


    # RUNNING

    if not has_Y:

        import csv, json
        from algorithm.initialization import progress_bar

        print("Running model...\n")
        
        runs = len(X)
        pf_dict = {}
        Y = np.zeros(runs)
        for path in (y_output_path, hv_gen_output_path, pf_output_path):
            path_dir = os.path.dirname(path)
            if path_dir:
                os.makedirs(path_dir, exist_ok=True)
        with open(y_output_path, "w", newline="") as f, open(hv_gen_output_path, "w", newline="") as hv_f:
            
            writer = csv.writer(f)
            writer.writerow(["id","mu","lambda","mut_prob","cx_prob","HV"])
            hv_writer = csv.writer(hv_f)
            hv_writer.writerow(["id","generation","HV","mu","lambda","mut_prob","cx_prob"])

            for i, x in enumerate(X):
                    
                progress_bar(i,runs)
                params = decode_params(x, base_parameters)

                try:
                    # Use different seed for each sample to ensure independent runs
                    run_seed = seed + i
                    params["seed"] = run_seed
                    _, logbook, _, pf, hv = run_ea(params)
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
                    
            with open(pf_output_path,"w") as f:      # writes once at the end
                json.dump(pf_dict,f,indent=2)

            
    else:
        print("Loading Y...")
        Y = np.loadtxt(y_output_path, delimiter=",", skiprows=1, usecols=-1)
        print(Y)

    Y = np.atleast_1d(Y)
    if Y.size == 0:
        raise RuntimeError(
            f"No HV values were found in '{y_output_path}'. "
            "Delete the partial output files and rerun the sensitivity experiment."
        )

    # EVALUATING

    print("\nAnalyzing...")
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order)
    print(Si)
    print_sobol_report(problem, Si)

    return Si

if __name__=="__main__":
    run()
