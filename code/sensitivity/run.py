from mondec.config_ea import DEFAULT
from mondec.ea import run_ea
from sensitivity.paths import X_INPUT_PATH, Y_OUTPUT_PATH, PF_OUTPUT_PATH
from sensitivity.saltelli import sample, decode_params, PROBLEM
from SALib.analyze import sobol

import os
import numpy as np

# supports multicore
CORES = 1       # range: [1,N], N: max number of cores
CORE_ID = 1     # range: [1,CORES]

SEED = 42       # if changed, it alters Saltelli matrix

def run():
    
    has_X = os.path.exists(X_INPUT_PATH)
    has_Y = os.path.exists(Y_OUTPUT_PATH)

 
    # SAMPLING
 
    if not has_X:
        print("Sampling...")
        X = sample(
            n_base_samples=1024,
            seed=SEED,
            calc_second_order=False
        )   # saved in X.txt
    else:
        print("Loading X...")
        X = np.loadtxt(X_INPUT_PATH)


    # RUNNING

    if not has_Y:

        import csv, json
        from mondec.initialization import progress_bar

        print("Running model...\n")
        runs = len(X)
        pf_dict = {}
        Y = np.zeros(runs)
        default = DEFAULT

        with open(Y_OUTPUT_PATH, "w", newline="") as f:
            
            writer = csv.writer(f)
            writer.writerow(["id","mu","lambda","mut_prob","cx_prob","HV"])

            if CORES==1:
        
                for i, x in enumerate(X):
                        
                    progress_bar(i,runs)
                    params = decode_params(x, default)

                    try:
                        _, _, _, pf, hv = run_ea(SEED, params)
                        writer.writerow([i, params["mu"], params["lambda"], params["mut_prob"], params["cx_prob"], round(hv,4)])
                        pf_dict[str(i)] = [ind.__str__() for ind in pf] 
                        Y[i] = round(hv,4)
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
    Si = sobol.analyze(PROBLEM, Y, calc_second_order=False)
    print(Si)

    return Si


if __name__=="__main__":
    run()