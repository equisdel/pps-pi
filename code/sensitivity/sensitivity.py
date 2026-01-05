import csv
import numpy as np
from mondec import run_ea, DEFAULT
from SALib.sample import saltelli
from SALib.analyze import sobol

PROBLEM = {
    'num_vars': 3,
    'names': ['mu', 'lambda', 'prob'],
    'bounds': [[50, 1000],  # tamaño poblacional
               [0.5,3],     # proporcional a mu
               [0.0,1.0]]   # punto de probabilidad entre cruce y mutacion
}

def run_sensibility():
    file = open("Y_results.csv", "w", newline="")
    writer = csv.writer(file)
    writer.writerow(["index", "parameters", "HV", "pareto front"])  # header
    np.random.seed(42)  # para que saltelli genere siempre lo mismo
    param_values = saltelli.sample(PROBLEM,1024,calc_second_order=False)    # solo nos interesa primer orden por ahora

    np.savetxt("param_values.txt", param_values)    # deja registrados los valores

    Y = np.zeros([param_values.shape[0]])

    print(Y)

    for i, X in enumerate(param_values):
        print(i)
        params = DEFAULT.copy()
        mu, prop_lambda, prob = int(X[0]),float(X[1]),float(X[2])
        params['mu'] = mu
        params['lambda'] = int(mu*prop_lambda)
        params['mut_prob'] = prob 
        params['cx_prob'] = 1-prob
        try:
            _,_,_,pareto_front,hv = run_ea(42,params)     # guardo solo el HV
        except ValueError:
            hv = 0.0
            pareto_front = []
        Y[i] = hv
        writer.writerow([i, X, hv, "                ", pareto_front])
        
    # Y guarda los outputs del modelo, param_values los inputs, problem las caracteristicas de la variacion
    file.close()
    return Y

# Y = run_sensibility()
#Si = sobol.analyze(PROBLEM, Y, calc_second_order=False)

