from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from mondec import run_ea

# saqué esto de la documentación

problem = {
    'num_vars': 3,
    'names': ['mu', 'lambda', 'prob'],
    'bounds': [[50, 1000],  # tamaño poblacional
               [0.5,3],     # proporcional a mu
               [0.0,1.0]]   # punto de probabilidad entre cruce y mutacion
}

param_values = saltelli.sample(problem,1024,calc_second_order=False)    # solo nos interesa primer orden por ahora

np.savetxt("param_values.txt", param_values)    # deja registrados los valores

Y = np.zeros([param_values.shape[0]])

print(Y)

for i, X in enumerate(param_values):
    Y[i] = run_ea(X)    # falta modificar con lo que tenía de los parámetros por defecto
    
# Y guarda los outputs del modelo, param_values los inputs, problem las caracteristicas de la variacion

Si = sobol.analyze(problem, Y)






