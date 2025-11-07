# análisis de incertidumbre con MonteCarlo
import numpy as np
from mondec import run_ea
import time

# valores originales de los parámetros en mondec.py
DEFAULT = {
    "pop_size": 100,
    "num_generations": 100,
    "hof_size": 10,
    "mu": 100,
    "lambda": 200,
    "mut_prob": 0.1,
    "cx_prob": 0.9,
}
"""
¿qué mas se podría alterar si no usaramos DEAP?
    "elitism": True,
    "selection_method": "tournament",
    ...
"""

def montecarlo(N: int, parameters, debug=False):
    """
    Módulo que ejecuta lo que hay en mondec.py N veces, variando los parámetros 
    de entrada según la distribución dada por el usuario (hasta ahora solo 
    distribuición normal). Guarda los resultados en una matriz de NxM donde:
    * M:    cantidad de parámetros a variar en el estudio (las demás quedan fijas)
    * N:    cantidad de muestras a tomar (variando los parámetros de entrada)
    * parameters:   diccionario que contiene los parámetros a variar y su distribución normal (media y desviación)
    """

    if debug:
        input_names = list(parameters.keys())
    
        for k in parameters.keys():
            print(k, type(k))

        print("input_names:", input_names)

    M = len(parameters)   
    input_values = np.empty((N, M))  # N filas: N muestras; M columnas: M parámetros a variar.

    # Primero, se toman N muestras random (combinaciones aleatorias de los parámetros a variar)
    for i in range(0,N):
        sample = []
        for param, dist in parameters.items():
            print(param)
            value = np.random.normal(dist["mean"], dist["std"])
            sample.append(value)
        input_values[i, :] = sample
    print(input_values)

    results = []

    # Una vez definidas las N muestras, se procede a las ejecuciones y el guardado de los resultados

    params = DEFAULT.copy() # considerar también los fijos
    for i in range(N):

        for j, value in enumerate(input_values[i,:]):
            print("value:",value)
            print("input_name[j]",input_names[j])
            params[input_names[j]] = value
        print(params)

        print("sleep...")
        time.sleep(10)
        output = run_ea(42,params)  # ejecuta el NSGA-III talcual como está en mondec.py
        results.append(output)

    # Dudas puntuales que tengo:
    # 1. ¿Qué hago con los resultados de cada ejecución?
    #    1.1. Análisis de incertidumbre: ver la varianza entre ejecuciones para definir qué tan robusto es el modelo
    #    1.2. Análisis de sensibilidad: hay que proceder distinto, posiblemente usar la librería de Sobol, pero igual hay que definir métricas para esto
    # 2. ¿Tendría que contemplar otros parámetros de entrada del algoritmo o con esos estamos bien?
    # 3. Calculo que para un N grande, que sería lo ideal porque es MonteCarlo, esto va a llevar muuucho tiempo ¿hay algún tema con eso? Vi que suelen usar surrogate models para este tipo de cosas.
    # 4. Probé calcular la métrica HV (HyperVolume) en el mondec.py como está. Da 0.0, capaz porque pongo mal el punto de referencia. ¿Está bien que toda la población sea el frente de Pareto o tendría que interpretarlo como un problema?
    # 5. Le pregunté a Virginia sobre el punto 4. Me dijo que les consulte primero si está bien que me ayude en estas cosas.

    return results


if __name__=="__main__":

    # parámetros a variar (las desviaciones estándar están mal)

    parameters = {
        "pop_size":         {"mean": 100,   "std": 20},
        "num_generations":  {"mean": 100,   "std": 20},
        "hof_size":         {"mean": 10,    "std": 20},
        "mu":               {"mean": 100,   "std": 20},
        "lambda":           {"mean": 200,   "std": 20},
        "mut_prob":         {"mean": 0.1,   "std": 20},
        "cx_prob":          {"mean": 0.9,   "std": 20}
    }

    N = 10
    
    print(montecarlo(3,parameters))