# análisis de incertidumbre con MonteCarlo
import numpy as np
from scipy import stats as sp_stats
from mondec import run_ea, DEFAULT
from plots import plot_evolution
import matplotlib.pyplot as plt
import time


# qué vamos a variar:
#  - tamaño poblacional: mu                 100,200,500,1000
#  - proporcion de lambda: lambda           1,1.5,2,10
#  - probabilidad de cruce vs. mutacion     0.1, 0.5, 0.9

# vamos a probar todas las combinaciones iterativamente y guardar los resultados, vamos a definir qué parámetros causan mayor variacion

# valores originales de los parámetros en mondec.py

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

    input_names = list(parameters.keys())
    
    if debug:
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

        #print("sleep...")
        #time.sleep(10)
        output, logbook, hof, pareto_front, hv  = run_ea(42,params)  # ejecuta el NSGA-III talcual como está en mondec.py
        results.append([output,hv])

    # Dudas puntuales que tengo:
    # 1. ¿Qué hago con los resultados de cada ejecución?
    #    1.1. Análisis de incertidumbre: ver la varianza entre ejecuciones para definir qué tan robusto es el modelo
    #    1.2. Análisis de sensibilidad: hay que proceder distinto, posiblemente usar la librería de Sobol, pero igual hay que definir métricas para esto
    # 2. ¿Tendría que contemplar otros parámetros de entrada del algoritmo o con esos estamos bien?
    # 3. Calculo que para un N grande, que sería lo ideal porque es MonteCarlo, esto va a llevar muuucho tiempo ¿hay algún tema con eso? Vi que suelen usar surrogate models para este tipo de cosas.
    # 4. Probé calcular la métrica HV (HyperVolume) en el mondec.py como está. Da 0.0, capaz porque pongo mal el punto de referencia. ¿Está bien que toda la población sea el frente de Pareto o tendría que interpretarlo como un problema?
    # 5. Le pregunté a Virginia sobre el punto 4. Me dijo que les consulte primero si está bien que me ayude en estas cosas.

    generations = logbook.select("gen")
    avg = np.array(logbook.select("avg"))
    min_ = np.array(logbook.select("min"))
    max_ = np.array(logbook.select("max"))

    # gráficos con respecto a nuestro método aisladamente
    plot_evolution(generations, avg, min_, max_)

    return results
