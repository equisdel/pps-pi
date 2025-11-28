# análisis de incertidumbre con MonteCarlo
import numpy as np
from scipy import stats as sp_stats
from mondec import run_ea, DEFAULT, calculate_hv
from plots import plot_evolution
import matplotlib.pyplot as plt
import time
from uncertainty import igd_plus, run_uncertainty
import pandas as pd


# qué vamos a variar:
#  - tamaño poblacional: mu                 100,200,500
#  - proporcion de lambda: lambda           1,  1.5,  2
#  - probabilidad de cruce vs. mutacion     0.1,0.5,0.9
# 9 pruebas

# vamos a probar todas las combinaciones iterativamente y guardar los resultados, vamos a definir qué parámetros causan mayor variacion

# valores originales de los parámetros en mondec.py

"""
¿qué mas se podría alterar si no usaramos DEAP?
    "elitism": True,
    "selection_method": "tournament",
    ...
"""

def run_sensitivity(parameters, N=1, debug=True):
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
    
    input_values = parameters.values()

    #print("input_names:", input_names)

    params = DEFAULT.copy() # considerar también los fijos
    for j, value in enumerate(input_values):
       # print("value:",value)
       # print("input_name[j]",input_names[j])
        params[input_names[j]] = value
    #print(params)

    hv_values, igd_plus_values, _, all_pareto_fronts, ideal_front  = run_uncertainty(N,params)  # ejecuta el NSGA-III talcual como está en mondec.py

    return [hv_values,igd_plus_values,ideal_front,calculate_hv(ideal_front)]

def display_results(results_df):
    """
    Muestra los resultados en un formato legible.
    """
    print("\n" + "="*120)
    print("RESULTADOS POR CONFIGURACIÓN")
    print("="*120)
    print(results_df.to_string(index=False))
    print("="*120 + "\n")

def process_results(results_list):
    """
    Procesa los resultados de sensibilidad usando análisis Sobol (SALib).
    
    Parámetros:
    - results_list: lista de dicts con resultados de cada prueba (id, mu, lambda, mut_prob, cx_prob, hv_mean, hv_std, etc.)
    
    Retorna:
    - df_results: DataFrame con indices de sensibilidad Sobol para HV e IGD+
    - df_rankings: DataFrame de ranking de factores por importancia
    """
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    
    # Convertir a DataFrame para facilitar manipulación
    df = pd.DataFrame(results_list)
    
    #print(df.to_string())
    print(df.head())

    # Extraer parámetros únicos y crear problema
    param_names = ["mu", "lambda", "mut_prob", "cx_prob"]
    
    # Crear rango de valores observados (min/max de cada parámetro en las pruebas)
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': [
            [df['mu'].min(), df['mu'].max()],
            [df['lambda'].min(), df['lambda'].max()],
            [df['mut_prob'].min(), df['mut_prob'].max()],
            [df['cx_prob'].min(), df['cx_prob'].max()]
        ]
    }
    
    # Extraer valores de salida
    X = df[param_names].values
    Y_hv = df['hv_mean'].values
    Y_igd = df['igd_plus_mean'].values
    
    # Análisis Sobol para HV
    Si_hv = sobol.analyze(problem, Y_hv, print_to_console=False)
    
    # Análisis Sobol para IGD+
    Si_igd = sobol.analyze(problem, Y_igd, print_to_console=False)
    
    # Crear DataFrame con índices de sensibilidad
    df_sensitivity = pd.DataFrame({
        'Parameter': param_names,
        'S1_HV': Si_hv['S1'],
        'S1_conf_HV': Si_hv['S1_conf'],
        'ST_HV': Si_hv['ST'],
        'ST_conf_HV': Si_hv['ST_conf'],
        'S2_HV': [Si_hv['S2'][i,i] if i < len(Si_hv['S2']) else np.nan for i in range(len(param_names))],
        'S1_IGD+': Si_igd['S1'],
        'S1_conf_IGD+': Si_igd['S1_conf'],
        'ST_IGD+': Si_igd['ST'],
        'ST_conf_IGD+': Si_igd['ST_conf'],
    })
    
    # Ranking de factores por importancia (S1 = efecto de primer orden, principal)
    df_ranking_hv = df_sensitivity[['Parameter', 'S1_HV', 'ST_HV']].copy()
    df_ranking_hv['Métrica'] = 'HV'
    df_ranking_hv = df_ranking_hv.sort_values('S1_HV', ascending=False).reset_index(drop=True)
    
    df_ranking_igd = df_sensitivity[['Parameter', 'S1_IGD+', 'ST_IGD+']].copy()
    df_ranking_igd.columns = ['Parameter', 'S1_HV', 'ST_HV']
    df_ranking_igd['Métrica'] = 'IGD+'
    df_ranking_igd = df_ranking_igd.sort_values('S1_HV', ascending=False).reset_index(drop=True)
    
    df_ranking = pd.concat([df_ranking_hv, df_ranking_igd], ignore_index=True)
    
    return df, df_sensitivity, df_ranking

if __name__=="__main__":

    N = 1

    mu_variations =     [100,150,200,250,500,1000]
    lambda_variations = [0.5,0.75,1.0,1.5,2.0,3.0]
    prob_variations =   [0.1,0.25,0.5,0.5,0.75,0.9]

   # mu_variations =     [100,200]# [100,200,500]
   # lambda_variations = [1.0]#[1.0,1.5,2.0]
   # prob_variations =   [0.1]#[0.1,0.5,0.9]


    total_pruebas = len(mu_variations)*len(lambda_variations)*len(prob_variations)

    results = []

    # variable_parameters: parámetros a variar junto con su distribución
    i = 0
    for mu in mu_variations:
        for lamb in lambda_variations:
            for prob in prob_variations:

                print(f"Prueba {i}; Valores: (mu:{mu}), (lambda:{int(mu*lamb)}), (mut_prob:{prob}), (cx_prob:{1-prob})")
                variable_parameters = {
                    "pop_size": mu,
                    "mu":       mu,          
                    "lambda":   int(mu*lamb), 
                    "mut_prob": prob,
                    "cx_prob":  1-prob,
                }
                output = run_sensitivity(variable_parameters)
                result = {
                    "id": i,
                    "mu": variable_parameters["mu"],
                    "lambda": variable_parameters["lambda"],
                    "mut_prob": variable_parameters["mut_prob"],
                    "cx_prob": variable_parameters["cx_prob"],
                    "pareto_front": output[2],
                    "pareto_front_hv": output[3],
                    "hv_mean": np.mean(output[0]),
                    "hv_std": np.std(output[0]),
                    "igd_plus_mean": np.mean(output[1]),
                    "igd_plus_std": np.std(output[1]),
                }
                results.append(result)
                i+=1

    # Convertir a DataFrame
    df_results = pd.DataFrame(results)
    
    print(f"\nTotal de pruebas completadas: {len(results)}")
    print(f"TIMESTAMP: {time.time()}\n")
    
    # Mostrar tabla de configuraciones y resultados brutos
    display_results(df_results)
    
    # Procesar con análisis Sobol
    print("\n" + "="*40)
    print("ANÁLISIS DE SENSIBILIDAD GLOBAL (SOBOL)")
    print("="*40 + "\n")
    
    df_config, df_sensitivity, df_ranking = process_results(results)
    
    print("Índices de Sensibilidad (S1 = efectos de primer orden, ST = efectos totales):\n")
    print(df_sensitivity.to_string(index=False))
    
    print("\n" + "="*120)
    print("RANKING DE FACTORES POR IMPORTANCIA")
    print("="*120 + "\n")
    print(df_ranking.to_string(index=False))
    
    # Guardar en archivos CSV
    timestamp = int(time.time())
    df_results.to_csv(f"sensitivity_results_{timestamp}.csv", index=False)
    df_sensitivity.to_csv(f"sensitivity_indices_{timestamp}.csv", index=False)
    df_ranking.to_csv(f"sensitivity_ranking_{timestamp}.csv", index=False)
    print(f"\n✓ Resultados guardados: sensitivity_results_{timestamp}.csv")
    print(f"✓ Índices guardados: sensitivity_indices_{timestamp}.csv")
    print(f"✓ Ranking guardado: sensitivity_ranking_{timestamp}.csv")

    