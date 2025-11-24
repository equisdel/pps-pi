# Análisis de incertidumbre: robustez del modelo
#
#   Metodología: ejecuta el algoritmo evolutivo N veces
#                usando la confuguración por defecto. Busca
#                determinar qué tan robusto es el modelo.
#
#   Métricas a medir:
#   1. HV:      HyperVolume; 
#               Mide cuánto volumen cubre el frente.
#   2. IGD+:    Inverse Gradient Distance;
#               Mide qué tan lejos está el frente de Pareto con respecto al ideal.
#               El frente ideal se obtiene por union de los N frentes hallados.
#   3. Spread:  Mide la diversidad del frente de pareto en cada muestra.
#
#   Objetivo:    Determinar qué tanto varían estas métricas naturalmente (robustez).

import numpy as np
from scipy import stats as sp_stats
from mondec import run_ea, DEFAULT
from plots import plot_evolution
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# METRICA IGD+

def union_pareto_fronts(pareto_fronts_list):
    # une todos los frentes de Pareto obtenidos en uno solo (se queda con puntos no dominados)
    from deap import tools
    
    all_individuals = []
    for pf in pareto_fronts_list:
        all_individuals.extend(pf)
    
    if len(all_individuals) == 0:
        return []
    
    nondominated = tools.sortNondominated(all_individuals, len(all_individuals), first_front_only=True)[0]
    print(f"La cantidad de individuos en la union de frentes de pareto es: {len(nondominated)}")
    return nondominated

def igd_plus(obtained_pf, ideal_pf):
    # calcula IGD+ para una muestra en particular
    
    # extrae los fitness values
    obtained_arr = np.array([ind.fitness.values for ind in obtained_pf])
    ideal_arr = np.array([ind.fitness.values for ind in ideal_pf])
    
    # matriz de distancias euclidianas entre cada par de puntos (obtenido e ideal)
    distances = cdist(ideal_arr, obtained_arr, metric='euclidean')
    min_distances = np.min(distances, axis=1)   # para cada punto del ideal, tomar la distancia mínima al frente obtenido
    igd_plus = np.mean(min_distances)           # promedio de las distancias
    
    return igd_plus

# METRICA SPREAD

def spread(obtained_pf, ideal_pf):
    # calcula spread para una muestra en particular

    # extrae los fitness values
    obtained_arr = np.array([ind.fitness.values for ind in obtained_pf])
    ideal_arr = np.array([ind.fitness.values for ind in ideal_pf])



    return spread


# METRICA HV (Hypervolume)

# hv se calcula en run_ea, sería bueno traerlo acá

# ESTADISTICAS

def compute_stats(metric_values):
    """Calcula estadísticas de HV y IGD+: media, mediana, std, percentiles, IC 95%."""
    metric_arr = np.array(metric_values)

    mean = np.mean(metric_arr)
    median = np.median(metric_arr)
    std = np.std(metric_arr, ddof=1)
    
    p10 = np.percentile(metric_arr, 10)
    p50 = np.percentile(metric_arr, 50)
    p90 = np.percentile(metric_arr, 90)
    
    # IC 95% con t-student
    n = len(metric_arr)
    se = std / np.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n-1)
    ic_lower = mean - t_crit * se
    ic_upper = mean + t_crit * se
    
    return {
        'mean': mean,
        'median': median,
        'std': std,
        'p10': p10,
        'p50': p50,
        'p90': p90,
        'ic_95_lower': ic_lower,
        'ic_95_upper': ic_upper,
        'n': n,
    }

def print_robustness(stats_dict,metric):

    print("\n" + "="*60)
    print(f"ANÁLISIS DE ROBUSTEZ: DISTRIBUCIÓN DE {metric.upper()}")
    print("="*60)
    print(f"Número de corridas: {stats_dict['n']}")
    print(f"\nMedia:              {stats_dict['mean']:.6f}")
    print(f"Mediana:            {stats_dict['median']:.6f}")
    print(f"Desvío estándar:    {stats_dict['std']:.6f}")
    print(f"\nPercentiles:")
    print(f"  P10:              {stats_dict['p10']:.6f}")
    print(f"  P50:              {stats_dict['p50']:.6f}")
    print(f"  P90:              {stats_dict['p90']:.6f}")
    print(f"\nIntervalo Confianza 95%:")
    print(f"  [{stats_dict['ic_95_lower']:.6f}, {stats_dict['ic_95_upper']:.6f}]")
    print(f"\nInterpretación:")
    if stats_dict['std'] < stats_dict['mean'] * 0.2:
        print("  ✓ Algoritmo es ROBUSTO (baja variabilidad)")
    elif stats_dict['std'] < stats_dict['mean'] * 0.5:
        print("  ~ Algoritmo tiene VARIABILIDAD MODERADA")
    else:
        print("  ✗ Algoritmo es INESTABLE (alta variabilidad)")
    if stats_dict['p10'] < stats_dict['mean'] * 0.5:
        print("  ⚠ A veces obtienes resultados pésimos (P10 << media)")
    print("="*60 + "\n")

def plot_robustness(values,metric):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.boxplot(values, vert=True)
    ax1.set_ylabel(f'{metric}')
    ax1.set_title(f'Boxplot de {metric}')
    ax1.grid(True, alpha=0.3)
    
    mean = np.mean(values)
    median = np.median(values)
    ax2.hist(values, bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Media: {mean:.4f}')
    ax2.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median:.4f}')
    ax2.set_xlabel(f'{metric}')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title(f'Histograma de {metric}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Distribución de {metric} - Análisis de Robustez', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{metric}_robustness.png', dpi=150, bbox_inches='tight')
    plt.show()


# EJECUCION DE LAS N PRUEBAS

def run_uncertainty(N=10):

    # El objetivo es medir la incertidumbre con distintas semillas, pero sin variar los parámetros
    seeds = [x for x in range(N)]
    all_pareto_fronts = []

    hv_values = []
    igd_plus_values = []
    spread_values = []

    for seed in seeds:
        output, logbook, hof, pareto_front, hv = run_ea(seed, DEFAULT)
        all_pareto_fronts.append(pareto_front)
        hv_values.append(hv)

    # Construir frente de referencia ideal
    ideal_front = union_pareto_fronts(all_pareto_fronts)
    #print(ideal_front)
    for pf in all_pareto_fronts:
        igd_plus_values.append(igd_plus(pf,ideal_front))
        spread_values.append(spread(pf, ideal_front))


    return hv_values, igd_plus_values, spread_values



if __name__=="__main__":

    # N: cantidad de muestras a tomar
    N = 30

    # prueba de incertidumbre
    print("\nEjecutando análisis de robustez con", N, "corridas...\n")
    uncertainty_hv_values, uncertainty_igd_plus_values, _ = run_uncertainty(N)

    print(uncertainty_igd_plus_values)

    # calcular y mostrar estadísticas
    hv_stats = compute_stats(uncertainty_hv_values)
    igd_plus_stats = compute_stats(uncertainty_igd_plus_values)

    print(igd_plus_stats)

    
    
    # imprimir y graficar robustez de HV
    print_robustness(hv_stats,"hypervolume")   # esto tiene que ser un análisis de robustez integral
    print_robustness(igd_plus_stats,"IGD+") 

    plot_robustness(uncertainty_hv_values,"hypervolume")
    plot_robustness(uncertainty_igd_plus_values,"IGD+")
    #plot_robustness(uncertainty_hv_values,"hypervolume")

    # prueba de sensibilidad

    # variable_parameters: parámetros a variar junto con su distribución
    variable_parameters = {
        "mu":               {"mean": 100,   "std": 0},
        "lambda":           {"mean": 200,   "std": 0},
        "mut_prob":         {"mean": 0.1,   "std": 0},
        "cx_prob":          {"mean": 0.9,   "std": 0}
    }


    
    # ejecución de las N pruebas
    #r = montecarlo(N,variable_parameters)
    #for i in r:
    #    print(i[1])