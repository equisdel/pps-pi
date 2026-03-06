import numpy as np
from scipy import stats as sp_stats
import csv
import json
from mondec.ea import run_ea, DEFAULT
from mondec.plots import plot_evolution
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

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

# ESTADISTICAS

def compute_stats(metric_values):
    """Calcula estadísticas de HV: media, mediana, std, percentiles, IC 95%."""
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
    plt.savefig(f'uncertainty/img/{metric}_robustness.png', dpi=150, bbox_inches='tight')
    plt.show()

# EJECUCION DE LAS N PRUEBAS

from mondec.initialization import progress_bar

def run_uncertainty(N=30,params=DEFAULT):

    seeds = [x for x in range(N)]
    hv_values = []
    hv_by_run = {}
    hv_long_rows = []

    for i, seed in enumerate(seeds):
        progress_bar(i,N)
        _, logbook, _, _, hv = run_ea(seed, params)
        hv_values.append(hv)
        gens = logbook.select("gen")
        hv_series = logbook.select("hv")

        hv_by_run[str(i)] = {
            "seed": seed,
            "hv_final": hv,
            "generations": gens,
            "hv_per_generation": hv_series,
        }

        for gen, hv_gen in zip(gens, hv_series):
            hv_long_rows.append([i, seed, gen, hv_gen, hv])

    with open("uncertainty/hv_by_generation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "seed", "generation", "hv", "hv_final"])
        writer.writerows(hv_long_rows)

    with open("uncertainty/hv_by_run.json", "w", encoding="utf-8") as f:
        json.dump(hv_by_run, f, indent=2)

    return hv_values

if __name__=="__main__":

    # N: cantidad de muestras a tomar
    N = 30

    # prueba de incertidumbre
    print("\nEjecutando análisis de robustez con", N, "corridas...\n")
    uncertainty_hv_values = run_uncertainty(N)

    # calcular y mostrar estadísticas
    hv_stats = compute_stats(uncertainty_hv_values)
  
    # imprimir y graficar robustez de HV
    print_robustness(hv_stats,"hypervolume")   # esto tiene que ser un análisis de robustez integral
    plot_robustness(uncertainty_hv_values,"hypervolume")
