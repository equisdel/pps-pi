"""
Este código es un análisis del espacio de soluciones de la instancia a trabajar.
Por defecto, esta instancia es JPetStore.

Emplea MonteCarlo para generar N soluciones al azar, calcular sus fitness, y extraer:
    ** Rangos de valores para cada fitness (media y varianza, máximo y mínimo).
    ** Matriz Pearson de correlación entre los objetivos.
    ** Matriz Spearman de correlación.
    ** PCA

Del análisis de estos datos se pueden extraer:
- Niveles de independencia de los objetivos con respecto al problema
- Precalentamiento del rango de valores entre los que se mueve cada uno de los objetivos
- P individuos con mejores fitness para inicializar la población

"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from mondec import evaluate, MAX_MICROSERVICES, N_CLASSES, OBJECTIVES

M = 1000    # muestras de Monte Carlo
SEED = 42

def plot_obj_distribution(objs):
    """
    Grafica la distribución (boxplot + estadísticas) de cada objetivo.
    objs: array de shape (M, 4) con M muestras y 4 objetivos
    """
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    for i in range(4):
        obj_values = objs[:, i]
        
        # Boxplot
        axs[i].boxplot(obj_values, vert=False, widths=0.5)
        
        # Estadísticas
        mean_val = np.mean(obj_values)
        min_val = np.min(obj_values)
        max_val = np.max(obj_values)
        std_val = np.std(obj_values)
        
        # Mostrar estadísticas como texto en la gráfica
        stats_text = f"μ={mean_val:.4f}\nσ={std_val:.4f}\nMin={min_val:.4f}\nMax={max_val:.4f}"
        axs[i].text(0.98, 0.97, stats_text, transform=axs[i].transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9, family='monospace')
        
        axs[i].set_xlabel('Value')
        axs[i].set_title(f'Objective {OBJECTIVES[i]} (n={len(obj_values)})')
        axs[i].grid(True, alpha=0.3)
    
    fig.suptitle('Distribution of Objectives (Monte Carlo Analysis)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('obj_distribution.png', bbox_inches='tight', dpi=150)
    plt.show()

def random_individual():
    r_ind = np.random.randint(0, MAX_MICROSERVICES, size=N_CLASSES)
    return r_ind

def main():
    np.random.default_rng(SEED)

    # genera N individuos (posibles soluciones) al azar
    individuals = [random_individual() for _ in range(M)]
    objs = []       # recolecta los valores de sus objetivos, o sea, 
                    # f(x1,x2,...,xn) = o1, o2, o3, o4
                    # para cada una de las M muestras
    t0 = time.time()
    print()
    for i, ind in enumerate(individuals):
        ind_eval = evaluate(ind)
        print(i,":",ind_eval)
        objs.append(ind_eval)
    print("M:",M)
    print("Evaluado en", time.time() - t0,"segundos.")
    
    # M filas: las muestras; 
    # P columnas: los 4 objetivos;
    objs = np.array(objs) 

    # calculo de las distribuciones de cada objetivo
    objs_T = objs.T
    avg = np.mean(objs_T, axis=1)
    min_ = np.min(objs_T, axis=1)
    max_ = np.max(objs_T, axis=1)
    print("min: ",min_)
    print("avg: ",avg)
    print("max: ",max_)
    plot_obj_distribution(objs) # se las grafica en un box plot

    # calculo de correlacion entre objetivos
    # matriz de correlación Pearson
    corr = np.corrcoef(objs, rowvar=False)
    print("Pearson:\n", corr)
    # matriz de correlación Spearman
    spear = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            spear[i,j] = stats.spearmanr(objs[:,i], objs[:,j]).correlation
    print("Spearman:\n", spear)

    # PCA: Principal Components Analysis
    scaler = StandardScaler()       
    X = scaler.fit_transform(objs)  # 1. escala los objetivos
    pca = PCA(whiten=True)          # 2. centraliza por defecto
    pca.fit(X)                      # 3. calcula PCA
    print("PCA explained:", pca.components_)
    print("PCA explained:", pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()
