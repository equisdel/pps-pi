"""
Este código es un análisis MonteCarlo del espacio de soluciones de la instancia a trabajar.
La idea es usarlo para precalentar el algoritmo con límites realistas para cada uno de los objetivos.

Explora el espacio al azar y registra mínimos y máximos. También pueden sacarse otros datos:
- Correlación entre objetivos (Pearson y Spearman)
- PCA
- Distribución de los objetivos (media, desviación, etc.)

Por el momento, se deshabilitan esas opciones analíticas para priorizar funcionalidad

"""

import numpy as np
import time
import sys
from mondec import evaluate, new_evaluate, init_partition, N_CLASSES, DEFAULT, INSTANCE, METADATA, N_OBJECTIVES

M = 1000000     # muestras de Monte Carlo
SEED = 42

def progress_bar(i, total, width=30):
    progress = (i + 1) / total
    filled = int(width * progress)
    bar = "█" * filled + "-" * (width - filled)
    percent = int(progress * 100)
    sys.stdout.write(f"\r|{bar}| {percent}% ({i+1}/{total})")
    sys.stdout.flush()

def random_individual(n=N_CLASSES):
    if DEFAULT["new_representation"]:
        return init_partition(n)
    else:
        return np.random.randint(0, n, size=n).tolist()

def run(m:int=M):
  
    np.random.default_rng(SEED)

    individuals = [random_individual() for _ in range(m)]
    objs = []       # recolecta los valores de sus objetivos, o sea, 
                    # f(x1,x2,...,xn) = o1, o2, o3, o4
                    # para cada una de las M muestras
    
    t0 = time.time()
    for i, ind in enumerate(individuals):
        ind_eval = new_evaluate(ind) if DEFAULT["new_representation"] else evaluate(ind)
        objs.append(ind_eval)
        if (m>100000):
            if i % 1000 == 999:
                progress_bar(i, m)
        else:
            progress_bar(i, m)
    tf = time.time()

    # M filas:      las muestras a tomar; 
    # P columnas:   los 4 objetivos;
    objs = np.array(objs) 

    # calculo de las distribuciones de cada objetivo
    objs_T = objs.T
    min_ = np.min(objs_T, axis=1)
    max_ = np.max(objs_T, axis=1)

    print()

    return min_, max_, tf-t0

if __name__ == "__main__":

    import json 

    with open(METADATA,"r") as f:
        metadata = json.load(f) 

    order = ["NED", "SM", "ICP", "IN"]

    # known limits
    old_min = [metadata["RANGE"]["MIN"][k] for k in order]
    old_max = [metadata["RANGE"]["MAX"][k] for k in order]

    #print(metadata)
    

    print(old_min)

    m = 10000000

    print("\nInstancia: ",INSTANCE," (N: ",N_CLASSES," clases)")
    print("Montecarlo corre con M:  ",m," muestras\n")

    new_min, new_max, t = run(m)

    update_needed = False

    for i in range(N_OBJECTIVES):
        if new_min[i]<old_min[i]:
            old_min[i]=new_min[i]
            update_needed = True
        if new_max[i]>old_max[i]:
            old_max[i]=new_max[i]
            update_needed = True

    if update_needed:
        print("update needed")

    print("Evaluado en",t,"segundo\n")
