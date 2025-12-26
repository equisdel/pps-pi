# prueba de correlacion
from mondec import run_ea, DEFAULT
import numpy as np


def measure_correlation(population_fits):
    corr = np.corrcoef(population_fits.T)
    print(corr)

if __name__=="__main__":

    pop, _, _, _, _ = run_ea(23,DEFAULT)
    measure_correlation(np.array([ind.fitness.values for ind in pop]))