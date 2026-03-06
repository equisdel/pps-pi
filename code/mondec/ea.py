# Librerias Estandar
import pickle
import random

# Librerias Externas
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from deap import base, creator, tools, algorithms

# Modulos Locales
from mondec.representations import Individual as IndividualClass, init_individual, individual_to_microservices, evaluate, mate, mutate
from mondec.config_ea import *
from mondec.config_instance import *
from mondec.initialization import *
from mondec.evaluation import evaluate
from mondec.ea_performance import hv
from mondec.plots import (
    plot_evolution,
    pareto_front_3d,
    plot_pareto_front,
    plot_radar_chart,
    plot_parallel_coordinates,
)

def deap_to_mondec(deap_ind):
    blocks_dict = {}
    for cls_idx, block_idx in enumerate(deap_ind):
        blocks_dict.setdefault(block_idx, set()).add(cls_idx)
    blocks = [list(b) for b in blocks_dict.values()]
    return IndividualClass(blocks)


def configure_nsga_iii(pop_size=100):

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti",base.Fitness,weights=(-1.0, +1.0, -1.0, -1.0))

    if not hasattr(creator, "Individual"):
        creator.create("Individual",IndividualClass,fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    toolbox.register("individual", lambda: init_individual(N_CLASSES))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)

    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("evaluate", evaluate)

    ref_points = tools.uniform_reference_points(nobj=N_OBJECTIVES, p=P)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    return toolbox


def run_ea(seed=None, parameters = {}):
    random.seed(seed)
    # inicialización del algoritmo genético: operadores, individuos y población inicial
    toolbox = configure_nsga_iii(int(parameters["pop_size"]))  
    # preparación de los datos a extraer/visualizar
    stats = tools.Statistics(lambda ind: ind)
    stats.register("avg", lambda inds: np.mean([ind.fitness.values for ind in inds], axis=0))
    stats.register("std", lambda inds: np.std([ind.fitness.values for ind in inds], axis=0))
    stats.register("min", lambda inds: np.min([ind.fitness.values for ind in inds], axis=0))
    stats.register("max", lambda inds: np.max([ind.fitness.values for ind in inds], axis=0))
    stats.register("hv", lambda inds: hv(tools.sortNondominated(list(inds), len(inds), first_front_only=True)[0]),
    )
    population = toolbox.population()  # Population size

    for ind in population:
        ind.fitness = creator.FitnessMulti()

    hof = tools.HallOfFame(int(parameters["hof_size"]))
    # configuración adicional del algoritmo genético
    num_generations = int(parameters["num_generations"])
    mut_prob = parameters["mut_prob"] 
    cx_prob = parameters["cx_prob"]  

    # ejecución
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=int(parameters["mu"]), lambda_=int(parameters["lambda"]), cxpb=cx_prob,
                                                    mutpb=mut_prob,
                                                    ngen=num_generations, stats=stats, halloffame=hof, verbose=False)

    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    pf_hv = hv(pareto_front)

    return population, logbook, hof, pareto_front, pf_hv

if __name__ == "__main__":
    
    seed = 42
 
    pop, logbook, hof, pareto_front, pf_hv = run_ea(seed, DEFAULT)

    # Pareto front stats
    pareto_solutions = [ind.fitness.values for ind in pareto_front]
    objectives = list(zip(*pareto_solutions))
    medians = [np.mean(obj) for obj in objectives]
    print("Pareto medians:", medians)

    # Best individual (from population & HOF)
    best_decomposition = tools.selBest(pop, k=1)[0]
    bd_partitions = individual_to_microservices(best_decomposition)  # no conditional
    print("BEST FROM PARETO FRONT", best_decomposition.fitness, len(bd_partitions), bd_partitions)

    best_hof = hof[0]
    bd_partitions = individual_to_microservices(best_hof)  # unified
    print("BEST FROM HOF", best_hof.fitness, len(bd_partitions), bd_partitions)

    # Logbook stats for plotting
    generations = logbook.select("gen")
    avg = np.array(logbook.select("avg"))
    min_ = np.array(logbook.select("min"))
    max_ = np.array(logbook.select("max"))

    # Plots
    #plot_evolution(generations, avg, min_, max_)
    #plot_pareto_front(pareto_front)

    cp = dict(
        population=pop,
        pareto_front=pareto_front,
        halloffame=hof,
        logbook=logbook,
        rndstate=random.getstate()
    )

    with open("experiment_database.pkl", "wb") as f:
        pickle.dump(cp, f)

    print("Hypervolume:", pf_hv)
    methods = ['M2M', 'FoSCI', 'CoGCN', 'Bunch', 'MEM']
    objectives = list(OBJECTIVES.values())
    scores = [
        [0.257, 0.054, 0.333, 1.857],   # M2M
        [0.516, 0.044, 0.478, 3.75],    # FoSCI
        [0.392, 0.091, 0.582, 2.533],   # CoGCN
        [0.667, np.nan, 0.477, 7.948],  # Bunch
        [1.0, 0.124, 0.434, 3.429]     # MEM
    ]
    methods.append("Our approach")
    scores.append(medians)

    #plot_parallel_coordinates(methods, scores, objectives)
    from redundancy.auxiliar import fitness_values
    for ind in pareto_front:
        f1 = np.array(ind.fitness.values)
        f2 = np.array(fitness_values(list(ind)))
        print(np.linalg.norm(f1 - f2), f1, f2)


