# Librerias Estandar
import pickle
import random
from collections import defaultdict

# Librerias Externas
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from pymoo.indicators.hv import HV
from deap import base, creator, tools, algorithms

# Modulos Locales
from metrics import sm, ifn, ned, icp
from hv import calculate_hv
from instance import *
from config import *
from plots import (
    plot_evolution,
    pareto_front_3d,
    plot_pareto_front,
    plot_radar_chart,
    plot_parallel_coordinates,
)


if DEFAULT["new_representation"]:
    print("canonical")
    from representations.canonical import Individual as IndividualClass, init_individual, individual_to_microservices, evaluate
    from operators import cx_blocks as mate, mut_move as mutate
else:
    print("original")
    from representations.original import Individual as IndividualClass, init_individual, individual_to_microservices, evaluate
    from operators import mutate_class_assignment as mutate
    from deap import tools
    mate = tools.cxTwoPoint


def configure_nsga_iii(pop_size=100):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, +1.0, -1.0, -1.0))
    creator.create("Individual", IndividualClass, fitness=creator.FitnessMulti)

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
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
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

    hv = calculate_hv(pareto_front)

    return population, logbook, hof, pareto_front, hv

if __name__ == "__main__":
    seeds = [42]

    for i, s in enumerate(seeds):
        print(f"EJECUCION ({i}/{len(seeds)}), SEMILLA: {s}")

        pop, logbook, hof, pareto_front, hv_ = run_ea(s, DEFAULT)

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
        plot_evolution(generations, avg, min_, max_)
        plot_pareto_front(pareto_front)

        cp = dict(
            population=pop,
            pareto_front=pareto_front,
            halloffame=hof,
            logbook=logbook,
            rndstate=random.getstate()
        )

        with open("experiment_database.pkl", "wb") as f:
            pickle.dump(cp, f)

        print("Hypervolume:", hv_)
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

        plot_parallel_coordinates(methods, scores, objectives)