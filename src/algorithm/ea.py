# Librerias
import pickle
import random
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from deap import base, creator, tools, algorithms

# Modulos Locales
from algorithm.representations import Individual as IndividualClass, init_individual, mate, mutate
from algorithm.config import *
from algorithm.initialization import *
from algorithm.fitness import evaluate
from algorithm.quality_indicators import hv

def configure_nsga_iii(pop_size):

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti",base.Fitness,weights=(-1.0, +1.0, -1.0, -1.0))

    if not hasattr(creator, "Individual"):
        creator.create("Individual",IndividualClass,fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    toolbox.register("individual", lambda: init_individual(N_CLASSES))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)

    #toolbox.register("clone", clone)        # usado internamente en VarOr
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("evaluate", evaluate)

    ref_points = tools.uniform_reference_points(nobj=M_OBJECTIVES, p=P)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    return toolbox

def run_ea(parameters=None):

    if parameters is None:
        parameters = DEFAULT

    random.seed(parameters["seed"])
    np.random.seed(parameters["seed"])

    # inicialización del algoritmo genético: operadores, individuo y población inicial
    toolbox = configure_nsga_iii(int(parameters["pop_size"]))

    # preparación de los datos a registrar por iteración
    stats = tools.Statistics(lambda ind: ind)
    stats.register("avg", lambda inds: np.mean([ind.fitness.values for ind in inds], axis=0))
    stats.register("std", lambda inds: np.std([ind.fitness.values for ind in inds], axis=0))
    stats.register("min", lambda inds: np.min([ind.fitness.values for ind in inds], axis=0))
    stats.register("max", lambda inds: np.max([ind.fitness.values for ind in inds], axis=0))
    stats.register("hv", lambda inds: hv(tools.sortNondominated(list(inds), len(inds), first_front_only=True)[0]),)
    
    population = toolbox.population()  # Population size

    for ind in population:
        ind.fitness = creator.FitnessMulti()

    # configuración adicional del algoritmo genético
    hof = tools.HallOfFame(int(parameters["hof_size"]))
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

    pop, logbook, hof, pareto_front, pf_hv = run_ea(DEFAULT)

    # Logbook stats for plotting
    generations = logbook.select("gen")
    avg = np.array(logbook.select("avg"))
    min_ = np.array(logbook.select("min"))
    max_ = np.array(logbook.select("max"))
    hv_ = np.array(logbook.select("hv"))
    igd_ = np.array(logbook.select("igd"))
    igd_plus = np.array(logbook.select("igd_plus"))

    cp = dict(
        population=pop,
        pareto_front=pareto_front,
        logbook=logbook,
        rndstate_python=random.getstate(),
        rndstate_numpy=np.random.get_state(),
    )

    with open("experiment_database.pkl", "wb") as f:
        pickle.dump(cp, f)

