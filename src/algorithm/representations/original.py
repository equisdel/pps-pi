from algorithm.config import *
from algorithm.fitness import *
from collections import defaultdict
import random

# Estructura
class Individual(list):

    def __str__(self):
        return list.__str__(self)

# Generación aleatoria
def init_individual(n_classes, seed=None):
    if isinstance(seed, random.Random):
        rng = seed
    elif seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    return Individual([rng.randint(0, MAX_MICROSERVICES - 1) for _ in range(n_classes)])

# Traducción a diccionario
def individual_to_microservices(individual):
    partitions = defaultdict(list)
    for class_id, microservice_id in enumerate(individual):
        partitions[microservice_id].append(CLASS_MAPPING.get(class_id))

    old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(sorted(partitions.keys()))}
    sequential_partitions = {old_to_new_id[old_id]: sorted(classes) for old_id, classes in partitions.items()}
    return sequential_partitions

# Evaluación
def evaluate(individual):
    partitions = individual_to_microservices(individual)

    ned_value = ned(partitions, N_CLASSES if DEFAULT["proportional_NED"] else None)
    sm_value  = sm(partitions, graph)
    icp_value = icp(partitions, graph)
    in_value  = ifn(partitions, graph)

    values = (ned_value, sm_value, icp_value, in_value)
    return values

# Operador de mutación
def mutate(individual):
    idx = random.randint(0, N_CLASSES - 1)
    individual[idx] = random.randint(0, MAX_MICROSERVICES - 1)
    return (individual,)

# Operador de cruzamiento
import deap.tools as tools
def mate(p1,p2):
    return tools.cxOnePoint(p1,p2)
