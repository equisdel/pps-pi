from config import *
from instance import *
from metrics import *
import random

# Estructura
class Individual:
    def __init__(self, blocks):
        self.blocks = frozenset(
            frozenset(b) for b in blocks if b
        )

    def copy(self):
        return Individual(self.blocks)

    def __hash__(self):
        return hash(self.blocks)

    def __eq__(self, other):
        return isinstance(other, Individual) and self.blocks == other.blocks

# Generación aleatoria
def init_individual(n_classes, k_range=(2,10)):
    k = random.randint(*k_range)
    blocks = [[] for _ in range(k)]

    for i in range(n_classes):
        blocks[random.randrange(k)].append(i)

    return Individual(blocks)

# Traducción a diccionario
def individual_to_microservices(individual: Individual):
    partitions = defaultdict(list)

    for ms_id, block in enumerate(individual.blocks):
        for class_id in block:
            partitions[ms_id].append(CLASS_MAPPING[class_id])

    return partitions

# Evaluación
def evaluate(individual):
    partitions = individual_to_microservices(individual)

    ned_value = ned(partitions, N_CLASSES if DEFAULT["proportional_NED"] else None)
    sm_value  = sm(partitions, graph)
    icp_value = icp(partitions, graph)
    in_value  = ifn(partitions, graph)

    values = (ned_value, sm_value, icp_value, in_value)
    return values