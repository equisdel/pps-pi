from deap import creator
import random

from mondec.config_ea import *
from mondec.config_instance import *
from mondec.metrics import *

# Estructura
class Individual:
    def __init__(self, blocks):
        self.blocks = frozenset(
            frozenset(b) for b in blocks if b
        )

    def copy(self):
        return Individual(self.blocks)

    def __str__(self):
        return str([list(block) for block in self.blocks])

    def __hash__(self):
        return hash(self.blocks)

    def __eq__(self, other):
        return isinstance(other, Individual) and self.blocks == other.blocks

# Generación aleatoria
def init_individual(n_classes:int, seed=42):
    k = random.randint(*(1,n_classes))
    blocks = [[] for _ in range(k)]

    for i in range(n_classes):
        blocks[random.randrange(k)].append(i)

    return Individual(blocks)


# Traducción a diccionario
def individual_to_microservices(individual: Individual):
    partitions = {}

    for ms_id, block in enumerate(individual.blocks):
        partitions[ms_id] = [CLASS_MAPPING[class_id] for class_id in block]

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

# Operador de mutación
def mutate(individual):
    blocks = [set(b) for b in individual.blocks]

    src = random.choice([b for b in blocks if len(b) > 1])
    dst = random.choice(blocks)

    cls = random.choice(tuple(src))
    src.remove(cls)
    dst.add(cls)

    return creator.Individual(blocks),

# Operador de cruzamiento
def mate(p1, p2):
    used = set()
    child_blocks = []

    for b1, b2 in zip(p1.blocks, p2.blocks):
        block = (set(b1) | set(b2)) - used
        if block:
            child_blocks.append(block)
            used |= block

    remaining = set(range(N_CLASSES)) - used
    if remaining:
        child_blocks.append(remaining)

    child1 = creator.Individual(child_blocks)
    child2 = creator.Individual(child_blocks.copy())

    return child1, child2