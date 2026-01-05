
import random
from instance import *
from deap import creator

def mutate_class_assignment(individual):
    idx = random.randint(0, N_CLASSES - 1)
    individual[idx] = random.randint(0, MAX_MICROSERVICES - 1)
    return individual

def mut_move(individual):
    blocks = [set(b) for b in individual.blocks]

    src = random.choice([b for b in blocks if len(b) > 1])
    dst = random.choice(blocks)

    cls = random.choice(tuple(src))
    src.remove(cls)
    dst.add(cls)

    return creator.Individual(blocks),

def cx_blocks(p1, p2):
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