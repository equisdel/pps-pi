from deap import creator, base
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from algorithm.config import *
from algorithm.fitness import *

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
def init_individual(n_classes, seed=None):
    rng = random.Random(seed) if seed is not None else random
    k = rng.randint(1, n_classes)
    blocks = [[] for _ in range(k)]

    for i in range(n_classes):
        blocks[rng.randrange(k)].append(i)

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
def mutate(individual,n=N_CLASSES,p_transfer=0.5):
    
    blocks = [set(b) for b in individual.blocks]

    mec_selector = random.uniform(0,1)

    if mec_selector <= p_transfer or (len(blocks)==1):    
    # transfer mechanism: move random class from Bs to Bt (Bt can be new)
        #print("transfer")
        cls = random.choice(list(range(n)))
        #print(cls)
        
        src = next(b for b in blocks if cls in b)
        blocks.remove(src)
        dst = random.choice(blocks+[set()])
        
        src.remove(cls)
        dst.add(cls)
        if len(dst)==1:
            blocks.append(dst)
        if len(src)!=0:
            blocks.append(src)

    else:
    # merge mechanism: combine two random classes
        #print("merge")
        b1 = random.choice(blocks)
        blocks.remove(b1)
        b2 = random.choice(blocks)
        blocks.remove(b2)
        b1_U_b2 = set.union(b1,b2)
        blocks.append(b1_U_b2)

    return creator.Individual(blocks),

def mate(pA, pB,n=N_CLASSES,p_merge=1):
    child_blocks = []
    unassigned = set(range(n))

    for a in pA.blocks:
        a_set = set(a)
        for b in pB.blocks:
            inter = a_set.intersection(b)
            if inter:
                child_blocks.append(inter)
                unassigned -= inter

    for cls in unassigned:
        child_blocks.append({cls})

    child1 = creator.Individual(child_blocks)
    child2 = creator.Individual(child_blocks.copy())

    if random.uniform(0,1) <= p_merge:
        child1 = mutate(child1,n)[0]
    if random.uniform(0,1) <= p_merge:
        child2 = mutate(child2,n)[0]

    return child1, child2


if __name__=="__main__":

    n = N_CLASSES
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, +1.0, -1.0, -1.0))
    creator.create("Individual", Individual, fitness=creator.FitnessMulti)
    
    # Initial population
    population = [init_individual(n) for _ in range(50)]
    population_copy = population.copy()
    
    # Mutation experiment
    mutation_ks = []
    for gen in range(100):
        idx = random.randint(0, 49)
        individual = population[idx]
        offspring, = mutate(individual, n)
        population[idx] = offspring
        avg_k = sum(len(ind.blocks) for ind in population) / 50
        mutation_ks.append(avg_k)
    
    mean_m = np.mean(mutation_ks)
    std_m = np.std(mutation_ks)
    cv_m = std_m / mean_m
    min_m = np.min(mutation_ks)
    max_m = np.max(mutation_ks)
    print(f"\nEstadísticas de k promedio para mutación:")
    print(f"Media: {mean_m:.2f}")
    print(f"Desv. Est.: {std_m:.2f}")
    print(f"Coef. Variación: {cv_m:.4f}")
    print(f"Mín: {min_m}, Máx: {max_m}")
    
    # Reset population
    population = population_copy
    
    # Crossover experiment
    crossover_ks = []
    for gen in range(100):
        idx1, idx2 = random.sample(range(50), 2)
        p1 = population[idx1]
        p2 = population[idx2]
        child1, child2 = mate(p1, p2, n, 0)
        # replace idx1 with child1
        population[idx1] = child1
        avg_k = sum(len(ind.blocks) for ind in population) / 50
        crossover_ks.append(avg_k)
    
    mean_c = np.mean(crossover_ks)
    std_c = np.std(crossover_ks)
    cv_c = std_c / mean_c
    min_c = np.min(crossover_ks)
    max_c = np.max(crossover_ks)
    print(f"\nEstadísticas de k promedio para cruzamiento:")
    print(f"Media: {mean_c:.2f}")
    print(f"Desv. Est.: {std_c:.2f}")
    print(f"Coef. Variación: {cv_c:.4f}")
    print(f"Mín: {min_c}, Máx: {max_c}")
    
    # Combined plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(range(len(mutation_ks)), mutation_ks, 'o-', color='steelblue', alpha=0.6, linewidth=1.5, markersize=4, label='Mutación')
    ax.axhline(y=mean_m, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Media Mutación = {mean_m:.2f}')
    ax.plot(range(len(crossover_ks)), crossover_ks, 'o-', color='red', alpha=0.6, linewidth=1.5, markersize=4, label='Cruzamiento')
    ax.axhline(y=mean_c, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'Media Cruzamiento = {mean_c:.2f}')
    ax.set_xlabel('Generación', fontsize=12, fontweight='bold')
    ax.set_ylabel('k promedio (Cantidad de microservicios)', fontsize=12, fontweight='bold')
    ax.set_title('Evolución del número promedio de microservicios durante 100 mutaciones y cruzamientos', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    #stats_text = (f'Mutación:\nMedia: {mean_m:.2f}\nDesv. Est.: {std_m:.2f}\nCoef. Var.: {cv_m:.3f}\nMín: {min_m}\nMáx: {max_m}\n\nCruzamiento:\nMedia: {mean_c:.2f}\nDesv. Est.: {std_c:.2f}\nCoef. Var.: {cv_c:.3f}\nMín: {min_c}\nMáx: {max_c}')
    #ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, max(n, max(max_m, max_c) + 1))
    plt.show()
"""
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
"""
