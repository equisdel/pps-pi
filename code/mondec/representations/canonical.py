from deap import creator, base
import random
import os
import matplotlib.pyplot as plt
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

    print("")
    n = 56

    creator.create("FitnessMulti",base.Fitness,weights=(-1.0, +1.0, -1.0, -1.0))
    creator.create("Individual",Individual,fitness=creator.FitnessMulti)
    
    # mutate
    sample_individual = init_individual(n)
    for _ in range(100):
        sample_individual = mutate(sample_individual,n)[0]
    print("ok")
    """
    """
    # Estadísticas
    mean_k = np.mean(coso)
    std_k = np.std(coso)
    cv_k = std_k / mean_k  # Coeficiente de variación
    min_k = np.min(coso)
    max_k = np.max(coso)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot de valores k por generación
    ax.plot(range(len(coso)), coso, 'o-', color='steelblue', alpha=0.6, 
            linewidth=1.5, markersize=4, label='k (cantidad de microservicios)')
 
    # Línea horizontal del promedio
    ax.axhline(y=mean_k, color='green', linestyle=':', linewidth=2, 
               alpha=0.7, label=f'Media = {mean_k:.2f}')
    
    # Etiquetas y título
    ax.set_xlabel('Generación', fontsize=12, fontweight='bold')
    ax.set_ylabel('k (Cantidad de microservicios)', fontsize=12, fontweight='bold')
    ax.set_title('Evolución del número de microservicios durante 100 mutaciones\n' + 
                 'Probabilidad Transfer/Merge = 0.5/0.5', 
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Leyenda con estadísticas
    stats_text = (f'Media: {mean_k:.2f}\nDesv. Est.: {std_k:.2f}\n'
                  f'Coef. Var.: {cv_k:.3f}\nMín: {min_k}\nMáx: {max_k}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, max(n, max_k + 1))
    

    plt.show()

    
    print(f"\nEstadísticas de k:")
    print(f"Media: {mean_k:.2f}")
    print(f"Desv. Est.: {std_k:.2f}")
    print(f"Coef. Variación: {cv_k:.4f}")
    print(f"Mín: {min_k}, Máx: {max_k}")

    
    coso = []
    # mate
    pA, pB = init_individual(n), init_individual(n)
    for _ in range(100):
        h, _ = mate(pA,pB,n)
        pA = h
        pB = init_individual(n)
        print(h)


    print(coso)
    
        
    # Estadísticas
    mean_k = np.mean(coso)
    std_k = np.std(coso)
    cv_k = std_k / mean_k  # Coeficiente de variación
    min_k = np.min(coso)
    max_k = np.max(coso)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot de valores k por generación
    ax.plot(range(len(coso)), coso, 'o-', color='steelblue', alpha=0.6, 
            linewidth=1.5, markersize=4, label='k (cantidad de microservicios)')
 
    # Línea horizontal del promedio
    ax.axhline(y=mean_k, color='green', linestyle=':', linewidth=2, 
               alpha=0.7, label=f'Media = {mean_k:.2f}')
    
    # Etiquetas y título
    ax.set_xlabel('Generación', fontsize=12, fontweight='bold')
    ax.set_ylabel('k (Cantidad de microservicios)', fontsize=12, fontweight='bold')
    ax.set_title('Evolución del número de microservicios durante 100 cruzamientos\n' + 
                 'Probabilidad de Merge correctivo = 1.0', 
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Leyenda con estadísticas
    stats_text = (f'Media: {mean_k:.2f}\nDesv. Est.: {std_k:.2f}\n'
                  f'Coef. Var.: {cv_k:.3f}\nMín: {min_k}\nMáx: {max_k}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, max(n, max_k + 1))
    

    plt.show()

    
    print(f"\nEstadísticas de k:")
    print(f"Media: {mean_k:.2f}")
    print(f"Desv. Est.: {std_k:.2f}")
    print(f"Coef. Variación: {cv_k:.4f}")
    print(f"Mín: {min_k}, Máx: {max_k}")



"""
"""

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
