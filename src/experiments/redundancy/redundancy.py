import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from redundancy.paths import PF_INPUT_PATH


#pareto_list = df['pareto front'].apply(_parse_pf_value).dropna().tolist()
#print("step1: ok")
"""

redundancy_rows = []

for idx, pf in enumerate(pareto_list):

    if pf is None or len(pf) == 0:
        continue

    pf_size = len(pf)

    # 1. Redundancia genotípica
    pf_geno = unique_genotypes(pf)
    geno_size = len(pf_geno)

    # 2. Redundancia por simetría
    pf_canon = unique_canonicals(pf)
    canon_size = len(pf_canon)

    # 3. Métricas
    red_dup = 1 - geno_size / pf_size
    red_sim = 1 - canon_size / geno_size if geno_size > 0 else 0
    red_total = 1 - canon_size / pf_size

    redundancy_rows.append([
        idx,
        pf_size,
        geno_size,
        canon_size,
        red_dup,
        red_sim,
        red_total
    ])

#ideal_pf = pareto_list

# ahora hacemos una lista que los junte eliminando individuos duplicados (en genotipo)

def flatten_pareto_fronts(pareto_fronts):
    return [ind for pf in pareto_fronts for ind in pf]

all_individuals = flatten_pareto_fronts(pareto_list)
print("Total individuos (con repetidos):", len(all_individuals))
print("step2: ok")


"""
def unique_individuals(individuals):
    seen = set()
    unique = []
    for ind in individuals:
        t = tuple(ind)
        if t not in seen:
            seen.add(t)
            unique.append(ind)
    return unique
all_individuals = unique_individuals(all_individuals)
print("Individuos únicos:", len(all_individuals))
print("step3: ok")

def save_individuals_txt(individuals, path):
    with open(path, "w", encoding="utf-8") as f:
        for ind in individuals:
            f.write(f"{ind}\n")
save_individuals_txt(all_individuals,"individuos.txt")
print(len(all_individuals))

# 3. Guardar a txt
save_individuals_txt(all_individuals, "all_unique_individuals.txt")
print("step4: ok")
"""
# 4. Calcular fitness (ned, sm, icp, ifn)
"""
import pickle
from metrics import ned,sm,icp,ifn

fitnesses = []
with open("monoliths/jpetstore/graph.pkl", 'rb') as file:
    graph = pickle.load(file)
    nodes_to_remove = [node for node in graph.nodes if 'test' in node.lower() or 'transition' in node.lower()]
    graph.remove_nodes_from(nodes_to_remove)

CLASS_MAPPING = {i: node for i, node in enumerate(graph.nodes)}

from collections import defaultdict

def list_to_partitions(individual, class_mapping=CLASS_MAPPING):
    """
    individual: lista donde index = class_id, value = microservice_id
    class_mapping: dict {class_id: class_name}
    """
    partitions = defaultdict(list)

    # 1. Agrupar clases por microservicio
    for class_id, microservice_id in enumerate(individual):
        partitions[microservice_id].append(class_mapping[class_id])

    # 2. Reindexar microservicios a IDs secuenciales
    old_to_new_id = {
        old_id: new_id
        for new_id, old_id in enumerate(sorted(partitions.keys()))
    }

    sequential_partitions = {
        old_to_new_id[old_id]: sorted(classes)
        for old_id, classes in partitions.items()
    }

    return sequential_partitions

for ind in all_individuals:
    partitions = list_to_partitions(ind)   # ← asumido por tu modelo
    fitness = (
        ned(partitions),
        sm(partitions, graph),
        icp(partitions, graph),
        ifn(partitions, graph),
    )
    fitnesses.append(fitness)

print("step5: ok")

# 5. Filtrar no dominados

# Orden: [ned, sm, icp, ifn]
MINIMIZE = [True, False, True, True]

def dominates(f_a, f_b, minimize):
    """
    f_a domina a f_b según criterio mixto min/max
    """
    not_worse = True
    strictly_better = False

    for a, b, is_min in zip(f_a, f_b, minimize):
        if is_min:
            if a > b:
                not_worse = False
                break
            if a < b:
                strictly_better = True
        else:  # maximizar
            if a < b:
                not_worse = False
                break
            if a > b:
                strictly_better = True

    return not_worse and strictly_better

def pareto_filter(individuals, fitnesses, minimize):
    non_dominated = []

    for i, f_i in enumerate(fitnesses):
        dominated = False
        for j, f_j in enumerate(fitnesses):
            if i != j and dominates(f_j, f_i, minimize):
                dominated = True
                break
        if not dominated:
            non_dominated.append(individuals[i])

    return non_dominated

ideal_pf = pareto_filter(all_individuals, fitnesses, MINIMIZE)

print(f"Frente de Pareto ideal final: {len(ideal_pf)} individuos")

# Guardar PF final
save_individuals_txt(ideal_pf, "ideal_pf_final.txt")

redundancy_df = pd.DataFrame(
    redundancy_rows,
    columns=[
        "index",
        "pf_size",
        "geno_size",
        "canon_size",
        "red_dup",
        "red_sim",
        "red_total"
    ]
)

redundancy_df.to_csv("redundancy.txt", index=False)
print("Redundancy analysis saved to redundancy.txt")


# eliminamos los no dominados según métricas de evaluación en metrics, pero evitando deap en lo posible

"""
# Call the union function with the parsed & wrapped list of pareto fronts.
ideal_pf = union_pareto_fronts(pareto_list) # toma lista de individuos

print("\n\n\nSummary of ideal pareto front:")
print(f"Type: {type(ideal_pf)}, Length: {len(ideal_pf)}")

# Save full representation to a file to avoid overflowing the console buffer
with open("ideal_pf_repr.txt", "w", encoding="utf-8") as fh:
    fh.write(repr(ideal_pf))

print("ideal_pf saved to ideal_pf_repr.txt")
print("end")
"""
"""
"""