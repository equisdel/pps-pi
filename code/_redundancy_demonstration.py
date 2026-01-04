import pandas as pd

PATH = "ideal_pf_final.txt"

# Se abre el frente de Pareto con individuos únicos 
with open(PATH):
    df = pd.read_csv(PATH,sep=r",\s+",header=None)
    df[0] = df[0].str.replace('[', '', regex=False)
    df[df.columns[-1]] = df[df.columns[-1]].str.replace(']', '', regex=False)
    df = df.astype(int)


from collections import defaultdict

canon_map = defaultdict(lambda: {
    "counter": 0,
    "rows": [],
    "size": 0,
    "fitness": tuple()
})

def canonical_partition(row):
    blocks = defaultdict(list)

    for i, _ in enumerate(row):
        blocks[row[i]].append(i)

    return frozenset(
        frozenset(indices) for indices in blocks.values()
    )



import pickle
from metrics import ned,sm,icp,ifn

with open("monoliths/jpetstore/graph.pkl", 'rb') as file:
    graph = pickle.load(file)
    nodes_to_remove = [node for node in graph.nodes if 'test' in node.lower() or 'transition' in node.lower()]
    graph.remove_nodes_from(nodes_to_remove)

CLASS_MAPPING = {i: node for i, node in enumerate(graph.nodes)}


def canon_to_partition(canon, class_mapping=CLASS_MAPPING):
    partitions = defaultdict(list)

    # 1. Agrupar clases por microservicio
    for microservice_id, microservice_group in enumerate(canon):
        for class_id in microservice_group:
            print(class_id)
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
    pass


def calculate_fitness(canon):
    partitions = canon_to_partition(canon)
    return (
        ned(partitions),
        sm(partitions, graph),
        icp(partitions, graph),
        ifn(partitions, graph),
    )

for idx, (_, row) in enumerate(df.iterrows()):
    canon = canonical_partition(row.values)

    canon_map[canon]["counter"] += 1
    canon_map[canon]["rows"].append(idx)   # o row.values.copy()
    if canon_map[canon]["counter"]==1:
        canon_map[canon]["fitness"] = calculate_fitness(canon)
        canon_map[canon]["size"] = len(canon)
    

for canon, data in canon_map.items():

    if data["counter"] > 1:
        print("Canonical partition:")
        print(sorted([sorted(b) for b in canon]))
        print("Size:",data["size"])
        print("Count:", data["counter"])
        print("Rows:")
        for r in data["rows"]:
            print(df.iloc[r].values)
        print("fitness:",data["fitness"])
        print("-" * 60)


print(f"\nLa cantidad de soluciones únicas es: {len(canon_map)}\n")
print(f"En total existen: {int(df.size/df.columns.size)}")
print(f"Es decir que la redundancia es de: {round(((180-82)/180)*100,2)}%")

import numpy as np

N = 24
C = np.zeros((N, N))
S = len(canon_map)

for canon, _ in canon_map.items():
    canon = sorted([sorted(b) for b in canon])
    for block in canon:
        for i in block:
            for j in block:
                C[i, j] += 1

C /= S

OUTPUT = "pareto_canonico_con_fitness.txt"

with open(OUTPUT, "w") as f:
    header = (
        f"{'ID':>3} | {'#MS':>3} | {'CNT':>4} | "
        f"{'NED':>6} | {'SM':>7} | {'ICP':>7} | {'IFN':>7} | "
        f"PARTITION\n"
    )
    f.write(header)
    f.write("-" * (len(header) + 20) + "\n")

    for i, (canon, data) in enumerate(canon_map.items()):
        canon_sorted = sorted([sorted(b) for b in canon])
        ned_v, sm_v, icp_v, ifn_v = data["fitness"]

        f.write(
            f"{i:3d} | "
            f"{data['size']:3d} | "
            f"{data['counter']:4d} | "
            f"{ned_v:6.3f} | "
            f"{sm_v:7.4f} | "
            f"{icp_v:7.4f} | "
            f"{ifn_v:7.4f} | "
            f"{canon_sorted}\n"
        )


import seaborn as sns
import matplotlib.pyplot as plt

sns.clustermap(C, cmap="viridis")
plt.title("Class co-assignment frequency")
plt.show()

