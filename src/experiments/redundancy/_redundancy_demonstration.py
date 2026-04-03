import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Usuario/Desktop/Delfina/Personal/PPS+PI/src/experiments/redundancy/data/redundancy_jpetstore_canonical.csv")

print(np.mean(df["red_dup"]))
print(np.mean(df["red_sim"]))
print(np.mean(df["red_total"]))
print(np.std(df["red_total"]))

print(df.head(10))

import matplotlib.pyplot as plt
#print(df["red_total"])
#print(np.array(df["red_total"]))


mean = np.mean(df["red_total"])
plt.hist(x=np.array(df["red_total"]),range=[0,1],edgecolor="black",alpha=0.7)
plt.ylabel("Ocurrencias")
plt.xlabel("Redundancia")
plt.title("Histograma de Redundancia")
plt.annotate(
    f"Promedio: {mean:.4f}",
    xy=(mean, 150),          # punto al que apunta (ajusta Y según tu histograma)
    xytext=(mean+0.05, 180), # posición del cartel
    arrowprops=dict(arrowstyle="->", color="red"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red"),
    color="red"
)
plt.axvline(np.mean(df["red_total"]), color='red', linestyle='--', linewidth=2, label=f'Media: {mean:.4f}')
#plt.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median:.4f}')
plt.show()

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
    

OUTPUT = "redundant_canonical_solutions.txt"

with open(OUTPUT, "w", encoding="utf-8") as f:

    for canon, data in canon_map.items():

        if data["counter"] > 1:
            f.write("Canonical partition:\n")
            f.write(f"{sorted([sorted(b) for b in canon])}\n")
            f.write(f"Size: {data['size']}\n")
            f.write(f"Count: {data['counter']}\n")
            f.write("Rows:\n")

            for r in data["rows"]:
                f.write(f"{df.iloc[r].values}\n")

            f.write(f"Fitness: {data['fitness']}\n")
            f.write("-" * 60 + "\n")

    f.write("\n")
    f.write(f"La cantidad de soluciones únicas es: {len(canon_map)}\n")
    f.write(f"En total existen: {int(df.size / df.columns.size)}\n")
    f.write(
        f"Es decir que la redundancia es de: "
        f"{round(((180 - 82) / 180) * 100, 2)}%\n"
    )

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


counts = [d["counter"] for d in canon_map.values()]
print("Total canónicas:", len(canon_map))
print("Únicas (counter=1):", sum(c == 1 for c in counts))
print("Redundantes (counter>1):", sum(c > 1 for c in counts))

# número de filas originales
print("Filas totales:", len(df))

# suma de counters = filas
print("Suma de counters:", sum(d["counter"] for d in canon_map.values()))
