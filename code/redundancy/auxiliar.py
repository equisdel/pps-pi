from mondec.metrics import ned,sm,icp,ifn
import ast
from collections import defaultdict
from mondec.representations import individual_to_microservices
from mondec.config_instance import CLASS_MAPPING, graph

def fitness_values(ind):
    partitions = list_to_partitions_new(ind)
    return [ned(partitions,24),sm(partitions,graph),icp(partitions,graph),ifn(partitions,graph)]

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

def list_to_partitions_new(individual, class_mapping=CLASS_MAPPING):
    """
    individual: list of lists, each sublist is a microservice containing class_ids
    class_mapping: dict {class_id: class_name}
    """
    #print(individual)
    partitions = {}

    for new_id, microservice in enumerate(individual):
        partitions[new_id] = sorted(class_mapping[c] for c in microservice)

    return partitions

def canonical(sol):
    #print(sol)
    mservices = {str(k): [] for k in range(len(sol))}

    for i, ms in enumerate(sol):
        mservices[f'{ms}'] = mservices[f'{ms}']+[i]

    #print([v for k,v in mservices.items() if v])
    return [v for k,v in mservices.items() if v]


def canonical_key_new(solution: list[list[int]]) -> tuple:
    """
    solution: [[...], [...], ...]
    returns: hashable canonical tuple
    """
    #print("\n",solution)
    return tuple(
        sorted(
            (tuple(sorted(ms)) for ms in solution),
            key=lambda x: (len(x), x)
        )
    )

registry = {}

def register_solution(solution):

    #print("\nSOLUTION: ",solution)
    solution = canonical(solution)
    key = canonical_key_new(solution)
    #print("\nSOLUTION: ",key)
    if key not in registry:
        registry[key] = {
            "count": 1,
            "n_services": len(solution),
            "fitness": fitness_values(solution)
        }
    else:
        registry[key]["count"] += 1

def pareto_dominated(points):
    n = points.shape[0]
    dominated = np.zeros(n, dtype=bool)

    order = np.argsort(points[:, 0])
    pts = points[order]

    best = np.inf * np.ones(points.shape[1])

    for i in range(n):
        p = pts[i]
        if np.all(p >= best):
            dominated[i] = True
        else:
            best = np.minimum(best, p)

    result = np.zeros(n, dtype=bool)
    result[order] = dominated
    return result

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

def coverage(A, B):
    return sum(any(dominates(a,b) for a in A) for b in B) / len(B)


if __name__=="__main__":

    import json
    import pickle
    import numpy as np
    from mondec.initialization import progress_bar
    from tabulate import tabulate
    path = "../data/PF.json"
    """
    with open(path, "r") as f:
        pf_raw = json.load(f)

        # parse strings into real lists

        i = 0
        for key, pf_list in pf_raw.items():
            progress_bar(i,len(pf_raw))
            for pf_str in pf_list:
                register_solution(ast.literal_eval(pf_str))
            i+=1


    #for k, v in registry.items():
    #    print(k, v["count"], v["n_services"], v["fitness"])

    print(len(registry))
    print(np.sum([v["count"] for v in registry.values()]))


    headers = ["ID", "#MS", "CNT", "NED", "SM", "ICP", "IFN", "PARTITION"]
    rows = []

    for k, v in registry.items():
        rows.append([
            k,
            v["n_services"],
            v["count"],
            f"{v['fitness'][0]:.3f}",
            f"{v['fitness'][1]:.4f}",
            f"{v['fitness'][2]:.4f}",
            f"{v['fitness'][3]:.4f}",
        ])

    table = tabulate(rows, headers=headers, tablefmt="github")

    with open("old_registry_table.txt", "w") as f:
        f.write(table)

    with open("old_registry.pkl", "wb") as f:
        pickle.dump(registry, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("old_registry.pkl", "rb") as f:
        registry = pickle.load(f)

    print(np.max([v['count'] for _, v in registry.items()]))
    print(np.mean([v['count'] for _, v in registry.items()]))
    print(np.min([v['count'] for _, v in registry.items()]))

    keys = list(registry.keys())

    objs = np.array([
        [v['fitness'][0], v['fitness'][1], v['fitness'][2], v['fitness'][3]]
        for v in registry.values()
    ])

    objs[:, 1] *= -1

    dominated_mask = pareto_dominated(objs)

    for i, k in enumerate(keys):
        progress_bar(i,len(keys))
        registry[k]["dominated"] = bool(dominated_mask[i])
    
    headers = ["ID", "#MS", "CNT", "NED", "SM", "ICP", "IFN", "DOMINATED"]
    rows = []

    for k, v in registry.items():
        rows.append([
            k,
            v["n_services"],
            v["count"],
            f"{v['fitness'][0]:.3f}",
            f"{v['fitness'][1]:.4f}",
            f"{v['fitness'][2]:.4f}",
            f"{v['fitness'][3]:.4f}",
            f"{v['dominated']}",
        ])


    table = tabulate(rows, headers=headers, tablefmt="github")

    with open("old_registry_table.txt", "w") as f:
        f.write(table)

    with open("old_registry.pkl", "wb") as f:
        pickle.dump(registry, f, protocol=pickle.HIGHEST_PROTOCOL)

    ##

    with open("old_registry.pkl", "rb") as f:
        registry = pickle.load(f)

    print(len(registry))
    print(sum(v["dominated"] for v in registry.values()))
    print(sum(not v["dominated"] for v in registry.values()))

    pareto_registry = {
        k: v for k, v in registry.items()
        if not v["dominated"]
    }

    #w = [1.0,1.0,1.0,.1]

    #pareto_registry = dict(
    #    sorted(pareto_registry.items(), 
    #           key=lambda x: 
    #                +w[0]*x[1]['fitness'][0]
    #                -w[1]*x[1]['fitness'][1]
    #                +w[2]*x[1]['fitness'][2]
    #                +w[3]*x[1]['fitness'][3],
    #            reverse=False)
    #)
    
    pareto_registry = dict(
        sorted(pareto_registry.items(), key=lambda x: x[1]['count'],reverse=True))

    headers = ["ID", "#MS", "CNT", "NED", "SM", "ICP", "IFN", "PARTITION"]
    rows = []

    for k, v in pareto_registry.items():
        rows.append([
            k,
            v["n_services"],
            v["count"],
            f"{v['fitness'][0]:.3f}",
            f"{v['fitness'][1]:.4f}",
            f"{v['fitness'][2]:.4f}",
            f"{v['fitness'][3]:.4f}",
        ])

    table = tabulate(rows, headers=headers, tablefmt="github")

    with open("old_pareto_registry_table.txt", "w") as f:
        f.write(table)

    with open("old_pareto_registry.pkl", "wb") as f:
        pickle.dump(pareto_registry, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("old_pareto_registry.pkl", "rb") as f:
        registry = pickle.load(f)

    print(np.mean([v['fitness'][0] for _, v in registry.items()]))
    print(np.mean([v['fitness'][1] for _, v in registry.items()]))
    print(np.mean([v['fitness'][2] for _, v in registry.items()]))
    print(np.mean([v['fitness'][3] for _, v in registry.items()]))


    with open("old_pareto_registry.pkl", "rb") as f:
        old_registry = pickle.load(f)

    with open("pareto_registry.pkl", "rb") as f:
        new_registry = pickle.load(f)

    print(coverage(new_registry.keys(), old_registry.keys()))
    print(coverage(old_registry.keys(), new_registry.keys()))
    """
"""
    import pandas as pd

    with open("redundancy/data/Y_old.csv", "rb") as f:
        old_Y = pd.read_csv(f)

    with open("redundancy/data/Y_new.csv", "rb") as f:
        new_Y = pd.read_csv(f)

    print(np.min(old_Y['HV']))
    print(np.min(new_Y['HV']))

    print(np.max(old_Y['HV']))
    print(np.max(new_Y['HV']))

    print(np.mean(old_Y['HV']))
    print(np.mean(new_Y['HV']))

    print(np.std(old_Y['HV']))
    print(np.std(new_Y['HV']))


    hv_old = old_Y['HV']
    hv_new = new_Y['HV']

    from scipy.stats import mannwhitneyu

    stat, p = mannwhitneyu(hv_old, hv_new, alternative="two-sided")
    print(p)

    import matplotlib.pyplot as plt
    import numpy as np



    import matplotlib.pyplot as plt

    plt.figure()

    data = [hv_old, hv_new]

    parts = plt.violinplot(data, showmeans=True, showmedians=False)

    colors = ["red", "blue"]

    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])

    plt.xticks([1, 2], ["Old", "New"])
    plt.ylabel("Hypervolume")
    plt.title("HV Distribution")

    plt.show()


    import matplotlib.pyplot as plt

    plt.figure()

    plt.hist(hv_old, bins=20, alpha=0.6, label="Old", color="red")
    plt.hist(hv_new, bins=20, alpha=0.6, label="New", color="blue")

    plt.xlabel("Hypervolume")
    plt.ylabel("Frequency")
    plt.title("HV Distribution: Old vs New")
    plt.legend()

    plt.show()
    import matplotlib.pyplot as plt

    plt.figure()

    plt.violinplot([hv_old, hv_new], showmeans=True)

    plt.xticks([1, 2], ["Old", "New"])
    plt.ylabel("Hypervolume")
    plt.title("HV Distribution")

    plt.show()



    import matplotlib.pyplot as plt

    plt.figure()

    plt.boxplot([hv_old, hv_new], labels=["Old", "New"])
    plt.ylabel("Hypervolume")
    plt.title("HV Comparison")

    plt.show()

    import numpy as np
    import matplotlib.pyplot as plt

    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x)+1) / len(x)
        return x, y

    x_old, y_old = ecdf(hv_old)
    x_new, y_new = ecdf(hv_new)

    plt.figure()
    plt.plot(x_old, y_old, label="Old", color="red")
    plt.plot(x_new, y_new, label="New", color="blue")

    plt.xlabel("Hypervolume")
    plt.ylabel("ECDF")
    plt.title("ECDF of Hypervolume")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()



"""

import pandas as pd
import json

with open("data/PF.json.csv", "rb") as f:
    old_Y = json.load(f)