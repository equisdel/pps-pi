import copy
import json
import os
import pickle

import numpy as np

# EVOLUTIVE ALGORITHM

POP_SIZE = 200

_DEFAULT_TEMPLATE = {           # configuracion por defecto
    "pop_size": POP_SIZE,
    "num_generations": 300,
    "hof_size": 10,
    "mu": POP_SIZE,             # mu es el tamano de la poblacion
    "lambda": POP_SIZE * 1.5,   # lambda es proporcional al tamano de la poblacion
    "mut_prob": 0.0,            # mut_prob es complemento de cx_prob (suman 1.0)
    "cx_prob": 1.0,
    "proportional_NED": True,   # modificacion #1
    "new_representation": True,# modificacion #2
    "MC_samples": 10000000,
    "seed": 42,
}
DEFAULT = copy.deepcopy(_DEFAULT_TEMPLATE)

P = 12
M_OBJECTIVES = 4
OBJECTIVES = {0: "NED", 1: "SM", 2: "ICP", 3: "IN"}

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSTANCE = None
GRAPH_FILENAME = None
METADATA = None
graph = None
N_CLASSES = 0
CLASS_MAPPING = {}
MAX_MICROSERVICES = 0


def _instance_paths(instance):
    return (
        os.path.join(_base_dir, f"monoliths/{instance}/graph.pkl"),
        os.path.join(_base_dir, f"monoliths/{instance}/metadata.json"),
    )


def _load_graph(instance):
    graph_filename, metadata = _instance_paths(instance)
    with open(graph_filename, "rb") as file:
        loaded_graph = pickle.load(file)

    nodes_to_remove = [
        node for node in loaded_graph.nodes
        if "test" in node.lower() or "transition" in node.lower()
    ]
    loaded_graph.remove_nodes_from(nodes_to_remove)
    return graph_filename, metadata, loaded_graph


def configure_runtime(instance="jpetstore", default_overrides=None):
    global INSTANCE, GRAPH_FILENAME, METADATA, graph
    global N_CLASSES, CLASS_MAPPING, MAX_MICROSERVICES

    INSTANCE = instance
    GRAPH_FILENAME, METADATA, graph = _load_graph(instance)

    nodes = sorted(graph.nodes)
    N_CLASSES = len(graph.nodes)
    CLASS_MAPPING = dict(enumerate(nodes))
    MAX_MICROSERVICES = N_CLASSES

    DEFAULT.clear()
    DEFAULT.update(copy.deepcopy(_DEFAULT_TEMPLATE))
    if default_overrides:
        DEFAULT.update(default_overrides)


configure_runtime()


def load_range_from_metadata():
    try:
        with open(METADATA, "r") as f:
            metadata = json.load(f)
        order = list(OBJECTIVES.values())
        mins = [metadata["RANGE"]["MIN"][k] for k in order]
        maxs = [metadata["RANGE"]["MAX"][k] for k in order]
        return np.array(mins), np.array(maxs)
    except Exception as e:
        print("ERROR: no range or no metadata file for this instance:", e)
        return np.zeros(M_OBJECTIVES), np.zeros(M_OBJECTIVES)
