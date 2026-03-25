import pickle
import json
import numpy as np
import os

# EVOLUTIVE ALGORITHM

POP_SIZE = 200

DEFAULT = {                     # configuración por defecto
    "pop_size": POP_SIZE,
    "num_generations": 300,
    "hof_size": 10,
    "mu": POP_SIZE,             # mu es el tamaño de la población
    "lambda": POP_SIZE*1.5,     # lambda es proporcional al tamaño de la población
    "mut_prob": 0.0,            # mut_prob es complemento de cx_prob (suman 1.0)
    "cx_prob": 1.0,
    "proportional_NED": True,    # modificación #1 
    "new_representation": False, # modificación #2
    "MC_samples": 10000000,
    "seed": 42
}

P = 12 
M_OBJECTIVES = 4    
OBJECTIVES = {0: 'NED',1: 'SM',2: 'ICP',3: 'IN',}

# INSTANCE

INSTANCE = "cargo"
# Resolve paths correctly (works from any subdirectory)
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Navigate to root code directory
GRAPH_FILENAME = os.path.join(_base_dir, f"monoliths/{INSTANCE}/graph.pkl")
METADATA = os.path.join(_base_dir, f"monoliths/{INSTANCE}/metadata.json")

with open(GRAPH_FILENAME, 'rb') as file:
    graph = pickle.load(file)

nodes_to_remove = [node for node in graph.nodes if 'test' in node.lower() or 'transition' in node.lower()]
graph.remove_nodes_from(nodes_to_remove)

nodes = sorted(graph.nodes)
N_CLASSES = len(graph.nodes)
CLASS_MAPPING = dict(enumerate(nodes))
MAX_MICROSERVICES = N_CLASSES

def load_range_from_metadata():
    try:
        with open(METADATA, "r") as f:
            metadata = json.load(f)
        order = list(OBJECTIVES.values())
        MINS = [metadata["RANGE"]["MIN"][k] for k in order]
        MAXS = [metadata["RANGE"]["MAX"][k] for k in order]
        return np.array(MINS), np.array(MAXS)
    except Exception as e:
        print("ERROR: no range or no metadata file for this instance:", e)
        return np.zeros(M_OBJECTIVES), np.zeros(M_OBJECTIVES)
