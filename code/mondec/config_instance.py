import pickle
import json
import numpy as np
from mondec.config_ea import N_OBJECTIVES, OBJECTIVES

INSTANCE = "jpetstore"
GRAPH_FILENAME = f"monoliths/{INSTANCE}/graph.pkl"
METADATA = f"monoliths/{INSTANCE}/metadata.json"

with open(GRAPH_FILENAME, 'rb') as file:
    graph = pickle.load(file)

nodes_to_remove = [node for node in graph.nodes if 'test' in node.lower() or 'transition' in node.lower()]
graph.remove_nodes_from(nodes_to_remove)

N_CLASSES = len(graph.nodes)
print(N_CLASSES)
CLASS_MAPPING = {i: node for i, node in enumerate(graph.nodes)}
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
        return np.zeros(N_OBJECTIVES), np.zeros(N_OBJECTIVES)

"""
def modify_metadata(new_data, mode="w"):

    try:
        # Load existing metadata
        try:
            with open(METADATA, "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}

        if mode == "w":
            # Overwrite only the keys in new_data, keep the rest intact
            metadata.update(new_data)
        elif mode == "a":
            # Merge recursively
            def recursive_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        recursive_update(d[k], v)
                    else:
                        d[k] = v
            recursive_update(metadata, new_data)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Save back to file
        with open(METADATA, "w") as f:
            json.dump(metadata, f, indent=4)
        print("Metadata updated successfully.")

    except Exception as e:
        print("Error modifying metadata:", e)

"""
if __name__=="__main__":
    print()

