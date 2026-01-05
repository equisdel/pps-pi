import pickle

INSTANCE = "jpetstore"
GRAPH_FILENAME = f"monoliths/{INSTANCE}/graph.pkl"
METADATA = f"monoliths/{INSTANCE}/metadata.json"

with open(GRAPH_FILENAME, 'rb') as file:
    graph = pickle.load(file)

nodes_to_remove = [node for node in graph.nodes if 'test' in node.lower() or 'transition' in node.lower()]
graph.remove_nodes_from(nodes_to_remove)

N_CLASSES = len(graph.nodes)
CLASS_MAPPING = {i: node for i, node in enumerate(graph.nodes)}
MAX_MICROSERVICES = N_CLASSES
