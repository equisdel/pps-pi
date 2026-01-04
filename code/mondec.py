# Librerias Estandar
import pickle
import random
import time
from collections import defaultdict
# Librerias Externas
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from pymoo.indicators.hv import HV
from deap import base, creator, tools, algorithms
# Modulos Locales
from metrics import sm, ifn, ned, icp, hv
from plots import (
    plot_evolution,
    pareto_front_3d,
    plot_pareto_front,
    plot_radar_chart,
    plot_parallel_coordinates,
)

# CONFIG

DEMO = True
GRAPH_FILENAME = "monoliths/cargo/graph.pkl"
METADATA = "monoliths/cargo/metadata.json"

POP_SIZE = 200
DEFAULT = {     # configuración por defecto
    "pop_size": POP_SIZE,
    "num_generations": 100,
    "hof_size": 10,
    "mu": POP_SIZE,             # mu es el tamaño de la población
    "lambda": POP_SIZE*1.5,     # lambda es proporcional al tamaño de la población
    "mut_prob": 0.9,            # mut_prob es complemento de cx_prob (suman 1.0)
    "cx_prob": 0.1,
    "proportional_NED": True,   # modificación #1 
    "new_representation": True, # modificación #2
}

with open(GRAPH_FILENAME, 'rb') as file:
    graph = pickle.load(file)
    nodes_to_remove = [node for node in graph.nodes if 'test' in node.lower() or 'transition' in node.lower()]
    graph.remove_nodes_from(nodes_to_remove)

# future work: pre-calentamiento con Montecarlo
MINS = [0.0,    0.0,    0.0,    0.0] #[0.0,0.0,0.0,0.4] - [ 0.0, 0.0, 0.3, 0.6]    # Para normalización
MAXS = [1.0,    0.5838, 0.7827, 2.5]   #[1.0,0.7542,0.7826087,5.0]#[ 1.7, 0.7, 0.8, 2.0]    # Para normalización

N_CLASSES = len(graph.nodes) 
MAX_MICROSERVICES = N_CLASSES   # máxima cantidad de bins: 24 (caso extremo, una clase por microservicio)
CLASS_MAPPING = {i: node for i, node in enumerate(graph.nodes)}

P = 12      # que es?
N_OBJECTIVES = 4              
OBJECTIVES = {
    0: 'NED',
    1: 'SM',
    2: 'ICP',
    3: 'IN',
}

time.sleep(10)

# _______________

# parametrizar "mapper", porque en cargo es "repository". esta ligado a la instancia.
def validate_mapper_constraint(individual):
    microservice_to_tables = defaultdict(set)
    mapper_classes = {i for i, cls in CLASS_MAPPING.items() if "repository" in cls.lower() and not "test" in cls.lower()}
    for class_idx, microservice_id in enumerate(individual):
        if class_idx in mapper_classes:
            microservice_to_tables[microservice_id].add(class_idx)
    table_to_microservices = defaultdict(set)
    for microservice_id, tables in microservice_to_tables.items():
        for table in tables:
            table_to_microservices[table].add(microservice_id)
    for microservices in table_to_microservices.values():
        if len(microservices) > 1:
            return False
    return True

def mutate_class_assignment(individual):
    idx = random.randint(0, N_CLASSES - 1)
    individual[idx] = random.randint(0, MAX_MICROSERVICES - 1)
    return individual,

def normalize(fitness):
    return [(v - mi)/(ma - mi + 1e-9) for v, mi, ma in zip(fitness, MINS, MAXS)]

def denormalize(fitness):
    return [v * (ma - mi) + mi for v, mi, ma in zip(fitness, MINS, MAXS)]


def evaluate(individual):

    partitions = individual_to_microservices(individual)    # pasa de lista a diccionario

    ned_value = ned(partitions, N_CLASSES if DEFAULT["proportional_NED"] else None)
    sm_value  = sm(partitions, graph)
    icp_value = icp(partitions, graph)
    in_value  = ifn(partitions, graph)
    
    if not validate_mapper_constraint(individual):
        return tuple([10000,-10000,10000,10000])     # Eliminar el individuo directamente (comentario original)

    values = [ned_value, sm_value, icp_value, in_value]

    return tuple(values)

def individual_to_microservices(individual):
    partitions = defaultdict(list)
    for class_id, microservice_id in enumerate(individual):
        partitions[microservice_id].append(CLASS_MAPPING.get(class_id))

    old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(sorted(partitions.keys()))}
    sequential_partitions = {old_to_new_id[old_id]: sorted(classes) for old_id, classes in partitions.items()}
    return sequential_partitions


class Partition:
    def __init__(self, blocks):
        self.blocks = frozenset(
            frozenset(b) for b in blocks if b
        )

    def copy(self):
        return Partition(self.blocks)

    def __hash__(self):
        return hash(self.blocks)

    def __eq__(self, other):
        return isinstance(other, Partition) and self.blocks == other.blocks


def partition_to_microservices(partition: Partition):
    partitions = defaultdict(list)

    for ms_id, block in enumerate(partition.blocks):
        for class_id in block:
            partitions[ms_id].append(CLASS_MAPPING[class_id])

    return partitions

def init_partition(n_classes, k_range=(2,10)):
    k = random.randint(*k_range)
    blocks = [[] for _ in range(k)]

    for i in range(n_classes):
        blocks[random.randrange(k)].append(i)

    return Partition(blocks)

def mut_move(individual):
    blocks = [set(b) for b in individual.blocks]

    src = random.choice([b for b in blocks if len(b) > 1])
    dst = random.choice(blocks)

    cls = random.choice(tuple(src))
    src.remove(cls)
    dst.add(cls)

    return creator.Individual(blocks),


def cx_blocks(p1, p2):
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


def new_evaluate(individual: Partition):
    partitions = partition_to_microservices(individual)

    ned_value = ned(partitions, N_CLASSES if DEFAULT["proportional_NED"] else None)
    sm_value  = sm(partitions, graph)
    icp_value = icp(partitions, graph)
    in_value  = ifn(partitions, graph)

    values = (ned_value, sm_value, icp_value, in_value)
    return values

def configure_nsga_iii(pop_size=100):
    # Maximized SM, minimized IN, minimized NED, minimized ICP
    if not DEFAULT["new_representation"]:
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, +1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 0, MAX_MICROSERVICES - 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=N_CLASSES)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)   # tamaño de la población: 100
        toolbox.register("mate", tools.cxTwoPoint)           # cruzamiento!
        toolbox.register("mutate", mutate_class_assignment)  # mutación!
        ref_points = tools.uniform_reference_points(nobj=N_OBJECTIVES, p=P)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points) #corregir en el paper
        toolbox.register("evaluate", evaluate)
        return toolbox
    else:
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, +1.0, -1.0, -1.0))
        creator.create("Individual", Partition, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 0, MAX_MICROSERVICES - 1)
        toolbox.register(
            "individual",
            lambda: creator.Individual(init_partition(N_CLASSES).blocks)
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
            n=pop_size
        )

        toolbox.register(
            "individual",
            lambda: creator.Individual(init_partition(N_CLASSES).blocks)
        )

        toolbox.register("mutate", mut_move)
        toolbox.register("mate", cx_blocks)
        ref_points = tools.uniform_reference_points(nobj=N_OBJECTIVES, p=P)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points) #corregir en el paper
        toolbox.register("evaluate", new_evaluate)
        return toolbox


def normalize_fitness(obj_id, obj_value):
    n_obj_value = obj_value-MINS[obj_id]/MAXS[obj_id]-MINS[obj_id]
    #print("obj_id,min,max,value,normalized_value:  ",obj_id,MINS[obj_id],MAXS[obj_id],obj_value,n_obj_value)
    return (obj_value-MINS[obj_id])/(MAXS[obj_id]-MINS[obj_id])

def normalized(fitness):
    return [normalize_fitness(i,f) for i,f in enumerate(fitness)]

def calculate_hv(pareto_front):
    
    normalized_front = np.array([normalized(ind.fitness.values) for ind in pareto_front])

    weights = (-1.0, +1.0, -1.0, -1.0)
    front_min = normalized_front.copy()

    for i, w in enumerate(weights):
        if w > 0:            # SM es un objetivo originalmente maximizado
            front_min[:, i] = -front_min[:, i]

    ref_point = np.max(front_min, axis=0) * 1.1  
    hv = HV(ref_point=ref_point)
    hv_value = hv(front_min)

    return hv_value


def run_ea(seed=None, parameters = {}):
    random.seed(seed)
    # inicialización del algoritmo genético: operadores, individuos y población inicial
    toolbox = configure_nsga_iii(int(parameters["pop_size"]))  
    # preparación de los datos a extraer/visualizar
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    population = toolbox.population()  # Population size
    hof = tools.HallOfFame(int(parameters["hof_size"]))
    # configuración adicional del algoritmo genético
    num_generations = int(parameters["num_generations"])
    mut_prob = parameters["mut_prob"] 
    cx_prob = parameters["cx_prob"]  

    # ejecución
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=int(parameters["mu"]), lambda_=int(parameters["lambda"]), cxpb=cx_prob,
                                                    mutpb=mut_prob,
                                                    ngen=num_generations, stats=stats, halloffame=hof, verbose=False)

    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    hv = calculate_hv(pareto_front)

    return population, logbook, hof, pareto_front, hv



if __name__ == "__main__":

    #seeds = [42,12,23,1,79,99,52,56,54,77,40,10,20,10,70,90,50,6,4,7]
    seeds = [42]
    # Ejecutar 20 corridas (comentario original)
    
    for i,s in enumerate(seeds):
        print("EJECUCION (",i,"/20).\nSEMILLA:",s)

        pop, logbook, hof, pareto_front, hv_ = run_ea(23,DEFAULT)

        #print("Densidad del frente de pareto: ",(len(pareto_front)/len(pop))*100,"%")
    
        # Una vez finalizada la ejecución:
        pop_fit = np.array([ind.fitness.values for ind in pop])
        pareto_solutions = [ind.fitness.values for ind in pareto_front]
        objectives = list(zip(*pareto_solutions))  # Now objectives[0] = all SM, [1] = IN, etc.
        medians = [np.mean(obj) for obj in objectives]
        print(medians)
        best_decomposition = tools.selBest(pop, k=1)[0]
        bd_partitions = partition_to_microservices(best_decomposition) if DEFAULT["new_representation"] else individual_to_microservices(best_decomposition)

        print("BEST FROM PARETO FRONT")
        print(best_decomposition.fitness)
        print(len(bd_partitions))
        print(bd_partitions)

        print("BEST FROM HOF")
        best_decomposition = hof[0]
        bd_partitions = partition_to_microservices(best_decomposition) if DEFAULT["new_representation"] else individual_to_microservices(best_decomposition)
        print(best_decomposition.fitness)
        print(len(bd_partitions))
        print(bd_partitions)

        generations = logbook.select("gen")
        avg = np.array(logbook.select("avg"))
        min_ = np.array(logbook.select("min"))
        max_ = np.array(logbook.select("max"))

        # gráficos con respecto a nuestro método aisladamente
        plot_evolution(generations, avg, min_, max_)
        plot_pareto_front(pareto_front)
        #pareto_front_3d(pareto_front)
        # comparación con otros métodos
        methods = ['M2M', 'FoSCI', 'CoGCN', 'Bunch', 'MEM']
        objectives = list(OBJECTIVES.values())
        scores = [
            [0.257, 0.054, 0.333, 1.857],   # M2M
            [0.516, 0.044, 0.478, 3.75],    # FoSCI
            [0.392, 0.091, 0.582, 2.533],   # CoGCN
            [0.667, np.nan, 0.477, 7.948],  # Bunch
            [1.0, 0.124, 0.434, 3.429]     # MEM
        ]
        methods.append("Our approach")
        scores.append(medians)
        #plot_radar_chart(methods, scores, objectives)
        plot_parallel_coordinates(methods, scores, objectives)
        cp = dict(population=pop, pareto_front=pareto_front, halloffame=hof,
                logbook=logbook, rndstate=random.getstate())

        print(hv_)

        with open("experiment_database.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)
