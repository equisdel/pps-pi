from mondec.config_ea import *
from mondec.config_instance import *
from mondec.metrics import *
from mondec.representations import individual_to_microservices
import json

with open(METADATA, "r") as f:
    data = json.load(f)
    MAPPER_KEYWORD = data["KEYWORD"]

def validate_mapper_constraint(individual):
    microservice_to_tables = defaultdict(set)
    mapper_classes = {i for i, cls in CLASS_MAPPING.items() if MAPPER_KEYWORD in cls.lower() and not "test" in cls.lower()}
    for class_idx, microservice_id in enumerate(individual):
        if class_idx in mapper_classes:
            microservice_to_tables[microservice_id].add(class_idx)
    table_to_microservices = defaultdict(set)
    for microservice_id, tables in microservice_to_tables.items():
        for table in tables:
            table_to_microservices[table].add(microservice_id)
    for microservices in table_to_microservices.values():
        if len(microservices) > 1:
            with open("constraint_violation.txt", "w") as f:  # registra individuos que violan constraint
                f.write("!\n")                
            return False
    return True

def evaluate(individual):       # ya llega traducido a diccionario

    partitions = individual_to_microservices(individual)

    ned_value = ned(partitions, N_CLASSES if DEFAULT["proportional_NED"] else None)
    sm_value  = sm(partitions, graph)
    icp_value = icp(partitions, graph)
    in_value  = ifn(partitions, graph)

    if not validate_mapper_constraint(partitions):
        return tuple([10000,-10000,10000,10000])     # Eliminar el individuo directamente (comentario original)

    values = (ned_value, sm_value, icp_value, in_value)
    return values