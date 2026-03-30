from importlib import import_module
from algorithm.config import DEFAULT 


module_name = "algorithm.representations.canonical" if DEFAULT["new_representation"] else "algorithm.representations.original"
_module = import_module(module_name)
Individual = _module.Individual
init_individual = _module.init_individual
individual_to_microservices = _module.individual_to_microservices
evaluate = _module.evaluate
mutate = _module.mutate
mate = _module.mate
clone = _module.Individual.copy
