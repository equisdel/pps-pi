from mondec.config_ea import DEFAULT

def load_representation():

    if DEFAULT["new_representation"]:
        module_name = "mondec.representations.canonical"
    else:
        module_name = "mondec.representations.original"

    # import dinámico
    import importlib
    module = importlib.import_module(module_name)

    return module.Individual, module.init_individual, module.individual_to_microservices, module.evaluate, module.mate, module.mutate
