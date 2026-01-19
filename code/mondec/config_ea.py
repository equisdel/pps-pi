POP_SIZE = 200

DEFAULT = {                     # configuración por defecto
    "pop_size": POP_SIZE,
    "num_generations": 100,
    "hof_size": 10,
    "mu": POP_SIZE,             # mu es el tamaño de la población
    "lambda": POP_SIZE*1.5,     # lambda es proporcional al tamaño de la población
    "mut_prob": 0.0,            # mut_prob es complemento de cx_prob (suman 1.0)
    "cx_prob": 1.0,
    "proportional_NED": True,   # modificación #1 
    "new_representation": False, # modificación #2
    "MC_samples": 10000000
}

P = 12      # que es?

N_OBJECTIVES = 4    

OBJECTIVES = {
    0: 'NED',
    1: 'SM',
    2: 'ICP',
    3: 'IN',
}