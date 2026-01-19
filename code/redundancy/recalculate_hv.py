from mondec.initialization import progress_bar
y_path = f"redundancy/data/Y_old_rep.csv"
pf_path = f"redundancy/data/PF_old_rep.json"

import ast
import json
import pandas as pd
from mondec.config_instance import load_range_from_metadata
import numpy as np
from redundancy.auxiliar import fitness_values
from pymoo.indicators.hv import HV


def hv(pareto_front):

    MINS, MAXS = load_range_from_metadata()

    front = np.array([fitness_values(ind) for ind in pareto_front])

    front[:, 1] *= -1           # negación de SM en los individuos del frente ()

    ref_point = np.array([
        MAXS[0],
        -MINS[1],
        MAXS[2],
        MAXS[3]
    ]) * 1.05

    hv = HV(ref_point=ref_point)
    hv_value = hv(front)

    return hv_value


df_y = pd.read_csv(y_path)

with open(pf_path, "r") as f:
    pf_raw = json.load(f)

new_hv = []
    
    
for i, (key, pf_list) in enumerate(pf_raw.items()):
    
    progress_bar(i,len(pf_raw))
    new_hv.append(round(hv(pf_list),4))    

df_y["HV_old"] = df_y["HV"]
df_y["HV"] = new_hv

out_path = "redundancy/data/Y_old_rep_recomputed.csv"
df_y.to_csv(out_path, index=False)

print("\nSaved:", out_path)
print(df_y.head())
