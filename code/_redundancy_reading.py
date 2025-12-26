import pandas as pd

df = pd.read_csv("redundancy.txt")

import numpy as np

print(np.mean(df["red_dup"]))
print(np.mean(df["red_sim"]))
print(np.mean(df["red_total"]))
print(np.std(df["red_total"]))