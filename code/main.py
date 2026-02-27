import pickle as pkl
import numpy as np


path = "registry.pkl"

with open (path,"rb") as registry:
    data = pkl.load(registry)



rows = []
for k,v in data.items():
    rows.append(v["fitness"])

cols = np.array(rows).T
print(cols)

print(np.min(cols[0]))  # NED
print(np.max(cols[0]))  # NED

print(np.min(cols[1]))  # NED
print(np.max(cols[1]))  # NED

print(np.min(cols[2]))  # NED
print(np.max(cols[2]))  # NED

print(np.min(cols[3]))  # NED
print(np.max(cols[3]))  # NED