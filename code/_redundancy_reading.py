import pandas as pd

df = pd.read_csv("redundancy.txt")

import numpy as np

print(np.mean(df["red_dup"]))
print(np.mean(df["red_sim"]))
print(np.mean(df["red_total"]))
print(np.std(df["red_total"]))

print(df.head(10))

import matplotlib.pyplot as plt
#print(df["red_total"])
#print(np.array(df["red_total"]))


mean = np.mean(df["red_total"])
plt.hist(x=np.array(df["red_total"]),range=[0,1],edgecolor="black",alpha=0.7)
plt.ylabel("Ocurrencias")
plt.xlabel("Redundancia")
plt.title("Histograma de Redundancia")
plt.annotate(
    f"Promedio: {mean:.4f}",
    xy=(mean, 150),          # punto al que apunta (ajusta Y según tu histograma)
    xytext=(mean+0.05, 180), # posición del cartel
    arrowprops=dict(arrowstyle="->", color="red"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red"),
    color="red"
)
plt.axvline(np.mean(df["red_total"]), color='red', linestyle='--', linewidth=2, label=f'Media: {mean:.4f}')
#plt.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median:.4f}')
plt.show()