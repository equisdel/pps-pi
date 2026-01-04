import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#df.sort_values(by=["HV"],axis=0,ascending=False,inplace=True)
#df.sort_values(by=["mu"],axis=0,ascending=True,inplace=True)
#df.sort_values(by=["mu"],axis=0,ascending=True,inplace=True)
#print(df.loc[[df['HV'].idxmax()]])      # highest HV
#print(df)


# 3.3. Visualizaciones separadas


def scatterplot(label="",x_values=[],y_values=[]):
    # ordenar puntos por mu
    order = np.argsort(x_values)
    x_sorted = x_values[order]
    y_sorted = y_values[order]
    # suavizado
    y_smooth = uniform_filter1d(y_sorted, size=100)
    # graficar
    plt.scatter(x_sorted, y_sorted, alpha=0.4)
    plt.plot(x_sorted, y_smooth, color='red', linewidth=2, label="Promedio suavizado")
    plt.xlabel(label)
    plt.ylabel("HV")
    plt.title(f"{label} vs HV con curva suavizada")
    plt.legend()
    plt.show()

mu_values = np.array(df['mu'])
lambda_values = np.array(df['lambda'])
prob_values = np.array(df['prob'])

scatterplot('mu',mu_values,hv_values)
scatterplot('lambda',lambda_values,hv_values)
scatterplot('prob de mutación',prob_values,hv_values)
"""
"""

plt.scatter(df["mu"], df["lambda"]*df["mu"], c=(df["HV"]), cmap=plt.get_cmap('RdYlGn'),alpha=0.1)
plt.xlabel("mu")
plt.ylabel("lambda")
plt.title("Regiones donde HV = 0")
plt.show()

# cargar datos
df = pd.read_csv("C:/Users/Usuario/Documents/Personal/PPS+PI/side_quest/data.csv")
df.columns = ["parameters","HV","index","pareto front"]
df = df.drop(columns=['pareto front','index'])

# convertir parámetros de string a float arrays
"""
"""
mu = mu_values
lam = lambda_values
prob = prob_values
hv = hv_values
print(len(mu),len(lam),len(prob),len(hv_values))
# normalizar HV
hv_norm = np.zeros_like(hv)
mask_nonzero = hv > 0
hv_norm[mask_nonzero] = (hv[mask_nonzero] - hv[mask_nonzero].min()) / (hv[mask_nonzero].max() - hv[mask_nonzero].min())

# generar colores RGBA
cmap = plt.get_cmap('RdYlGn')
point_colors = np.array([cmap(h) if h>0 else (0,0,0,1) for h in hv_norm])  # negro para HV=0

# plot 3D
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(mu, lam, prob, c=point_colors, s=50, depthshade=True)
ax.set_xlabel("mu")
ax.set_ylabel("lambda")
ax.set_zlabel("prob")
ax.set_title("HV según parámetros (rojo=peor, verde=mejor, negro=error)")
plt.show()
"""

from SALib.analyze import sobol

problem = {
    'num_vars': 3,
    'names': ['mu', 'lambda', 'prob'],
    'bounds': [[50, 1000],  # tamaño poblacional
               [0.5,3],     # proporcional a mu
               [0.0,1.0]]   # punto de probabilidad entre cruce y mutacion
}

#Si = sobol.analyze(problem, hv_values, calc_second_order=False)
#print(Si)