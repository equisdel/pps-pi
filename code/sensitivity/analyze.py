import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
from SALib.analyze import sobol
from sensitivity.saltelli import PROBLEM
from sensitivity.paths import Y_OUTPUT_PATH

Y_OUTPUT_PATH = "redundancy/data/Y_new_rep_recomputed.csv"

def parse_params(s):
    s = s.strip("[]")         # saca corchetes
    return np.fromstring(s, sep=" ")  # separa por espacios

def get_output(orig_path=Y_OUTPUT_PATH):

    df = pd.read_csv(orig_path)
    df = df.drop(columns=['id',"HV_old"])

    return df

## plots

def plot_output_3d(df):

    mu_range = np.linspace(df["mu"].min(), df["mu"].max(), 60)
    lambda_range = np.linspace(df["lambda"].min()/df["mu"].min(), df["lambda"].max()/df["mu"].max(), 60)
    prob_range = np.linspace(df["mut_prob"].min(), df["mut_prob"].max(), 60)
    MU, LAM, PROB = np.meshgrid(mu_range, lambda_range, prob_range)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df["mu"], df["lambda"]/df["mu"], df["mut_prob"],c=df["HV"],cmap='RdYlGn',s=30,alpha=0.9,label="Éxitos")
    ax.set_xlabel("mu")
    ax.set_ylabel("lambda")
    ax.set_zlabel("mut_prob")

    plt.show()

def plot_histogram_hv(df):
    from scipy.stats import gaussian_kde
    y=df['HV']
    plt.figure(figsize=(8,5))
    plt.hist(y, bins=30, alpha=0.5, density=True)
    kde = gaussian_kde(y)
    x = np.linspace(y.min(),y.max(), 300)
    plt.plot(x, kde(x))
    mean = np.mean(y)
    std = np.std(y)
    plt.axvline(mean, color='black', linestyle='--', label='Media')
    plt.axvline(mean + std, color='grey', linestyle=':', label='Media ±1σ')
    plt.axvline(mean - std, color='grey', linestyle=':')
    plt.legend()
    plt.xlabel("HV")
    plt.ylabel("Densidad")
    plt.title("Distribución del HV con media y varianza")
    plt.tight_layout()
    plt.show()

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

# 3.2/3.3 Distribución de HV - Relaciones individuales de hiperparámetros con HV

def describe_elite(df):
    q99 = df['HV'].quantile(0.99)
    elite = df[df['HV'] >= q99]
    elite_stats = elite.describe()
    print("\nEstadísticas del top 1%:")
    print(elite_stats,"\n")

if __name__=="__main__":
    
    output = get_output()   # obtiene resultados de las pruebas de sensibility.py
    y = output['HV']

    mu_values = np.array(output['mu'])
    lambda_values = np.array(output['lambda']/output["mu"])
    prob_values = np.array(output['mut_prob'])

    plot_output_3d(output)
    plot_histogram_hv(output)
    scatterplot('mu',mu_values,y)
    scatterplot('lambda',lambda_values,y)
    scatterplot('prob de mutación',prob_values,y)

    describe_elite(output)

    # 3.4. Análisis de sensibilidad
    y = y.to_numpy()
    Si = sobol.analyze(PROBLEM, y, calc_second_order=False)
    print(Si)
    print(np.mean(output["HV"]))
