import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
from SALib.analyze import sobol
from experiments.sensitivity.run import PROBLEM
from experiments.sensitivity.paths import Y_OUTPUT_PATH
from experiments.sensitivity.sobol_report import print_sobol_report, plot_sobol_indices
from matplotlib.animation import FuncAnimation


def parse_params(s):
    s = s.strip("[]")         # saca corchetes
    return np.fromstring(s, sep=" ")  # separa por espacios

def get_output(orig_path=Y_OUTPUT_PATH):

    df = pd.read_csv(orig_path)
    df = df.drop(columns=['id'])
    #df['lambda'] = df['mu']/df["lambda"]
    return df

## plots

def plot_output_3d(df):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set fixed limits based on full dataset
    ax.set_xlim(df["mu"].min(), df["mu"].max())
    ax.set_ylim((df["lambda"]/df["mu"]).min(), (df["lambda"]/df["mu"]).max())
    ax.set_zlim(df["mut_prob"].min(), df["mut_prob"].max())
    
    ax.set_xlabel("mu")
    ax.set_ylabel("lambda / mu")
    ax.set_zlabel("prob. mut.")
    
    def update(percentile):
        ax.clear()
        # Reset fixed limits and labels
        ax.set_xlim(df["mu"].min(), df["mu"].max())
        ax.set_ylim((df["lambda"]/df["mu"]).min(), (df["lambda"]/df["mu"]).max())
        ax.set_zlim(df["mut_prob"].min(), df["mut_prob"].max())
        ax.set_xlabel("mu")
        ax.set_ylabel("lambda / mu")
        ax.set_zlabel("prob. mut.")
        
        q_high = df['HV'].quantile(1 - percentile/100)
        q_low = df['HV'].quantile(percentile/100)
        
        top = df[df['HV'] >= q_high]
        bottom = df[df['HV'] <= q_low]
        
        ax.scatter(top["mu"], top["lambda"]/top["mu"], top["mut_prob"], c='green', s=30, alpha=0.6)
        ax.scatter(bottom["mu"], bottom["lambda"]/bottom["mu"], bottom["mut_prob"], c='red', s=30, alpha=0.6)
        
        ax.set_title(f"Configuraciones de hiperparámetros (Percentil: {percentile}%)")
        
        return ax,
    
    # Oscillate between 1 and 50, repeated for longer duration
    frames = (list(range(1, 51)) + list(range(50, 0, -1))) * 2  # Repeat twice for ~40 seconds at 5 fps
    ani = FuncAnimation(fig, update, frames=frames, interval=200)
    
    # Save the animation as GIF (requires pillow)
    #ani.save('sensitivity_animation.gif', writer='pillow', fps=5)
    
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
    q95 = df['HV'].quantile(0.99)
    elite = df[df['HV'] >= q95]
    elite_stats = elite.describe()
    print("\nEstadísticas del top 5%:")
    print(elite_stats,"\n")

def describe_bottom(df):
    q5 = df['HV'].quantile(0.01)
    elite = df[df['HV'] <= q5]
    elite_stats = elite.describe()
    print("\nEstadísticas del bottom 5%:")
    print(elite_stats,"\n")

if __name__=="__main__":
    
    output = get_output()   # obtiene resultados de las pruebas de sensibility.py
    y = output['HV']

    mu_values = np.array(output['mu'])
    lambda_values = np.array(output['lambda']/output["mu"])
    prob_values = np.array(output['mut_prob'])


    plot_output_3d(output)
    describe_elite(output)
    describe_bottom(output)

    # 3.4. Análisis de sensibilidad
    y = y.to_numpy()
    Si = sobol.analyze(PROBLEM, y, calc_second_order=False)
    print(Si)
    print_sobol_report(PROBLEM, Si)
    print(np.mean(output["HV"]))
