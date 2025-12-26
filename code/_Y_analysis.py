import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
from SALib.analyze import sobol
from code.analysis._sensitivity import PROBLEM

# lectura y escritura de datos en formato .csv
ORIG_OUTPUT_PATH = "Y_results.csv"
DEST_OUTPUT_PATH = "Y.csv"
OUTPUT_CONFIG = {'drop_errors': False, 'actual_params': False, 'save_to_csv': True}

def parse_params(s):
    s = s.strip("[]")         # saca corchetes
    return np.fromstring(s, sep=" ")  # separa por espacios

def get_output(orig_path=ORIG_OUTPUT_PATH,dest_path=DEST_OUTPUT_PATH,config=OUTPUT_CONFIG):

    df = pd.read_csv(orig_path)
    df.columns = ['parameters','HV','index', 'pareto front']
    df = df.drop(columns=['index','pareto front'])

    parameters = df['parameters'].apply(parse_params)
    df['mu'] = parameters.apply(lambda t:       t[0] if not config['actual_params'] else int(t[0]))
    df['lambda'] = parameters.apply(lambda t:   t[1] if not config['actual_params'] else int(t[1]*t[0]))
    df['p_mut'] = parameters.apply(lambda t:    t[2])
    df['p_cx'] = parameters.apply(lambda t:     1-t[2])
    df.drop(columns=['parameters'],inplace=True)
    df = df.reindex(['mu','lambda','p_mut','p_cx','HV'],axis=1)
    if not config['actual_params']:
        df.drop(columns=['p_cx'],inplace=True)
        df.rename(columns={'p_mut': 'prob'},inplace=True)
    if config['drop_errors']:
        df = df[df['HV']!=0]
    if config['save_to_csv']:
        df.to_csv(DEST_OUTPUT_PATH)

    return df

###### ANALISIS ######

# 3.0. Visualización de resultados

def plot_output_3d(df,only_errors=False):

    mu_range = np.linspace(df["mu"].min(), df["mu"].max(), 60)
    lambda_range = np.linspace(df["lambda"].min(), df["lambda"].max(), 60)
    prob_range = np.linspace(df["prob"].min(), df["prob"].max(), 60)
    MU, LAM, PROB = np.meshgrid(mu_range, lambda_range, prob_range)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if not only_errors:
        sc = ax.scatter(df["mu"], df["lambda"], df["prob"],c=df["HV"],cmap='RdYlGn',s=30,alpha=0.9,label="Éxitos")
    df_errors = df[df["HV"] == 0]
    ax.scatter(df_errors["mu"], df_errors["lambda"], df_errors["prob"],color="black",s=25,alpha=0.9,label="Errores")

    ax.set_xlabel("mu")
    ax.set_ylabel("lambda")
    ax.set_zlabel("prob")
    ax.set_title("Región conflictiva en el espacio de hiperparámetros")
    ax.legend()

    plt.show()

def plot_histogram_hv(df):
    from scipy.stats import gaussian_kde
    y=df['HV']
    plt.figure(figsize=(8,5))
    plt.hist(y, bins=30, alpha=0.5, density=True)
    kde = gaussian_kde(y)
    x = np.linspace(0,1, 300)
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

# 3.1. Errores internos de NSGA-III

def describe_errors(df):
    errors = df[df['HV']==0]       # se conservan solo instancias de error
    errors = errors.drop(columns=['HV'])
    errors_stats = errors.describe()      # se extraen qué valores tipicamente producen errores
    print("\nEstadísticas de los errores:")
    print(f"Tasa de errores: {round((len(errors)/df.shape[0])*100,3)}% -- ({len(errors)} ocurrencias)")
    print(errors_stats,"\n")

# 3.2/3.3 Distribución de HV - Relaciones individuales de hiperparámetros con HV

def describe_elite(df):
    q90 = df['HV'].quantile(0.90)
    elite = df[df['HV'] >= q90]
    elite_stats = elite.describe()
    print("\nEstadísticas del top 10%:")
    print(elite_stats,"\n")

if __name__=="__main__":
    
    output = get_output()   # obtiene resultados de las pruebas de sensibility.py
    y = output['HV']

    mu_values = np.array(output['mu'])
    lambda_values = np.array(output['lambda'])
    prob_values = np.array(output['prob'])

    #plot_output_3d(output,False)
    #plot_histogram_hv(output)
    #scatterplot('mu',mu_values,y)
    #scatterplot('lambda',lambda_values,y)
    #scatterplot('prob de mutación',prob_values,y)

    describe_errors(output)
    describe_elite(output)

    #plot_histogram_hv(output)

    # 3.4. Análisis de sensibilidad

    y = y.to_numpy()
    Si = sobol.analyze(PROBLEM, y, calc_second_order=False)
    print(Si)