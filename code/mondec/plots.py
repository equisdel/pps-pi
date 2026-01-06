import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mondec.config_ea import OBJECTIVES


def plot_evolution(generations, avg, min_, max_):
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    axs = axs.ravel()

    for i in range(4):
        axs[i].plot(generations, avg[:, i], label='Avg', color='blue')
        axs[i].fill_between(generations, min_[:, i], max_[:, i], color='blue', alpha=0.2, label='Min-Max')
        axs[i].set_title(f'Objective {OBJECTIVES.get(i)}')
        axs[i].set_xlabel('Generation')
        axs[i].set_ylabel('Fitness')
        axs[i].legend()
        axs[i].grid(True)

    fig.suptitle('Objective Evolution Over Generations')
    #plt.savefig('evolution.png', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def scatterplot_inpaired(population):
    print()

def plot_pareto_front(pareto_front):
    pareto_array = np.array([ind.fitness.values for ind in pareto_front])
    for ind in pareto_array:
        plt.plot(list(OBJECTIVES.keys()), ind, color='red', alpha=0.4)

    # Set custom x-axis tick labels
    plt.xticks(list(OBJECTIVES.keys()), list(OBJECTIVES.values()))
    plt.xlabel("Objectives")
    plt.ylabel("Fitness Value")
    plt.title("Pareto Front (Parallel Coordinates)")
    plt.grid(True)
    plt.legend(["ParetoFront"], loc='upper right')
    plt.tight_layout()
    #plt.savefig('pareto_front.png', bbox_inches='tight')
    plt.show()

"""
def plot_pareto_v2(pareto_front):
    pareto_array = np.array([ind.fitness.values for ind in pareto_front])

    # Convert to DataFrame for easier handling with seaborn
    df = pd.DataFrame(pareto_array, columns=list(OBJECTIVES.values()))

    # Create pairwise scatter plots
    sns.pairplot(df)
    plt.suptitle("Pareto Front (Pairwise Scatter Plots)", y=1.02)
    plt.tight_layout()
    #plt.savefig('pareto_front_pairwise.png', bbox_inches='tight')
    plt.show()
"""

def pareto_front_3d(pareto_front):
    # Assuming 'pareto_front' contains the Pareto front individuals
    pareto_array = np.array([ind.fitness.values for ind in pareto_front])

    # Extract the first three objectives for 3D plotting
    x = pareto_array[:, 0]
    y = pareto_array[:, 1]
    z = pareto_array[:, 2]
    color = pareto_array[:, 3]  # The fourth objective (color encoding)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=color, cmap='viridis', alpha=0.7)

    # Set labels
    ax.set_xlabel(list(OBJECTIVES.values())[0])
    ax.set_ylabel(list(OBJECTIVES.values())[1])
    ax.set_zlabel(list(OBJECTIVES.values())[2])
    ax.set_title("Pareto Front (3D Scatter Plot with Fourth Objective Encoded by Color)")

    # Add color bar
    plt.colorbar(sc, label=list(OBJECTIVES.values())[3])

    plt.tight_layout()
    #plt.savefig('pareto_front_3d_colored.png', bbox_inches='tight')
    plt.show()


def plot_radar_chart(method_names, objective_matrix, objective_names):
    """
    Plots a radar chart comparing multiple methods over multiple objectives.

    Parameters:
    - method_names: List of strings with method names
    - objective_matrix: 2D list/array [method][objective] with raw scores
    - objective_names: List of objective names (e.g., ['ICP', 'SM', 'IFN', 'NED'])
    """
    data = np.array(objective_matrix)

    # Invert objectives where lower is better: ICP, IFN, NED
    invert_idx = [i for i, name in enumerate(objective_names) if name in ['ICP', 'IFN', 'NED']]
    for i in invert_idx:
        if objective_names[i] == 'IFN':
            data[:, i] = 1 - data[:, i] / np.max(data[:, i])
        else:
            data[:, i] = 1 - data[:, i]

    # Close the radar shape
    N = len(objective_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    categories = [
        f"{'1 - ' if i in invert_idx else ''}{objective_names[i]}"
        for i in range(N)
    ]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, values in enumerate(data):
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, label=method_names[i])
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title("Comparison of Decomposition Methods", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    #plt.savefig('comparison.png', bbox_inches='tight')
    plt.show()

def plot_parallel_coordinates(method_names, objective_matrix, objective_names):
    """
    Plots a parallel coordinates chart comparing methods on multiple objectives.

    Parameters:
    - method_names: List of strings with method names
    - objective_matrix: 2D list/array [method][objective] with raw scores
    - objective_names: List of objective names (e.g., ['ICP', 'SM', 'IFN', 'NED'])
    """
    df = pd.DataFrame(objective_matrix, columns=objective_names)
    df["Method"] = method_names

    # Normalize values for comparison
    df_norm = df.copy()
    for col in objective_names:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df_melt = df_norm.melt(id_vars="Method", var_name="Objective", value_name="Value")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_melt, x="Objective", y="Value", hue="Method", marker="o")
    plt.title("Parallel Coordinates Plot of Objective Scores")
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('parallel.png', bbox_inches='tight')
    plt.show()
