import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

new_source = "sensitivity/data/HV_by_generation_cargo_new.csv"
old_source = "sensitivity/data/HV_by_generation_cargo_old.csv"

def summarize_hv(data: pd.DataFrame):
    data = data.sort_values(["id", "generation"]).copy()

    gen_summary = (
        data.groupby("generation", as_index=False)["HV"]
        .agg(avg_hv_30_runs="mean", std_hv_30_runs="std")
        .sort_values("generation")
    )

    auc_per_run = (
        data.groupby("id")[["generation", "HV"]]
        .apply(lambda g: np.trapezoid(g["HV"].to_numpy(), g["generation"].to_numpy()))
        .reset_index(name="auc_hv")
    )

    auc_stats = {
        "mean_auc_30_runs": float(auc_per_run["auc_hv"].mean()),
        "std_auc_30_runs": float(auc_per_run["auc_hv"].std()),
    }

    return gen_summary, auc_per_run, auc_stats


new_data = pd.read_csv(new_source)
old_data = pd.read_csv(old_source)

new_data = new_data[new_data['generation']<=100]
old_data = old_data[old_data['generation']<=100]

new_gen_summary, new_auc_per_run, new_auc_stats = summarize_hv(new_data)
old_gen_summary, old_auc_per_run, old_auc_stats = summarize_hv(old_data)

print("AUC (new):", new_auc_stats)
print("AUC (old):", old_auc_stats)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    new_gen_summary["generation"],
    new_gen_summary["avg_hv_30_runs"],
    color="blue",
    label="Version nueva",
    linewidth=2,
)
ax.plot(
    old_gen_summary["generation"],
    old_gen_summary["avg_hv_30_runs"],
    color="orange",
    label="Version antigua",
    linewidth=2,
)

ax.fill_between(
    new_gen_summary["generation"],
    new_gen_summary["avg_hv_30_runs"] - new_gen_summary["std_hv_30_runs"],
    new_gen_summary["avg_hv_30_runs"] + new_gen_summary["std_hv_30_runs"],
    color="blue",
    alpha=0.2,
)
ax.fill_between(
    old_gen_summary["generation"],
    old_gen_summary["avg_hv_30_runs"] - old_gen_summary["std_hv_30_runs"],
    old_gen_summary["avg_hv_30_runs"] + old_gen_summary["std_hv_30_runs"],
    color="orange",
    alpha=0.2,
)

ax.set_xlabel("Generacion", fontsize=12)
ax.set_ylabel("Hypervolumen promedio", fontsize=12)
ax.set_title("Convergencia HV: nueva vs antigua", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 5))
try:
    ax2.boxplot([new_auc_per_run["auc_hv"], old_auc_per_run["auc_hv"]], tick_labels=["Nueva", "Antigua"])
except TypeError:
    ax2.boxplot([new_auc_per_run["auc_hv"], old_auc_per_run["auc_hv"]], labels=["Nueva", "Antigua"])
ax2.set_ylabel("AUC de HV por ejecucion")
ax2.set_title("Distribucion de AUC (30 ejecuciones)")
ax2.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

print(new_gen_summary["std_hv_30_runs"])
print(old_gen_summary["std_hv_30_runs"])

#new_data = "sensitivity/data/Y_new_cargo_last.csv"
#old_data = "sensitivity/data/Y_old_cargo_last.csv"

# Extract final HV for each of the 30 runs
#new_final_hv = pd.read_csv(new_data)["HV"].to_numpy()
#old_final_hv = pd.read_csv(old_data)["HV"].to_numpy()

new_final_hv = new_data[new_data['generation']==100]["HV"].to_numpy()
old_final_hv = old_data[old_data['generation']==100]["HV"].to_numpy()


# Perform t-test for statistical significance
t_stat, p_value = stats.ttest_rel(old_final_hv, new_final_hv)
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3e}")

data = [old_final_hv, new_final_hv]

fig3, ax3 = plt.subplots(figsize=(8, 6))
parts = ax3.violinplot(data, showmeans=True, showmedians=False)

colors = ["red", "blue"]

for i, body in enumerate(parts['bodies']):
    body.set_facecolor(colors[i])

ax3.set_xticks([1, 2])
ax3.set_xticklabels(["Original", "Nuevo"])
ax3.set_ylabel("HV final")
ax3.set_title(f"Distribución del Hypervolume Final a lo Largo de {len(new_final_hv)} Ejecuciones")
ax3.grid(True, linestyle="--", alpha=0.3)

# Add p-value annotation
significance = "significant" if p_value < 0.05 else "not significant"
#ax3.text(0.5, 0.95, f'p-value: {p_value:.3f} ({significance})', transform=ax3.transAxes, ha='center', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Add min/mean/max stat boxes (bottom-right)
old_stats_text = (
    f"Original\n"
    f"max: {old_final_hv.max():.6f}\n"
    f"mean: {old_final_hv.mean():.6f}\n"
    f"min: {old_final_hv.min():.6f}"
)
new_stats_text = (
    f"Nuevo\n"
    f"max: {new_final_hv.max():.6f}\n"
    f"mean: {new_final_hv.mean():.6f}\n"
    f"min: {new_final_hv.min():.6f}"
)

print(old_stats_text)
print(new_stats_text)

ax3.text(
    0.42, 0.03, old_stats_text,
    transform=ax3.transAxes,
    ha="center", va="bottom",
    fontsize=9, color="black",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2, edgecolor="none")
)
ax3.text(
    0.58, 0.03, new_stats_text,
    transform=ax3.transAxes,
    ha="center", va="bottom",
    fontsize=9, color="black",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.2, edgecolor="none")
)

plt.tight_layout()
plt.show()
