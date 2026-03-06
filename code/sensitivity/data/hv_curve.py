import pandas as pd
import matplotlib.pyplot as plt

new_source = f'sensitivity/data/HV_by_generation_new.csv'
old_source = f'sensitivity/data/HV_by_generation_old.csv'

new_data = pd.read_csv(new_source)
old_data = pd.read_csv(old_source)

new_gen_summary = (
    new_data.groupby("generation", as_index=False)["HV"]
        .agg(avg_hv_30_runs="mean", std_hv_30_runs="std")
        .sort_values("generation")
)

old_gen_summary = (
    old_data.groupby("generation", as_index=False)["HV"]
        .agg(avg_hv_30_runs="mean", std_hv_30_runs="std")
        .sort_values("generation")
)

fig, ax = plt.subplots(figsize=(10,6))

# plot mean curves
ax.plot(new_gen_summary['generation'], new_gen_summary['avg_hv_30_runs'],
        color='blue', label='Versión nueva', linewidth=2)
ax.plot(old_gen_summary['generation'], old_gen_summary['avg_hv_30_runs'],
        color='orange', label='Versión antigua', linewidth=2)

# add shaded std bands
ax.fill_between(new_gen_summary['generation'],
                new_gen_summary['avg_hv_30_runs'] - new_gen_summary['std_hv_30_runs'],
                new_gen_summary['avg_hv_30_runs'] + new_gen_summary['std_hv_30_runs'],
                color='blue', alpha=0.2)
ax.fill_between(old_gen_summary['generation'],
                old_gen_summary['avg_hv_30_runs'] - old_gen_summary['std_hv_30_runs'],
                old_gen_summary['avg_hv_30_runs'] + old_gen_summary['std_hv_30_runs'],
                color='orange', alpha=0.2)

# labels and title
ax.set_xlabel('Generación', fontsize=12)
ax.set_ylabel('Hypervolumen promedio', fontsize=12)
ax.set_title('Convergencia HV: nueva vs antigua', fontsize=14, fontweight='bold')

# legend and grid
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
