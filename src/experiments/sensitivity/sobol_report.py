from __future__ import annotations

from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_sobol_summary(problem: Dict[str, Any], si: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = problem["names"]
    df = pd.DataFrame(
        {
            "param": names,
            "S1": si["S1"],
            "S1_conf": si["S1_conf"],
            "ST": si["ST"],
            "ST_conf": si["ST_conf"],
        }
    )
    df["interaction_gap"] = df["ST"] - df["S1"]
    return df.sort_values("ST", ascending=False).reset_index(drop=True)


def print_sobol_report(problem: Dict[str, Any], si: Dict[str, np.ndarray]) -> pd.DataFrame:
    summary = build_sobol_summary(problem, si)
    ranked_by_s1 = summary.sort_values("S1", ascending=False)["param"].tolist()
    ranked_by_st = summary.sort_values("ST", ascending=False)["param"].tolist()

    print("\nSobol summary (higher = more influence)\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nRanking by S1:", " > ".join(ranked_by_s1))
    print("Ranking by ST:", " > ".join(ranked_by_st))
    print(
        "Largest interaction gap (ST-S1):",
        summary.sort_values("interaction_gap", ascending=False).iloc[0]["param"],
    )
    return summary


def plot_sobol_indices(problem: Dict[str, Any], si: Dict[str, np.ndarray]) -> None:
    names = problem["names"]
    # Translate parameter names to Spanish
    name_map = {
        'mu': 'mu',
        'lambda_prop': 'prop. lambda',
        'prob': 'prob. mut.'
    }
    names_translated = [name_map.get(name, name) for name in names]
    
    # Sort by ST descending for better visualization
    sort_idx = np.argsort(si["ST"])[::-1]
    names_sorted = [names_translated[i] for i in sort_idx]
    s1_sorted = si["S1"][sort_idx]
    s1_conf_sorted = si["S1_conf"][sort_idx]
    st_sorted = si["ST"][sort_idx]
    st_conf_sorted = si["ST_conf"][sort_idx]
    
    y = np.arange(len(names_sorted))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_s1 = ax.barh(
        y - height / 2,
        s1_sorted,
        height=height,
        xerr=s1_conf_sorted,
        capsize=0,  # Remove caps
        label="S1 (Primer orden)",
        color='skyblue',
        alpha=0.8,
    )
    bars_st = ax.barh(
        y + height / 2,
        st_sorted,
        height=height,
        xerr=st_conf_sorted,
        capsize=0,  # Remove caps
        label="ST (Total)",
        color='salmon',
        alpha=0.8,
    )
    
    # Add value labels above the bars
    for bar, value in zip(bars_s1, s1_sorted):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() - 0.05, 
                f'{value:.3f}', ha='left', va='top', fontsize=9, color='black')
    for bar, value in zip(bars_st, st_sorted):
        ax.text(bar.get_width()+ 0.05, bar.get_y() + bar.get_height() - 0.05, 
                f'{value:.3f}', ha='left', va='top', fontsize=9, color='black')
    
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel("Índice de Sensibilidad de Sobol")
    ax.set_xlim(0, 1)  # fixed scale from 0 to 1
    ax.set_title("Índices de Sensibilidad de Sobol (Ordenados por Efecto Total)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plt.show()
