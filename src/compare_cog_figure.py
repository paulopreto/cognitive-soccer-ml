"""
-------------------------------------------------------------------------------
Statistical Comparison Plot: Cognitive Functions vs. Goal-Related Performance Clusters
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script performs non-parametric statistical comparisons (Mann-Whitney U Test)
between two performance groups ('High' vs. 'Low') for a set of cognitive
function metrics. It visualizes these comparisons using boxplots overlaid with individual
data points, enabling the visualization of central tendencies, dispersion, and potential
outliers across performance groups in football players.

Key Features:
--------------
- Handles multiple cognitive variables and target performance outcomes simultaneously.
- Applies cluster inversion logic for metrics where higher values are negative indicators.
- Automatically performs Mann-Whitney U tests for each comparison, reporting statistics.
- Annotates each subplot with p-values for immediate significance interpretation.
- Displays individual player data points (stripplot) with jitter for clarity.
- Outliers for specific metrics are highlighted distinctly (e.g., 'Acuracia Go').
- Generates a grid of subplots (5x4).

Inputs:
--------
- CSV File: Dataset_cog_clusters.csv
    Required columns:
        - Cognitive variables: 'Memory span', 'Acuracia Go', 'Acuracia nogo',
                               'Capacidade de rastreamento', 'Flexibilidade cognitiva (B-A)'
        - Clustering labels for performance outcomes: 'gols_feitos', 'gols_sofridos',
                                                     'gols_companheiros', 'saldo_gols'

Outputs:
---------
- Multi-panel figure (5x4 grid) showing statistical comparison plots.
- Figure saved as 'comparacoes_todas_juntas.png' in specified directory.

-------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from scipy.stats import mannwhitneyu

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
_project_root = Path(__file__).resolve().parent.parent
df = pd.read_csv(_project_root / "data" / "Dataset_cog_clusters.csv")

# Convert 'Flexibilidade cognitiva (B-A)' from milliseconds to seconds
df["Flexibilidade cognitiva (B-A)"] = df["Flexibilidade cognitiva (B-A)"] / 1000

# Scale accuracies to percentages
df["Acuracia Go"] = df["Acuracia Go"] * 100
df["Acuracia nogo"] = df["Acuracia nogo"] * 100

# Plot settings (white background, no grids)
sns.set(style="white", font_scale=1.0)

# Cognitive variables of interest and their corresponding Y-axis labels
features = [
    "Memory span",
    "Acuracia Go",
    "Acuracia nogo",
    "Capacidade de rastreamento",
    "Flexibilidade cognitiva (B-A)",
]

feature_labels = {
    "Memory span": "VWM (au)",
    "Acuracia Go": "Sustained Attention (%)",
    "Acuracia nogo": "Impulsivity (%)",
    "Capacidade de rastreamento": "Tracking Capacity (au)",
    "Flexibilidade cognitiva (B-A)": "Cognitive Flexibility (s)",
}

# Target variables and plot titles
comparisons = {
    "gols_feitos": "Individual Goals",
    "gols_sofridos": "Conceded Goals",
    "gols_companheiros": "Goals by Teammates",
    "saldo_gols": "Net Goals",
}

# Custom color palette for 'Low' and 'High' performance groups
palette = ["#D62728", "#1F77B4"]

# Output folder for saving the figure
save_folder = _project_root / "figures"
save_folder.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Statistical Test Function
# ---------------------------
def run_stat_test(group0, group1):
    """
    Performs Mann-Whitney U Test between two independent groups.
    Returns the test name, statistic value, and p-value.
    """
    stat, p = mannwhitneyu(group0, group1, alternative="two-sided")
    return "Mann-Whitney", stat, p


# ---------------------------
# Figure Creation
# ---------------------------
fig, axes = plt.subplots(
    nrows=len(features), ncols=len(comparisons), figsize=(22, 14), sharey="row"
)
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# ---------------------------
# Plotting Loop per Feature and Target Variable
# ---------------------------
for row_idx, feature in enumerate(features):
    for col_idx, (target_var, target_label) in enumerate(comparisons.items()):
        ax = axes[row_idx, col_idx]

        # Prepare plotting data for current feature and target variable
        plot_data = df[[feature, target_var]].dropna().copy()

        # For 'Conceded Goals', invert clusters only on X-axis representation
        if target_var == "gols_sofridos":
            plot_data["target_var_temp"] = plot_data[target_var].apply(lambda x: 1 - x)
        else:
            plot_data["target_var_temp"] = plot_data[target_var]

        # Grouping data for statistical test
        group0 = plot_data[plot_data["target_var_temp"] == 0][feature]
        group1 = plot_data[plot_data["target_var_temp"] == 1][feature]

        # Run Mann-Whitney U Test
        test_name, stat, p = run_stat_test(group0, group1)
        print(
            f"[{feature} vs {target_label}] -> Test: {test_name}, stat = {stat:.3f}, p = {p:.4f}"
        )

        # Specific Y-axis limits for 'Acuracia Go' due to low variance
        if feature == "Acuracia Go":
            ymin, ymax = 91.5, 102
            ax.set_ylim(ymin, ymax)

        # Plot Boxplot (without outliers) and Stripplot (individual data points)
        sns.boxplot(
            x="target_var_temp",
            y=feature,
            data=plot_data,
            ax=ax,
            palette=palette,
            width=0.6,
            showfliers=False,
        )
        sns.stripplot(
            x="target_var_temp",
            y=feature,
            data=plot_data,
            ax=ax,
            color="black",
            size=3,
            jitter=True,
            alpha=0.6,
        )

        # Remove top/right spines for cleaner aesthetic
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Remove left spine for non-first columns
        if col_idx != 0:
            ax.spines["left"].set_visible(False)
        else:
            ax.spines["left"].set_visible(True)

        # Add light grid lines on Y-axis
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="lightgray")
        ax.set_axisbelow(True)

        # Highlight outliers for 'Acuracia Go' with special red marker
        if feature == "Acuracia Go":
            outlier_y = ymin + 0.4
            for x_pos in [0, 1]:
                group_data = plot_data[plot_data["target_var_temp"] == x_pos][feature]
                outliers = group_data[(group_data < ymin) | (group_data > ymax)]
                for _ in outliers:
                    jitter = np.random.uniform(-0.08, 0.08)
                    ax.plot(
                        x_pos + jitter,
                        outlier_y,
                        "o",
                        markerfacecolor="red",
                        markeredgecolor="black",
                        markersize=5,
                    )

        # Annotate p-value in each subplot
        ax.text(
            0.5,
            ax.get_ylim()[0] + 0.02,
            f"p = {p:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        # Add subplot titles (first row only)
        if row_idx == 0:
            ax.set_title(target_label, fontsize=14, fontweight="bold")
        else:
            ax.set_title("")

        # Customize X-axis labels for bottom row
        ax.set_xlabel("")
        ax.set_xticks([0, 1])
        if row_idx == len(features) - 1:
            ax.set_xticklabels(["Low", "High"], fontsize=10)
        else:
            ax.set_xticklabels([])

        # Set Y-axis labels for first column only
        if col_idx == 0:
            ax.set_ylabel(feature_labels.get(feature, feature), fontsize=11)
        else:
            ax.set_ylabel("")

# ---------------------------
# Unified Legend Creation
# ---------------------------
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=palette[0],
        label="Low",
        markersize=10,
    ),
    plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=palette[1],
        label="High",
        markersize=10,
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        linestyle="",
        label="Individual Performance",
        markersize=6,
        alpha=0.6,
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="red",
        linestyle="",
        label="Outlier",
        markersize=6,
    ),
]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=False)

# Adjust layout to avoid overlap
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.align_ylabels(axes[:, 0])

# Save Figure
filename = save_folder / "comparacoes_todas_juntas.png"
fig.savefig(filename, dpi=300, bbox_inches="tight")
plt.show()
print(f"\n[Image saved at: {filename}]")
