"""
-------------------------------------------------------------------------------
Clusters groups comparisson figure
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script generates violin plots combined with jittered scatter points to visualize 
the distribution of various goal-related performance metrics across two distinct 
performance groups (clusters). Each subplot represents a specific performance indicator, 
displaying the separation between 'Superior' and 'Inferior' groups as identified 
through clustering analysis.

Key Features:
--------------
- Visual differentiation of clusters.
- Overlay of individual data points with jitter to avoid overlap.
- Adaptive label and color inversion for metrics where higher values represent worse performance.

Inputs:
--------
- Excel files containing columns for the metric of interest and corresponding cluster labels:
    - 'gols_feitos'      → Individual Goals per Game
    - 'gols_sofridos'    → Conceded Goals per Game
    - 'gols_companheiros'→ Goals by Teammates per Game
    - 'saldo_gols'       → Net Goals per Game
    - 'cluster'          → Cluster assignment (0 or 1)

Outputs:
---------
- 2x2 grid of violin plots with jittered data points, suitable for scientific articles.

-------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_vertical_clustered_with_violin(ax, df, data_col, cluster_col, y_label, title, invert_colors_labels=False):
    """
    Plots a vertical violin plot with jittered scatter points to visualize the distribution 
    of a performance metric across clusters.
    
    Parameters:
    - ax: The matplotlib axis to plot on.
    - df: DataFrame containing the data.
    - data_col: Name of the column with the metric to be plotted.
    - cluster_col: Name of the column with cluster labels (0 or 1).
    - y_label: Label for the Y-axis.
    - title: Title of the subplot.
    - invert_colors_labels: Boolean to switch colors and labels when higher values indicate worse performance.
    """

    # Extract metric values and cluster labels
    data = df[data_col].values
    clusters = df[cluster_col].values
    unique_clusters = np.unique(clusters)

    # Define color mapping for clusters 
    base_color_map = {
        0: '#D62728',  # Red - High performance
        1: '#1F77B4'   # Blue - Low performance
    }

    # Define label mapping for clusters
    base_label_map = {
        0: 'Inferior',
        1: 'Superior'
    }

    # Invert colors and labels if metric represents a negative performance indicator (goals conceded)
    if invert_colors_labels:
        color_map = {k: base_color_map[1 - k] for k in base_color_map}
        label_map = {k: base_label_map[1 - k] for k in base_label_map}
    else:
        color_map = base_color_map
        label_map = base_label_map

    # Marker styles for scatter points per cluster
    markers = ['o', 's']
    marker_map = {cluster: markers[i % len(markers)] for i, cluster in enumerate(unique_clusters)}

    # Plot violin plots for each cluster without internal details (just density shape)
    for cluster in unique_clusters:
        cluster_data = data[clusters == cluster]
        sns.violinplot(y=cluster_data, ax=ax, color=color_map[cluster], alpha=0.3, linewidth=0, inner=None)

    # Apply horizontal jitter to scatter points for better visibility
    horizontal_jitter = 0.05
    x_positions = np.random.uniform(-horizontal_jitter, horizontal_jitter, size=len(data))

    # Overlay scatter points per cluster with jitter and cluster-specific colors/markers
    for cluster in unique_clusters:
        cluster_mask = (clusters == cluster)
        ax.scatter(x_positions[cluster_mask], data[cluster_mask],
                   color=color_map[cluster],
                   marker=marker_map[cluster],
                   edgecolor='k', linewidth=0.5,
                   alpha=0.8,
                   s=40,
                   label=label_map[cluster])

    # Plot aesthetics adjustments
    ax.set_xticks([])  # Remove X-axis ticks (since it's a vertical layout)
    ax.set_xlabel('')
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.tick_params(axis='y', labelsize=12)

    # Add legend indicating performance groups
    ax.legend(title='Performance Groups', fontsize=12, title_fontsize=13)

    # Add light horizontal grid lines for reference
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# -------------------
# Main Execution Block
# -------------------

# File paths for datasets containing goal-related metrics and clusters
path_gf = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\gf_cluster.xlsx'
path_gs = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\gs_cluster.xlsx'
path_gc = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\gc_cluster.xlsx'
path_sg = 'C:\\Users\\rafae\\OneDrive\\Documentos\\Mestrado\\Cogfut\\statistics\\ML\\data\\Normal\\sg_cluster.xlsx'

# Load datasets into pandas DataFrames
gf_data = pd.read_excel(path_gf)
gs_data = pd.read_excel(path_gs)
gc_data = pd.read_excel(path_gc)
sg_data = pd.read_excel(path_sg)

# Create figure with 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Generate plots for each performance metric dataset
plot_vertical_clustered_with_violin(axs[0, 0], gf_data, 'gols_feitos', 'cluster', 
                                    'Individual goals per game', 'Individual Goals')

plot_vertical_clustered_with_violin(axs[0, 1], gs_data, 'gols_sofridos', 'cluster', 
                                    'Conceded goals per game', 'Conceded Goals', invert_colors_labels=True)

plot_vertical_clustered_with_violin(axs[1, 0], gc_data, 'gols_companheiros', 'cluster', 
                                    'Goals by teammates per game', 'Goals by Teammates')

plot_vertical_clustered_with_violin(axs[1, 1], sg_data, 'saldo_gols', 'cluster', 
                                    'Net team goals per game', 'Net Goals')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


