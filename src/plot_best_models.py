"""
-------------------------------------------------------------------------------
Comparative visualization of models with best balanced accuracies
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script generates a comparative visualization of balanced accuracy values 
for various machine learning algorithms predicting different performance outcomes.
The visualization comprises four vertically aligned subplots, each corresponding
 to a different target outcome.

Key Features:
--------------
- For each target variable, it plots the best balanced accuracy obtained from 
  different combinations of cognitive functions.
- Displays error bars representing standard deviation for each point.
- Highlights the algorithm responsible for the best accuracy.
- Consolidates a unified legend summarizing algorithm labels and colors.

Inputs:
--------
- Excel files containing mean values ('medias_*.xlsx') for balanced accuracy.
- Excel files containing standard deviations ('Dp_*.xlsx') for balanced accuracy.

Outputs:
---------
- Multi-panel scatter plot with error bars.

-------------------------------------------------------------------------------
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_balanced_accuracy(ax, dataset, dataset_dp, palette,
                           ylabel="Balanced accuracy (%)", 
                           title="", show_xlabel=False, show_legend=False):
    """
    Plots balanced accuracy results for cognitive function combinations on a given axis.
    
    Parameters:
    - ax: The matplotlib axis to plot on.
    - dataset: Dictionary containing the mean metric values.
    - dataset_dp: Dictionary containing the standard deviations.
    - palette: Color palette mapping algorithm names to colors.
    - ylabel: Label for the Y-axis.
    - title: Title of the subplot.
    - show_xlabel: Whether to display the X-axis label.
    - show_legend: Whether to display the legend for this subplot.
    """

    # Extract balanced accuracy values and corresponding standard deviations
    balanced_accuracy = dataset['balanced_accuracy'].copy()
    balanced_accuracy_dp = dataset_dp['balanced_accuracy'].copy()

    # Identify the maximum balanced accuracy per row and corresponding algorithm
    balanced_accuracy["Max Value"] = balanced_accuracy.iloc[:, 1:].max(axis=1)
    balanced_accuracy["Algorithms"] = balanced_accuracy.iloc[:, 1:].idxmax(axis=1)

    # Retrieve standard deviation corresponding to the max value
    balanced_accuracy["Standard Deviation"] = balanced_accuracy.apply(
        lambda row: balanced_accuracy_dp.loc[row.name, row["Algorithms"]],
        axis=1
    )

    # Mapping for variable name abbreviations
    sigla_map = {
        "Acuracia_Go": "SA",
        "Flexibilidade_cognitiva_(B-A)": "CF",
        "Capacidade_de_rastreamento": "TC",
        "Acuracia_nogo": "I",
        "Memory_span": "VWM"
    }

    # Translate variable names to abbreviations for cleaner axis labels
    def translate_variable(v):
        for k, v_abbr in sigla_map.items():
            v = v.replace(k, v_abbr)
        return v

    # Support both 'variables' and 'variaveis' column names for backward compatibility
    col_var = "variables" if "variables" in balanced_accuracy.columns else "variaveis"
    balanced_accuracy[col_var] = balanced_accuracy[col_var].apply(translate_variable)
    balanced_accuracy[col_var] = balanced_accuracy[col_var].str.replace("_", "-", regex=False)

    # Sort the data by balanced accuracy values for better visual distribution
    balanced_accuracy = balanced_accuracy.sort_values(by="Max Value")

    # Plot scatter points for each variable combination
    scatter = sns.scatterplot(
        data=balanced_accuracy,
        x=col_var,
        y="Max Value",
        hue="Algorithms",
        palette=palette,
        s=70,
        edgecolor="black",
        ax=ax,
        legend="auto" if show_legend else False
    )

    # Plot vertical error bars indicating standard deviation
    ax.errorbar(
        balanced_accuracy[col_var],
        balanced_accuracy["Max Value"],
        yerr=balanced_accuracy["Standard Deviation"],
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.5
    )

    # Adjust X-axis label formatting
    ax.set_xticklabels(balanced_accuracy[col_var], rotation=30, ha='right', fontsize=9)

    # Conditionally display X-axis label
    ax.set_xlabel("Cognitive functions combinations", fontsize=9 if show_xlabel else 0)

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Remove unnecessary plot spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return scatter


# -------------------
# Main Execution Block
# -------------------

# Define base directory containing results files (pathlib for cross-platform)
base_path = Path("D:/Processamento_mestrado_Sports_Science/final_analysis/results_CV")

# Target variable codes and corresponding subplot titles
siglas = ['gf', 'gs', 'gc', 'sg']
titles = [
    "Individual Goals",
    "Conceded Goals",
    "Goals by Teammates",
    "Net Goals"
]

# Color mapping for algorithms
algorithms_colors = {
    'KNN': '#27AE60',           
    'Random forest': '#8c564b', 
    'LogisticRegression': '#2ca02c',
    'SVM': '#4C72B0',           
    'Xgboost': '#9467bd',       
    'Neural network': '#FF7F0E',   
    'Naive':  '#D62828'        
}

# Prepare a 4-row subplot figure for visualizing each target metric
fig, axs = plt.subplots(4, 1, figsize=(13, 15), sharex=False)

# Dictionary to store unique legend handles and labels
all_handles = {}

# Iterate over each target variable (sigla) and generate plots
for i, sigla in enumerate(siglas):
    # Construct file paths for mean values and standard deviations
    path_means = base_path / f"medias_{sigla}.xlsx"
    path_stddevs = base_path / f"Dp_{sigla}.xlsx"

    # Read all sheets from Excel files into dictionaries
    xls_means = pd.ExcelFile(path_means)
    xls_stddevs = pd.ExcelFile(path_stddevs)

    dataset = {sheet_name: pd.read_excel(xls_means, sheet_name=sheet_name) for sheet_name in xls_means.sheet_names}
    dataset_dp = {sheet_name: pd.read_excel(xls_stddevs, sheet_name=sheet_name) for sheet_name in xls_stddevs.sheet_names}

    # Plot the balanced accuracy for the current target variable
    scatter = plot_balanced_accuracy(
        axs[i],
        dataset,
        dataset_dp,
        palette=algorithms_colors,
        title=titles[i],
        show_xlabel=(i == len(siglas) - 1),
        show_legend=True
    )

    # Extract and store unique legend handles and labels for the global legend
    handles_i, labels_i = axs[i].get_legend_handles_labels()
    for handle, label in zip(handles_i, labels_i):
        # Standardize algorithm labels for final legend
        if label == 'Neural network':
            label_ = 'MLP'
        elif label == 'Random forest':
            label_ = 'RF'
        elif label == 'Naive':
            label_ = 'GNB'
        elif label == 'Xgboost':
            label_ = 'XGBoost'
        else:
            label_ = label
        
        if label_ not in all_handles:
            all_handles[label_] = handle

    # Remove subplot-specific legends to avoid redundancy
    if axs[i].legend_ is not None:
        axs[i].legend_.remove()

# Add a consolidated legend for all subplots
fig.legend(
    all_handles.values(),
    all_handles.keys(),
    loc='center right',
    bbox_to_anchor=(1, 0.5),
    fontsize=9,
    title="Algorithms",
    title_fontsize=10
)

# Adjust layout to accommodate legend outside plot area
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()
