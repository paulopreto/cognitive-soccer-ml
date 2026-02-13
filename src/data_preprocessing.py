"""
------------------------------------------------------------------------------
Dataset Preparation Pipeline for Cognitive-Match Analysis with Clustering Labels
------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro (adapted)

Description:
-------------
This script automates the preparation of datasets for machine learning pipelines
involving cognitive metrics and on-field performance outcomes (goals scored, goals conceded, etc.).
It generates `.pkl` files containing train/test splits with labels defined by KMeans clustering
or a binary 'Half' split (based on performance ranks).

Main Workflow:
---------------
1. Load base dataset containing cognitive function measures and field performance metrics.
2. For each combination of cognitive variables:
    a. Normalize cognitive data.
    b. Perform KMeans clustering on field metrics (per team) to generate labels.
    c. (Optional) Use "Half" method to split top/bottom performers.
    d. Split into train/test sets (75/25).
    e. Save data into `.pkl` files for future ML model training.
3. Supports:
    - KMeans clustering (default)
    - Binary 'Half' labeling
    - Oversampling (SMOTE-ready datasets)

Inputs:
-------
- path_to_data: CSV file with full dataset (cognitive + performance)
- path_save: Output directory to save datasets (.pkl)
- colunas_func_cog: List of cognitive variables to be used in each iteration
- n_clusters_Desempenho_campo: Number of clusters for field performance variables
- oversample, normal, metade: Flags to determine which labeling strategy to apply.

Output:
-------
- `.pkl` files containing:
    [X_train, y_train, X_test, y_test]
-------------------------------------------------------------------------------
"""

# === Imports ===
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from itertools import combinations


# -------------------------------
# Generate all possible combinations of a list of columns
# -------------------------------
def generate_combinations(cols):
    all_combinations = []
    for r in range(1, len(cols) + 1):
        comb = list(combinations(cols, r))
        all_combinations.extend(comb)
    return all_combinations


# -------------------------------
# Apply KMeans clustering to a DataFrame and add cluster labels
# -------------------------------
def kmeans_cluster(dados, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(dados)
    dados_com_clusters = dados.copy()
    dados_com_clusters["cluster"] = kmeans.labels_
    return dados_com_clusters


# -------------------------------
# Apply KMeans clustering per team (split by midpoint; default assumes two equal-sized teams)
# -------------------------------
def kmeans_cluster_teams(dados, n_clusters, first_team_size=None):
    n = len(dados)
    if first_team_size is not None:
        n1 = first_team_size
    else:
        n1 = n // 2
    n2 = n - n1
    if n < 4:
        raise ValueError(
            f"kmeans_cluster_teams: need at least 4 rows (got {n}). "
            "Clustering requires at least 2 samples per team."
        )
    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"kmeans_cluster_teams: each team must have at least 2 rows. "
            f"Got team sizes {n1}, {n2} (total {n})."
        )
    dados_1team = dados.iloc[:n1].copy()
    dados_2team = dados.iloc[n1:].copy()

    scaler_1team = StandardScaler()
    dados_1team_norm = scaler_1team.fit_transform(dados_1team)
    dados_team_1 = kmeans_cluster(pd.DataFrame(dados_1team_norm), n_clusters)
    dados_team_1 = reorganizar_labels(dados_team_1, n_clusters)

    scaler_2team = StandardScaler()
    dados_2team_norm = scaler_2team.fit_transform(dados_2team)
    dados_team_2 = kmeans_cluster(pd.DataFrame(dados_2team_norm), n_clusters)
    dados_team_2 = reorganizar_labels(dados_team_2, n_clusters)

    dados_com_clusters = pd.concat([dados_team_1, dados_team_2])
    return dados_com_clusters


# -------------------------------
# Add Binary Cluster Label Splitting Top/Bottom Half (per Team)
# -------------------------------
def add_cluster_half(dados):
    d_sorted = dados.sort_values(by=0, ascending=False)
    midpoint = len(d_sorted) // 2
    d_sorted["cluster"] = 0
    d_sorted.iloc[:midpoint, d_sorted.columns.get_loc("cluster")] = 1
    return d_sorted.sort_index()


# -------------------------------
# Apply Half Labeling per Team
# -------------------------------
def label_half_teams(dados, n_clusters, first_team_size=None):
    n = len(dados)
    n1 = (first_team_size if first_team_size is not None else n // 2)
    n2 = n - n1
    if n1 < 1 or n2 < 1:
        raise ValueError(f"label_half_teams: invalid split (n1={n1}, n2={n2}, n={n}).")
    dados_1team = dados.iloc[:n1].copy()
    dados_2team = dados.iloc[n1:].copy()

    scaler_1team = StandardScaler()
    dados_team_1 = add_cluster_half(
        pd.DataFrame(scaler_1team.fit_transform(dados_1team))
    )

    scaler_2team = StandardScaler()
    dados_team_2 = add_cluster_half(
        pd.DataFrame(scaler_2team.fit_transform(dados_2team))
    )

    dados_com_clusters = pd.concat([dados_team_1, dados_team_2])
    return dados_com_clusters


# -------------------------------
# KMeans Clustering Applied to Each Column Individually
# -------------------------------
def kmeans_cluster_columns(dados, n_clusters):
    dados_com_clusters = pd.DataFrame(index=dados.index)
    for coluna in dados.columns:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(dados[[coluna]])
        dados_com_clusters[str(coluna) + "_cluster"] = kmeans.labels_
    return dados_com_clusters


# -------------------------------
# Reorganize Cluster Labels from High-to-Low Ordering
# -------------------------------
def reorganizar_labels(dados, n_clusters):
    dados_ordenados = dados.sort_values(by=dados.columns[0], ascending=False)
    ordem_original = dados_ordenados[dados.columns[1]].unique()
    nova_ordem = list(range(n_clusters - 1, -1, -1))
    dados_reorganizados = dados_ordenados.copy()
    for i, label in enumerate(ordem_original):
        dados_reorganizados.loc[
            dados_ordenados[dados.columns[1]] == label, dados.columns[1]
        ] = nova_ordem[i]
    return dados_reorganizados.sort_index()


# -------------------------------
# Main Execution Function for Dataset Preparation
# -------------------------------
def run(
    path_to_data,
    path_save,
    n_clusters_Desempenho_campo=2,
    oversample=False,
    normal=False,
    metade=False,
    colunas_func_cog=None,
):
    path_to_data = Path(path_to_data)
    path_save = Path(path_save)
    full_data = pd.read_csv(path_to_data)

    # Select Cognitive Variables
    X_cognitivo = full_data[colunas_func_cog]
    scaler_cog = StandardScaler()
    X_cognitivo_norm = scaler_cog.fit_transform(X_cognitivo)

    # Select Field Performance Variables
    Y_campo = full_data[
        ["gols_feitos", "gols_sofridos", "gols_companheiros", "saldo_gols"]
    ]

    # Generate KMeans Clusters (Default) for Field Performance per Team
    Y_gf = kmeans_cluster_teams(
        Y_campo[["gols_feitos"]], n_clusters_Desempenho_campo
    ).reset_index(drop=True)
    Y_gs = kmeans_cluster_teams(
        Y_campo[["gols_sofridos"]], n_clusters_Desempenho_campo
    ).reset_index(drop=True)
    Y_gc = kmeans_cluster_teams(
        Y_campo[["gols_companheiros"]], n_clusters_Desempenho_campo
    ).reset_index(drop=True)
    Y_sg = kmeans_cluster_teams(
        Y_campo[["saldo_gols"]], n_clusters_Desempenho_campo
    ).reset_index(drop=True)

    path_save.mkdir(parents=True, exist_ok=True)

    # Save Datasets (Oversample Scenario)
    if oversample:
        datasets = [("gf", Y_gf), ("gs", Y_gs), ("gc", Y_gc), ("sg", Y_sg)]
        for name, Y in datasets:
            X_train, X_test, y_train, y_test = train_test_split(
                X_cognitivo, Y, test_size=0.25, random_state=0
            )
            with open(path_save / f"cogfut_{name}.pkl", "wb") as f:
                pickle.dump([X_train, y_train["cluster"], X_test, y_test["cluster"]], f)
        return ()

    # Save Datasets (Normal Scenario)
    if normal:
        datasets = [("gf", Y_gf), ("gs", Y_gs), ("gc", Y_gc), ("sg", Y_sg)]
        for name, Y in datasets:
            X_train, X_test, y_train, y_test = train_test_split(
                X_cognitivo, Y, test_size=0.25, random_state=0
            )
            with open(path_save / f"cogfut_{name}.pkl", "wb") as f:
                pickle.dump([X_train, y_train["cluster"], X_test, y_test["cluster"]], f)

    # Save Datasets (Metade Scenario)
    if metade:
        Y_gf = label_half_teams(Y_campo[["gols_feitos"]], n_clusters_Desempenho_campo)
        Y_gs = label_half_teams(Y_campo[["gols_sofridos"]], n_clusters_Desempenho_campo)
        Y_gc = label_half_teams(
            Y_campo[["gols_companheiros"]], n_clusters_Desempenho_campo
        )
        Y_sg = label_half_teams(Y_campo[["saldo_gols"]], n_clusters_Desempenho_campo)

        datasets = [("gf", Y_gf), ("gs", Y_gs), ("gc", Y_gc), ("sg", Y_sg)]
        for name, Y in datasets:
            X_train, X_test, y_train, y_test = train_test_split(
                X_cognitivo_norm, Y, test_size=0.25, random_state=0
            )
            with open(path_save / f"cogfut_{name}.pkl", "wb") as f:
                pickle.dump([X_train, y_train["cluster"], X_test, y_test["cluster"]], f)


# -------------------------------
# Batch Execution over all Combinations
# -------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    colunas_func_cog = [
        "Memory span",
        "Acuracia Go",
        "Acuracia nogo",
        "Capacidade de rastreamento",
        "Flexibilidade cognitiva (B-A)",
    ]
    combinations_list = generate_combinations(colunas_func_cog)
    path_csv = data_dir / "Dataset_cog_clusters.csv"
    if not path_csv.is_file():
        path_csv = data_dir / "dataset.csv"
    path_base_save = data_dir / "ML_datasets"

    for combinacao in combinations_list:
        combinacao_str = "_".join([col.replace(" ", "_") for col in combinacao])
        path_save = path_base_save / combinacao_str

        run(
            path_to_data=path_csv,
            path_save=path_save,
            n_clusters_Desempenho_campo=2,
            oversample=False,
            normal=True,
            metade=False,
            colunas_func_cog=list(combinacao),
        )
