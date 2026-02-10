"""
-------------------------------------------------------------------------------
Summary Statistics Extraction and Consolidation from Nested Result Files
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script automates the extraction, calculation, and consolidation of performance metrics 
from multiple Excel result files stored in nested directories. It computes mean and standard 
deviation values for various classification metrics across different machine learning algorithms 
and hyperparameter tuning attempts.

The script organizes these summary statistics into structured DataFrames, annotates each 
with the maximum value locations, and saves the consolidated results into Excel files. 
It also contains utility functions for statistical tests and visualization of results.

Key Features:
--------------
- Recursive traversal of directories to locate result Excel files.
- Extraction of metrics means and standard deviations per tuning attempt.
- Annotation of maximum values and their positions in result tables.
- Export of summary statistics to Excel files with multiple sheets.
- Functions for normality tests, ANOVA, Tukey post hoc, and plotting (optional).

Usage:
-------
Set the root directory path containing your result folders in 'path_dir_combinacoes' and 
run the script. The consolidated Excel files will be saved in the specified directory.

-------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import MultiComparison
import os

def save_table_as_image(data, columns, title, save_path, footer_text=None):
    """
    Save a pandas DataFrame as a PNG image with title and optional footer.
    """
    fig, ax = plt.subplots(figsize=(12, 2 + (len(data) * 0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data.values, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.subplots_adjust(top=0.85)
    plt.suptitle(title, fontsize=14, y=0.98)
    if footer_text:
        plt.figtext(0.5, 0.01, footer_text, wrap=True, horizontalalignment='center', fontsize=12)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def criar_variaveis_vazias(metrics_list, columns):
    """
    Create multiple empty dictionaries of DataFrames for storing results.
    """
    dicionario_abas = {}
    for metric in metrics_list:
        dicionario_abas[metric] = pd.DataFrame(columns=columns)
    
    return (dicionario_abas, dicionario_abas.copy(), dicionario_abas.copy(),
            dicionario_abas.copy(), dicionario_abas.copy(), dicionario_abas.copy(),
            dicionario_abas.copy(), dicionario_abas.copy())

def adicionar_maior_valor(dicionario_dataframes):
    """
    Append a row indicating the maximum value and its position for each DataFrame in the dictionary.
    """
    for sheet_name, df in dicionario_dataframes.items():
        df_numeric = df.iloc[:, 1:]  # skip first column with labels
        
        if df_numeric.empty:
            continue
        
        max_value = df_numeric.values.max()
        max_pos = df_numeric.values.argmax()
        max_row, max_col = divmod(max_pos, df_numeric.shape[1])
        
        row_label = df.iloc[max_row, 0]
        col_label = df_numeric.columns[max_col]
        
        new_row = {col: '' for col in df.columns}
        new_row[col_label] = f'Max value: {max_value} (Row: {row_label}, Column: {col_label})'
        
        new_row_df = pd.DataFrame([new_row], index=[''])
        df = pd.concat([df, new_row_df], ignore_index=False)
        dicionario_dataframes[sheet_name] = df
    
    return dicionario_dataframes

def salvar_em_xlsx(output_file, dicionario_dataframes):
    """
    Save a dictionary of DataFrames to an Excel file with each DataFrame as a separate sheet.
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in dicionario_dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f'File "{output_file}" saved successfully.')

def extrair_salvar_medias(path_dir_combinacoes):
    """
    Traverse directory tree, extract mean and std deviation of metrics from Excel files,
    consolidate and save them in new Excel files.
    """
    metrics = ['accuracy', 'balanced_accuracy', 'f1', 'mcc', 'precision', 'recall', 'roc_auc',
               'precision_class_0', 'precision_class_1', 'recall_class_0', 'recall_class_1',
               'f1_class_0', 'f1_class_1']

    columns = ['variables', 'Naive', 'Random forest', 'KNN', 'Logistic Regression', 'SVM', 'Neural network', 'Xgboost']

    # Create empty containers for means and std devs per tuning attempt prefix
    resultados_gf, resultados_gs, resultados_gc, resultados_sg, resultados_gf_dp, resultados_gs_dp, resultados_gc_dp, resultados_sg_dp = criar_variaveis_vazias(metrics, columns)

    for root, dirs, _ in os.walk(path_dir_combinacoes):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for _, _, files in os.walk(dir_path):
                result_files = [f for f in files if 'resultados' in f]
                for tentativa_file in result_files:
                    prefix = tentativa_file.split('_')[0]
                    file_path = os.path.join(dir_path, tentativa_file)
                    sheets = pd.read_excel(file_path, sheet_name=None)
                    for metric_name, df_metric in sheets.items():
                        mean_vals = df_metric.mean()
                        std_vals = df_metric.std()

                        variable_row = pd.Series({'variables': dir_name})
                        mean_row = pd.concat([variable_row, mean_vals])
                        mean_row = pd.DataFrame(mean_row).transpose()
                        std_row = pd.concat([variable_row, std_vals])
                        std_row = pd.DataFrame(std_row).transpose()

                        if prefix == 'gf':
                            resultados_gf[metric_name] = pd.concat([resultados_gf[metric_name], mean_row], axis=0)
                            resultados_gf_dp[metric_name] = pd.concat([resultados_gf_dp[metric_name], std_row], axis=0)
                        elif prefix == 'gs':
                            resultados_gs[metric_name] = pd.concat([resultados_gs[metric_name], mean_row], axis=0)
                            resultados_gs_dp[metric_name] = pd.concat([resultados_gs_dp[metric_name], std_row], axis=0)
                        elif prefix == 'gc':
                            resultados_gc[metric_name] = pd.concat([resultados_gc[metric_name], mean_row], axis=0)
                            resultados_gc_dp[metric_name] = pd.concat([resultados_gc_dp[metric_name], std_row], axis=0)
                        elif prefix == 'sg':
                            resultados_sg[metric_name] = pd.concat([resultados_sg[metric_name], mean_row], axis=0)
                            resultados_sg_dp[metric_name] = pd.concat([resultados_sg_dp[metric_name], std_row], axis=0)

    # Add rows with max value info
    resultados_gf = adicionar_maior_valor(resultados_gf)
    resultados_gs = adicionar_maior_valor(resultados_gs)
    resultados_gc = adicionar_maior_valor(resultados_gc)
    resultados_sg = adicionar_maior_valor(resultados_sg)

    # Save consolidated means and std deviations to Excel files
    salvar_em_xlsx(os.path.join(path_dir_combinacoes, 'Medias_gf.xlsx'), resultados_gf)
    salvar_em_xlsx(os.path.join(path_dir_combinacoes, 'Medias_gs.xlsx'), resultados_gs)
    salvar_em_xlsx(os.path.join(path_dir_combinacoes, 'Medias_gc.xlsx'), resultados_gc)
    salvar_em_xlsx(os.path.join(path_dir_combinacoes, 'Medias_sg.xlsx'), resultados_sg)

    salvar_em_xlsx(os.path.join(path_dir_combinacoes, 'Dp_gf.xlsx'), resultados_gf_dp)
    salvar_em_xlsx(os.path.join(path_dir_combinacoes, 'Dp_gs.xlsx'), resultados_gs_dp)
    salvar_em_xlsx(os.path.join(path_dir_combinacoes, 'Dp_gc.xlsx'), resultados_gc_dp)
    salvar_em_xlsx(os.path.join(path_dir_combinacoes, 'Dp_sg.xlsx'), resultados_sg_dp)

if __name__ == "__main__":
    # Root folder containing subfolders with Excel result files
    path_dir_combinacoes = 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\results_CV\\'

    # Extract means and std deviations, consolidate and save
    extrair_salvar_medias(path_dir_combinacoes)
