"""
-------------------------------------------------------------------------------
Machine Learning Pipeline: Cross-Validation and Model Evaluation
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script performs cross-validation and evaluation of multiple machine learning
algorithms on datasets related to sports science. It supports oversampling techniques,
uses optimized hyperparameters loaded from CSV files, and calculates various
performance metrics using stratified K-Fold cross-validation.

Algorithms included:
- Naive Bayes
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- XGBoost

Main functionalities:
- Load datasets and corresponding best hyperparameters
- Run stratified cross-validation for each classifier pipeline
- Optionally apply SMOTE oversampling
- Calculate multiple metrics including accuracy, precision, recall, F1, ROC AUC,
  balanced accuracy, and Matthews correlation coefficient (MCC)
- Save detailed raw results and aggregated means to Excel files

-------------------------------------------------------------------------------
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import ast
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
import math

def save_raw_results_to_xlsx(results, path_save, identifier):
    """
    Save raw results to an XLSX file, with each sheet representing a metric.

    Args:
        results (dict): Dictionary with results for each algorithm.
        path_save (str): Path where to save the XLSX file.
        identifier (str): Identifier for the file name.
    """
    # Create directory if it does not exist
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    output_xlsx_name = f"{identifier}_resultados.xlsx"
    output_path = os.path.join(path_save, output_xlsx_name)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Use 'Naive' results keys to get metric names
        for metric in results['Naive'][0].keys():
            df_metric = pd.DataFrame({
                'Naive': [results['Naive'][i][metric] for i in range(len(results['Naive']))],
                'Random forest': [results['Random forest'][i][metric] for i in range(len(results['Random forest']))],
                'KNN': [results['KNN'][i][metric] for i in range(len(results['KNN']))],
                'Logistic Regression': [results['Logistic Regression'][i][metric] for i in range(len(results['Logistic Regression']))],
                'SVM': [results['SVM'][i][metric] for i in range(len(results['SVM']))],
                'Neural network': [results['Neural network'][i][metric] for i in range(len(results['Neural network']))],
                'Xgboost': [results['Xgboost'][i][metric] for i in range(len(results['Xgboost']))],
            })
            df_metric.to_excel(writer, sheet_name=metric, index=False)

def extract_params_dict(line):
    """
    Extracts hyperparameters dictionary from a CSV line string.

    Args:
        line (str): String line from CSV.

    Returns:
        dict: Hyperparameters as dictionary.
    """
    # Split the line by comma
    elements = line.split(',')

    # Join back from the 3rd element and strip extra quotes
    params_str = ','.join(elements[2:]).strip('""')

    # Safely evaluate string as Python dict
    params_dict = ast.literal_eval(params_str)
    
    return params_dict

def extract_params_dict_correct(line):
    """
    Alternative method to extract hyperparameters dictionary from string.

    Args:
        line (str): String representing dictionary.

    Returns:
        dict: Hyperparameters dictionary.
    """
    params_dict = ast.literal_eval(line)
    
    return params_dict

def compute_metric_means(dicts):
    """
    Calculates mean of each metric from a list of dictionaries, ignoring NaN values.

    Args:
        dicts (list of dict): List of metric dictionaries.

    Returns:
        dict: Dictionary of mean metrics.
    """
    media_dict = {}

    if not dicts:
        return media_dict

    # Compute mean per metric ignoring NaNs
    for metric in dicts[0]:
        values = [d[metric] for d in dicts if not math.isnan(d[metric])]
        if values:
            media_dict[metric] = sum(values) / len(values)
        else:
            media_dict[metric] = float('nan')

    return media_dict

def save_results_to_xlsx(results, path_save):
    """
    Save aggregated results to an XLSX file, each sheet is a dataset identifier.

    Args:
        results (dict): Dictionary with results per identifier.
        path_save (str): Path to save the file.
    """
    with pd.ExcelWriter(path_save, engine='openpyxl') as writer:
        for identifier, data in results.items():
            df = pd.DataFrame([data])
            df.to_excel(writer, sheet_name=identifier, index=False)

def aggregate_and_save_results(path_save, resultados_naive, resultados_random_forest,
                               resultados_knn, resultados_logistica, resultados_svm,
                               resultados_rede_neural, resultados_xgboost, identifier):
    """
    Aggregate results from each algorithm, save means and raw results.

    Args:
        path_save (str): Folder path to save results.
        resultados_* (list): List of metric dicts per algorithm.
        identifier (str): Dataset identifier.
    """
    results_dict = {
        'Naive': resultados_naive,
        'Random forest': resultados_random_forest,
        'KNN': resultados_knn,
        'Logistic Regression': resultados_logistica,
        'SVM': resultados_svm,
        'Neural network': resultados_rede_neural,
        'Xgboost': resultados_xgboost
    }

    # Calculate mean metrics for each algorithm
    mean_results = {algo: compute_metric_means(res) for algo, res in results_dict.items()}

    output_xlsx_name = f"{identifier}_medias.xlsx"
    output_path = os.path.join(path_save, output_xlsx_name)
    save_results_to_xlsx(mean_results, output_path)
    
    # Save the raw detailed results separately
    save_raw_results_to_xlsx(results_dict, path_save, identifier)


def run_cross_validation(path_data, path_param, path_save, n_clusters_Desempenho_campo=2, oversample=False,
                        normal=False, metade=False, n_splits_kfold=3, run_knn_only=False):
    """
    Main function to load datasets and parameters, run cross-validation, and save results.

    Args:
        path_data (str): Folder path with dataset pickles.
        path_param (str): Folder path with hyperparameter CSVs.
        path_save (str): Folder path to save results.
        n_clusters_Desempenho_campo (int): Number of clusters (unused here).
        oversample (bool): Whether to apply SMOTE oversampling.
        normal (bool): Unused here.
        metade (bool): Unused here.
        n_splits_kfold (int): Number of CV folds.
        run_knn_only (bool): If True, only run KNN classifier.
    """

    # Paths for each dataset variant
    paths = [os.path.join(path_data, 'cogfut_gf.pkl'),
             os.path.join(path_data, 'cogfut_gs.pkl'),
             os.path.join(path_data, 'cogfut_gc.pkl'),
             os.path.join(path_data, 'cogfut_sg.pkl')]
    
    paths_param = [os.path.join(path_param,'gf_parametros.csv'),
                   os.path.join(path_param,'gs_parametros.csv'),
                   os.path.join(path_param,'gc_parametros.csv'),
                   os.path.join(path_param,'sg_parametros.csv')]
    
    # Create save directory if missing
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    for i, (path, paths_params) in enumerate(zip(paths, paths_param)):
        
        # Extract dataset identifier: 'gf', 'gs', etc.
        identifier = path.split('_')[-1].split('.')[0]
        
        # Load dataset (training + test sets concatenated)
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        X_cog = np.concatenate((X_cog_treinamento, X_cog_teste), axis=0)
        y_campo = np.concatenate((y_campo_treinamento, y_campo_teste), axis=0)
        
        # Prepare lists to collect results for each algorithm
        resultados_naive = []
        resultados_random_forest = []
        resultados_knn = []
        resultados_logistica = []
        resultados_svm = []
        resultados_rede_neural = []
        resultados_xgboost = []
        
        # Load best hyperparameters from CSV
        best_param = pd.read_csv(paths_params, sep=',')
        
        # Define scoring metrics for cross-validation
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro'),
            'roc_auc': 'roc_auc',
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'mcc': make_scorer(matthews_corrcoef),
            'precision_class_0': make_scorer(precision_score, average=None, labels=[0]),
            'precision_class_1': make_scorer(precision_score, average=None, labels=[1]),
            'recall_class_0': make_scorer(recall_score, average=None, labels=[0]),
            'recall_class_1': make_scorer(recall_score, average=None, labels=[1]),
            'f1_class_0': make_scorer(f1_score, average=None, labels=[0]),
            'f1_class_1': make_scorer(f1_score, average=None, labels=[1])
        }
        
        # Run 30 iterations of stratified KFold CV with different seeds
        for j in range(30):
            print(f"Iteration {j}")
            kfold = StratifiedKFold(n_splits=n_splits_kfold, shuffle=True, random_state=j)
            
            if run_knn_only:
                # Only run KNN pipeline
                
                ######################################################################################
                # KNN Classifier Pipeline
                ######################################################################################
                try:
                    params = extract_params_dict(best_param.iloc[2,0])
                except:
                    params = extract_params_dict_correct(best_param.iloc[2,2])
                
                knn = KNeighborsClassifier(**params)
                
                if oversample:
                    pipeline = Pipeline([
                        ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                        ('scaler', StandardScaler()),
                        ('clf', knn)
                    ])
                else:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('clf', knn)
                    ])
                
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
                # Retry with lower k_neighbors in SMOTE if NaNs detected
                if np.any(np.isnan(scores['test_accuracy'])) and oversample:
                    pipeline = Pipeline([
                        ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                        ('scaler', StandardScaler()),
                        ('clf', knn)
                    ])
                    scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
                
                resultados_knn.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
            
                # Save results and exit early since only KNN was requested
                if identifier == 'gf':
                    aggregate_and_save_results(path_save, resultados_naive, resultados_random_forest, resultados_knn,
                                                  resultados_logistica, resultados_svm, resultados_rede_neural,
                                                  resultados_xgboost, 'gf')
                if identifier == 'gs':
                    aggregate_and_save_results(path_save, resultados_naive, resultados_random_forest, resultados_knn,
                                                  resultados_logistica, resultados_svm, resultados_rede_neural,
                                                  resultados_xgboost, 'gs')
                if identifier == 'gc':
                    aggregate_and_save_results(path_save, resultados_naive, resultados_random_forest, resultados_knn,
                                                  resultados_logistica, resultados_svm, resultados_rede_neural,
                                                  resultados_xgboost, 'gc')
                if identifier == 'sg':
                    aggregate_and_save_results(path_save, resultados_naive, resultados_random_forest, resultados_knn,
                                                  resultados_logistica, resultados_svm, resultados_rede_neural,
                                                  resultados_xgboost, 'sg')
                return()

            ######################################################################################
            # Naive Bayes Classifier Pipeline
            ######################################################################################
            naive = GaussianNB()
            
            if oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', naive)
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', naive)
                ])
            
            scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            # Retry with lower k_neighbors in SMOTE if NaNs detected
            if np.any(np.isnan(scores['test_accuracy'])) and oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                    ('scaler', StandardScaler()),
                    ('clf', naive)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            resultados_naive.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
            
            ######################################################################################
            # Random Forest Classifier Pipeline
            ######################################################################################
            try:
                params = extract_params_dict(best_param.iloc[1,0])
            except:
                params = extract_params_dict_correct(best_param.iloc[1,2])
            
            random_forest = RandomForestClassifier(**params)
            
            if oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', random_forest)
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', random_forest)
                ])
            
            scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            if np.any(np.isnan(scores['test_accuracy'])) and oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                    ('scaler', StandardScaler()),
                    ('clf', random_forest)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            resultados_random_forest.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
            
            ######################################################################################
            # KNN Classifier Pipeline
            ######################################################################################
            try:
                params = extract_params_dict(best_param.iloc[2,0])
            except:
                params = extract_params_dict_correct(best_param.iloc[2,2])
            
            knn = KNeighborsClassifier(**params)
            
            if oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', knn)
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', knn)
                ])
            
            scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            if np.any(np.isnan(scores['test_accuracy'])) and oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                    ('scaler', StandardScaler()),
                    ('clf', knn)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            resultados_knn.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
            
            ######################################################################################
            # Logistic Regression Pipeline
            ######################################################################################
            try:
                params = extract_params_dict(best_param.iloc[3,0])
            except:
                params = extract_params_dict_correct(best_param.iloc[3,2])
            
            logistica = LogisticRegression(**params)
            
            if oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', logistica)
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', logistica)
                ])
            
            scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            if np.any(np.isnan(scores['test_accuracy'])) and oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                    ('scaler', StandardScaler()),
                    ('clf', logistica)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            resultados_logistica.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
            
            ######################################################################################
            # Support Vector Machine Pipeline
            ######################################################################################
            try:
                params = extract_params_dict(best_param.iloc[4,0])
            except:
                params = extract_params_dict_correct(best_param.iloc[4,2])
            
            svm = SVC(**params)
            
            if oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', svm)
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', svm)
                ])
            
            scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            if np.any(np.isnan(scores['test_accuracy'])) and oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                    ('scaler', StandardScaler()),
                    ('clf', svm)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            resultados_svm.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
            
            ######################################################################################
            # Neural Network (MLPClassifier) Pipeline
            ######################################################################################
            try:
                params = extract_params_dict(best_param.iloc[5,0])
            except:
                params = extract_params_dict_correct(best_param.iloc[5,2])
            
            rede_neural = MLPClassifier(**params)
            
            if oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', rede_neural)
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', rede_neural)
                ])
            
            scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            if np.any(np.isnan(scores['test_accuracy'])) and oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                    ('scaler', StandardScaler()),
                    ('clf', rede_neural)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            resultados_rede_neural.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
            
            ######################################################################################
            # XGBoost Classifier Pipeline
            ######################################################################################
            try:
                params = extract_params_dict(best_param.iloc[6,0])
            except:
                params = extract_params_dict_correct(best_param.iloc[6,2])
            
            Xgboost = XGBClassifier(**params)
            
            if oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                    ('scaler', StandardScaler()),
                    ('clf', Xgboost)
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', Xgboost)
                ])
            
            scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            if np.any(np.isnan(scores['test_accuracy'])) and oversample:
                pipeline = Pipeline([
                    ('smote', SMOTE(sampling_strategy='minority', random_state=0, k_neighbors=3)),
                    ('scaler', StandardScaler()),
                    ('clf', Xgboost)
                ])
                scores = cross_validate(pipeline, X_cog, y_campo, cv=kfold, scoring=scoring, n_jobs=-1)
            
            resultados_xgboost.append({metric: np.mean(scores['test_' + metric]) for metric in scoring})
          
        # Save aggregated results after all folds for each dataset identifier
        if identifier == 'gf':
            aggregate_and_save_results(path_save, resultados_naive, resultados_random_forest, resultados_knn,
                                          resultados_logistica, resultados_svm, resultados_rede_neural,
                                          resultados_xgboost, 'gf')
                
        if identifier == 'gs':
            aggregate_and_save_results(path_save, resultados_naive, resultados_random_forest, resultados_knn,
                                          resultados_logistica, resultados_svm, resultados_rede_neural,
                                          resultados_xgboost, 'gs')
                
        if identifier == 'gc':
            aggregate_and_save_results(path_save, resultados_naive, resultados_random_forest, resultados_knn,
                                          resultados_logistica, resultados_svm, resultados_rede_neural,
                                          resultados_xgboost, 'gc')
                      
        if identifier == 'sg':
            aggregate_and_save_results(path_save, resultados_naive, resultados_random_forest, resultados_knn,
                                          resultados_logistica, resultados_svm, resultados_rede_neural,
                                          resultados_xgboost, 'sg')

if __name__ == "__main__":

    # Run the main cross-validation pipeline with specified parameters
    run_cross_validation(
        path_data='D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\ML_datasets\\Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)',
        path_param='D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)',
        path_save='D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)\\resultados_ml',
        n_clusters_Desempenho_campo=2,
        n_splits_kfold=5,
        oversample=False,
        normal=True,
        metade=False,
        run_knn_only=True
    )


