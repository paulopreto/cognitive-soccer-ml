"""
------------------------------------------------------------------------------
Grid Search Pipeline with Nested Cross-Validation for Cognitive-Match Analysis
------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script performs hyperparameter optimization using Grid Search with 
Stratified K-Fold Cross-Validation for various classification algorithms.
It loads pre-processed datasets (.pkl) generated from previous clustering pipelines,
applies oversampling (SMOTE) if selected, standardizes the data, and trains classifiers.
The best hyperparameters and balanced accuracy scores are saved in CSV files.

Supported Classifiers:
----------------------
- Gaussian Naive Bayes
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- XGBoost

Main Workflow:
---------------
1. Load the dataset from `.pkl` files (cognitive predictors and performance labels).
2. Concatenate training and test sets for cross-validation.
3. For each selected algorithm:
    a. Define hyperparameter grid.
    b. Apply oversampling (optional).
    c. Apply standardization.
    d. Perform Stratified K-Fold Cross-Validation.
    e. Save the best hyperparameters and balanced accuracy scores in CSV.
4. CSV file will be updated if already exists.

Inputs:
-------
- path_data: Directory containing `.pkl` datasets (gf, gs, gc, sg variants)
- path_save: Directory to save results CSV
- n_clusters_Desempenho_campo: Number of clusters used for labeling
- oversample: Whether to apply SMOTE balancing
- n_splits_kfold: Number of folds in StratifiedKFold

Outputs:
--------
- CSV files (per target variable) containing best hyperparameters and balanced accuracy
-------------------------------------------------------------------------------
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pickle
import numpy as np
import pandas as pd
import os
import csv

# -------------------------------
# Function to Run Grid Search per Algorithm and Dataset
# -------------------------------
def run_GridSearch(path_data, path_save, n_splits_kfold, n_clusters_Desempenho_campo=2,
                   naive=False, randomforest=False, knn=False, Logistic_Regression=False,
                   SVM=False, MLP=False, XGboost=False, oversample=False, normal=False, metade=False):
    
    # Define dataset file paths (4 targets)
    paths = [os.path.join(path_data, 'cogfut_gf.pkl'),
             os.path.join(path_data, 'cogfut_gs.pkl'),
             os.path.join(path_data, 'cogfut_gc.pkl'),
             os.path.join(path_data, 'cogfut_sg.pkl')]
    
    for i, path in enumerate(paths):
        identificador = path.split('_')[-1].split('.')[0]  # Extracting identifier (gf, gs, gc, sg)
        
        with open(path, 'rb') as f:
            X_cog_treinamento, y_campo_treinamento, X_cog_teste, y_campo_teste = pickle.load(f)
    
        # Concatenate train and test data for cross-validation
        X_cog = np.concatenate((X_cog_treinamento, X_cog_teste), axis=0)
        y_campo = np.concatenate((y_campo_treinamento, y_campo_teste), axis=0)
        
        # === Classifiers Selection ===
        if naive:
            ml_algorithm = GaussianNB()
            parametros = {}
            grid_search(ml_algorithm, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
        
        if randomforest:
            ml_algorithm = RandomForestClassifier()
            parametros = {'criterion': ['gini', 'entropy'],
                          'n_estimators': [10, 40, 100, 150],
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 5, 10]}
            grid_search(ml_algorithm, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
            
        if knn:
            ml_algorithm = KNeighborsClassifier()
            parametros = {'n_neighbors': [1, 2, 3, 5, 10, 20],
                          'p': [1, 2]}
            grid_search(ml_algorithm, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
            
        if Logistic_Regression:
            ml_algorithm = LogisticRegression()
            parametros = {'tol': [0.01, 0.001, 0.0001],
                          'C': [1.0, 1.5, 2.0],
                          'solver': ['lbfgs', 'sag', 'saga']}
            grid_search(ml_algorithm, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
        
        if SVM:
            ml_algorithm = SVC()
            parametros = {'tol': [0.001, 0.0001, 0.00001],
                          'C': [1.0, 1.5, 2.0],
                          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
            grid_search(ml_algorithm, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
        
        if MLP:
            ml_algorithm = MLPClassifier()
            parametros = {'max_iter': [100, 500, 1000, 1500],
                          'activation': ['relu', 'logistic', 'tanh'],
                          'solver': ['adam', 'sgd'],
                          'hidden_layer_sizes': [(5,5), (10,10), (25,25), (50,50)],
                          'random_state': [0]}
            grid_search(ml_algorithm, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
            print('Grid Search Complete:', identificador)

        if XGboost:
            ml_algorithm = XGBClassifier()
            parametros = {'max_depth': [4, 6, 8],
                          'learning_rate': [0.05, 0.1, 0.15],
                          'n_estimators': [100, 200],
                          'min_child_weight': [1, 5]}
            grid_search(ml_algorithm, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold)
            print('Grid Search Complete:', identificador)

# -------------------------------
# Grid Search Execution per Algorithm
# -------------------------------
def grid_search(ml_algorithm, parametros, X_cog, y_campo, identificador, path_save, oversample, n_splits_kfold):
    kfold = StratifiedKFold(n_splits=n_splits_kfold, shuffle=True, random_state=0)
    os.makedirs(path_save, exist_ok=True)
    
    # Define Pipeline Steps
    if oversample:
        steps = [('smote', SMOTE(sampling_strategy='minority', random_state=0)),
                 ('scaler', StandardScaler()),
                 ('clf', ml_algorithm)]
    else:
        steps = [('scaler', StandardScaler()), ('clf', ml_algorithm)]

    pipeline = Pipeline(steps)
    parametros = {f'clf__{k}': v for k, v in parametros.items()}

    grid_search = GridSearchCV(estimator=pipeline, cv=kfold, param_grid=parametros,
                               scoring='balanced_accuracy', refit='f1_macro')
    
    grid_search.fit(X_cog, y_campo)
    melhores_parametros = grid_search.best_params_
    melhor_resultado = grid_search.best_score_
    
    print(f"{ml_algorithm.__class__.__name__} - Best Params: {melhores_parametros}")
    print(f"{ml_algorithm.__class__.__name__} - Best Balanced Accuracy: {melhor_resultado}")
    
    # Save Best Params to CSV
    nome_arquivo_csv = f"{identificador}_parametros.csv"
    caminho_arquivo = os.path.join(path_save, nome_arquivo_csv)
    
    melhores_parametros = {k.replace('clf__', ''): v for k, v in melhores_parametros.items()}
    
    linha_nova = [ml_algorithm.__class__.__name__, melhor_resultado, melhores_parametros]
    
    if os.path.isfile(caminho_arquivo):
        df = pd.read_csv(caminho_arquivo)
        df_existing = df[df['Nome do Algoritmo'] == ml_algorithm.__class__.__name__]
        
        if not df_existing.empty:
            df.loc[df['Nome do Algoritmo'] == ml_algorithm.__class__.__name__, 
                   ['Melhor Resultado', 'Melhores Parametros']] = melhor_resultado, melhores_parametros
        else:
            df = pd.concat([df, pd.DataFrame([linha_nova], columns=df.columns)], ignore_index=True)
    else:
        df = pd.DataFrame([linha_nova], columns=['Nome do Algoritmo', 'Melhor Resultado', 'Melhores Parametros'])
    
    df.to_csv(caminho_arquivo, index=False)

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    run_GridSearch(n_clusters_Desempenho_campo=2, n_splits_kfold=5,
                   naive=False,
                   randomforest=False,
                   knn=True,
                   Logistic_Regression=False,
                   SVM=False,
                   MLP=False,
                   XGboost=False,
                   oversample=False,
                   normal=True,
                   metade=False,
                   path_data='D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\ML_datasets\\Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)',
                   path_save='D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)2\\')
