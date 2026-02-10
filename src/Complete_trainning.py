"""
-------------------------------------------------------------------------------
Complete trainning Pipeline: Hyperparameter Optimization and Nested Cross-Validation
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script automates the process of hyperparameter optimization and model evaluation 
using Grid Search and Nested Cross-Validation for multiple machine learning algorithms. 

It systematically iteracts through combinations of datasets, applies hyperparameter 
search, and evaluates the selected models using stratified K-Fold Cross-Validation. 
The optimized parameters and validation results are saved in structured directories 
for further statistical analysis.

Machine Learning algorithms considered:
- Naive Bayes
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- XGBoost

Pipeline Steps:
---------------
1. Iterate through all dataset combinations.
2. Perform Grid Search to identify the best hyperparameters for each algorithm.
3. Evaluate models using Nested Cross-Validation with the optimized hyperparameters.
4. Save the best parameters and cross-validation results for statistical analysis.

Note:
- Oversampling and undersampling options are configurable.
- The script is designed for structured datasets in the context of sports science performance prediction.

-------------------------------------------------------------------------------
"""

import GridSearch
import Cross_validation
import os

if __name__ == "__main__":

    # Root directory containing dataset combinations for model evaluation
    path_dir_combinacoes = 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\ML_datasets'
    
    # Iterate through each subdirectory (representing a dataset combination)
    for root, dirs, files in os.walk(path_dir_combinacoes):
        for dir_name in dirs:
            
            # Define paths for current dataset, parameter storage, and results storage
            path_data = os.path.join(path_dir_combinacoes, dir_name)
            path_param = os.path.join('D:\\Processamento_mestrado_Sports_Science\\final_analysis\\best_param\\', dir_name)
            path_save_grid = path_param  # Save Grid Search results in parameter directory
            path_save_valid = os.path.join('D:\\Processamento_mestrado_Sports_Science\\final_analysis\\results_CV\\', dir_name)
            
            # ---------------------------
            # Hyperparameter Optimization (Grid Search)
            # ---------------------------
            # Perform exhaustive search to find the best hyperparameters for the selected models.
            # The search is executed with 5-fold Stratified Cross-Validation.
            GridSearch.run_GridSearch(
                n_clusters_Desempenho_campo=2,  # Number of clusters for target performance variable
                n_splits_kfold=5,               # Number of folds in K-Fold Cross-Validation
                naive=True,                     # Enable Naive Bayes
                randomforest=True,              # Enable Random Forest
                knn=True,                       # Enable K-Nearest Neighbors
                Logistic_Regression=True,       # Enable Logistic Regression
                SVM=True,                       # Enable Support Vector Machine
                MLP=True,                       # Enable Multi-Layer Perceptron (Neural Network)
                XGboost=True,                   # Enable XGBoost
                oversample=False,               # Disable oversampling techniques
                normal=True,                    # Use entire dataset (no undersampling)
                metade=False,                   # Do not reduce dataset size by half
                path_data=path_data,            # Path to dataset file
                path_save=path_save_grid        # Directory to save best parameters
            )

            # ---------------------------
            # Nested Cross-Validation (Model Evaluation)
            # ---------------------------
            # Evaluate models using Nested Cross-Validation with outer 5-folds,
            # applying the best hyperparameters obtained from Grid Search.
            Cross_validation.run_validacao_cruzada(
                path_data=path_data,                # Path to dataset file
                path_param=path_param,              # Path to best hyperparameters file
                path_save=path_save_valid,          # Directory to save validation results
                n_clusters_Desempenho_campo=2,      # Number of clusters for target performance variable
                n_splits_kfold=5,                   # Number of outer folds for Nested CV
                oversample=False,                   # Disable oversampling techniques
                normal=True,                        # Use entire dataset
                metade=False                        # Do not reduce dataset size by half
            )
