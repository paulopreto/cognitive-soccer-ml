"""
-------------------------------------------------------------------------------
Script: Generate Confusion Matrices using Leave-One-Out Cross-Validation (LOO)
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script automates the generation of a single image containing four confusion 
matrices, each corresponding to a specific cognitive-performance dataset. The 
evaluation is conducted using Leave-One-Out Cross-Validation (LOO) and utilizes 
pre-optimized hyperparameters for different machine learning algorithms.

The pipeline includes:
1. Loading pre-saved datasets and their respective best hyperparameters.
2. Constructing and training models within a standardized pipeline (scaling + classifier).
3. Executing LOO validation to predict each sample.
4. Computing and visualizing confusion matrices and associated metrics (accuracy, 
   balanced accuracy, precision, recall).
5. Saving the final figure containing all 4 matrices for further analysis.

Algorithms Supported:
- Naive Bayes
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- XGBoost

Special Considerations:
- For dataset 'gf', the classifier is forced to MLP regardless of the selected algorithm.
- Label inversion is applied to the 'gs' dataset to match domain-specific interpretation.

-------------------------------------------------------------------------------
"""

import os
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, accuracy_score,
                             balanced_accuracy_score)

# Import machine learning classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def load_best_params(path_csv, algorithm_name):
    """
    Load the best hyperparameters for a given algorithm from a CSV file.
    """
    with open(path_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Algorithm Name'] == algorithm_name:
                return eval(row['Best Params'])
    raise ValueError(f"Parameters for {algorithm_name} not found in {path_csv}")

def get_model_by_name(name, params):
    """
    Instantiate a machine learning model given its name and hyperparameters.
    """
    if name == 'GaussianNB':
        return GaussianNB(**params)
    elif name == 'RandomForestClassifier':
        return RandomForestClassifier(**params)
    elif name == 'KNeighborsClassifier':
        return KNeighborsClassifier(**params)
    elif name == 'LogisticRegression':
        return LogisticRegression(**params)
    elif name == 'SVC':
        return SVC(**params)
    elif name == 'MLPClassifier':
        return MLPClassifier(**params)
    elif name == 'XGBClassifier':
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported algorithm: {name}")

def get_title_from_identifier(identifier):
    """
    Return a descriptive title based on dataset identifier.
    """
    titles = {
        'gf': "Individual Goals",
        'gs': "Conceded Goals",
        'gc': "Goals by Teammates",
        'sg': "Net Goals",
    }
    return titles.get(identifier, f"Confusion Matrix - {identifier.upper()}")

def run_all_LOO_matrices(datasets_info, algorithm_name, path_save_img_final):
    """
    Execute LOO validation across multiple datasets, generate confusion matrices,
    and save the final image.
    """
    # Adjust global tick label sizes for plots
    plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})

    # Create a 2x2 grid layout for plotting confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Iterate through datasets information
    for idx, info in enumerate(datasets_info):
        path_data = info['path_data']
        path_csv = info['path_best_params_csv']
        identifier = info['identificador']

        # Load dataset (train/test splits) from .pkl file
        with open(os.path.join(path_data, f"cogfut_{identifier}.pkl"), 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)

        # Concatenate train and test splits for full data LOO evaluation
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        # Force model selection to MLPClassifier for 'gf' dataset
        if identifier == 'gf':
            model_name_to_use = 'MLPClassifier'
        else:
            model_name_to_use = algorithm_name

        # Load best hyperparameters for current dataset and algorithm
        best_params = load_best_params(os.path.join(path_csv, f"{identifier}_parametros.csv"), model_name_to_use)
        model = get_model_by_name(model_name_to_use, best_params)

        # Build preprocessing and classification pipeline
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', model)])

        # Initialize Leave-One-Out cross-validator
        loo = LeaveOneOut()
        y_true, y_pred = [], []

        # Execute manual Leave-One-Out Cross-Validation loop
        for train_idx, test_idx in loo.split(X):
            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]
            pipeline.fit(X_train_cv, y_train_cv)
            pred = pipeline.predict(X_test_cv)
            y_true.append(y_test_cv[0])
            y_pred.append(pred[0])

        # Compute Confusion Matrix and metrics
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)

        # Adjust class label ordering for 'gs' dataset
        if identifier == 'gs':
            class_labels = ["High", "Low"]
        else:
            class_labels = ["Low", "High"]

        # Plot the confusion matrix in the respective subplot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=axes[idx], cmap='Reds', colorbar=False, values_format='d')

        # Increase font size of matrix cell values
        for text in disp.text_.ravel():
            text.set_fontsize(16)

        # Adjust tick label sizes on axes
        axes[idx].tick_params(axis='both', which='major', labelsize=14)

        # Set title and axis labels for the subplot
        axes[idx].set_title(get_title_from_identifier(identifier), fontsize=16, weight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=14)
        axes[idx].set_ylabel('True Label', fontsize=14)

        # Prepare metrics summary text box for display next to the matrix
        metrics_text = f"Accuracy: {acc:.2%}\nBalanced Acc.: {balanced_acc:.2%}\n\n"
        for i, label in enumerate(class_labels):
            metrics_text += f"{label}:\n"
            metrics_text += f"  - Precision: {precision[i]:.2f}\n"
            metrics_text += f"  - Recall:    {recall[i]:.2f}\n\n"

        # Place the metrics box on the plot area
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        axes[idx].text(1.05, 0.5, metrics_text.strip(), transform=axes[idx].transAxes,
                       fontsize=12, bbox=props, verticalalignment='center')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Ensure target directory exists and save the final figure
    if not os.path.exists(path_save_img_final):
        os.makedirs(path_save_img_final)
    full_path = os.path.join(path_save_img_final, f"confusion_matrices_LOO_{algorithm_name}.png")
    plt.savefig(full_path, dpi=300)
    plt.show()
    print(f"[OK] Final figure saved: {full_path}")

# ====================
# Execution Entry Point
# ====================
if __name__ == "__main__":
    run_all_LOO_matrices(
        datasets_info=[
            {
                'path_data': 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\ML_datasets\\Memory_span_Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)',
                'path_best_params_csv': 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\best_param\\Memory_span_Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)',
                'identificador': 'gf'
            },
            {
                'path_data': 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\ML_datasets\\Memory_span',
                'path_best_params_csv': 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\best_param\\Memory_span',
                'identificador': 'gs'
            },
            {
                'path_data': 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\ML_datasets\\Acuracia_nogo_Capacidade_de_rastreamento',
                'path_best_params_csv': 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\best_param\\Acuracia_nogo_Capacidade_de_rastreamento',
                'identificador': 'gc'
            },
            {
                'path_data': 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\data\\ML_datasets\\Memory_span_Acuracia_nogo_Flexibilidade_cognitiva_(B-A)',
                'path_best_params_csv': 'D:\\Processamento_mestrado_Sports_Science\\final_analysis\\best_param\\Memory_span_Acuracia_nogo_Flexibilidade_cognitiva_(B-A)',
                'identificador': 'sg'
            },
        ],
        algorithm_name='KNeighborsClassifier',
        path_save_img_final='D:\\Processamento_mestrado_Sports_Science\\final_analysis\\matriz_final_LOO'
    )


