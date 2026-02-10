"""
-------------------------------------------------------------------------------
Complete Training Pipeline: Hyperparameter Optimization and Nested Cross-Validation
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script automates the process of hyperparameter optimization and model evaluation
using Grid Search and Nested Cross-Validation for multiple machine learning algorithms.

It systematically iterates through combinations of datasets, applies hyperparameter
search, and evaluates the selected models using stratified K-Fold Cross-Validation.
The optimized parameters and validation results are saved in structured directories
for further statistical analysis.

Nested Cross-Validation is used for small sample size mitigation: it provides an
unbiased estimate of generalization performance by keeping a separate outer loop
for evaluation and an inner loop for hyperparameter selection, reducing the risk
of overfitting and optimistic bias in small cohorts (e.g. N=44).

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

-------------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

# Ensure src directory is on path when run from project root
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import hyperparameter_tuning
import nested_cv_evaluation


def run_training_pipeline(path_dir_combinations, path_best_param_base, path_results_cv_base):
    """
    Run hyperparameter tuning and nested CV evaluation for each dataset combination.

    Paths can be str or pathlib.Path; uses pathlib internally for cross-platform
    portability (macOS, Windows, Linux).
    """
    base_combinations = Path(path_dir_combinations)
    base_param = Path(path_best_param_base)
    base_results = Path(path_results_cv_base)

    for subdir in sorted(base_combinations.iterdir()):
        if not subdir.is_dir():
            continue
        dir_name = subdir.name
        path_data = str(base_combinations / dir_name)
        path_param = str(base_param / dir_name)
        path_save_valid = str(base_results / dir_name)

        # Hyperparameter Optimization (Grid Search)
        hyperparameter_tuning.run_GridSearch(
            n_clusters_Desempenho_campo=2,
            n_splits_kfold=5,
            naive=True,
            randomforest=True,
            knn=True,
            Logistic_Regression=True,
            SVM=True,
            MLP=True,
            XGboost=True,
            oversample=False,
            normal=True,
            metade=False,
            path_data=path_data,
            path_save=path_param
        )

        # Nested Cross-Validation (Model Evaluation)
        nested_cv_evaluation.run_cross_validation(
            path_data=path_data,
            path_param=path_param,
            path_save=path_save_valid,
            n_clusters_Desempenho_campo=2,
            n_splits_kfold=5,
            oversample=False,
            normal=True,
            metade=False
        )


if __name__ == "__main__":
    # Default paths: use project-relative paths for portability
    path_dir_combinations = os.environ.get(
        "PATH_ML_DATASETS",
        str(PROJECT_ROOT / "data" / "ML_datasets")
    )
    path_best_param_base = os.environ.get(
        "PATH_BEST_PARAM",
        str(PROJECT_ROOT / "best_param")
    )
    path_results_cv_base = os.environ.get(
        "PATH_RESULTS_CV",
        str(PROJECT_ROOT / "results_CV")
    )

    run_training_pipeline(path_dir_combinations, path_best_param_base, path_results_cv_base)
