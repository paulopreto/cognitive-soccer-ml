# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Generates a Single Image with 4 Confusion Matrices using Nested Cross-Validation
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro (adapted)

Description:
-------------
This script automates the process of performing Nested Cross-Validation (6 repetitions)
with inner hyperparameter optimization (Grid Search) and outer model evaluation.
For each dataset, a confusion matrix is generated based on predictions aggregated
over the entire Nested CV procedure.

The script generates a single figure containing 4 confusion matrices, each corresponding
to a different dataset configuration. The figure is saved as a high-resolution image.

Pipeline Steps:
---------------
1. Iterate through each dataset.
2. For each dataset:
    a. Perform Nested Cross-Validation (5-fold Outer, 5-fold Inner).
    b. Repeat the Nested CV 6 times with different random seeds.
    c. Aggregate predictions from all folds and repetitions.
    d. Compute performance metrics (Accuracy, Balanced Accuracy, Precision, Recall).
    e. Plot confusion matrix with metric annotations.
3. Save the final image containing all confusion matrices.

Supported Algorithms:
---------------------
- KNeighborsClassifier
- MLPClassifier (only for 'gf' dataset)

-------------------------------------------------------------------------------
"""

import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# -------------------------------
# Function to Instantiate Model by Name
# -------------------------------
def get_model_by_name(name):
    if name == "GaussianNB":
        return GaussianNB()
    elif name == "RandomForestClassifier":
        return RandomForestClassifier()
    elif name == "KNeighborsClassifier":
        return KNeighborsClassifier()
    elif name == "LogisticRegression":
        return LogisticRegression()
    elif name == "SVC":
        return SVC()
    elif name == "MLPClassifier":
        return MLPClassifier()
    elif name == "XGBClassifier":
        return XGBClassifier()
    else:
        raise ValueError(f"Unsupported algorithm: {name}")


# -------------------------------
# Function to Define Grid Search Parameter Space per Algorithm
# -------------------------------
def get_param_grid(name):
    if name == "KNeighborsClassifier":
        return {"clf__n_neighbors": [1, 2, 3, 5, 10, 20], "clf__p": [1, 2]}
    elif name == "MLPClassifier":
        return {
            "clf__max_iter": [100, 500, 1000, 1500],
            "clf__activation": ["relu", "logistic", "tanh"],
            "clf__solver": ["adam", "sgd"],
            "clf__hidden_layer_sizes": [(5, 5), (10, 10), (25, 25), (50, 50)],
            "clf__random_state": [0],
        }
    else:
        raise ValueError(f"No grid defined for: {name}")


# -------------------------------
# Function to Get Plot Title from Dataset Identifier
# -------------------------------
def get_title_from_identifier(identifier):
    titles = {
        "gf": "Individual Goals",
        "gs": "Conceded Goals",
        "gc": "Goals by Teammates",
        "sg": "Net Goals",
    }
    return titles.get(identifier, f"Confusion Matrix - {identifier.upper()}")


# -------------------------------
# Main Function to Run Nested Cross-Validation and Plot Confusion Matrices
# -------------------------------
def run_all_NestedCV_matrices(datasets_info, algorithm_name, path_save_img_final):
    plt.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14})

    # Create 2x2 grid for the four confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Define seeds for the 6 repetitions of Nested CV
    seeds = [42 + i for i in range(6)]

    # Iterate through each dataset configuration
    for idx, info in enumerate(datasets_info):
        path_data = Path(info["path_data"])
        identifier = info["identificador"]

        # Load dataset (pickle file format)
        with open(path_data / f"cogfut_{identifier}.pkl", "rb") as f:
            X_train, y_train, X_test, y_test = pickle.load(f)

        # Concatenate train and test sets for full dataset evaluation
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        # Encode labels if they are not numeric
        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)

        # Algorithm selection (MLP forced for 'gf' dataset)
        if identifier == "gf":
            model_name = "MLPClassifier"
        else:
            model_name = algorithm_name

        model = get_model_by_name(model_name)
        param_grid = get_param_grid(model_name)

        y_true_all, y_pred_all = [], []

        # Nested Cross-Validation (6 repetitions)
        for seed in seeds:
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            # Outer loop (model evaluation)
            for train_idx, test_idx in outer_cv.split(X, y):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                # Define pipeline: StandardScaler + Classifier
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])

                # Hyperparameter Optimization (Grid Search with Inner CV)
                grid_search = GridSearchCV(
                    pipe,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring="balanced_accuracy",
                    n_jobs=-1,
                )

                grid_search.fit(X_tr, y_tr)
                best_model = grid_search.best_estimator_
                y_pr = best_model.predict(X_te)

                y_true_all.extend(y_te)
                y_pred_all.extend(y_pr)

        # Compute Final Metrics
        cm = confusion_matrix(y_true_all, y_pred_all)
        acc = accuracy_score(y_true_all, y_pred_all)
        balanced_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        precision = precision_score(
            y_true_all, y_pred_all, average=None, zero_division=0
        )
        recall = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)

        # Adjust class labels display order for 'gs' identifier
        if identifier == "gs":
            class_labels = ["High", "Low"]
        else:
            class_labels = ["Low", "High"]

        # Plot Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=axes[idx], cmap="Reds", colorbar=False, values_format="d")

        # Increase font size inside cells
        for text in disp.text_.ravel():
            text.set_fontsize(16)

        # Axis label formatting
        axes[idx].tick_params(axis="both", which="major", labelsize=14)
        axes[idx].set_title(
            get_title_from_identifier(identifier), fontsize=16, weight="bold"
        )
        axes[idx].set_xlabel("Predicted Label", fontsize=14)
        axes[idx].set_ylabel("True Label", fontsize=14)

        # Annotate Metrics beside each Confusion Matrix
        metrics_text = f"Accuracy: {acc:.2%}\nBalanced Acc.: {balanced_acc:.2%}\n\n"
        for i, label in enumerate(class_labels):
            metrics_text += f"{label}:\n"
            metrics_text += f"  - Precision: {precision[i]:.2f}\n"
            metrics_text += f"  - Recall:    {recall[i]:.2f}\n\n"

        props = dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"
        )
        axes[idx].text(
            1.05,
            0.5,
            metrics_text.strip(),
            transform=axes[idx].transAxes,
            fontsize=12,
            bbox=props,
            verticalalignment="center",
        )

    # Adjust layout and save final figure
    plt.tight_layout()
    path_save_img_final = Path(path_save_img_final)
    path_save_img_final.mkdir(parents=True, exist_ok=True)
    full_path = path_save_img_final / f"confusion_matrices_NestedCV_{algorithm_name}.png"
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"[OK] Final figure saved: {full_path}")


# -------------------------------
# Default best combinations per target (article Table 1 / Figure 8)
# -------------------------------
BEST_COMBO_GF = "Memory_span_Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)"
BEST_COMBO_GS = "Memory_span"
BEST_COMBO_GC = "Acuracia_nogo_Capacidade_de_rastreamento"
BEST_COMBO_SG = "Memory_span_Acuracia_nogo_Flexibilidade_cognitiva_(B-A)"


def run_nested_cv_figure(ml_datasets_dir, path_save_img_final, algorithm_name="KNeighborsClassifier"):
    """
    Run nested CV and save confusion matrices figure (reproduces article Figure 8).

    Uses the best cognitive-feature combination per target from the article.
    ml_datasets_dir and path_save_img_final can be str or pathlib.Path.
    """
    ml_datasets_dir = Path(ml_datasets_dir)
    path_save_img_final = Path(path_save_img_final)
    datasets_info = [
        {"path_data": str(ml_datasets_dir / BEST_COMBO_GF), "identificador": "gf"},
        {"path_data": str(ml_datasets_dir / BEST_COMBO_GS), "identificador": "gs"},
        {"path_data": str(ml_datasets_dir / BEST_COMBO_GC), "identificador": "gc"},
        {"path_data": str(ml_datasets_dir / BEST_COMBO_SG), "identificador": "sg"},
    ]
    run_all_NestedCV_matrices(
        datasets_info=datasets_info,
        algorithm_name=algorithm_name,
        path_save_img_final=str(path_save_img_final),
    )


# ===== EXECUTION =====
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    ml_datasets_dir = project_root / "data" / "ML_datasets"
    path_save = project_root / "figures"
    run_nested_cv_figure(
        ml_datasets_dir=ml_datasets_dir,
        path_save_img_final=path_save,
        algorithm_name="KNeighborsClassifier",
    )
