"""
------------------------------------------------------------------------------
Anti-Overfitting Validation: Permutation Test for Model Significance
------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script provides an "anti-overfitting" audit by mathematically testing whether
the model learns real patterns rather than noise. It uses a permutation test:
labels are shuffled at random many times (e.g. 1,000) and the model is re-evaluated
on each shuffled dataset. The true model's score is then compared to the distribution
of permutation scores; a low empirical p-value indicates that the model performs
significantly better than chance.

This addresses reviewer concerns about overfitting in small samples (N=44) by
providing empirical evidence that the classifier's performance is not due to
random chance.

Reference: Ojala & Garriga (2010). Permutation tests for studying classifier
performance. J. Mach. Learn. Res.

Usage:
------
Run from project root: python -m src.validate_overfitting
Or call run_permutation_test() with desired paths.

Output:
-------
- figures/permutation_test_proof.png (histogram of permutation scores + true score)
- Console: empirical p-value and interpretation.
-------------------------------------------------------------------------------
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import permutation_test_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


def run_permutation_test(
    path_pkl, n_permutations=1000, output_dir=None, random_state=0
):
    """
    Run permutation test to assess whether model performance is better than chance.

    Loads processed data (U-17 elite soccer players), fits a pipeline (StandardScaler
    + KNN as the best-performing model in the paper), and compares the true score
    against 1,000 permutations with shuffled labels. Saves a histogram to
    figures/permutation_test_proof.png.

    Parameters
    ----------
    path_pkl : str or pathlib.Path
        Path to a .pkl file containing (X_train, y_train, X_test, y_test).
    n_permutations : int
        Number of label permutations (default 1000).
    output_dir : str or pathlib.Path, optional
        Directory to save the figure. Default: project_root/figures.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (true_score, perm_scores, pvalue)
    """
    path_pkl = Path(path_pkl)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if output_dir is None:
        output_dir = project_root / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(path_pkl, "rb") as f:
        X_train, y_train, X_test, y_test = pickle.load(f)

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    n_samples = len(y)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, p=2)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    score, perm_scores, pvalue = permutation_test_score(
        pipeline,
        X,
        y,
        cv=cv,
        n_permutations=n_permutations,
        random_state=random_state,
        n_jobs=-1,
    )

    # Build figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_scores, bins=30, color="lightgray", edgecolor="black", alpha=0.8)
    ax.axvline(score, color="red", linewidth=2, label=f"True model (score={score:.3f})")
    ax.set_xlabel("Balanced accuracy (permuted labels)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    p_label = f"{pvalue:.3f}" if pvalue >= 0.001 else "< 0.001"
    title = f"Permutation Test (N={n_samples}): Empirical p-value = {p_label}"
    if pvalue < 0.05:
        title += " — Model significantly better than chance."
    ax.set_title(title, fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "permutation_test_proof.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Permutation test completed. True score: {score:.4f}")
    print(f"     Empirical p-value: {pvalue:.4f}")
    print(f"     Figure saved: {out_path}")
    if pvalue < 0.05:
        print(
            "     Conclusion: Model performance is significantly better than chance (p < 0.05)."
        )
    else:
        print("     Conclusion: Cannot reject null (performance may be due to chance).")

    return score, perm_scores, pvalue


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "ML_datasets"

    default_pkl = None
    if data_dir.is_dir():
        for sub in sorted(data_dir.iterdir()):
            if sub.is_dir():
                pkl_path = sub / "cogfut_gf.pkl"
                if pkl_path.is_file():
                    default_pkl = pkl_path
                    break

    if default_pkl is None:
        fallbacks = [
            project_root / "data" / "cogfut_gf.pkl",
            data_dir
            / "Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)"
            / "cogfut_gf.pkl",
        ]
        for candidate in fallbacks:
            if candidate.is_file():
                default_pkl = candidate
                break

    if default_pkl is None:
        raise FileNotFoundError(
            "No processed .pkl file found. Run data preprocessing first (e.g. data_preprocessing.run) "
            "or set PATH_PKL to a valid cogfut_*.pkl path."
        )

    import os

    path_pkl_env = os.environ.get("PATH_PKL")
    run_permutation_test(
        path_pkl=Path(path_pkl_env) if path_pkl_env else default_pkl,
        n_permutations=1000,
        output_dir=project_root / "figures",
    )
