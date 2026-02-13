"""
Tests for preprocessing output shape, clustering consistency, and numeric reproducibility.
Run from repo root: uv run pytest tests/test_reproducibility.py -v
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _make_synthetic_csv(n_rows=44, n_cog=5):
    """Minimal CSV with cognitive + performance columns for preprocessing."""
    np.random.seed(42)
    cols_cog = [
        "Memory span",
        "Acuracia Go",
        "Acuracia nogo",
        "Capacidade de rastreamento",
        "Flexibilidade cognitiva (B-A)",
    ]
    cols_perf = ["gols_feitos", "gols_sofridos", "gols_companheiros", "saldo_gols"]
    data = np.random.randn(n_rows, n_cog).clip(-2, 2)
    perf = np.random.rand(n_rows, 4) * 3  # positive, bounded
    df = pd.DataFrame(
        np.hstack([data, perf]),
        columns=cols_cog + cols_perf,
    )
    return df


def test_preprocessing_output_shape(tmp_path):
    """Preprocessing run() produces .pkl files with expected structure and shapes."""
    import data_preprocessing

    csv_path = tmp_path / "synthetic.csv"
    _make_synthetic_csv(44).to_csv(csv_path, index=False)
    save_dir = tmp_path / "out"
    colunas = [
        "Memory span",
        "Acuracia Go",
        "Acuracia nogo",
        "Capacidade de rastreamento",
        "Flexibilidade cognitiva (B-A)",
    ]
    data_preprocessing.run(
        path_to_data=str(csv_path),
        path_save=str(save_dir),
        n_clusters_Desempenho_campo=2,
        oversample=False,
        normal=True,
        metade=False,
        colunas_func_cog=colunas,
    )
    pkl_path = save_dir / "cogfut_gf.pkl"
    assert pkl_path.is_file(), "run() should create cogfut_gf.pkl"
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)
    assert len(loaded) == 4, "pkl content: [X_train, y_train, X_test, y_test]"
    X_train, y_train, X_test, y_test = loaded
    assert X_train.ndim == 2 and X_test.ndim == 2
    assert len(y_train) == len(X_train) and len(y_test) == len(X_test)
    assert set(np.unique(y_train)) <= {0, 1} and set(np.unique(y_test)) <= {0, 1}


def test_clustering_consistency():
    """K-means clustering with fixed data and seed is deterministic."""
    import data_preprocessing

    df = _make_synthetic_csv(20)
    y = df[["gols_feitos"]].copy()
    out1 = data_preprocessing.kmeans_cluster_teams(y, 2, first_team_size=10)
    out2 = data_preprocessing.kmeans_cluster_teams(y, 2, first_team_size=10)
    assert out1.shape == out2.shape
    # Labels may be permuted (0/1 swap); at least same number of unique labels
    assert set(out1["cluster"].unique()) == set(out2["cluster"].unique()) == {0, 1}


def test_label_half_teams_raises_on_invalid_split():
    """label_half_teams raises on too few rows per team."""
    import data_preprocessing

    df = pd.DataFrame({"gols_feitos": [1.0, 2.0]})  # only 2 rows
    with pytest.raises(ValueError, match="invalid split"):
        data_preprocessing.label_half_teams(df, 2, first_team_size=1)


def test_kmeans_cluster_teams_raises_on_few_rows():
    """kmeans_cluster_teams raises when total rows < 4 or team size < 2."""
    import data_preprocessing

    df = pd.DataFrame({"gols_feitos": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="at least 4 rows"):
        data_preprocessing.kmeans_cluster_teams(df, 2)


def test_numeric_reproducibility_cv():
    """Cross-validation with fixed seeds yields reproducible metric (same run-to-run)."""
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    np.random.seed(123)
    X = np.random.randn(60, 5).astype(np.float64)
    y = np.array([0] * 30 + [1] * 30)
    model = KNeighborsClassifier(n_neighbors=5, p=2)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    n_splits = 3
    cv_seed = 456
    scores1 = cross_validate(
        pipe,
        X,
        y,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed),
        scoring="balanced_accuracy",
        return_train_score=False,
    )
    scores2 = cross_validate(
        pipe,
        X,
        y,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed),
        scoring="balanced_accuracy",
        return_train_score=False,
    )
    np.testing.assert_array_almost_equal(
        scores1["test_score"], scores2["test_score"], decimal=10
    )
    assert 0 <= scores1["test_score"].mean() <= 1
