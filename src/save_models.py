"""
-------------------------------------------------------------------------------
Save fitted pipelines to disk for later inference.
-------------------------------------------------------------------------------
After hyperparameter tuning and nested CV, this script fits one pipeline per
dataset combination and target (gf, gs, gc, sg) using the best KNN parameters,
and saves them under models/<combo_name>/<identifier>.joblib.

Use these saved models for prediction on new data (e.g. in tests or apps).
-------------------------------------------------------------------------------
"""

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .nested_cv_evaluation import extract_params_dict, extract_params_dict_correct
except ImportError:
    from nested_cv_evaluation import extract_params_dict, extract_params_dict_correct

IDENTIFIERS = ("gf", "gs", "gc", "sg")
# CSV row index for KNN in best_param CSV (0=Naive, 1=RF, 2=KNN, ...)
KNN_ROW = 2


def _load_data(pkl_path: Path):
    """Load X, y from a cogfut_*.pkl file (train + test concatenated)."""
    with open(pkl_path, "rb") as f:
        X_train, y_train, X_test, y_test = pickle.load(f)
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    return X, y


def _build_knn_pipeline(params: dict, oversample: bool = False):
    """Build a scaler + KNN pipeline (no SMOTE by default for inference)."""
    clf = KNeighborsClassifier(**params)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run_save_models(
    path_ml_datasets: str | Path,
    path_best_param: str | Path,
    path_models_dir: str | Path,
    *,
    oversample: bool = False,
) -> None:
    """
    For each dataset combination, fit a KNN pipeline per target (gf, gs, gc, sg)
    using best params and save to models/<combo_name>/<identifier>.joblib.

    Paths can be str or pathlib.Path. Uses pathlib internally.
    """
    base_data = Path(path_ml_datasets)
    base_param = Path(path_best_param)
    models_dir = Path(path_models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for subdir in sorted(base_data.iterdir()):
        if not subdir.is_dir():
            continue
        combo_name = subdir.name
        param_dir = base_param / combo_name
        if not param_dir.is_dir():
            continue
        out_combo = models_dir / combo_name
        out_combo.mkdir(parents=True, exist_ok=True)

        for identifier in IDENTIFIERS:
            pkl_path = subdir / f"cogfut_{identifier}.pkl"
            param_file = param_dir / f"{identifier}_parametros.csv"
            if not pkl_path.is_file() or not param_file.is_file():
                continue
            X, y = _load_data(pkl_path)
            best_param = pd.read_csv(param_file, sep=",")
            try:
                params = extract_params_dict(best_param.iloc[KNN_ROW, 0])
            except Exception:
                params = extract_params_dict_correct(best_param.iloc[KNN_ROW, 2])
            pipeline = _build_knn_pipeline(params, oversample=oversample)
            pipeline.fit(X, y)
            out_path = out_combo / f"{identifier}.joblib"
            joblib.dump(pipeline, out_path)
            print(f"  Saved {out_path}")


if __name__ == "__main__":
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    run_save_models(
        path_ml_datasets=_root / "data" / "ML_datasets",
        path_best_param=_root / "best_param",
        path_models_dir=_root / "models",
    )
