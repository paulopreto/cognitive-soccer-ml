"""
Smoke tests: ensure all main modules can be imported when run from project root.
Run from repo root: uv run pytest tests/ -v   or   python -m pytest tests/ -v
"""

import sys
from pathlib import Path


# Add project root and src so imports work
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_import_data_preprocessing():
    import data_preprocessing

    assert hasattr(data_preprocessing, "run")


def test_import_hyperparameter_tuning():
    import hyperparameter_tuning

    assert hasattr(hyperparameter_tuning, "run_GridSearch")


def test_import_nested_cv_evaluation():
    import nested_cv_evaluation

    assert hasattr(nested_cv_evaluation, "run_cross_validation")


def test_import_train_final_models():
    import train_final_models

    assert hasattr(train_final_models, "run_training_pipeline")


def test_import_validate_overfitting():
    import validate_overfitting

    assert hasattr(validate_overfitting, "run_permutation_test")


def test_import_save_models():
    import save_models

    assert hasattr(save_models, "run_save_models")
