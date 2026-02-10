"""
If models/ exists and contains joblib files, check they can be loaded and predict.
Run after: uv run python run_pipeline.py  (or at least Step 2b).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.mark.skipif(
    not MODELS_DIR.is_dir(), reason="models/ not present; run pipeline first"
)
def test_models_dir_exists_and_has_joblib():
    joblib_files = list(MODELS_DIR.rglob("*.joblib"))
    assert joblib_files, "models/ should contain at least one .joblib file"


@pytest.mark.skipif(
    not MODELS_DIR.is_dir(), reason="models/ not present; run pipeline first"
)
def test_load_one_model_and_predict():
    import joblib
    import numpy as np

    joblib_files = list(MODELS_DIR.rglob("*.joblib"))
    if not joblib_files:
        pytest.skip("No .joblib files in models/")
    path = joblib_files[0]
    pipeline = joblib.load(path)
    scaler = pipeline.named_steps.get("scaler")
    n_features = getattr(scaler, "n_features_in_", 5)
    X = np.zeros((2, n_features))
    pred = pipeline.predict(X)
    assert pred.shape == (2,)
    assert pred.dtype in (np.int64, np.int32, np.object_)
