#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------
Orchestrator: Full pipeline for cognitive soccer ML (data → nested CV → validation)
-------------------------------------------------------------------------------
Run the complete study with a single command: python run_pipeline.py

Steps:
  1. Data processing (prepare .pkl datasets from cognitive + performance data)
  2. Nested CV evaluation (hyperparameter tuning + cross-validation)
  2a. Nested CV confusion matrices (Figure 8)
  2b. Save fitted models to models/<combo>/<identifier>.joblib
  3. Overfitting validation (permutation test to prove model > chance)

All paths are relative to the project root. Uses pathlib for cross-platform
portability (macOS, Windows, Linux).
-------------------------------------------------------------------------------
"""

from pathlib import Path

# Ensure package is importable when run as script from project root
if __name__ == "__main__":
    import sys

    _root = Path(__file__).resolve().parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from src.pipeline import run

    run(_root)
