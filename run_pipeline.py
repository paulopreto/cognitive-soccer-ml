#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------
Orchestrator: Full pipeline for cognitive soccer ML (data → nested CV → validation)
-------------------------------------------------------------------------------
Run the complete study with a single command: python run_pipeline.py

Steps:
  1. Data processing (prepare .pkl datasets from cognitive + performance data)
  2. Nested CV evaluation (hyperparameter tuning + cross-validation)
  2b. Save fitted models to models/<combo>/<identifier>.joblib
  3. Overfitting validation (permutation test to prove model > chance)

All paths are relative to the project root. Uses pathlib for cross-platform
portability (macOS, Windows, Linux).
-------------------------------------------------------------------------------
"""

import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Paths (cross-platform via pathlib)
    data_dir = project_root / "data"
    path_csv = data_dir / "Dataset_cog_clusters.csv"
    if not path_csv.is_file():
        path_csv = data_dir / "dataset.csv"
    ml_datasets_dir = data_dir / "ML_datasets"
    best_param_dir = project_root / "best_param"
    results_cv_dir = project_root / "results_CV"

    # ----- Step 1: Data processing -----
    print("Step 1: Processing data...")
    try:
        import data_preprocessing

        colunas = [
            "Memory span",
            "Acuracia Go",
            "Acuracia nogo",
            "Capacidade de rastreamento",
            "Flexibilidade cognitiva (B-A)",
        ]
        combo = list(colunas[:2])
        path_save_combo = ml_datasets_dir / "_".join(c.replace(" ", "_") for c in combo)
        if path_csv.is_file():
            data_preprocessing.run(
                path_to_data=str(path_csv),
                path_save=str(path_save_combo),
                n_clusters_Desempenho_campo=2,
                oversample=False,
                normal=True,
                metade=False,
                colunas_func_cog=combo,
            )
            print("  Data processing completed.")
        else:
            print(
                "  Skipping data processing (no CSV found). Using existing .pkl if present."
            )
    except Exception as e:
        print(f"  Warning: Data processing step failed: {e}")
        print("  Continuing with existing data if available.")

    # ----- Step 2: Nested CV evaluation -----
    print("Step 2: Validating model robustness (Nested CV evaluation)...")
    try:
        import train_final_models

        train_final_models.run_training_pipeline(
            path_dir_combinations=str(ml_datasets_dir),
            path_best_param_base=str(best_param_dir),
            path_results_cv_base=str(results_cv_dir),
        )
        print("  Nested CV evaluation completed.")
    except Exception as e:
        print(f"  Warning: Nested CV step failed: {e}")
        print("  You may need to run hyperparameter tuning first or check paths.")

    # ----- Step 2b: Save fitted models to models/ -----
    print("Step 2b: Saving fitted models to models/...")
    try:
        import save_models

        models_dir = project_root / "models"
        save_models.run_save_models(
            path_ml_datasets=str(ml_datasets_dir),
            path_best_param=str(best_param_dir),
            path_models_dir=str(models_dir),
        )
        print("  Models saved to models/.")
    except Exception as e:
        print(f"  Warning: Save models step failed: {e}")

    # ----- Step 3: Overfitting validation (permutation test) -----
    print("Step 3: Validating model robustness (permutation test)...")
    try:
        import validate_overfitting

        pkl_path = None
        if ml_datasets_dir.is_dir():
            for sub in ml_datasets_dir.iterdir():
                if sub.is_dir():
                    cand = sub / "cogfut_gf.pkl"
                    if cand.is_file():
                        pkl_path = str(cand)
                        break
        if pkl_path:
            validate_overfitting.run_permutation_test(
                path_pkl=pkl_path,
                n_permutations=1000,
                output_dir=str(project_root / "figures"),
            )
            print(
                "  Permutation test completed. See figures/permutation_test_proof.png"
            )
        else:
            print("  Skipping permutation test (no .pkl found). Run Step 1 first.")
    except Exception as e:
        print(f"  Warning: Permutation test failed: {e}")

    print("\nPipeline run finished.")


if __name__ == "__main__":
    main()
