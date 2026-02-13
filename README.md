# [Are Cognitive Functions Capable of Predicting Overall Effectiveness and Player Contributions in Small-Sided Soccer Games? A Machine Learning Approach](https://bv.fapesp.br/en/scholarships/209042/are-cognitive-functions-capable-of-predicting-overall-effectiveness-and-player-contributions-in-sma/)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Português:** [README em português](docs/README.pt.md)

## Project Overview

This repository contains data and Python scripts developed for predicting the performance of young soccer players in small-sided games through the analysis of cognitive functions using machine learning algorithms. The shared data is part of the master’s thesis project of student Rafael Luiz Martins Monteiro, funded by the São Paulo Research Foundation (FAPESP), grant #2021/15134-9. The dissertation is entitled: **"Are cognitive functions capable of predicting overall effectiveness and player contributions in small-sided soccer games? A machine learning approach."**

Data was collected using computerized tests with software such as [PEBL](https://pebl.sourceforge.net/) and [PsychoPy](https://www.psychopy.org/), focusing on assessments of sustained attention, visuospatial memory, impulsivity, cognitive flexibility, and multiple object tracking capacity. The tests used included the **Corsi Block-Tapping Test**, **Go/No-Go**, **Trail Making Test (PEBL)**, and **Multiple Object Tracking (PsychoPy)**.

On-field evaluation was performed through **3 vs. 3 small-sided games without goalkeepers**. Each match lasted 4 minutes. After every round, teams were randomly shuffled, ensuring that players did not repeatedly play with or against the same teammates. The objective of this evaluation was to extract each athlete’s individual performance through collective interactions among players. This protocol was proposed and validated by [Wilson et al. 2021](https://onlinelibrary.wiley.com/doi/full/10.1111/sms.13969).

<p align="center">
  <img src="./figures/Figure_1.png" alt="Field" />
</p>

<p align="center">
  <img src="./figures/Figure_3.png" alt="Graph" />
</p>


## Development setup (Linux, Windows, macOS)

**Environment:** Python 3.12.x, supported on Linux, Windows, and macOS. The project uses [uv](https://docs.astral.sh/uv/) for dependency management (recommended) and pins versions in `requirements.txt` for reproducibility.

**Linux/macOS** — from the project root, run once:
```bash
bash install.sh
```
This installs uv (if needed), creates `.venv`, installs dependencies, and runs a quick sanity check.

**Windows** — [install uv](https://docs.astral.sh/uv/getting-started/installation/), then in the project root:
```powershell
uv venv --python 3.12.12 --clear; uv sync
```

**Run the full pipeline** (recommended):
```bash
uv run python run_pipeline.py
```
Or: `make run`. This runs, in order: (1) data processing, (2) nested cross-validation evaluation, (2b) saving fitted models to `models/`, (3) overfitting validation (permutation test).

To run only the training step: `uv run python -m src.train_final_models`. To run only the permutation test: `uv run python -m src.validate_overfitting`. To only save trained models (after Step 2): `uv run python -m src.save_models`.

**Run tests** (imports + optional checks on `models/` if present):
```bash
uv run pytest tests/ -v
```
Or: `make test`.

Use **`make run`** or **`make test`** (not `./Makefile`).

**Lint and format** (Ruff): `make lint-format` (or `make lint` then `make format`). Without Make: `uv run ruff check --fix . && uv run ruff format .`

## Repository structure (PEP 8, snake_case)

| Path | Description |
|------|-------------|
| `run_pipeline.py` | **Single entry point**: runs data processing → nested CV → save models → permutation test. |
| `requirements.txt` | Python dependencies (also defined in `pyproject.toml`; use `uv sync` when using uv). |
| `src/data_preprocessing.py` | Dataset preparation and `.pkl` generation from cognitive + performance data. |
| `src/hyperparameter_tuning.py` | Grid search for best hyperparameters per algorithm. |
| `src/nested_cv_evaluation.py` | Nested cross-validation (small sample size mitigation); saves CV results. |
| `src/train_final_models.py` | Orchestrates hyperparameter tuning and nested CV over all dataset combinations. |
| `src/validate_overfitting.py` | **Anti-overfitting audit**: permutation test; saves `figures/permutation_test_proof.png` and reports empirical p-value. |
| `src/plot_best_models.py` | Plots best balanced accuracy by cognitive combination. |
| `src/loo_validation.py` | Leave-one-out validation and confusion matrices. |
| `src/save_models.py` | Fits pipelines with best params and saves them to `models/<combo>/<id>.joblib`. |
| `models/` | Fitted pipelines (created by pipeline Step 2b); use for inference. |
| `tests/` | Smoke tests (imports, optional model load). Run with `uv run pytest tests/ -v` or `make test`. |

Other scripts in `src/` (e.g. `cluster_figure.py`, `compare_cog_figure.py`, `nested_cv_confusion_matrix.py`, `generate_ml_models_reports.py`, `statistics_top3_models.py`) support additional figures and reports. Paths use `pathlib` for cross-platform compatibility (macOS, Windows, Linux).

## Naming conventions

Dataset column and directory names (e.g. `Acuracia_Go`, `gols_feitos`, `Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)`) follow the original study for reproducibility. For English meanings, units, and variable descriptions, see the [Data dictionary](docs/DATA_DICTIONARY.md).

## Data Access

All scripts used for data processing and validation are in the [src](./src/) folder. Each script includes a short description and usage notes. Adjust file paths to your local setup if needed.

The anonymized data used for processing and final analyses are available in the [data](./data/) folder. This folder contains both **raw data** and **processed data** used in the file `dataset.csv`.

The best hyperparameters obtained from GridSearch are provided in the [best_param](./best_param/) folder.

Machine learning training results and statistical analyses are available in the [results_CV](./results_CV/) folder.

Figures used in the publication can be accessed in the [figures](./figures/) folder. After running the pipeline, `figures/permutation_test_proof.png` is generated by the overfitting validation step (permutation test with 1,000 shuffles); it supports the claim that the model performs significantly better than chance (e.g. for reviewer response).

**Saved models** (optional): After running the full pipeline, fitted KNN pipelines are written to `models/<dataset_combo>/gf.joblib`, `gs.joblib`, `gc.joblib`, `sg.joblib`. Load with `joblib.load(path)` for prediction on new data.

## Acknowledgements

The authors would like to thank the coaching staff and athletes for their participation and valuable contributions to this study. We also acknowledge the financial support from the São Paulo Research Foundation (FAPESP), grants:

- [#2024/17521-8](https://bv.fapesp.br/en/scholarships/224064/)
- [#2024/15658-6](https://bv.fapesp.br/en/scholarships/224975/)
- [#2021/15134-9](https://bv.fapesp.br/en/scholarships/209042/)
- [#2019/22262-3](https://bv.fapesp.br/51740)
- [#2019/17729-0](https://bv.fapesp.br/51021)

We also thank the Coordination for the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) – Funding Code 001, and the Bavarian Academic Center for Latin America (BAYLAT).

## Contact

For further information or questions, please contact [rafaell_mmonteiro@usp.br](mailto:rafaell_mmonteiro@usp.br).
