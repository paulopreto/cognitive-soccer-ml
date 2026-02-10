# [Are Cognitive Functions Capable of Predicting Overall Effectiveness and Player Contributions in Small-Sided Soccer Games? A Machine Learning Approach](https://bv.fapesp.br/en/scholarships/209042/are-cognitive-functions-capable-of-predicting-overall-effectiveness-and-player-contributions-in-sma/)

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

This project uses [uv](https://docs.astral.sh/uv/) with Python 3.12.12.

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


# [As funções cognitivas são capazes de predizer a efetividade geral e as contribuições do atleta para a equipe em jogos reduzidos de futebol? Uma abordagem utilizando aprendizado de máquina.](https://bv.fapesp.br/pt/bolsas/209042/as-funcoes-cognitivas-sao-capazes-de-predizer-a-efetividade-geral-e-contribuicao-do-atleta-para-a-eq/)

## Visão geral do projeto

Este repositório contém dados e códigos desenvolvidos em Python para a predição do desempenho de jovens jogadores em jogos reduzidos por meio da análise de funções cognitivas utilizando algoritmos de aprendizado de máquina. Os dados compartilhados são resultado do projeto de mestrado do aluno Rafael Luiz Martins Monteiro com fomento da Fundação de Amparo à Pesquisa do Estado de São Paulo, processo #2021/15134-9. A dissertação é intitulada: "As funções cognitivas são capazes de predizer a efetividade geral e as contribuições do atleta para a equipe em jogos reduzidos de futebol? Uma abordagem utilizando aprendizado de máquina.".

Foram utilizados dados obtidos por meio de testes computadorizados com softwares como PEBL (https://pebl.sourceforge.net/)  e PsychoPy (https://www.psychopy.org/), focando em avaliações de atenção sustentada, memória visuoespacial, impulsividade, flexibilidade cognitiva e capacidade de rastreamento de objetos múltiplos. Os testes utilizados foram: Blocos de Corsi, Go/ No Go e teste de trilhas no PEBL e rastreamento de objetos múltiplos no PsychoPy.

A avaliação em campo foi feita por meio de jogos reduzidos de 3 x 3 jogadores sem goleiros. Cada jogo tinha duração de 4 minutos. Após cada rodada os times eram misturados de forma aleatória, dessa forma os jogadores não jogavam com ou contra os mesmos colegas. O objetivo da avaliação foi extrair a performance individual de cada atleta por meio da interação coletiva entre os jogadores. Este protoclo foi proposto e validado por Wilson et al. 2021 (https://onlinelibrary.wiley.com/doi/full/10.1111/sms.13969).


## Configuração para desenvolvimento (Linux, Windows, macOS)

O projeto usa [uv](https://docs.astral.sh/uv/) com Python 3.12.12. Na raiz do projeto: **Linux/macOS** — `bash install.sh` (instala uv, cria `.venv`, instala dependências e testa). **Windows** — instale o uv e rode `uv venv --python 3.12.12 --clear` e depois `uv sync`. Para executar o pipeline completo: `uv run python run_pipeline.py`. Para apenas treino: `uv run python -m src.train_final_models`. Para apenas o teste de permutação (validação de overfitting): `uv run python -m src.validate_overfitting`.

## Acesso aos dados
Todos os códigos utilizados para o processamento de dados podem ser acessados na pasta [src](./src/). Cada código possui uma breve descrição de suas funcionalidades e de como utiliza-lo. Lembre-se sempre de ajustar os caminhos dos arquivos para executa-los em seu próprio computador. Os dados anonimizados utilizados para o processamento e trabalho final podem ser acessados na pasta [data](./data/). Dentro desta pasta tem os dados brutos e os dados processados que foram utilizados para análise no arquivo 'dataset.csv'. Os melhores hiperparâmetros resultantes do GridSearch podem ser encontrados na pasta [best_param](./best_param/). Os resultados do treinamento dos modelos de aprendizado de máquina assim como as análises estatísticas podem ser acessados na pasta [results_CV](./results_CV/). As figuras do artigo podem ser acessadas na pasta [figures](./figures/). 


### Agradecimentos
Os autores agradecem à comissão técnica e aos atletas pela participação e contribuições fundamentais para este estudo. Além disso, agradecemos o apoio financeiro da Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP), processos: [#2024/17521-8](https://bv.fapesp.br/pt/bolsas/224064/analise-do-desenvolvimento-cognitivo-na-predicao-de-variaveis-tecnicas-taticas-e-fisicas-de-jogadore/), [#2024/15658-6](https://bv.fapesp.br/pt/bolsas/224975/classificacao-de-esforcos-em-corrida-com-aprendizado-maquina-por-meio-da-forca-de-reacao-do-solo-e-r/), [#2021/15134-9](https://bv.fapesp.br/pt/bolsas/209042/as-funcoes-cognitivas-sao-capazes-de-predizer-a-efetividade-geral-e-contribuicao-do-atleta-para-a-eq/), [#2019/22262-3](https://bv.fapesp.br/51740), e [#2019/17729-0](https://bv.fapesp.br/51021). Também agradecemos à Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) – Código de Financiamento 001, e ao Centro Acadêmico Bavariano para a América Latina (BAYLAT).

## Contato
Para mais informações ou dúvidas, por favor, entre em contato pelo e-mail [rafaell_mmonteiro@usp.br](mailto:rafaell_mmonteiro@usp.br).

