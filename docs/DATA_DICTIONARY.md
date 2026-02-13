# Data dictionary

This document describes the variables used in the cognitive-soccer-ml pipeline. Column and directory names in the repository follow the original study (mixed Portuguese/English) for reproducibility; English equivalents and meanings are given below.

## Cognitive variables (predictors)

| Column name (dataset) | English / meaning | Units / scale | Source |
|----------------------|-------------------|--------------|--------|
| **Memory span** | Visuospatial working memory (VWM) | Arbitrary units (au); higher = better | Corsi Block-Tapping Test (PEBL) |
| **Acuracia Go** | Sustained attention (Go trial accuracy) | 0–1 (proportion correct); sometimes scaled to 0–100% in figures | Go/No-Go (PEBL) |
| **Acuracia nogo** | Impulsivity control (No-Go trial accuracy) | 0–1 (proportion correct); sometimes scaled to 0–100% in figures | Go/No-Go (PEBL) |
| **Capacidade de rastreamento** | Multiple object tracking capacity | Arbitrary units (au); higher = better | Multiple Object Tracking (PsychoPy) |
| **Flexibilidade cognitiva (B-A)** | Cognitive flexibility (set-shifting) | Milliseconds (ms); in some scripts converted to seconds (s). Lower = faster/better | Trail Making Test Part B minus Part A (PEBL) |

## Performance / field variables (targets)

Targets are derived from small-sided game outcomes. Each is first used to form two clusters (e.g. high vs. low) via K-means; the cluster label is the target for classification.

| Column name (dataset) | English / meaning | Units / scale |
|----------------------|-------------------|--------------|
| **gols_feitos** | Individual goals per game (gf) | Goals per game; continuous, then clustered |
| **gols_sofridos** | Conceded goals per game (gs) | Goals per game (higher = worse); continuous, then clustered |
| **gols_companheiros** | Goals by teammates per game (gc) | Goals per game; continuous, then clustered |
| **saldo_gols** | Net (team) goals per game (sg) | Goal difference per game; continuous, then clustered |

## Identifiers used in the codebase

- **gf**, **gs**, **gc**, **sg**: Short identifiers for the four target outcomes (gols_feitos, gols_sofridos, gols_companheiros, saldo_gols).
- **cluster**: Binary cluster label (0 or 1) assigned by K-means to one of the performance variables above; used as the classification target in ML pipelines.
- **best_param**, **parametros**: Folders/files containing the best hyperparameters from GridSearch per algorithm and per target (e.g. `gf_parametros.csv`).

## Directory naming

Dataset combination folders under `data/ML_datasets/` and `best_param/` are named by concatenating cognitive variable names with underscores, e.g. `Memory_span_Acuracia_Go`, `Capacidade_de_rastreamento_Flexibilidade_cognitiva_(B-A)`. These names are preserved for exact reproduction of the article’s results.
