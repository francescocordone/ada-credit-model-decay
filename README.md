# How long does a credit scoring model stay reliable?

Individual project for the course Advanced Data Analytics, by Francesco Cordone.

A bank builds a default model on past borrowers and then uses it for years. This project
replicates that situation on Lending Club data: a logistic regression and an XGBoost model
are trained on 36-month loans issued in 2009-2011 and tested on every quarter from 2012Q1
to 2015Q4, to measure when their performance starts to drop and why.

## Installation

Python 3.11 or later.

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Linux, macOS, Nuvolos)
pip install -r requirements.txt
```

## How to reproduce

Run the stages in this order from the repository root. Each stage loads what the previous
one saved.

| Order | File | What it does | Output |
| --- | --- | --- | --- |
| 1 | `code/01_get_data.py` | Downloads the raw file and keeps 36-month loans issued 2009-2015 | `data/interim/lc_36m_2009_2015.parquet` |
| 2 | `code/02_clean.ipynb` | Target, cohort, features, occupation groups, sample | `data/processed/loans_clean.parquet` |
| 3 | `code/03_models.ipynb` | Logistic regression and XGBoost on 2009-2011 | `data/processed/scores.parquet`, `models/` |
| 4 | `code/04_monitoring.ipynb` | Gini, calibration and PSI per quarter | `data/processed/monitoring.csv`, `figures/` |
| 5 | `code/05_drift.ipynb` | Why the models degrade | `figures/` |

```
python code/01_get_data.py
jupyter nbconvert --to notebook --execute --inplace code/02_clean.ipynb
```

The notebooks can also be opened in Jupyter and run with Restart & Run All.

## Data

Source: Kaggle dataset [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
file `accepted_2007_to_2018Q4.csv.gz`. Licence: CC0 Public Domain. Downloaded on 16 November 2026.

The raw file is not stored in this repository: `code/01_get_data.py` downloads it into
`data/raw/`. No Kaggle account or API key is needed. If the download link stops working,
download the file by hand from the Kaggle page and place it in `data/raw/`.
