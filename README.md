# How long does a credit scoring model stay reliable?

Individual project for the course Advanced Data Analytics, by Francesco Cordone.

A bank builds a default model on past borrowers and then uses it for years. This project
replicates that situation on Lending Club data: a logistic regression and an XGBoost model
are trained on 36-month loans issued in 2009-2011, frozen, and tested on every quarter from
2012Q1 to 2015Q4, to measure when their performance starts to drop and why.

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
one saved and saves what the next one needs.

```
python code/01_get_data.py
jupyter nbconvert --to notebook --execute --inplace code/02_clean.ipynb
jupyter nbconvert --to notebook --execute --inplace code/03_models.ipynb
jupyter nbconvert --to notebook --execute --inplace code/04_monitoring.ipynb
jupyter nbconvert --to notebook --execute --inplace code/05_drift.ipynb
jupyter nbconvert --to notebook --execute --inplace code/06_extensions.ipynb
```

The notebooks can also be opened in Jupyter and run with Restart & Run All.

| Stage | File | What it does | Main output | Time |
| --- | --- | --- | --- | --- |
| 1 | `code/01_get_data.py` | Downloads the raw file and keeps the 36-month loans issued 2009-2015 | `data/interim/lc_36m_2009_2015.parquet` | 3-10 min (download) |
| 2 | `code/02_clean.ipynb` | Target, cohort, features, occupation groups, sample | `data/processed/loans_clean.parquet` | 20 s |
| 3 | `code/03_models.ipynb` | Logistic regression, XGBoost and the Lending Club grade as a benchmark | `data/processed/scores.parquet`, `models/` | 10 s |
| 4 | `code/04_monitoring.ipynb` | Gini, calibration and PSI per quarter, with traffic lights | `data/processed/monitoring.csv` | 1 min |
| 5 | `code/05_drift.ipynb` | Why the models degrade: population, relationship, level, occupation, PCA | tables and figures | 4 min |
| 6 | `code/06_extensions.ipynb` | Gaussian process trend and borrower clusters | tables and figures | 1 min |

`code/helpers.py` holds the paths, the feature lists and every function used by more than
one stage. Every figure and result table is written to `figures/` by the code; nothing in
that folder is edited by hand. The seed is 42 everywhere.

## Data

Source: Kaggle dataset [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
file `accepted_2007_to_2018Q4.csv.gz` (393 MB). Licence: CC0 Public Domain.
Downloaded on 16 November 2026.

The raw file is not stored in this repository: `code/01_get_data.py` downloads it into
`data/raw/`. No Kaggle account or API key is needed. If the download link stops working,
download the file by hand from the Kaggle page and place it in `data/raw/`.

The project works on a sample: all the 28,538 loans issued in 2009-2011 and at most 12,000
loans per quarter from 2012, 211,505 loans in total.

## Compute

Developed and run on a laptop: Intel Core i5-1135G7 (4 cores), 8 GB of RAM, Windows 11,
Python 3.14.3, no GPU. The whole pipeline runs in about 15 minutes, most of it the download.

## Use of AI

The code in this repository was drafted with the help of Claude (Anthropic). The full
declaration of AI use is in the paper.
