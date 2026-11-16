from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Paths are built from the repository root, so the code behaves the same
# when it is run from the root folder, from code/ or from a notebook.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

# Git does not track empty folders, so a fresh clone needs them created.
for folder in [RAW, INTERIM, PROCESSED, FIGURES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42

# The models are built on loans issued in these years only. Every later
# quarter is the monitoring period and is never used to fit them.
DEV_YEARS = [2009, 2010, 2011]

# The same feature lists are used by every stage, so that the models, the
# stability checks and the drift analysis always look at the same columns.
NUMERIC_FEATURES = [
    "loan_amnt", "log_annual_inc", "dti", "fico", "emp_length_years",
    "log_revol_bal", "revol_util", "open_acc", "total_acc",
    "delinq_2yrs", "inq_last_6mths", "pub_rec", "credit_history_years",
]
CATEGORICAL_FEATURES = ["home_ownership", "purpose", "verification_status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The most common answer of each categorical feature, in the same order as
# CATEGORICAL_FEATURES. The logistic regression leaves these out, so every
# other category is read as a difference from the typical borrower.
REFERENCE_CATEGORIES = ["MORTGAGE", "debt_consolidation", "Not Verified"]


def make_logit():
    # Missing numbers get the median plus a "was missing" flag, so the model
    # can learn whether a missing answer is itself a signal. Scaling makes
    # the coefficients comparable: each one is the effect of one standard
    # deviation. No class weights: they would distort the probabilities.
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    one_hot = OneHotEncoder(drop=REFERENCE_CATEGORIES, handle_unknown="ignore")
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", one_hot, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000))])


def make_xgb(n_trees):
    # Trees need no scaling and XGBoost handles missing numbers by itself,
    # so the only preparation is the one-hot encoding of the categories.
    # The output is kept dense because XGBoost would read the empty cells
    # of a sparse matrix as missing values instead of zeros.
    encode = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)
    model = XGBClassifier(n_estimators=n_trees, max_depth=3, learning_rate=0.05,
                          eval_metric="logloss", random_state=SEED)
    return Pipeline([("encode", encode), ("model", model)])


def gini(target, score):
    # Gini = 2 * AUC - 1: 0 for a random ranking, 1 for a perfect one.
    return 2 * roc_auc_score(target, score) - 1


def calibration_table(target, prob, bins=10):
    # Loans are sorted by predicted probability and cut into equal groups:
    # in a calibrated model the two rates are close in every group.
    table = pd.DataFrame({"target": np.asarray(target), "prob": np.asarray(prob)})
    table["bucket"] = pd.qcut(table["prob"], bins, labels=False, duplicates="drop") + 1
    return table.groupby("bucket").agg(
        loans=("target", "size"),
        mean_pd=("prob", "mean"),
        default_rate=("target", "mean"),
    )
