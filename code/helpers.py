import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
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

# A rare category that a model has not seen in training (a loan purpose that
# disappears in some years) is encoded like the reference category.
# scikit-learn repeats this warning at every prediction.
warnings.filterwarnings("ignore", message="Found unknown categories")


def make_logit():
    # Missing numbers get the median plus a "was missing" flag, so the model
    # can learn whether a missing answer is itself a signal. Scaling makes
    # the coefficients comparable: each one is the effect of one standard
    # deviation. No class weights: they would distort the probabilities.
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    one_hot = OneHotEncoder(drop=REFERENCE_CATEGORIES, handle_unknown="ignore", sparse_output=False)
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", one_hot, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)
    # This solver finds the same coefficients as the default one, but several
    # times faster when there are many more loans than features. It matters
    # in stage 5, where the model is refitted hundreds of times.
    model = LogisticRegression(solver="newton-cholesky", max_iter=1000)
    return Pipeline([("preprocess", preprocess), ("model", model)])


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


def bootstrap_gini(target, score, n_boot=200, seed=SEED):
    # The Gini of one quarter is an estimate. Resampling the loans with
    # replacement shows how much it would move with a different sample.
    target, score = np.asarray(target), np.asarray(score)
    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(target), len(target))
        resampled.append(gini(target[rows], score[rows]))
    low, high = np.percentile(resampled, [2.5, 97.5])
    return gini(target, score), low, high


def bootstrap_coefficients(model, X, target, n_boot=100, seed=SEED):
    # Same idea for the coefficients of a logistic regression: refit it on
    # resampled loans and see how much each coefficient moves. A change
    # smaller than this interval can be noise.
    X, target = np.asarray(X), np.asarray(target)
    rng = np.random.default_rng(seed)
    coefficients = clone(model).fit(X, target).coef_[0]
    resampled = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(target), len(target))
        resampled.append(clone(model).fit(X[rows], target[rows]).coef_[0])
    low, high = np.percentile(resampled, [2.5, 97.5], axis=0)
    return coefficients, low, high


def psi(reference, current, bins=10):
    # Population Stability Index: how different the distribution of
    # `current` is from `reference`. 0 means identical.
    reference, current = pd.Series(reference), pd.Series(current)
    if pd.api.types.is_numeric_dtype(reference):
        # The bins are the deciles of the reference sample. A variable with
        # few distinct values (many loans at zero) simply gets fewer bins.
        deciles = np.unique(reference.quantile(np.linspace(0.1, 0.9, bins - 1)))
        edges = np.concatenate([[-np.inf], deciles, [np.inf]])
        # Missing values become their own bin, so a change in how often a
        # field is left empty also counts as a shift.
        reference = pd.cut(reference, edges).astype(str)
        current = pd.cut(current, edges).astype(str)
    expected = reference.value_counts(normalize=True)
    actual = current.value_counts(normalize=True)
    expected, actual = expected.align(actual, fill_value=0)
    # A small floor avoids log(0) when a bin is empty on one side.
    expected, actual = expected.clip(lower=1e-4), actual.clip(lower=1e-4)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def traffic_light(value, amber, red):
    if value < amber:
        return "green"
    if value < red:
        return "amber"
    return "red"


# One name and one colour per model, the same in every figure.
MODEL_LABELS = {
    "pd_logit": "Logistic regression",
    "pd_xgb": "XGBoost",
    "pd_grade": "Lending Club grade",
}
MODEL_COLORS = {"pd_logit": "#2a78d6", "pd_xgb": "#eb6834", "pd_grade": "#1baf7a"}
LIGHT_COLORS = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b"}
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
# A single hue from light to dark for magnitudes (heatmaps).
BLUES = LinearSegmentedColormap.from_list("blues", ["#eef4fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])


def set_plot_style():
    # Quiet axes and grid, so that the data is the only loud thing.
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "text.color": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK_SOFT, "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SOFT, "ytick.labelcolor": INK_SOFT,
        "lines.linewidth": 2, "lines.markersize": 5,
        "legend.frameon": False,
    })


def label_quarters(ax, quarters, offset=0.0):
    # Sixteen quarter labels are too many: the year is written once, under
    # its first quarter. `offset` is 0.5 when the cells of a heatmap are
    # drawn between integer positions.
    positions = [i + offset for i in range(len(quarters))]
    labels = [quarter[:4] if quarter.endswith("Q1") else "" for quarter in quarters]
    ax.set_xticks(positions, labels)
