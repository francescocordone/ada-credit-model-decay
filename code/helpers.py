from pathlib import Path

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
