import urllib.request

import pandas as pd

from helpers import INTERIM, RAW

# Public download link of the Kaggle dataset wordsforthewise/lending-club
# (accepted loans only). It needs no account and no API key.
URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "wordsforthewise/lending-club/accepted_2007_to_2018Q4.csv.gz"
)
RAW_FILE = RAW / "accepted_2007_to_2018Q4.csv.gz"
OUT_FILE = INTERIM / "lc_36m_2009_2015.parquet"

FIRST_YEAR, LAST_YEAR = 2009, 2015

# Only what is known when the borrower applies, plus the lender's own grade
# and the final outcome. Everything recorded after the loan is issued
# (payments, recoveries, last FICO score, hardship and settlement plans) is
# never loaded: it would leak the outcome into the features. The interest
# rate and the installment are left out too, because they are set by
# Lending Club's own model and carry the same information as the grade.
USEFUL_COLUMNS = [
    # identifiers and outcome
    "id", "issue_d", "term", "loan_status",
    # the lender's own assessment, kept only as a benchmark
    "grade", "sub_grade",
    # the application
    "loan_amnt", "purpose", "annual_inc", "verification_status", "dti",
    "emp_title", "emp_length", "home_ownership",
    # the credit report at origination
    "fico_range_low", "fico_range_high", "earliest_cr_line",
    "revol_bal", "revol_util", "open_acc", "total_acc",
    "delinq_2yrs", "mths_since_last_delinq", "inq_last_6mths", "pub_rec",
    # credit report fields that Lending Club started to publish later:
    # loaded so that stage 2 can show from which year each one exists
    "tot_cur_bal", "tot_coll_amt", "total_rev_hi_lim", "avg_cur_bal",
    "mort_acc", "bc_util", "percent_bc_gt_75", "acc_open_past_24mths",
    "mths_since_recent_inq",
]


def download():
    if RAW_FILE.exists():
        print(f"Already downloaded: {RAW_FILE.name}")
        return
    print(f"Downloading {RAW_FILE.name} (about 375 MB)...")
    urllib.request.urlretrieve(URL, RAW_FILE)
    print(f"Saved {RAW_FILE.stat().st_size / 1e6:.0f} MB")


def explore():
    # A first look at 1,000 rows is enough to choose columns and filters
    # without loading a file that is larger than 1 GB once decompressed.
    head = pd.read_csv(RAW_FILE, nrows=1000, low_memory=False)
    summary = pd.DataFrame({
        "dtype": head.dtypes.astype(str),
        "missing_%": (head.isna().mean() * 100).round(1),
    })
    print(f"\nFirst 1,000 rows: {head.shape[1]} columns")
    print(summary.to_string())
    print("\nloan_status:", sorted(head["loan_status"].dropna().unique()))
    print("term:", sorted(head["term"].dropna().unique()))
    issue_d = pd.to_datetime(head["issue_d"], format="%b-%Y")
    print("issue_d in these rows:", issue_d.min().date(), "to", issue_d.max().date())


def filter_and_save():
    # The file does not fit comfortably in memory, so it is read in blocks
    # and only the rows we need are kept from each block.
    kept = []
    rows_read = rows_36m = 0
    first_date, last_date = pd.Timestamp.max, pd.Timestamp.min
    chunks = pd.read_csv(RAW_FILE, usecols=USEFUL_COLUMNS, dtype={"id": str},
                         chunksize=200_000, low_memory=False)
    for chunk in chunks:
        # The file ends with a few summary lines that are not loans.
        chunk = chunk[chunk["issue_d"].notna()].copy()
        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y")
        rows_read += len(chunk)
        first_date = min(first_date, chunk["issue_d"].min())
        last_date = max(last_date, chunk["issue_d"].max())

        chunk = chunk[chunk["term"].str.strip() == "36 months"]
        rows_36m += len(chunk)
        chunk = chunk[chunk["issue_d"].dt.year.between(FIRST_YEAR, LAST_YEAR)]
        kept.append(chunk)

    loans = pd.concat(kept, ignore_index=True)
    loans.to_parquet(OUT_FILE, index=False)

    # Stage 2 continues this count, so the paper can report how many rows
    # every rule removes, from the raw file to the final sample.
    row_counts = pd.Series({
        "Loans in the raw file": rows_read,
        "36-month loans": rows_36m,
        f"Issued {FIRST_YEAR}-{LAST_YEAR}": len(loans),
    }, name="rows")
    row_counts.to_csv(INTERIM / "row_counts.csv", index_label="step")

    print(f"\nRaw file covers {first_date:%b %Y} to {last_date:%b %Y}")
    print(row_counts.to_string())
    print(f"Saved {OUT_FILE.name} ({OUT_FILE.stat().st_size / 1e6:.0f} MB)")

    year = loans["issue_d"].dt.year.rename("issue_year")
    print("\nRows per year:")
    print(year.value_counts().sort_index().to_string())
    print("\nloan_status by year:")
    print(pd.crosstab(loans["loan_status"], year).to_string())


if __name__ == "__main__":
    download()
    explore()
    filter_and_save()
