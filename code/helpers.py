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
