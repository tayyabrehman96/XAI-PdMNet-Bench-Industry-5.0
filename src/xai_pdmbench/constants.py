"""UCI AI4I 2020 identifiers and HTTPS endpoints."""

from __future__ import annotations

# UCI Machine Learning Repository
UCI_DATASET_ID: int = 601
AI4I_CSV_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)
AI4I_LANDING_PAGE: str = (
    "https://archive.ics.uci.edu/dataset/601/"
    "ai4i+2020+predictive+maintenance+dataset"
)
UCI_DOI: str = "10.24432/C5HS5C"

# Optional checkout-local mirror (repository root-relative). Prefer UCI HTTPS for the
# freshest canonical file when online; bundled copy helps offline reproducibility.
AI4I_BUNDLED_CSV_REPO_RELATIVE: str = "data/ai4i/ai4i2020.csv"

# Failure-mode indicator columns — label leakage if used as inputs (B2 drops them).
FAULT_COLUMNS: tuple[str, ...] = ("TWF", "HDF", "PWF", "OSF", "RNF")
