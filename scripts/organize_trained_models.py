#!/usr/bin/env python3
"""Collect all available trained artifacts into trained_models/."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINED = ROOT / "trained_models"
AI4I_DST = TRAINED / "ai4i"
PHM_DST = TRAINED / "phm2010"

AI4I_EXPECTED = [
    "logistic_regression.joblib",
    "random_forest_ctgan.joblib",
    "hist_gradient_boosting_ctgan.joblib",
    "xgboost_ctgan.json",
    "numpy_logistic_weighted.npz",
    "numpy_gaussian_nb.npz",
    "numpy_knn_k5.npz",
    "alexnet1d_safe.pt",
    "alexnet1d_safe_weights.pt",
    "tabtransformer.pt",
    "tabtransformer_weights.pt",
    "cnn_lstm_thr031.pt",
    "cnn_lstm_thr031_weights.pt",
]

PHM_EXPECTED = [
    "xgboost_phm.json",
    "cnn1d_phm.pt",
    "cnn1d_phm_weights.pt",
    "rcnn_phm.pt",
    "rcnn_phm_weights.pt",
    "transfer_ai4i_to_phm.pt",
    "transfer_ai4i_to_phm_weights.pt",
]


def ensure_folders():
    AI4I_DST.mkdir(parents=True, exist_ok=True)
    PHM_DST.mkdir(parents=True, exist_ok=True)


def rows(expected, folder, dataset):
    out = []
    for name in expected:
        path = folder / name
        out.append(
            {
                "dataset": dataset,
                "artifact": name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "present": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return out


def write_csv(path, rows_):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "artifact", "path", "present", "bytes"])
        writer.writeheader()
        writer.writerows(rows_)


def main():
    ensure_folders()
    ai4i_rows = rows(AI4I_EXPECTED, AI4I_DST, "AI4I 2020")
    phm_rows = rows(PHM_EXPECTED, PHM_DST, "PHM 2010")
    all_rows = ai4i_rows + phm_rows
    write_csv(TRAINED / "ALL_TRAINED_MODELS_INVENTORY.csv", all_rows)
    write_csv(ROOT / "results" / "trained_model_artifacts.csv", all_rows)
    write_csv(ROOT / "results" / "ai4i_model_details.csv", ai4i_rows)
    write_csv(ROOT / "results" / "phm2010_model_details.csv", phm_rows)
    print("Organized trained models under", TRAINED)


if __name__ == "__main__":
    main()
