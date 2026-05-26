#!/usr/bin/env python3
"""Produce code-generated reproducibility tables from the local checkout.

This script is intentionally dependency-light: it uses pandas/numpy/openpyxl
only, so it runs in the bundled Codex Python environment.

Important reproducibility rule: this script does not copy performance numbers
out of a PDF. Tables written here are generated from files and code available
in the repository. The full training scripts remain the source for rerunning
the heavier models when those dependencies and datasets are available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "data" / "ai4i" / "ai4i2020.csv"
PHM_DIR = ROOT / "data" / "phm2010"
TRAINED_MODELS_DIR = ROOT / "trained_models"
OUT = ROOT / "results"
FAULT_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")
SEED = 42


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip() for c in df.columns})


def clean_ai4i_benchmark_rows(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    mf = d["Machine failure"].astype(int)
    faults_present = d[list(FAULT_COLUMNS)].astype(int).sum(axis=1) >= 1
    bad_rnf = (d["RNF"].astype(int) == 1) & (mf == 0)
    bad_mf = (mf == 1) & (~faults_present)
    return d[~(bad_rnf | bad_mf)].reset_index(drop=True)


def build_features_b2(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    dd = d.copy()
    dd = dd.drop(columns=[c for c in ("UID", "UDI", "Product ID") if c in dd.columns], errors="ignore")
    dd = dd.drop(columns=[c for c in FAULT_COLUMNS if c in dd.columns], errors="ignore")
    y = dd["Machine failure"].astype(int).to_numpy()
    dd = dd.drop(columns=["Machine failure"])

    torque = pd.to_numeric(dd["Torque [Nm]"], errors="coerce")
    rpm = pd.to_numeric(dd["Rotational speed [rpm]"], errors="coerce")
    air = pd.to_numeric(dd["Air temperature [K]"], errors="coerce")
    proc = pd.to_numeric(dd["Process temperature [K]"], errors="coerce")
    wear = pd.to_numeric(dd["Tool wear [min]"], errors="coerce")

    power = torque * rpm
    dd["Power_proxy"] = power
    dd["RPM_over_torque"] = rpm / torque.clip(lower=1e-6)
    dd["Log1p_torque"] = np.log1p(np.maximum(torque.fillna(0.0), 0.0))
    dd["Log1p_rpm"] = np.log1p(np.maximum(rpm.fillna(0.0), 0.0))
    dd["Log_power_proxy"] = np.log1p(np.maximum(power.fillna(0.0), 0.0))

    mx = float(np.nanmax(wear.to_numpy())) if wear.notna().any() else 1.0
    mx = mx if np.isfinite(mx) and mx > 0 else 1.0
    wear_safe = wear.fillna(0.0)
    dd["Wear_norm"] = wear_safe / max(mx, 1e-6)
    dd["Log1p_wear"] = np.log1p(np.maximum(wear_safe, 0.0))
    dd["Torque_x_wear"] = torque.fillna(0.0) * wear_safe
    dd["Rpm_x_wear"] = rpm.fillna(0.0) * wear_safe
    dd["Power_per_wear"] = power / (np.maximum(wear_safe, 0.0) + 1.0)

    air_safe = air.clip(lower=1.0)
    delta = proc - air
    dd["Proc_x_torque"] = proc * torque
    dd["Temp_x_torque"] = delta * torque
    dd["Delta_temp_K"] = delta
    dd["Thermal_ratio"] = delta / air_safe
    dd["Temp_product_K2"] = air * proc
    dd["Inv_air_temp"] = 1.0 / air_safe
    dd["Delta_temp_sq"] = delta**2
    dd["Process_over_air"] = proc / air_safe
    dd["Thermal_per_wear"] = delta / (wear_safe.clip(lower=0.0) + 1.0)

    cat_cols = [c for c in dd.columns if dd[c].dtype == object]
    X_num = dd.drop(columns=cat_cols, errors="ignore")
    X_cat = pd.get_dummies(dd[cat_cols], drop_first=False) if cat_cols else pd.DataFrame(index=dd.index)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.astype(np.float64), y


def stratified_split(y: np.ndarray, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    train_parts, test_parts = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train = np.concatenate(train_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int = 1200, lr: float = 0.05) -> tuple[np.ndarray, float]:
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y == 1, neg / pos, 1.0)
    weights = weights / weights.mean()
    for _ in range(epochs):
        p = sigmoid(x @ w + b)
        err = (p - y) * weights
        w -= lr * ((x.T @ err) / len(y) + 1e-4 * w)
        b -= lr * float(err.mean())
    return w, b


def predict_knn(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int = 5) -> np.ndarray:
    probs = np.empty(len(x_test), dtype=np.float64)
    chunk = 256
    for start in range(0, len(x_test), chunk):
        xt = x_test[start : start + chunk]
        d2 = ((xt[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2)
        nn = np.argpartition(d2, kth=min(k, len(x_train) - 1), axis=1)[:, :k]
        probs[start : start + chunk] = y_train[nn].mean(axis=1)
    return probs


def predict_gaussian_nb(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    classes = [0, 1]
    logps = []
    for cls in classes:
        xc = x_train[y_train == cls]
        mu = xc.mean(axis=0)
        var = xc.var(axis=0) + 1e-6
        prior = math.log(max(len(xc), 1) / len(x_train))
        ll = -0.5 * (np.log(2 * np.pi * var) + ((x_test - mu) ** 2) / var).sum(axis=1)
        logps.append(prior + ll)
    logits = np.vstack(logps).T
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((y_true == 1).sum()), 1)
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.trapezoid(precision, recall))


def metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(y_true, score),
        "pr_auc": pr_auc(y_true, score),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def csv_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.rglob("*.csv")):
        rel = path.relative_to(ROOT).as_posix()
        if rel == "data/ai4i2020.csv" or rel.startswith("data/ai4i/"):
            role = "primary AI4I dataset"
            usable = True
        elif rel.startswith("data/phm2010/"):
            role = "secondary PHM 2010 dataset/derived table"
            usable = True
        elif rel.startswith("results/"):
            role = "generated result table"
            usable = True
        else:
            role = "not used by the reproducibility workflow"
            usable = False
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "used_in_repro_pipeline": usable,
                "role": role,
            }
        )
    return pd.DataFrame(rows)


def dataset_inventory() -> pd.DataFrame:
    phm_files = []
    phm_combined = PHM_DIR / "phm2010_combined.csv"
    phm_raw = PHM_DIR / "raw"
    if phm_combined.exists():
        phm_files.append(phm_combined)
    if phm_raw.exists():
        phm_files.extend(
            p
            for p in phm_raw.rglob("*")
            if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".dat"}
        )
    if PHM_DIR.exists():
        phm_files.extend(
            p
            for p in PHM_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in {".csv", ".npy", ".npz"}
        )
    # deduplicate while preserving counts
    phm_files = list(dict.fromkeys(phm_files))
    return pd.DataFrame(
        [
            {
                "dataset": "AI4I 2020",
                "expected_location": "data/ai4i/ai4i2020.csv",
                "present": DATA_PATH.exists(),
                "file_count": 1 if DATA_PATH.exists() else 0,
                "project_role": "Leakage-safe tabular CNC predictive-maintenance benchmark",
            },
            {
                "dataset": "PHM 2010",
                "expected_location": "data/phm2010/",
                "present": len(phm_files) > 0,
                "file_count": len(phm_files),
                "project_role": "Tool-independent real CNC milling validation benchmark",
            },
        ]
    )


def model_inventory() -> pd.DataFrame:
    expected = [
        ("AI4I 2020", "xgboost_ctgan", "XGBoost proposed model", ".json/.ubj/.pkl/.joblib"),
        ("AI4I 2020", "random_forest_ctgan", "Random Forest baseline", ".pkl/.joblib"),
        ("AI4I 2020", "hist_gradient_boosting_ctgan", "HistGradientBoosting baseline", ".pkl/.joblib"),
        ("AI4I 2020", "logistic_regression", "Linear baseline", ".pkl/.joblib"),
        ("AI4I 2020", "alexnet1d_safe", "Leakage-safe 1D-AlexNet", ".pt/.keras/.h5"),
        ("AI4I 2020", "tabtransformer", "Proposed deep tabular model", ".pt/.keras/.pth"),
        ("AI4I 2020", "cnn_lstm_thr031", "Exploratory pseudo-temporal diagnostic", ".pt/.keras/.h5"),
        ("PHM 2010", "xgboost_phm", "Statistical PHM baseline", ".json/.ubj/.pkl/.joblib"),
        ("PHM 2010", "cnn1d_phm", "Temporal 1D-CNN baseline", ".keras/.h5/.pt/.pth"),
        ("PHM 2010", "rcnn_phm", "Proposed RCNN model", ".keras/.h5/.pt/.pth"),
        ("PHM 2010", "transfer_ai4i_to_phm", "Staged transfer model", ".keras/.h5/.pt/.pth"),
        ("AI4I 2020", "numpy_logistic_weighted", "Dependency-free release baseline", ".npz"),
        ("AI4I 2020", "numpy_gaussian_nb", "Dependency-free release baseline", ".npz"),
        ("AI4I 2020", "numpy_knn_k5", "Dependency-free release baseline", ".npz"),
    ]
    model_files = []
    if TRAINED_MODELS_DIR.exists():
        model_files.extend(p for p in TRAINED_MODELS_DIR.rglob("*") if p.is_file())
    rows = []
    for dataset, model_id, role, extensions in expected:
        matches = [p for p in model_files if model_id.lower() in p.stem.lower()]
        rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "project_role": role,
                "expected_extensions": extensions,
                "present_in_repo": len(matches) > 0,
                "matched_files": "; ".join(p.relative_to(ROOT).as_posix() for p in matches),
            }
        )
    return pd.DataFrame(rows)


def run_fast_baselines(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    train_idx, test_idx = stratified_split(y)
    x_train, x_test = X.iloc[train_idx].to_numpy(), X.iloc[test_idx].to_numpy()
    y_train, y_test = y[train_idx], y[test_idx]
    x_train, x_test = standardize(x_train, x_test)

    w, b = fit_logistic(x_train, y_train)
    rows = []
    scores = {
        "numpy_logistic_weighted": sigmoid(x_test @ w + b),
        "numpy_gaussian_nb": predict_gaussian_nb(x_train, y_train, x_test),
        "numpy_knn_k5": predict_knn(x_train, y_train, x_test, k=5),
    }
    for name, score in scores.items():
        row = {"model": name, "family": "fast local baseline", "threshold": 0.5}
        row.update(metrics(y_test, score))
        rows.append(row)
    return pd.DataFrame(rows)


def reproduction_status(datasets: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    phm_present = bool(datasets.loc[datasets["dataset"] == "PHM 2010", "present"].iloc[0])
    any_release_model = bool(models["present_in_repo"].any())
    return pd.DataFrame(
        [
            {
                "component": "AI4I dataset",
                "status": "ready" if DATA_PATH.exists() else "missing",
                "code_first_rule": "computed from data/ai4i/ai4i2020.csv",
            },
            {
                "component": "PHM 2010 dataset",
                "status": "ready" if phm_present else "missing real data files",
                "code_first_rule": "generated from bundled PHM 2010 source arrays and CSV tables",
            },
            {
                "component": "trained release models",
                "status": "present" if any_release_model else "missing artifacts",
                "code_first_rule": "metrics should be regenerated from model files or training code",
            },
            {
                "component": "pdf-derived tables",
                "status": "not generated here",
                "code_first_rule": "no PDF-derived metrics are written by this script",
            },
        ]
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = normalize_columns(pd.read_csv(DATA_PATH))
    clean = clean_ai4i_benchmark_rows(raw)
    X_b2, y_b2 = build_features_b2(clean)

    inventory = csv_inventory()
    datasets = dataset_inventory()
    models = model_inventory()
    status = reproduction_status(datasets, models)
    feature_summary = pd.DataFrame(
        [
            {"item": "raw_rows", "value": len(raw)},
            {"item": "clean_rows", "value": len(clean)},
            {"item": "dropped_inconsistent_rows", "value": len(raw) - len(clean)},
            {"item": "raw_failures", "value": int(raw["Machine failure"].sum())},
            {"item": "clean_failures", "value": int(y_b2.sum())},
            {"item": "clean_normals", "value": int((y_b2 == 0).sum())},
            {"item": "b2_feature_count", "value": X_b2.shape[1]},
        ]
    )
    features = pd.DataFrame({"feature": list(X_b2.columns)})
    fast = run_fast_baselines(X_b2, y_b2)
    trained_metrics_path = OUT / "ai4i_trained_model_metrics.csv"
    trained_metrics = (
        pd.read_csv(trained_metrics_path)
        if trained_metrics_path.exists()
        else pd.DataFrame()
    )
    deep_metrics_path = OUT / "ai4i_deep_pt_model_metrics.csv"
    deep_metrics = (
        pd.read_csv(deep_metrics_path)
        if deep_metrics_path.exists()
        else pd.DataFrame()
    )
    phm_metrics_path = OUT / "phm_model_metrics.csv"
    phm_metrics = pd.read_csv(phm_metrics_path) if phm_metrics_path.exists() else pd.DataFrame()
    phm_cfg_path = OUT / "phm_training_config.csv"
    phm_cfg = pd.read_csv(phm_cfg_path) if phm_cfg_path.exists() else pd.DataFrame()
    trained_artifacts_path = OUT / "trained_model_artifacts.csv"
    trained_artifacts = (
        pd.read_csv(trained_artifacts_path)
        if trained_artifacts_path.exists()
        else pd.DataFrame()
    )

    inventory.to_csv(OUT / "csv_inventory.csv", index=False)
    datasets.to_csv(OUT / "dataset_inventory.csv", index=False)
    models.to_csv(OUT / "model_inventory.csv", index=False)
    status.to_csv(OUT / "reproduction_status.csv", index=False)
    feature_summary.to_csv(OUT / "dataset_feature_summary.csv", index=False)
    features.to_csv(OUT / "b2_features.csv", index=False)
    fast.to_csv(OUT / "fast_recomputed_baselines.csv", index=False)

    with pd.ExcelWriter(OUT / "reproduction_results_bundle.xlsx", engine="openpyxl") as writer:
        inventory.to_excel(writer, sheet_name="csv_inventory", index=False)
        datasets.to_excel(writer, sheet_name="dataset_inventory", index=False)
        models.to_excel(writer, sheet_name="model_inventory", index=False)
        feature_summary.to_excel(writer, sheet_name="dataset_summary", index=False)
        features.to_excel(writer, sheet_name="b2_features", index=False)
        status.to_excel(writer, sheet_name="reproduction_status", index=False)
        fast.to_excel(writer, sheet_name="fast_local_baselines", index=False)
        if not trained_metrics.empty:
            trained_metrics.to_excel(writer, sheet_name="ai4i_trained_metrics", index=False)
        if not deep_metrics.empty:
            deep_metrics.to_excel(writer, sheet_name="ai4i_deep_pt_metrics", index=False)
        if not phm_metrics.empty:
            phm_metrics.to_excel(writer, sheet_name="phm_model_metrics", index=False)
        if not phm_cfg.empty:
            phm_cfg.to_excel(writer, sheet_name="phm_training_cfg", index=False)
        if not trained_artifacts.empty:
            trained_artifacts.to_excel(writer, sheet_name="trained_artifacts", index=False)

    manifest = {
        "primary_dataset": str(DATA_PATH.relative_to(ROOT)),
        "second_dataset_expected": "data/phm2010/",
        "trained_models_expected": "trained_models/",
        "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file()),
        "note": "Code-first bundle: all metrics are generated from local CSV/data arrays and trained artifacts; no performance table is copied from the PDF.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Wrote results to", OUT)
    print(feature_summary.to_string(index=False))
    print("\nDataset inventory:")
    print(datasets.to_string(index=False))
    print("\nModel inventory:")
    print(models.to_string(index=False))
    print("\nReproduction status:")
    print(status.to_string(index=False))
    print("\nFast recomputed local baselines:")
    print(fast[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
