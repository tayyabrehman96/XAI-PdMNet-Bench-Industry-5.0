#!/usr/bin/env python3
"""Train dependency-free AI4I baseline models and export real weight files.

This is not a replacement for the full benchmark model zoo. It exists so a clean
checkout can immediately produce actual model artifacts from code + CSV using
only numpy/pandas.

Outputs:
  trained_models/ai4i/numpy_logistic_weighted.npz
  trained_models/ai4i/numpy_gaussian_nb.npz
  trained_models/ai4i/numpy_knn_k5.npz
  trained_models/ai4i/fast_feature_columns.json
  results/fast_release_model_metrics.csv
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_pdmbench.data import clean_ai4i_benchmark_rows, normalize_columns  # noqa: E402
from xai_pdmbench.features import build_features_b2  # noqa: E402


DATA = ROOT / "data" / "ai4i2020.csv"
if not DATA.exists():
    DATA = ROOT / "data" / "ai4i" / "ai4i2020.csv"
MODELS = ROOT / "trained_models" / "ai4i"
RESULTS = ROOT / "results"
SEED = 42


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


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int = 1500, lr: float = 0.05) -> tuple[np.ndarray, float]:
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


def fit_gaussian_nb(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    params = {}
    for cls in (0, 1):
        xc = x[y == cls]
        params[f"mu_{cls}"] = xc.mean(axis=0)
        params[f"var_{cls}"] = xc.var(axis=0) + 1e-6
        params[f"prior_{cls}"] = np.array([max(len(xc), 1) / len(x)], dtype=np.float64)
    return params


def predict_gaussian_nb(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    logps = []
    for cls in (0, 1):
        mu = params[f"mu_{cls}"]
        var = params[f"var_{cls}"]
        prior = math.log(float(params[f"prior_{cls}"][0]))
        ll = -0.5 * (np.log(2 * np.pi * var) + ((x - mu) ** 2) / var).sum(axis=1)
        logps.append(prior + ll)
    logits = np.vstack(logps).T
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


def predict_knn(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int = 5) -> np.ndarray:
    probs = np.empty(len(x_test), dtype=np.float64)
    for start in range(0, len(x_test), 256):
        xt = x_test[start : start + 256]
        d2 = ((xt[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2)
        nn = np.argpartition(d2, kth=min(k, len(x_train) - 1), axis=1)[:, :k]
        probs[start : start + 256] = y_train[nn].mean(axis=1)
    return probs


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
    return float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))


def metric_row(model: str, y_true: np.ndarray, prob: np.ndarray) -> dict[str, float | str]:
    pred = (prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "dataset": "AI4I 2020",
        "model": model,
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(y_true, prob),
        "pr_auc": pr_auc(y_true, prob),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = clean_ai4i_benchmark_rows(normalize_columns(pd.read_csv(DATA)))
    X, y = build_features_b2(df)
    train_idx, test_idx = stratified_split(y)
    x_train_raw = X.iloc[train_idx].to_numpy(np.float64)
    x_test_raw = X.iloc[test_idx].to_numpy(np.float64)
    y_train = y[train_idx]
    y_test = y[test_idx]
    x_train, x_test, mean, scale = standardize(x_train_raw, x_test_raw)

    rows = []

    w, b = fit_logistic(x_train, y_train)
    prob = sigmoid(x_test @ w + b)
    np.savez(MODELS / "numpy_logistic_weighted.npz", weights=w, bias=b, mean=mean, scale=scale)
    rows.append(metric_row("numpy_logistic_weighted", y_test, prob))

    nb = fit_gaussian_nb(x_train, y_train)
    prob = predict_gaussian_nb(nb, x_test)
    np.savez(MODELS / "numpy_gaussian_nb.npz", mean=mean, scale=scale, **nb)
    rows.append(metric_row("numpy_gaussian_nb", y_test, prob))

    prob = predict_knn(x_train, y_train, x_test, k=5)
    np.savez(MODELS / "numpy_knn_k5.npz", X_train=x_train, y_train=y_train, mean=mean, scale=scale, k=np.array([5]))
    rows.append(metric_row("numpy_knn_k5", y_test, prob))

    (MODELS / "fast_feature_columns.json").write_text(
        json.dumps(list(X.columns), indent=2), encoding="utf-8"
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "fast_release_model_metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print("Wrote fast release model artifacts to", MODELS)


if __name__ == "__main__":
    main()
