#!/usr/bin/env python3
"""Train AI4I deep models and save .pt model/weight files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
deps_override = os.environ.get("XAI_DEPS_PATH", "").strip()
if deps_override:
    sys.path.insert(0, deps_override)
elif (ROOT / ".deps_runtime").exists():
    sys.path.insert(0, str(ROOT / ".deps_runtime"))
elif (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xai_pdmbench.data import clean_ai4i_benchmark_rows, normalize_columns
from xai_pdmbench.features import build_features_b2


DATA = ROOT / "data" / "ai4i" / "ai4i2020.csv"
OUT_MODELS = ROOT / "trained_models" / "ai4i"
OUT_RESULTS = ROOT / "results"
SEED = 42
TABULAR_EPOCHS = int(os.environ.get("AI4I_TORCH_TABULAR_EPOCHS", "30"))
CNN_LSTM_EPOCHS = int(os.environ.get("AI4I_TORCH_CNN_LSTM_EPOCHS", "20"))


def sanitize_feature_names(columns) -> list[str]:
    clean = []
    seen = {}
    for col in columns:
        name = str(col)
        for bad in "[]<>":
            name = name.replace(bad, "")
        name = name.replace(" ", "_").replace("/", "_per_")
        count = seen.get(name, 0)
        seen[name] = count + 1
        clean.append(name if count == 0 else f"{name}_{count}")
    return clean


def stratified_split(y: np.ndarray, test_size: float = 0.2):
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


def standardize(train, test):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def roc_auc(y_true, score):
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))


def pr_auc(y_true, score):
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((y_true == 1).sum()), 1)
    return float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))


def metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
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


class AlexNet1DSafe(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 192, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(192, 192, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(192, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(1)


class TabTransformerSmall(nn.Module):
    def __init__(self, n_features, dim=32, heads=4):
        super().__init__()
        self.value = nn.Linear(1, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_features, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_features * dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        z = self.value(x.unsqueeze(-1)) + self.pos
        return self.head(self.encoder(z)).squeeze(1)


class CNNLSTM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_features, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 32, 3, padding=1), nn.ReLU())
        self.lstm1 = nn.LSTM(32, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        return self.head(z[:, -1, :]).squeeze(1)


def train_model(model, train_loader, pos_weight, epochs=18):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def predict(model, x):
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            xb = torch.tensor(x[start : start + 512], dtype=torch.float32)
            outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs)


def make_windows(x, y, window=10):
    xs, ys = [], []
    for i in range(0, len(x) - window + 1):
        xs.append(x[i : i + window])
        ys.append(y[i + window - 1])
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def save_artifacts(model_id, model, config, metric_row):
    payload = {
        "model_id": model_id,
        "config": config,
        "metrics": metric_row,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, OUT_MODELS / f"{model_id}.pt")
    torch.save(model.state_dict(), OUT_MODELS / f"{model_id}_weights.pt")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)

    df = clean_ai4i_benchmark_rows(normalize_columns(pd.read_csv(DATA)))
    X, y = build_features_b2(df)
    feature_columns = sanitize_feature_names(X.columns)
    X.columns = feature_columns
    train_idx, test_idx = stratified_split(y)
    x_train, x_test, mean, scale = standardize(X.iloc[train_idx].to_numpy(np.float32), X.iloc[test_idx].to_numpy(np.float32))
    y_train = y[train_idx].astype(np.float32)
    y_test = y[test_idx].astype(int)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    rows = []
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=128,
        shuffle=True,
    )

    for model_id, cls in [("alexnet1d_safe", AlexNet1DSafe), ("tabtransformer", TabTransformerSmall)]:
        print("Training", model_id)
        model = train_model(cls(x_train.shape[1]), train_loader, pos_weight, epochs=TABULAR_EPOCHS)
        prob = predict(model, x_test)
        row = {"dataset": "AI4I 2020", "model": model_id}
        row.update(metrics(y_test, prob))
        rows.append(row)
        save_artifacts(
            model_id,
            model,
            {
                "features": feature_columns,
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "epochs": TABULAR_EPOCHS,
                "seed": SEED,
            },
            row,
        )

    print("Training cnn_lstm_thr031")
    xw, yw = make_windows(X.to_numpy(np.float32), y.astype(np.float32), window=10)
    w_train_idx, w_test_idx = stratified_split(yw.astype(int))
    xw_train_raw, xw_test_raw = xw[w_train_idx], xw[w_test_idx]
    yw_train, yw_test = yw[w_train_idx], yw[w_test_idx].astype(int)
    flat_train = xw_train_raw.reshape(-1, xw.shape[2])
    w_mean = flat_train.mean(axis=0)
    w_scale = flat_train.std(axis=0)
    w_scale[w_scale < 1e-12] = 1.0
    xw_train = (xw_train_raw - w_mean) / w_scale
    xw_test = (xw_test_raw - w_mean) / w_scale
    w_pos_weight = float((yw_train == 0).sum() / max((yw_train == 1).sum(), 1))
    w_loader = DataLoader(
        TensorDataset(torch.tensor(xw_train, dtype=torch.float32), torch.tensor(yw_train, dtype=torch.float32)),
        batch_size=128,
        shuffle=True,
    )
    cnn_lstm = train_model(CNNLSTM(xw.shape[2]), w_loader, w_pos_weight, epochs=CNN_LSTM_EPOCHS)
    prob = predict(cnn_lstm, xw_test)
    row = {"dataset": "AI4I 2020", "model": "cnn_lstm_thr031"}
    row.update(metrics(yw_test, prob, threshold=0.31))
    rows.append(row)
    save_artifacts(
        "cnn_lstm_thr031",
        cnn_lstm,
        {
            "features": feature_columns,
            "window": 10,
            "mean": w_mean.tolist(),
            "scale": w_scale.tolist(),
            "threshold": 0.31,
            "epochs": CNN_LSTM_EPOCHS,
            "seed": SEED,
        },
        row,
    )

    pd.DataFrame(rows).to_csv(OUT_RESULTS / "ai4i_deep_pt_model_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"model": "alexnet1d_safe", "epochs": TABULAR_EPOCHS, "seed": SEED},
            {"model": "tabtransformer", "epochs": TABULAR_EPOCHS, "seed": SEED},
            {"model": "cnn_lstm_thr031", "epochs": CNN_LSTM_EPOCHS, "seed": SEED},
        ]
    ).to_csv(OUT_RESULTS / "ai4i_deep_training_config.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Wrote .pt models to", OUT_MODELS)


if __name__ == "__main__":
    main()
