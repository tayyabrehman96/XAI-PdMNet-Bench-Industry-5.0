#!/usr/bin/env python3
"""Train PHM 2010 release models and export real model artifacts.

Required inputs:
  data/phm2010/phm2010_feature_table.csv
  data/phm2010/phm2010_windows_6x500.csv

Outputs:
  trained_models/phm2010/xgboost_phm.json
  trained_models/phm2010/cnn1d_phm.pt
  trained_models/phm2010/cnn1d_phm_weights.pt
  trained_models/phm2010/rcnn_phm.pt
  trained_models/phm2010/rcnn_phm_weights.pt
  trained_models/phm2010/transfer_ai4i_to_phm.pt
  trained_models/phm2010/transfer_ai4i_to_phm_weights.pt
  results/phm_model_metrics.csv
  results/phm_training_config.csv
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
deps_override = os.environ.get("XAI_DEPS_PATH", "").strip()
if deps_override:
    sys.path.insert(0, deps_override)
elif (ROOT / ".deps_runtime").exists():
    sys.path.insert(0, str(ROOT / ".deps_runtime"))
elif (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

from xai_pdmbench.data import clean_ai4i_benchmark_rows, normalize_columns  # noqa: E402
from xai_pdmbench.features import build_features_b2  # noqa: E402


FEATURES_CSV = ROOT / "data" / "phm2010" / "phm2010_feature_table.csv"
WINDOWS_CSV = ROOT / "data" / "phm2010" / "phm2010_windows_6x500.csv"
AI4I_CSV = ROOT / "data" / "ai4i" / "ai4i2020.csv"
MODEL_DIR = ROOT / "trained_models" / "phm2010"
RESULTS_DIR = ROOT / "results"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_EPOCHS = int(os.environ.get("PHM_CNN_EPOCHS", "35"))
RCNN_EPOCHS = int(os.environ.get("PHM_RCNN_EPOCHS", "35"))
TRANSFER_PRETRAIN_EPOCHS = int(os.environ.get("PHM_TRANSFER_PRETRAIN_EPOCHS", "15"))
TRANSFER_FREEZE_EPOCHS = int(os.environ.get("PHM_TRANSFER_FREEZE_EPOCHS", "10"))
TRANSFER_FINETUNE_EPOCHS = int(os.environ.get("PHM_TRANSFER_FINETUNE_EPOCHS", "20"))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val(y: np.ndarray, val_frac: float = 0.2, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train - mean) / std, (val - mean) / std, (test - mean) / std, mean, std


def metric_row(dataset: str, model: str, y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | str]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (prob >= threshold).astype(int)
    return {
        "dataset": dataset,
        "model": model,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
    }


class CNN1DPHM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


class RCNNPHM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x).transpose(1, 2)
        z, _ = self.bilstm(z)
        return self.head(z[:, -1, :]).squeeze(1)


class AI4IEncoder(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AI4IClassifier(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.encoder = AI4IEncoder(in_dim)
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)


class TransferPHM(nn.Module):
    def __init__(self, phm_dim: int, ai4i_dim: int, encoder: AI4IEncoder) -> None:
        super().__init__()
        self.phm_proj = nn.Linear(phm_dim, ai4i_dim)
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(64, 64), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.phm_proj(x)
        z = self.encoder(z)
        return self.head(z).squeeze(1)


def train_binary_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    pos_weight: float,
    lr: float = 1e-3,
    patience: int = 15,
) -> nn.Module:
    from sklearn.metrics import average_precision_score

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

    best_state = None
    best_score = -1.0
    wait = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        all_prob = []
        all_y = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
                all_prob.append(prob)
                all_y.append(yb.numpy())
        p = np.concatenate(all_prob)
        y = np.concatenate(all_y)
        score = float(average_precision_score(y, p))
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_prob(model: nn.Module, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            prob = torch.sigmoid(model(xb)).detach().cpu().numpy()
            outs.append(prob)
    return np.concatenate(outs)


def save_torch_artifacts(model_id: str, model: nn.Module, payload: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, MODEL_DIR / f"{model_id}.pt")
    torch.save(model.state_dict(), MODEL_DIR / f"{model_id}_weights.pt")


def train_xgboost_tabular(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    neg = int((y_train == 0).sum())
    pos = max(int((y_train == 1).sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        random_state=SEED,
        tree_method="hist",
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)
    model.save_model(MODEL_DIR / "xgboost_phm.json")
    return model.predict_proba(x_test)[:, 1]


def load_phm_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FEATURES_CSV.exists() or not WINDOWS_CSV.exists():
        raise SystemExit(
            "Missing PHM CSVs. Run scripts/prepare_phm2010_release_dataset.py first."
        )
    feat = pd.read_csv(FEATURES_CSV)
    win = pd.read_csv(WINDOWS_CSV)
    return feat.sort_values(["cutter", "cut_index"]).reset_index(drop=True), win.sort_values(
        ["cutter", "cut_index"]
    ).reset_index(drop=True)


def build_ai4i_pretrain_data() -> tuple[np.ndarray, np.ndarray]:
    if not AI4I_CSV.exists():
        raise SystemExit(f"Missing AI4I CSV: {AI4I_CSV}")
    df = clean_ai4i_benchmark_rows(normalize_columns(pd.read_csv(AI4I_CSV)))
    X, y = build_features_b2(df)
    return X.to_numpy(np.float32), y.astype(np.int64)


def main() -> None:
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feat_df, win_df = load_phm_data()

    meta_cols = ["cutter", "cut_index", "wear", "failure_label"]
    phm_feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    x_phm = feat_df[phm_feature_cols].to_numpy(np.float32)
    y_phm = feat_df["failure_label"].to_numpy(np.int64)

    seq_cols = [c for c in win_df.columns if c.startswith("ch")]
    x_seq = win_df[seq_cols].to_numpy(np.float32).reshape(-1, 6, 500)
    y_seq = win_df["failure_label"].to_numpy(np.int64)

    train_mask = feat_df["cutter"].isin(["C1", "C4"]).to_numpy()
    test_mask = feat_df["cutter"].eq("C6").to_numpy()
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise SystemExit("Expected PHM split not found. Need C1/C4 train and C6 test rows.")

    x_train_tab = x_phm[train_mask]
    y_train_tab = y_phm[train_mask]
    x_test_tab = x_phm[test_mask]
    y_test = y_phm[test_mask]

    xgb_prob = train_xgboost_tabular(x_train_tab, y_train_tab, x_test_tab)
    metrics_rows = [metric_row("PHM 2010", "xgboost_phm", y_test, xgb_prob)]

    train_seq = x_seq[train_mask]
    train_y = y_seq[train_mask]
    test_seq = x_seq[test_mask]
    test_y = y_seq[test_mask]
    tr_idx, val_idx = split_train_val(train_y, val_frac=0.2, seed=SEED)

    seq_train = train_seq[tr_idx]
    seq_val = train_seq[val_idx]
    y_seq_train = train_y[tr_idx].astype(np.float32)
    y_seq_val = train_y[val_idx].astype(np.float32)

    seq_mean = seq_train.mean(axis=(0, 2), keepdims=True)
    seq_std = seq_train.std(axis=(0, 2), keepdims=True)
    seq_std[seq_std < 1e-12] = 1.0
    seq_train_s = (seq_train - seq_mean) / seq_std
    seq_val_s = (seq_val - seq_mean) / seq_std
    seq_test_s = (test_seq - seq_mean) / seq_std

    pos_weight_seq = float((y_seq_train == 0).sum() / max((y_seq_train == 1).sum(), 1))
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(seq_train_s, dtype=torch.float32),
            torch.tensor(y_seq_train, dtype=torch.float32),
        ),
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(seq_val_s, dtype=torch.float32),
            torch.tensor(y_seq_val, dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=False,
    )

    cnn = train_binary_model(
        CNN1DPHM(),
        train_loader,
        val_loader,
        epochs=CNN_EPOCHS,
        pos_weight=pos_weight_seq,
        lr=1e-3,
        patience=15,
    )
    cnn_prob = predict_prob(cnn, seq_test_s)
    metrics_rows.append(metric_row("PHM 2010", "cnn1d_phm", test_y, cnn_prob))
    save_torch_artifacts(
        "cnn1d_phm",
        cnn,
        {
            "model_id": "cnn1d_phm",
            "config": {
                "epochs": CNN_EPOCHS,
                "seed": SEED,
                "split": "C1+C4 train, C6 test",
                "window_length": 500,
            },
            "metrics": metrics_rows[-1],
            "state_dict": cnn.state_dict(),
        },
    )

    rcnn = train_binary_model(
        RCNNPHM(),
        train_loader,
        val_loader,
        epochs=RCNN_EPOCHS,
        pos_weight=pos_weight_seq,
        lr=1e-3,
        patience=15,
    )
    rcnn_prob = predict_prob(rcnn, seq_test_s)
    metrics_rows.append(metric_row("PHM 2010", "rcnn_phm", test_y, rcnn_prob))
    save_torch_artifacts(
        "rcnn_phm",
        rcnn,
        {
            "model_id": "rcnn_phm",
            "config": {
                "epochs": RCNN_EPOCHS,
                "seed": SEED,
                "split": "C1+C4 train, C6 test",
                "window_length": 500,
            },
            "metrics": metrics_rows[-1],
            "state_dict": rcnn.state_dict(),
        },
    )

    ai4i_x, ai4i_y = build_ai4i_pretrain_data()
    pre_tr_idx, pre_val_idx = split_train_val(ai4i_y, val_frac=0.2, seed=SEED)
    ai4i_tr = ai4i_x[pre_tr_idx]
    ai4i_val = ai4i_x[pre_val_idx]
    ai4i_y_tr = ai4i_y[pre_tr_idx].astype(np.float32)
    ai4i_y_val = ai4i_y[pre_val_idx].astype(np.float32)
    ai4i_tr_s, ai4i_val_s, _, ai4i_mean, ai4i_std = standardize(
        ai4i_tr, ai4i_val, ai4i_val
    )
    ai4i_pos_w = float((ai4i_y_tr == 0).sum() / max((ai4i_y_tr == 1).sum(), 1))
    pretrain_loader = DataLoader(
        TensorDataset(
            torch.tensor(ai4i_tr_s, dtype=torch.float32),
            torch.tensor(ai4i_y_tr, dtype=torch.float32),
        ),
        batch_size=128,
        shuffle=True,
    )
    preval_loader = DataLoader(
        TensorDataset(
            torch.tensor(ai4i_val_s, dtype=torch.float32),
            torch.tensor(ai4i_y_val, dtype=torch.float32),
        ),
        batch_size=256,
        shuffle=False,
    )
    pre_model = train_binary_model(
        AI4IClassifier(ai4i_x.shape[1]),
        pretrain_loader,
        preval_loader,
        epochs=TRANSFER_PRETRAIN_EPOCHS,
        pos_weight=ai4i_pos_w,
        lr=1e-3,
        patience=8,
    )

    phm_tr_idx, phm_val_idx = split_train_val(y_train_tab, val_frac=0.2, seed=SEED)
    phm_tr = x_train_tab[phm_tr_idx]
    phm_val = x_train_tab[phm_val_idx]
    phm_y_tr = y_train_tab[phm_tr_idx].astype(np.float32)
    phm_y_val = y_train_tab[phm_val_idx].astype(np.float32)
    phm_tr_s, phm_val_s, phm_test_s, phm_mean, phm_std = standardize(phm_tr, phm_val, x_test_tab)

    transfer = TransferPHM(phm_dim=phm_tr_s.shape[1], ai4i_dim=ai4i_x.shape[1], encoder=pre_model.encoder)
    transfer = transfer.to(DEVICE)

    for p in transfer.encoder.parameters():
        p.requires_grad = False
    train_freeze_loader = DataLoader(
        TensorDataset(
            torch.tensor(phm_tr_s, dtype=torch.float32),
            torch.tensor(phm_y_tr, dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
    )
    val_transfer_loader = DataLoader(
        TensorDataset(
            torch.tensor(phm_val_s, dtype=torch.float32),
            torch.tensor(phm_y_val, dtype=torch.float32),
        ),
        batch_size=128,
        shuffle=False,
    )
    pos_w_transfer = float((phm_y_tr == 0).sum() / max((phm_y_tr == 1).sum(), 1))
    transfer = train_binary_model(
        transfer,
        train_freeze_loader,
        val_transfer_loader,
        epochs=TRANSFER_FREEZE_EPOCHS,
        pos_weight=pos_w_transfer,
        lr=1e-3,
        patience=8,
    )

    for p in transfer.encoder.parameters():
        p.requires_grad = True
    transfer = train_binary_model(
        transfer,
        train_freeze_loader,
        val_transfer_loader,
        epochs=TRANSFER_FINETUNE_EPOCHS,
        pos_weight=pos_w_transfer,
        lr=1e-4,
        patience=10,
    )
    transfer_prob = predict_prob(transfer, phm_test_s)
    metrics_rows.append(metric_row("PHM 2010", "transfer_ai4i_to_phm", y_test, transfer_prob))
    save_torch_artifacts(
        "transfer_ai4i_to_phm",
        transfer,
        {
            "model_id": "transfer_ai4i_to_phm",
            "config": {
                "pretrain_epochs_ai4i": TRANSFER_PRETRAIN_EPOCHS,
                "freeze_epochs_phm": TRANSFER_FREEZE_EPOCHS,
                "finetune_epochs_phm": TRANSFER_FINETUNE_EPOCHS,
                "seed": SEED,
                "split": "C1+C4 train, C6 test",
            },
            "metrics": metrics_rows[-1],
            "state_dict": transfer.state_dict(),
            "ai4i_mean": ai4i_mean.tolist(),
            "ai4i_std": ai4i_std.tolist(),
            "phm_mean": phm_mean.tolist(),
            "phm_std": phm_std.tolist(),
        },
    )

    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / "phm_model_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"model": "xgboost_phm", "epochs": 0, "seed": SEED},
            {"model": "cnn1d_phm", "epochs": CNN_EPOCHS, "seed": SEED},
            {"model": "rcnn_phm", "epochs": RCNN_EPOCHS, "seed": SEED},
            {
                "model": "transfer_ai4i_to_phm",
                "epochs": TRANSFER_PRETRAIN_EPOCHS + TRANSFER_FREEZE_EPOCHS + TRANSFER_FINETUNE_EPOCHS,
                "seed": SEED,
            },
        ]
    ).to_csv(RESULTS_DIR / "phm_training_config.csv", index=False)

    (MODEL_DIR / "phm_feature_columns.json").write_text(
        json.dumps(phm_feature_cols, indent=2), encoding="utf-8"
    )

    print("Wrote PHM artifacts to", MODEL_DIR)
    print(pd.DataFrame(metrics_rows).to_string(index=False))


if __name__ == "__main__":
    main()
