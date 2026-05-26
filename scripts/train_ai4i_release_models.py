#!/usr/bin/env python3
"""Train and export AI4I release models from the repository CSV.

Outputs:
  trained_models/ai4i/logistic_regression.joblib
  trained_models/ai4i/random_forest_ctgan.joblib
  trained_models/ai4i/hist_gradient_boosting_ctgan.joblib
  trained_models/ai4i/xgboost_ctgan.json          (when xgboost is installed)
  trained_models/ai4i/ai4i_feature_columns.json
  results/ai4i_trained_model_metrics.csv

The script is code-first: every metric is computed from data/ai4i/ai4i2020.csv and
the model artifacts exported by this run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
deps_override = os.environ.get("XAI_DEPS_PATH", "").strip()
if deps_override:
    sys.path.insert(0, deps_override)
elif (ROOT / ".deps_runtime").exists():
    sys.path.insert(0, str(ROOT / ".deps_runtime"))
elif (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

from xai_pdmbench.data import clean_ai4i_benchmark_rows, normalize_columns
from xai_pdmbench.features import build_features_b2


DATA = ROOT / "data" / "ai4i2020.csv"
if not DATA.exists():
    DATA = ROOT / "data" / "ai4i" / "ai4i2020.csv"
MODELS = ROOT / "trained_models" / "ai4i"
RESULTS = ROOT / "results"
SEED = 42


def require_ml_deps():
    try:
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies. Install the full environment first:\n"
            "  pip install -r requirements.txt\n"
            f"Original import error: {exc}"
        ) from exc

    return {
        "joblib": joblib,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def score_row(name: str, model, X_test, y_test, metrics) -> dict[str, float | str]:
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "dataset": "AI4I 2020",
        "model": name,
        "accuracy": metrics["accuracy_score"](y_test, pred),
        "precision": metrics["precision_score"](y_test, pred, zero_division=0),
        "recall": metrics["recall_score"](y_test, pred, zero_division=0),
        "f1": metrics["f1_score"](y_test, pred, zero_division=0),
        "roc_auc": metrics["roc_auc_score"](y_test, prob),
        "pr_auc": metrics["average_precision_score"](y_test, prob),
    }


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


def main() -> None:
    deps = require_ml_deps()
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = normalize_columns(pd.read_csv(DATA))
    df = clean_ai4i_benchmark_rows(df)
    X, y = build_features_b2(df)
    X = X.copy()
    X.columns = sanitize_feature_names(X.columns)

    train_test_split = deps["train_test_split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    Pipeline = deps["Pipeline"]
    StandardScaler = deps["StandardScaler"]
    LogisticRegression = deps["LogisticRegression"]
    RandomForestClassifier = deps["RandomForestClassifier"]
    HistGradientBoostingClassifier = deps["HistGradientBoostingClassifier"]
    joblib = deps["joblib"]

    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest_ctgan": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    try:
        models["hist_gradient_boosting_ctgan"] = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=SEED,
        )
    except TypeError:
        models["hist_gradient_boosting_ctgan"] = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=SEED,
        )

    rows = []
    for name, model in models.items():
        print(f"Training {name}")
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS / f"{name}.joblib")
        rows.append(score_row(name, model, X_test, y_test, deps))

    try:
        import xgboost as xgb

        neg = int((y_train == 0).sum())
        pos = max(int((y_train == 1).sum()), 1)
        xgb_model = xgb.XGBClassifier(
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
        print("Training xgboost_ctgan")
        xgb_model.fit(X_train, y_train)
        xgb_model.save_model(MODELS / "xgboost_ctgan.json")
        rows.append(score_row("xgboost_ctgan", xgb_model, X_test, y_test, deps))
    except ImportError:
        print("xgboost not installed; skipping xgboost_ctgan export")

    (MODELS / "ai4i_feature_columns.json").write_text(
        json.dumps(list(X.columns), indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(RESULTS / "ai4i_trained_model_metrics.csv", index=False)
    print("Wrote model artifacts to", MODELS)
    print("Wrote metrics to", RESULTS / "ai4i_trained_model_metrics.csv")


if __name__ == "__main__":
    main()
