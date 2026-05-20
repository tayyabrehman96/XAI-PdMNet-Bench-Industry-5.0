# XAI-PdMNet-Bench — System architecture

This document matches the six-stage pipeline described in the manuscript: **ingestion → leakage-safe features → CTGAN audit → multi-model training → held-out evaluation → SHAP + operator report + dashboard**.

---

## High-level data flow (Mermaid)

The diagram below renders automatically on GitHub when viewing this file.

```mermaid
flowchart TB
  subgraph S1["Stage 1 — Ingestion & cleaning"]
    UCI["UCI CSV / ucimlrepo id=601"]
    CLN["Ileri-style row filter\n(RNF/MF inconsistency)"]
    UCI --> CLN
  end

  subgraph S2["Stage 2 — Leakage-safe features"]
    B1["Track B1 — repro / may include TWF…RNF"]
    B2["Track B2 — drop TWF…RNF\n+ 17 engineered scalars + Type OHE"]
    CLN --> B1
    CLN --> B2
  end

  subgraph S3["Stage 3 — Imbalance handling"]
    CTGAN["CTGAN minority synthesis\n+ KS audit"]
    SMOTE["SMOTE / class weights / focal (CNN)"]
    B2 --> CTGAN
    B2 --> SMOTE
  end

  subgraph S4["Stage 4 — Models"]
    TAB["Tabular: LR, RF, HGB, XGB"]
    CNN1D["1D-CNN zoo — balanced CV"]
    SEQ["CNN–LSTM on sliding windows\n(pseudo-temporal)"]
    CTGAN --> TAB
    CTGAN --> CNN1D
    CTGAN --> SEQ
  end

  subgraph S5["Stage 5 — Evaluation"]
    HOLD["Shared held-out split\nn=1993 imbalanced"]
    MET["F1, ROC-AUC, PR-AUC\nthreshold τ* on val (CNN)"]
    TAB --> HOLD
    SEQ --> HOLD
    HOLD --> MET
  end

  subgraph S6["Stage 6 — XAI & reporting"]
    SHAP["SHAP TreeExplainer (XGBoost)\nGradientExplainer optional (CNN)"]
    RPT["Template-based 5-section\noperator report (JSON payload)"]
    UI["Streamlit / Plotly dashboard"]
    MET --> SHAP
    SHAP --> RPT
    RPT --> UI
  end
```

---

## Component responsibilities

| Stage | Input | Output | Notes |
|-------|--------|--------|-------|
| 1 | Raw AI4I rows | `df_clean` | ~27 rows dropped; see `clean_ai4i_basepaper` in [`src/xai_pdmbench/data.py`](../src/xai_pdmbench/data.py) |
| 2 | Clean frame | `X_b1`/`X_b2`, `y` | B2 aligns with manuscript 25-dim tabular design |
| 3 | Training split | Augmented tensors / tables | CTGAN trained on failure-only rows; marginal KS on 25 features |
| 4 | Augmented training | Fitted models | Stratified 68/12/20 split; seeds fixed in notebook |
| 5 | Test split | Metrics CSV, curves | PR-AUC primary for imbalance; CNN uses tuned τ* |
| 6 | Best backbone + test batch | SHAP + report + figures | Production path: XGBoost + TreeExplainer |

---

## Pseudocode alignment (Colab)

| Colab step | Maps to |
|------------|---------|
| Step 1 | `ARTIFACTS`, seeds, progress tracker |
| Step 2 | Load CSV, `clean_ai4i_basepaper`, `build_features_b*` |
| Step 3 | Shallow ML + 1D-CNN CV |
| Step 4 | CNN–LSTM + optional CTGAN/SMOTE branches |
| Step 5–6 | Metrics, ROC/PR, SHAP, export |

---

## Optional bitmap figure

To add a raster pipeline image for talks or the GitHub README:

1. Export from your paper (`fig1_pipeline.png`) or regenerate with `generate_figures.py`.
2. Save as `docs/assets/pipeline.png`.
3. Reference it from the root `README.md` if desired.

Place a `.gitkeep` in `docs/assets/` so the folder exists before you add the PNG.
