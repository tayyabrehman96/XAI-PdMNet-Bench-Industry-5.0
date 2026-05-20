"""
Generate predictive_maintenance_ai4i2020.ipynb for Google Colab / Jupyter.
Run from repo root: python colab/gen_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(text: str) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "predictive_maintenance_ai4i2020.ipynb"

cells = []

cells.append(
    md(
        r"""
# AI4I 2020 — Predictive maintenance (base paper repro + proposed pipeline)

**Dataset:** [UCI ML Repository — AI4I 2020 (ID 601)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).

**Base paper:** Ileri, Altun, Narin (2024), *Applied Sciences* — DOI [10.3390/app14114899](https://doi.org/10.3390/app14114899).

**This notebook**
- **B1:** broader feature set for proximity to Table 4 (may include fault-mode columns → **label leakage** — documented).
- **B2:** leakage-safe pipeline (drop `TWF`…`RNF`) + **17 engineered scalars** (loads, thermal stress, wear interactions, logs) + one-hot **Type** — see §2b vs base paper.
- **Resume:** checkpoints + `cv_progress.json` under `ARTIFACTS` — use **Google Drive** path so Colab disconnects do not wipe progress.
- **Sliding windows:** consecutive dataframe rows — **not** guaranteed physical machine timelines (AI4I rows are independent datapoints); document honestly in your paper.

---

### Colab run order (pipeline map)

| Step | Block | What happens |
|------|--------|----------------|
| **0** | Optional download | Fetch a **zip** of `models/` + `checkpoints/` from your URL (no manual Drive upload) |
| **1** | Setup | Seeds, `ARTIFACTS`, tracker |
| **2** | Data | **Ingest** AI4I → **clean** (Ileri rules) → **B1/B2 preprocessing** (features) |
| **3** | Train baselines | Shallow ML + **1D-CNN** zoo (balanced CV, per-fold **scaling**) |
| **4** | Train proposal | **CNN–LSTM** on windows (**class weights / focal**, checkpoints) |
| **5** | Test & compare | Held-out metrics, **side-by-side** vs LeNet CV, augmentation / LR–RF–XGB–**HGB**, **auto best-model pick** (baseline vs contribution) |
| **6** | Calibration & curves | ROC/PR, confusion @ tuned threshold, **reliability diagram** |
| **7** | XAI & replay | SHAP, optional LLM, Plotly |
| **8** | Export | Excel **`results_summary.xlsx`**, CSV bundle |
| **9** | Checklist | Reviewer readiness CSV |
| **10** | Final bundle | One folder with code/configs/models/checkpoints/results/figures/reports + expert audit; plus **`RESEARCH_EXPORT_PACK/`** (all CSV + all figures in two subfolders for manuscripts) |
"""
    )
)

cells.append(
    md(
        r"""
## 1. Dataset — AI4I 2020 (Predictive Maintenance)

| Item | Detail |
|------|--------|
| **Source** | [UCI ML Repository, ID 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| **Citation** | Matzka, S. (2020). UCI ML Repository. DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C) |
| **License** | CC BY 4.0 |
| **Size** | 10,000 rows × 14 attributes (synthetic, industry-style PdM) |
| **Target** | **Machine failure** binary (0 = normal, 1 = fault) |
| **Imbalance** | ~**3.4%** positives — optimize **recall / F1 / PR-AUC**, not accuracy alone |
| **Sensor / process inputs** | Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], product **Type** (L/M/H), identifiers |

**Leakage warning:** Columns **TWF, HDF, PWF, OSF, RNF** are failure-mode flags logically tied to **Machine failure**. For scientifically valid models (**track B2**), they must **not** be used as inputs. **Track B1** may retain them only to approximate the base paper Table 4 setting — report this clearly in your manuscript.

**B2 feature engineering (no leakage):** adds **17** scalars from raw sensors before **Type** encoding — **Power_proxy**, **RPM_over_torque**, **Log_power_proxy**, **Log1p_torque**, **Log1p_rpm**, **Wear_norm**, **Log1p_wear**, **Torque_x_wear**, **Rpm_x_wear**, **Power_per_wear**, **Delta_temp_K**, **Thermal_ratio**, **Temp_product_K2**, **Inv_air_temp**, **Delta_temp_sq**, **Process_over_air**, **Thermal_per_wear**.

**Download in Google Colab:** Yes — fetch over **HTTPS** from the official UCI archive (public, **CC BY 4.0**, no login). This notebook first tries **`ucimlrepo`** (`fetch_ucirepo(id=601)`). That API often returns **features + target only** (no **TWF…RNF** columns); the next step **automatically reloads the full `ai4i2020.csv`** from UCI so cleaning and **B1** features work. You can skip straight to CSV by setting **`PDM_AI4I_CSV`** or uncommenting `wget` in the load cell.

**Cleaning (Ileri et al.):** Rows where **RNF** and **Machine failure** disagree, or **Machine failure** = 1 with no fault flag set, are removed before experiments (see code cell below).
"""
    )
)

cells.append(
    md(
        r"""
## 2. Gaps in the base paper (what they did not address)

**Reference:** Ileri, Altun, Narin (2024), *Appl. Sci.* — 1D-CNN + balancing + normalization on AI4I 2020; best ~**98.5%** accuracy / **98.32%** F1 with 1D-LeNet under balanced + normalized MF classification.

| Topic | Base paper (2024) | Open gap → our direction |
|-------|-------------------|---------------------------|
| **Temporal structure** | Single-row 1D-CNN (stateless) | **Sliding windows + CNN–LSTM** for sequence-shaped inputs |
| **Imbalance** | Random undersampling / balanced subsets | **Class weights**, **focal loss**, **SMOTE**, **CTGAN**, **F1 / cost-based** thresholds |
| **Explainability** | None on deep models | **SHAP GradientExplainer** on CNN–LSTM |
| **Operator communication** | Class labels / scores only | **LLM** briefings from SHAP + readings |
| **System demo** | Static plots / tables | **Digital twin style** replay (Plotly / Streamlit in paper) |
| **Framing** | Industry 4.0 benchmark | **Industry 5.0** human-centric AI narrative |

This notebook **reproduces-style** the LeNet branch (B1/B2 configurable) then implements the **proposal stack** (B2 + windows + CNN–LSTM + SHAP + optional LLM).
"""
    )
)

cells.append(
    md(
        r"""
## 2b. Base paper vs this notebook — inputs, imbalance, and “fault” handling

**Reference:** Ileri, Altun, Narin (2024), *Appl. Sci.* focus on **1D-CNN + normalization + balancing** on AI4I for machine-failure classification (strong scores when training is **class-balanced**).

| Aspect | Base paper style (what they emphasize) | What we add here (beyond that scope) |
|--------|----------------------------------------|--------------------------------------|
| **Input vector** | Essentially **raw process sensors + product Type** (then scaling / balancing in training) | **Track B2:** same raw fields plus **17 leakage-safe engineered scalars** (power/load, thermal stress, wear interactions, log-scales) — see feature cell |
| **Fault-type columns (`TWF`…`RNF`)** | Dataset includes them; some reproductions **risk label leakage** if used as inputs | **B2 explicitly drops them**; we model **fault risk** via **stress proxies** (temperature gap, mechanical load, wear) instead of future-known fault flags |
| **Imbalance** | Undersampling / **balanced** training regimes | **Class weights**, optional **focal loss**, **SMOTE**, **CTGAN**, **F1 and cost-based** threshold tuning on validation |
| **Temporal modeling** | Single-row **1D-CNN** | **Sliding windows + CNN–LSTM** |
| **Results reporting** | Accuracy / F1 under balancing | **ROC/PR-AUC**, **Brier**, **ECE**, **recall @ capped FPR**, **cost-optimal** metrics, **MC-dropout** uncertainty |
| **Transparency** | Classical CNN outputs | **SHAP**, optional **LLM** briefing, **Plotly** replay |

Use **B1** only when you intentionally approximate a **high-dimensional / leakage-prone** reproduction setting; use **B2** for **deployment-style** claims and your contribution narrative.
"""
    )
)

cells.append(
    md(
        r"""
## 3. Our contributions (paper-ready bullets)

1. **CTGAN / generative augmentation** on minority failures (vs SMOTE-only prior art on this dataset in many works).
2. **CNN–LSTM** on **W-step** sliding windows (local patterns + sequence context); base paper used flat 1D-CNN only.
3. **Deep SHAP** (GradientExplainer) attributions aligned with sensor channels — supports operator trust.
4. **SHAP → LLM** pipeline: structured attributions → short actionable maintenance text (optional OpenAI cell).
5. **Full baseline zoo in code:** **DT, SVM, k-NN** + **1D-LeNet, 1D-AlexNet (adapted), VGG-mini** (CV + checkpoints) beside **CNN–LSTM / SMOTE / CTGAN / LR–RF–XGB**.
6. **End-to-end artifact trail**: metrics tables, ROC/PR, confusion matrices, SHAP bar plot, replay HTML — suitable for **Results** and **System architecture** figures.
7. **Process-informed B2 features** (leakage-safe): **17 engineered scalars** on top of raw sensors + **Type** one-hot — power/load, thermal stress, wear–load interactions, log transforms — **without** using fault-flag columns (`TWF`…`RNF`).
8. **Training & evaluation extras**: optional **focal loss** for imbalance, **cost-sensitive** thresholding (FN vs FP), **Brier score + ECE** (calibration), **recall at capped FPR**, **MC Dropout** predictive uncertainty.

*(Align numbering with your final manuscript from `research_analysis.pdf`.)*
"""
    )
)

cells.append(
    md(
        r"""
## 4. System architecture (conceptual + implemented blocks)

**End-to-end flow (proposal)**

`AI4I 2020` → **clean + encode** → `(B1 leaked features | B2 safe features)`

→ **Base reproduction:** **DT / SVM / k-NN** + **1D-LeNet / 1D-AlexNet / VGG-mini** (balanced MF, 5-fold CV, min-max & standard scaling branches for CNNs)

→ **Proposal:** **balance / augment** (SMOTE • CTGAN demo) → **CNN–LSTM** on `[batch, W, F]` windows

→ **evaluate** (Acc, P, R, F1, ROC/PR-AUC, **Brier, ECE**, **recall@FPR**, cost-optimal threshold, **MC-dropout uncertainty**) → **SHAP** → **optional LLM report** → **Plotly replay** → **`paper_bundle/` exports**

**CNN–LSTM block (this notebook)** — input shape `(W=10, F=num_B2_features)`:

| Stage | Layer | Role |
|-------|--------|------|
| Front | Conv1D 64, k=3, ReLU | Local contiguous patterns across timesteps |
| | Conv1D 32, k=3, ReLU | Deeper local features |
| | MaxPool1D 2 | Downsample temporal axis |
| Mid | LSTM 64 (return sequences) | Early temporal memory |
| | Dropout 0.3 | Regularize |
| | LSTM 32 | Late temporal summary |
| Tail | Dropout 0.2 → Dense 16 ReLU → Dense 1 sigmoid | Fault probability |

**Base 1D-CNN repro branch:** **LeNet / AlexNet-style / VGG-mini** on `(n_features, 1)` with per-fold scaler, per-arch checkpoints, and `cv_progress.json` resume (keys include architecture name). **Shallow DT / SVM / k-NN** use the same balanced MF subset (one draw) with 5-fold CV.

**What is still “paper polish” (next steps outside code):** formal pipeline diagram (PowerPoint / draw.io), MDPI Sensors narrative for Industry 5.0 pillars, expert ratings for LLM outputs, Streamlit screenshots if you deploy locally.
"""
    )
)

cells.append(
    md(
        r"""
## 5. No-reupload mode (reuse existing results/models)

Set these environment variables **once** in Colab before running cells:

- `PDM_REUSE_RESULTS=1` (default): load cached CSV/checkpoints/SHAP cache where possible.
- `PDM_FORCE_REFRESH_DATA=1`: ignore cached dataset and fetch/download again.
- `PDM_FORCE_CNN_RETRAIN=1`: force retraining CNN-LSTM even if checkpoint exists.
- `PDM_FORCE_CTGAN_REFIT=1`: force CTGAN fit even if saved model exists.
- `PDM_FORCE_TABULAR_REFIT=1`: refit LR/RF/XGB/HGB even if `baselines_last_timestep.csv` exists.
- `PDM_FORCE_SMOTE_MLP_REFIT=1`: redo SMOTE+MLP even if cached metrics/`_smote_flat_train.npz` exist.
- `PDM_PREVIEW_PRIOR_RESULTS=1`: print CSV/JSON previews from **`ARTIFACTS`** (Step 1b cell).

**CNN–LSTM extras**

- `PDM_CNN_LOSS=focal` (default) or `bce` — focal focuses on hard minority examples; `bce` is plain cross-entropy.
- `PDM_FOCAL_GAMMA=2`, `PDM_FOCAL_ALPHA=0.75` — focal hyperparameters (when `PDM_CNN_LOSS=focal`).
- `PDM_USE_CLASS_WEIGHT=0` or `1` — empty/unset: **off when focal**, **on when BCE** (override explicitly if needed).
- `PDM_FN_COST` / `PDM_FP_COST` (defaults **10** / **1**) — validation cost grid → **test** metrics at cost-optimal threshold.
- `PDM_MC_SAMPLES=20` — MC Dropout forward passes for uncertainty (`0` disables).

**CNN–LSTM: early stopping, LR on plateau, fit checks**

- `PDM_ES_MONITOR=val_auc` (default) or `val_loss` — **val_auc** is usually better under heavy imbalance; **val_loss** matches older notebook behavior.
- `PDM_ES_PATIENCE=15` — epochs without improvement before stopping (default tuned for plateau + ReduceLR).
- `PDM_ES_MIN_DELTA` — default **0.002** if monitoring `val_auc`, else **1e-4** for `val_loss` (minimum change to count as improvement).
- `PDM_RLRP_FACTOR=0.5`, `PDM_RLRP_PATIENCE` (default ≈ one-third of ES patience, min 3) — **ReduceLROnPlateau** when metrics stall (helps **underfitting** / escape plateaus).
- `PDM_MIN_LR=1e-6` — floor for learning rate after reductions.
- `PDM_FITCHECK_GAP_WINDOW=3`, `PDM_FITCHECK_MIN_EPOCHS=5` — rolling gap window for `[fit-check]` lines.

**1D-CNN CV folds** — `ReduceLROnPlateau` + `min_delta` early stopping: `PDM_ES_PATIENCE_1DCNN`, `PDM_ES_MIN_DELTA_1DCNN`, `PDM_RLRP_FACTOR_1DCNN`, `PDM_MIN_LR_1DCNN`.

**1D-CNN protocol (vs Ileri et al. baseline)**

- `PDM_N_NEG_BALANCE` (default **402**) — negatives sampled per balanced MF subset (Table 3).
- `PDM_N_BALANCE_REPS` — random balanced subsets (**330** failures + **402** normals, Table 3); the paper uses **100** reps and reports the **maximum** metric across reps — default here is **5** for speed (`tables/ileri_2024_comparison_summary.csv` explains mean vs max). Use **`PDM_N_BALANCE_REPS=100`** for closer parity (long run).

**Human-centric LLM report**

- `REGOLO_API_KEY` **or** `OPENAI_API_KEY` — set via Colab **Secrets** / env only; never commit keys.
- With **`REGOLO_API_KEY`** set, **`PDM_LLM_BASE_URL`** defaults to **`https://api.regolo.ai/v1`** if unset.
- `PDM_LLM_MODEL` — Regolo chat model id (e.g. **`Llama-3.3-70B-Instruct`**) or OpenAI model name when using OpenAI.
- `PDM_LLM_BASE_URL` or `OPENAI_BASE_URL` — override API base (OpenAI-compatible paths).
- `PDM_LLM_PROVIDER=regolo|openai|auto` — `auto` picks Regolo base URL when only `REGOLO_API_KEY` is set.

**Optional: pull checkpoints from the web (Colab — no manual upload)**

Host a **zip** whose paths match this repo layout, e.g. top-level folders **`models/`**, **`checkpoints/`**, optional **`tables/cv_progress.json`**.

- `PDM_REMOTE_MODELS_ZIP_URL` — HTTPS URL to that zip; extracted under `ARTIFACTS` **before** training cells (run Step 0 cell).
- `PDM_REMOTE_CV_PROGRESS_URL` — optional direct URL to a `cv_progress.json` only.

Or use one switch:

- `PDM_RUN_MODE=fast` → **max reuse**: sets `PDM_REUSE_RESULTS=1` and clears `PDM_FORCE_*` refit flags (checkpoints, SMOTE/MLP cache, tabular CSV, etc.).
- `PDM_RUN_MODE=resume` → default; continue from saved checkpoints/caches.
- `PDM_RUN_MODE=full` → recompute heavy steps (data refresh + CNN retrain + CTGAN refit).

**Extensive experiments (one switch)**

- `PDM_EXTENSIVE_EXPERIMENTS=1` (also `true` / `yes` / `on`) — **before** the setup cell (or at the very top of it), raises default budgets via **`setdefault`** only: `PDM_N_BALANCE_REPS=25`, `PDM_EPOCHS_1DCNN`/`PDM_EPOCHS_LENET=100`, `PDM_EPOCHS_CNN=120`, slightly higher ES / ReduceLR patience. Any variable you set **earlier** still wins. For paper-style **100** balance reps, set `PDM_N_BALANCE_REPS=100` yourself (long run).

**Reporting (which model to highlight)**

- `PDM_BEST_MODEL_METRIC=f1` (default), `pr_auc`, or `roc_auc` — chooses **best contribution** model on held-out test (Step 5d).
- `PDM_PRIMARY_BASELINE=lenet` — narrative hook for Ileri et al. comparison (does not change training).

Recommended for everyday reruns (no file replace):

```python
import os
os.environ["PDM_REUSE_RESULTS"] = "1"
os.environ["PDM_FORCE_REFRESH_DATA"] = "0"
os.environ["PDM_FORCE_CNN_RETRAIN"] = "0"
os.environ["PDM_FORCE_CTGAN_REFIT"] = "0"
```

This lets you duplicate/run cells and continue from saved artifacts instead of starting from zero.

**Final paper bundle folder**

- `PDM_FINAL_BUNDLE_NAME=FINAL_ROUND_BUNDLE` controls the final consolidated folder name under `ARTIFACTS`.
- `PDM_RESEARCH_PACK_NAME=RESEARCH_EXPORT_PACK` — Step 10 also fills **`ARTIFACTS/<this>/csv/`** and **`figures/`** with every table CSV and every saved plot (easy zip for supervisors / LaTeX).
- Run **Step 10** after all training/evaluation cells to collect **code/configs/models/checkpoints/CSV/Excel/figures/reports/logs** in one clean folder.
"""
    )
)

cells.append(
    code(
        r"""
# --- Optional: mount Google Drive for persistent checkpoints (recommended on Colab)
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# os.environ['PDM_ARTIFACTS'] = '/content/drive/MyDrive/XAI_PdM_artifacts'

!pip -q install ucimlrepo pandas numpy scikit-learn imbalanced-learn matplotlib seaborn plotly openpyxl tqdm xgboost shap sdv openai tensorflow>=2.15
"""
    )
)

cells.append(
    code(
        r"""
from __future__ import annotations

import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

warnings.filterwarnings("ignore")

# Stream prints immediately while cells run (Colab/Jupyter often buffers stdout).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
try:
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def _show_df(title: str, df: pd.DataFrame) -> None:
    # Print full table as text (always works); try rich Colab display if available.
    print(title, flush=True)
    print(df.to_string(), flush=True)
    try:
        from IPython.display import display

        display(df)
    except Exception:
        pass


SEED = int(os.environ.get("PDM_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)

_pdm_a = os.environ.get("PDM_ARTIFACTS")
if _pdm_a:
    _default_art = Path(_pdm_a).expanduser()
elif Path("/content").exists():
    _default_art = Path("/content/artifacts")
else:
    _default_art = Path.cwd() / "pdm_artifacts"

# Dedicated experiment folder per run/model set to keep artifacts separated.
_exp_name = os.environ.get("PDM_EXPERIMENT_NAME", "ai4i_research_run").strip()
ARTIFACTS = (_default_art / _exp_name).resolve()
FINAL_BUNDLE = ARTIFACTS / os.environ.get("PDM_FINAL_BUNDLE_NAME", "FINAL_ROUND_BUNDLE").strip()
for sub in ("figures", "tables", "models", "checkpoints", "paper_bundle", "final_round_bundle"):
    (ARTIFACTS / sub).mkdir(parents=True, exist_ok=True)
for sub in (
    "code",
    "models",
    "checkpoints",
    "results_csv",
    "results_excel",
    "figures",
    "reports",
    "configs",
    "logs",
):
    (FINAL_BUNDLE / sub).mkdir(parents=True, exist_ok=True)

PROGRESS_PATH = ARTIFACTS / "tables" / "cv_progress.json"
CONFIG_PATH = ARTIFACTS / "tables" / "run_config.json"


def set_tf_seed(s: int = SEED) -> None:
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(s)
    except Exception:
        pass


set_tf_seed(SEED)

print("ARTIFACTS ->", ARTIFACTS, flush=True)
print("FINAL BUNDLE ->", FINAL_BUNDLE, flush=True)
print("EXPERIMENT NAME ->", _exp_name, flush=True)

# One-switch run control:
# - fast: quick rerun, prefer reuse
# - resume: continue from checkpoints/caches (default)
# - full: recompute everything heavy
RUN_MODE = os.environ.get("PDM_RUN_MODE", "resume").strip().lower()
if RUN_MODE not in {"fast", "resume", "full"}:
    RUN_MODE = "resume"
print("RUN MODE ->", RUN_MODE, flush=True)

if RUN_MODE == "fast":
    os.environ["PDM_REUSE_RESULTS"] = "1"
    os.environ.setdefault("PDM_FORCE_REFRESH_DATA", "0")
    os.environ.setdefault("PDM_FORCE_CNN_RETRAIN", "0")
    os.environ.setdefault("PDM_FORCE_CTGAN_REFIT", "0")
    os.environ.setdefault("PDM_FORCE_TABULAR_REFIT", "0")
    os.environ.setdefault("PDM_FORCE_SMOTE_MLP_REFIT", "0")
    print(
        "FAST MODE ON: reuse caches/checkpoints (set any PDM_FORCE_*=1 earlier in this session to override).",
        flush=True,
    )

# Extensive experiments: larger CV repeat budget + longer DL training (setdefault only — explicit env wins).
_ext_raw = os.environ.get("PDM_EXTENSIVE_EXPERIMENTS", "0").strip().lower()
PDM_EXTENSIVE_EXPERIMENTS = _ext_raw in {"1", "true", "yes", "on"}
if PDM_EXTENSIVE_EXPERIMENTS:
    os.environ.setdefault("PDM_N_BALANCE_REPS", "25")
    os.environ.setdefault("PDM_EPOCHS_1DCNN", "100")
    os.environ.setdefault("PDM_EPOCHS_LENET", "100")
    os.environ.setdefault("PDM_EPOCHS_CNN", "120")
    os.environ.setdefault("PDM_ES_PATIENCE", "22")
    os.environ.setdefault("PDM_RLRP_PATIENCE", "8")
    print(
        "EXTENSIVE EXPERIMENTS ON: bumped default N_BALANCE_REPS / EPOCHS_* / ES patience "
        "(override anytime by setting those env vars before this cell).",
        flush=True,
    )


class ProgressTracker:
    # Tracks completed (balance_rep x norm x CV fold) for resume after Colab disconnect.

    def __init__(self, path: Path):
        self.path = path
        self.done = {}
        if path.exists():
            try:
                self.done = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.done = {}

    def is_done(self, key: str) -> bool:
        return key in self.done and self.done[key].get("status") == "done"

    def mark_done(self, key: str, payload: dict) -> None:
        self.done[key] = {"status": "done", **payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.done, indent=2), encoding="utf-8")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.done, indent=2), encoding="utf-8")


tracker = ProgressTracker(PROGRESS_PATH)


def save_versions() -> None:
    import importlib.metadata as im

    rows = []
    for pkg in (
        "tensorflow",
        "pandas",
        "numpy",
        "scikit-learn",
        "imbalanced-learn",
        "xgboost",
        "shap",
        "sdv",
        "openai",
    ):
        try:
            rows.append((pkg, im.version(pkg)))
        except Exception:
            rows.append((pkg, "n/a"))
    pd.DataFrame(rows, columns=["package", "version"]).to_csv(
        ARTIFACTS / "paper_bundle" / "versions.csv", index=False
    )


save_versions()
CONFIG_PATH.write_text(
    json.dumps(
        {
            "seed": SEED,
            "experiment_name": _exp_name,
            "artifacts": str(ARTIFACTS),
            "final_bundle": str(FINAL_BUNDLE),
            "run_mode": RUN_MODE,
            "extensive_experiments": PDM_EXTENSIVE_EXPERIMENTS,
            "note": "Edit N_BALANCE_REPS, EPOCHS_LENET, etc. below before long runs.",
        },
        indent=2,
    ),
    encoding="utf-8",
)

# Quick hint if a previous run already wrote tables/figures (reuse with PDM_RUN_MODE=fast).
_tab_dir = ARTIFACTS / "tables"
_fig_dir = ARTIFACTS / "figures"
if _tab_dir.exists():
    _nc = len(list(_tab_dir.glob("*.csv")))
    _nj = len(list(_tab_dir.glob("*.json")))
    _np = len(list(_fig_dir.glob("*.png"))) if _fig_dir.exists() else 0
    if _nc + _nj + _np > 0:
        print(
            f"Found existing artifacts: {_nc} CSV, {_nj} JSON (tables), {_np} PNG (figures). "
            "Use RUN_MODE=fast + same PDM_ARTIFACTS/PDM_EXPERIMENT_NAME to avoid retraining.",
            flush=True,
        )
"""
    )
)

cells.append(
    md(
        r"""
## Step 1b — Load prior results from disk (optional)

If training already finished once and **`ARTIFACTS`** points to that folder (same `PDM_EXPERIMENT_NAME`):

1. Set **`PDM_RUN_MODE=fast`** at the top (or in Colab before imports) — forces **`PDM_REUSE_RESULTS=1`** and clears **`PDM_FORCE_*`** refit flags.
2. Run setup → data → features cells so **`X_b2`**, windows, and **`model_cnn`** exist where later cells need them.
3. Heavy steps skip when caches exist: **CV tracker**, **CNN checkpoint**, **SMOTE+MLP metrics**, **tabular baselines CSV**, **SHAP `.npz`**, **CTGAN `.pkl`**.

To **force** refit only one part: e.g. `PDM_FORCE_TABULAR_REFIT=1`, `PDM_FORCE_SMOTE_MLP_REFIT=1`, `PDM_FORCE_CNN_RETRAIN=1`.

Set **`PDM_PREVIEW_PRIOR_RESULTS=1`** and re-run the next cell to print **heads** of saved CSVs/JSON (no training).
"""
    )
)

cells.append(
    code(
        r"""
def load_artifact_tables_preview(root: Path | None = None, max_rows: int = 6) -> dict:
    root = root or ARTIFACTS
    priority = [
        "best_models_summary.csv",
        "ileri_2024_comparison_summary.csv",
        "1dcnn_cv_runs.csv",
        "shallow_cv_runs.csv",
        "cnn_lstm_metrics_flat.csv",
        "master_results_summary.csv",
        "baselines_last_timestep.csv",
        "mlp_smote_metrics.csv",
        "side_by_side_ab_metrics.csv",
    ]
    out = {}
    for name in priority:
        p = root / "tables" / name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            out[name] = df.head(max_rows)
        except Exception as e:
            out[name] = f"(read error: {e})"
    return out


def load_artifact_json_preview(root: Path | None = None) -> dict:
    root = root or ARTIFACTS
    keys = [
        "best_models_recommendation.json",
        "cnn_lstm_metrics.json",
        "cnn_lstm_fit_diagnostics.json",
    ]
    out = {}
    for name in keys:
        p = root / "paper_bundle" / name
        if not p.exists():
            continue
        try:
            out[name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            out[name] = {"error": str(e)}
    return out


# Set PDM_PREVIEW_PRIOR_RESULTS=1 to print cached tables after a completed run (no training).
if os.environ.get("PDM_PREVIEW_PRIOR_RESULTS", "0").strip() == "1":
    _prev_tab = load_artifact_tables_preview()
    _prev_js = load_artifact_json_preview()
    print("--- CSV preview (heads) ---", flush=True)
    for k, v in _prev_tab.items():
        print("\n>>", k, flush=True)
        print(v if isinstance(v, str) else v.to_string(), flush=True)
    print("\n--- JSON preview ---", flush=True)
    for k, v in _prev_js.items():
        print("\n>>", k, flush=True)
        print(json.dumps(v, indent=2)[:4000], flush=True)
else:
    print(
        "Tip: after a full run, set PDM_PREVIEW_PRIOR_RESULTS=1 re-run this cell to print saved CSV/JSON heads.",
        flush=True,
    )
"""
    )
)

cells.append(
    md(
        r"""
## Step 0 — Optional: download pretrained checkpoints (Colab)

Set **`PDM_REMOTE_MODELS_ZIP_URL`** to an HTTPS link (your Drive “direct download”, GitHub Release asset, or lab server).  
The notebook extracts into **`ARTIFACTS`** so the next cells **load** weights instead of training from scratch (still set `PDM_REUSE_RESULTS=1`).

Skip this step if you train entirely inside this runtime.
"""
    )
)

cells.append(
    code(
        r"""
import shutil
import urllib.request
import zipfile

REMOTE_ZIP = os.environ.get("PDM_REMOTE_MODELS_ZIP_URL", "").strip()
REMOTE_CV = os.environ.get("PDM_REMOTE_CV_PROGRESS_URL", "").strip()

if REMOTE_ZIP:
    dest_zip = ARTIFACTS / "tables" / "_downloaded_checkpoints.zip"
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading checkpoints zip ->", dest_zip, flush=True)
    urllib.request.urlretrieve(REMOTE_ZIP, dest_zip)
    staging = ARTIFACTS / "_remote_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(staging)
    # Merge models/, checkpoints/, tables/ relative to zip root
    for sub in ("models", "checkpoints", "tables"):
        src = staging / sub
        if src.exists():
            dst = ARTIFACTS / sub
            dst.mkdir(parents=True, exist_ok=True)
            for p in src.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(src)
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
            print("  merged folder:", sub, flush=True)
    # Flat zip (files at root): copy *.keras / *.h5 / *.pkl into sensible folders
    for p in staging.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".keras":
            shutil.copy2(p, ARTIFACTS / "models" / p.name)
            print("  copied", p.name, "-> models/", flush=True)
        elif suf in {".h5", ".weights"} or p.name.endswith(".weights.h5"):
            shutil.copy2(p, ARTIFACTS / "checkpoints" / p.name)
            print("  copied", p.name, "-> checkpoints/", flush=True)
        elif suf == ".pkl":
            shutil.copy2(p, ARTIFACTS / "models" / p.name)
            print("  copied", p.name, "-> models/", flush=True)
    shutil.rmtree(staging, ignore_errors=True)
    print("Remote zip merged under ARTIFACTS.", flush=True)
else:
    print("PDM_REMOTE_MODELS_ZIP_URL not set — skip remote checkpoint download.", flush=True)

if REMOTE_CV:
    print("Downloading cv_progress.json ->", PROGRESS_PATH, flush=True)
    urllib.request.urlretrieve(REMOTE_CV, PROGRESS_PATH)
    tracker = ProgressTracker(PROGRESS_PATH)
    print("Reloaded tracker keys:", len(tracker.done), flush=True)
"""
    )
)

cells.append(
    md(
        r"""
## Step 1 — Data ingest, cleaning, and preprocessing (B1 / B2)

1. **Ingest:** `ucimlrepo` and/or full **`ai4i2020.csv`** from UCI (HTTPS).  
2. **Cleaning:** drop inconsistent **Machine failure** vs fault-flag rows (Ileri et al.).  
3. **Preprocessing:** build **B1** (repro / leakage-prone) vs **B2** (contribution: **no** `TWF`…`RNF`, + engineered scalars + **Type** one-hot).
"""
    )
)

cells.append(
    code(
        r"""
from ucimlrepo import fetch_ucirepo

FAULT_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# --- Dataset load (Colab-safe): official UCI HTTPS. Primary = ucimlrepo id=601.
# Fallback examples (uncomment one if fetch_ucirepo fails):
# !wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv -O ai4i2020.csv
# ... then set: os.environ["PDM_AI4I_CSV"] = "ai4i2020.csv" before this cell, or pass path below.

AI4I_FULL_CSV = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

_CSV_PATH = os.environ.get("PDM_AI4I_CSV", "").strip()
REUSE_RESULTS = os.environ.get("PDM_REUSE_RESULTS", "1") == "1"
FORCE_REFRESH_DATA = os.environ.get("PDM_FORCE_REFRESH_DATA", "0") == "1"

# Apply mode presets unless explicitly overridden by env
if RUN_MODE == "full":
    REUSE_RESULTS = False
    FORCE_REFRESH_DATA = True
elif RUN_MODE in {"fast", "resume"}:
    REUSE_RESULTS = True
    if "PDM_FORCE_REFRESH_DATA" not in os.environ:
        FORCE_REFRESH_DATA = False

_AUTO_CSV = ARTIFACTS / "tables" / "ai4i2020_cached.csv"


def clean_ai4i_basepaper(frame: pd.DataFrame) -> pd.DataFrame:
    # Remove inconsistent MF / fault-flag rows (Ileri et al. narrative).
    d = frame.copy()
    mf = d["Machine failure"].astype(int)
    faults_present = d[FAULT_COLS].astype(int).sum(axis=1) >= 1
    bad_rnf = (d["RNF"].astype(int) == 1) & (mf == 0)
    bad_mf = (mf == 1) & (~faults_present)
    removed = int((bad_rnf | bad_mf).sum())
    out = d[~(bad_rnf | bad_mf)].reset_index(drop=True)
    print("Dropped inconsistent rows:", removed, "Remaining:", out.shape[0])
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Harmonize common UCI column spellings.
    m = {}
    for c in df.columns:
        lc = c.strip()
        m[c] = lc
    out = df.rename(columns=m)
    return out


# --- Load AI4I data (priority: explicit CSV -> cached CSV -> ucimlrepo -> full CSV fallback)
if _CSV_PATH:
    _df = pd.read_csv(_CSV_PATH)
    print("Loaded CSV (PDM_AI4I_CSV):", _CSV_PATH, _df.shape)
elif REUSE_RESULTS and _AUTO_CSV.exists() and (not FORCE_REFRESH_DATA):
    _df = pd.read_csv(_AUTO_CSV)
    print("Loaded cached CSV:", _AUTO_CSV, _df.shape)
else:
    _ds = fetch_ucirepo(id=601)
    X_raw = _ds.data.features.copy()
    y_raw = _ds.data.targets.copy()
    _target_candidates = [
        "Machine failure",
        "machine failure",
        "Machine Failure",
    ]
    target_col = next(c for c in _target_candidates if c in y_raw.columns)
    _df = pd.concat([X_raw, y_raw[target_col]], axis=1)
    _df.rename(columns={target_col: "Machine failure"}, inplace=True)
    print("Loaded via ucimlrepo(id=601)", _df.shape)

_df = normalize_columns(_df)

# ucimlrepo often splits X/y so fault-mode columns (TWF…RNF) are missing — cleaning needs them.
_missing_fault = [c for c in FAULT_COLS if c not in _df.columns]
if _missing_fault:
    print(
        "Reloading full AI4I CSV (official UCI mirror) — ucimlrepo merge lacked:",
        _missing_fault,
    )
    _df = pd.read_csv(AI4I_FULL_CSV)
    _df = normalize_columns(_df)

_still = [c for c in FAULT_COLS if c not in _df.columns]
if _still:
    raise ValueError(
        "AI4I dataframe still missing fault-mode columns after CSV load: "
        + str(_still)
        + ". Check network or set PDM_AI4I_CSV to a local ai4i2020.csv path."
    )

# Cache full dataframe for no-reupload reruns
try:
    _AUTO_CSV.parent.mkdir(parents=True, exist_ok=True)
    _df.to_csv(_AUTO_CSV, index=False)
    print("Cached dataframe ->", _AUTO_CSV)
except Exception as _cache_e:
    print("CSV cache skipped:", _cache_e)

# Ensure fault columns exist as int
for c in FAULT_COLS:
    if c in _df.columns:
        _df[c] = _df[c].astype(int)

_df["Machine failure"] = _df["Machine failure"].astype(int)

df_clean = clean_ai4i_basepaper(_df)
print(df_clean.shape, "Failures:", int(df_clean["Machine failure"].sum()))


def build_features_b1(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    # Paper proximity track; retains fault-mode columns (potential leakage).
    dd = d.copy()
    drop_ids = [c for c in ["UID", "UDI", "Product ID"] if c in dd.columns]
    dd = dd.drop(columns=drop_ids, errors="ignore")
    y = dd["Machine failure"].astype(int).values
    dd = dd.drop(columns=["Machine failure"])
    cat_cols = [c for c in dd.columns if dd[c].dtype == object]
    X_num = dd.drop(columns=cat_cols, errors="ignore")
    if cat_cols:
        X_cat = pd.get_dummies(dd[cat_cols], drop_first=False)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = X_num.reset_index(drop=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.astype(np.float32), y


def build_features_b2(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    # Leakage-safe contribution track + rich process-informed scalars (no TWF…RNF).
    # Base paper (Ileri et al.) uses essentially raw inputs + scaling/balancing; here we add
    # 17 engineered fields for stress/load/fault-risk proxies without fault-type flags.
    dd = d.copy()
    dd = dd.drop(columns=[c for c in ["UID", "UDI", "Product ID"] if c in dd.columns], errors="ignore")
    dd = dd.drop(columns=[c for c in FAULT_COLS if c in dd.columns], errors="ignore")
    y = dd["Machine failure"].astype(int).values
    dd = dd.drop(columns=["Machine failure"])

    torque_col = next((c for c in dd.columns if "Torque" in c), None)
    rpm_col = next((c for c in dd.columns if "Rotational speed" in c or "speed" in c.lower()), None)
    air_col = next((c for c in dd.columns if "air temperature" in c.lower()), None)
    proc_col = next((c for c in dd.columns if "process temperature" in c.lower()), None)
    wear_col = next((c for c in dd.columns if "tool wear" in c.lower()), None)

    dd = dd.copy()
    t_num = pd.to_numeric(dd[torque_col], errors="coerce") if torque_col else None
    r_num = pd.to_numeric(dd[rpm_col], errors="coerce") if rpm_col else None
    a = pd.to_numeric(dd[air_col], errors="coerce") if air_col else None
    p = pd.to_numeric(dd[proc_col], errors="coerce") if proc_col else None
    w = pd.to_numeric(dd[wear_col], errors="coerce") if wear_col else None

    if t_num is not None and r_num is not None:
        power = t_num * r_num
        dd["Power_proxy"] = power
        dd["RPM_over_torque"] = r_num / t_num.clip(lower=1e-6)
        dd["Log1p_torque"] = np.log1p(np.maximum(t_num.fillna(0.0), 0.0))
        dd["Log1p_rpm"] = np.log1p(np.maximum(r_num.fillna(0.0), 0.0))
        dd["Log_power_proxy"] = np.log1p(np.maximum(power.fillna(0.0), 0.0))

    if w is not None:
        w_raw = w
        mx = float(np.nanmax(w_raw.to_numpy())) if w_raw.notna().any() else 1.0
        if not np.isfinite(mx) or mx <= 0:
            mx = 1.0
        w_safe = w_raw.fillna(0.0)
        dd["Wear_norm"] = w_safe / max(mx, 1e-6)
        dd["Log1p_wear"] = np.log1p(np.maximum(w_safe, 0.0))
        if t_num is not None and r_num is not None:
            dd["Torque_x_wear"] = t_num.fillna(0.0) * w_safe
            dd["Rpm_x_wear"] = r_num.fillna(0.0) * w_safe
            dd["Power_per_wear"] = (t_num * r_num) / (np.maximum(w_safe, 0.0) + 1.0)

    if a is not None and p is not None:
        aa = a.clip(lower=1.0)
        dtemp = p - a
        dd["Delta_temp_K"] = dtemp
        dd["Thermal_ratio"] = dtemp / aa
        dd["Temp_product_K2"] = a * p
        dd["Inv_air_temp"] = 1.0 / aa
        dd["Delta_temp_sq"] = dtemp ** 2
        dd["Process_over_air"] = p / aa
        if w is not None:
            w_d = w.fillna(0.0).clip(lower=0.0)
            dd["Thermal_per_wear"] = dtemp / (w_d + 1.0)

    cat_cols = [c for c in dd.columns if dd[c].dtype == object]
    X_num = dd.drop(columns=cat_cols, errors="ignore")
    if cat_cols:
        X_cat = pd.get_dummies(dd[cat_cols], drop_first=False)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = X_num.reset_index(drop=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.astype(np.float32), y


X_b1, y_b1 = build_features_b1(df_clean)
X_b2, y_b2 = build_features_b2(df_clean)
print("B1 shape:", X_b1.shape, "B2 shape:", X_b2.shape)
"""
    )
)

cells.append(
    code(
        r"""
# --- EDA (saved high-DPI)

plt.figure(figsize=(6, 4))
df_clean["Machine failure"].value_counts().rename({0: "No failure", 1: "Failure"}).plot(
    kind="bar", color=["#4caf50", "#f44336"], title="Machine failure class distribution (cleaned)"
)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(ARTIFACTS / "figures" / "class_distribution.png", dpi=300)
plt.show()

num_df = df_clean.select_dtypes(include=[np.number]).drop(columns=["Machine failure"], errors="ignore")
plt.figure(figsize=(10, 8))
sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
plt.title("Numeric feature correlation")
plt.tight_layout()
plt.savefig(ARTIFACTS / "figures" / "correlation_heatmap.png", dpi=300)
plt.show()
"""
    )
)

cells.append(
    md(
        r"""
## Step 3 — Training: shallow ML + 1D-CNN zoo (balanced CV + fine-tuning)

**Fine-tuning / preprocessing here:** each CV fold fits its own **MinMax** or **Standard** scaler on training rows only; 1D-CNN weights are saved per fold/architecture for resume.

**Included in this notebook**

| Group | Models | Notes |
|-------|--------|--------|
| **Shallow (Ileri et al.)** | Decision Tree, **SVM (RBF)**, **k-NN** | Same balanced MF subset, **5-fold CV**, `StandardScaler` before SVM/kNN |
| **Deep 1D-CNN (Ileri et al.)** | **1D-LeNet**, **1D-AlexNet** (adapted), **VGG-style mini** | Row-wise `(n_features, 1)`; configurable via `PDM_1DCNN_MODELS` |
| **Proposal** | **CNN–LSTM** on sliding windows, **MLP + SMOTE**, optional **CTGAN**, **LR / RF / XGB** | B2 leakage-safe features for contribution track |
| **XAI / NL** | **SHAP GradientExplainer**, **LLM** report cell (`OPENAI_API_KEY` optional) | |

**Environment:** `PDM_1DCNN_MODELS=lenet,alexnet,vggmini` (comma-separated). Default in code is **all three** — set to `lenet` only on slow runtimes.

Checkpoints: `checkpoints/{arch}_{PIPELINE}_rep{r}_{norm}_fold{f}.weights.h5`

Resume: same `cv_progress.json`; keys distinguish arch + shallow model name.

---

### How this step relates to Ileri et al. (*Appl. Sci.* **2024**, DOI [10.3390/app14114899](https://doi.org/10.3390/app14114899))

| Aspect | Paper (headline Table 4: LeNet, MF, balanced + norm) | This notebook |
|--------|-----------------------------------------------------|---------------|
| **Cleaning** | Drops inconsistent RNF/MF rows (~9973 rows) | Same rule |
| **Balanced MF subset** | **330** positives + **402** negatives (Table 3) | Same counts (`N_NEG_BALANCE=402`, all failure indices) |
| **Normalization** | Min–max **or** z-score; Table 4 keeps **max** of the two per setting | Both **`minmax`** and **`standard`** (z-score) branches; each fold fits scaler on train only |
| **Deep models** | 1D-LeNet / AlexNet / **VGG16** | LeNet / AlexNet / **VGG-mini** (lighter block — cite as “VGG-style”, not identical depth) |
| **CV** | **5-fold** | **5-fold** stratified |
| **Random balance repeats** | **100** draws; table reports **maximum** CV score across repeats (optimistic) | Default **`PDM_N_BALANCE_REPS=5`**; we summarize **mean ± std** and **max over reps** in `tables/ileri_2024_comparison_summary.csv` |
| **Inputs** | Sensor + Type features for MF (fault-type columns are labels in dataset; paper focuses MF / per-fault tasks) | **`PDM_PIPELINE=B2`** = leakage-safe sensors + engineered features + Type one-hot (preferred for honest MF prediction). **`B1`** widens inputs — only for repro experiments |
| **Proposal models** | Not in paper | CNN–LSTM on synthetic windows, SMOTE/CTGAN, etc. — report on **natural split** or tuned threshold; **do not** compare those numbers directly to paper Table 4 without a protocol paragraph |

**How to improve / reposition results**

1. **Match their optimism only when intentional:** run **`PDM_N_BALANCE_REPS=100`** and compare your **max-over-reps** column to their **98.50% / 98.32%** headline.
2. **Report mean ± std across reps** for scientific honesty (our CSV does both).
3. **Tune 1D-CNN:** increase **`PDM_EPOCHS_1DCNN`**, adjust **`PDM_BATCH_SIZE`**, try **`PDM_PIPELINE=B1`** strictly for “closest feature dimensionality” ablations — not for deployment claims.
4. **Separate paragraphs in the manuscript:** “Reproduction track” (balanced CV, Ileri-style) vs “Contribution track” (B2, windows, imbalance-realistic test metrics, SHAP, calibration).
"""
    )
)

cells.append(
    code(
        r"""
import tensorflow as tf
from tensorflow.keras import layers, Model

PIPELINE = os.environ.get("PDM_PIPELINE", "B2")  # B1 or B2
N_BALANCE_REPS = int(os.environ.get("PDM_N_BALANCE_REPS", "5"))
N_SPLITS = 5
EPOCHS_1DCNN = int(os.environ.get("PDM_EPOCHS_1DCNN", os.environ.get("PDM_EPOCHS_LENET", "60")))
BATCH_SIZE = int(os.environ.get("PDM_BATCH_SIZE", "64"))
N_NEG_BALANCE = int(os.environ.get("PDM_N_NEG_BALANCE", "402"))

_arch_raw = os.environ.get("PDM_1DCNN_MODELS", "lenet,alexnet,vggmini").strip()
ARCHS = [a.strip().lower() for a in _arch_raw.split(",") if a.strip()]
print("1D-CNN architectures:", ARCHS, flush=True)

X_full, y_full = (X_b1, y_b1) if PIPELINE == "B1" else (X_b2, y_b2)
fail_idx = np.where(y_full == 1)[0]
ok_idx = np.where(y_full == 0)[0]
print(
    "PIPELINE",
    PIPELINE,
    "Failures:",
    len(fail_idx),
    "Normals:",
    len(ok_idx),
    flush=True,
)


def build_lenet1d(n_features: int) -> Model:
    inp = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(inp)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(84, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out)


def build_alexnet1d_small(n_features: int) -> Model:
    # Five Conv1D blocks + MaxPool (AlexNet-style), sized for short tabular sequences.
    inp = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(48, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(192, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(192, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(84, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out)


def build_vggmini1d(n_features: int) -> Model:
    # Two double-conv VGG-style blocks + tail (compact).
    def double_conv(inp_layer, filters):
        z = layers.Conv1D(filters, 3, activation="relu", padding="same")(inp_layer)
        z = layers.Conv1D(filters, 3, activation="relu", padding="same")(z)
        return layers.MaxPooling1D(2)(z)

    inp = layers.Input(shape=(n_features, 1))
    x = double_conv(inp, 32)
    x = double_conv(x, 64)
    x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out)


BUILDERS = {
    "lenet": build_lenet1d,
    "alexnet": build_alexnet1d_small,
    "vggmini": build_vggmini1d,
}


def train_1dcnn_fold(
    build_fn,
    arch_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    scaler_kind: str,
    n_features: int,
    ckpt_path: Path,
    epochs: int,
):
    if scaler_kind == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_va = scaler.transform(X_val).astype(np.float32)
    X_tr = X_tr[..., np.newaxis]
    X_va = X_va[..., np.newaxis]

    model = build_fn(n_features)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    if ckpt_path.exists():
        try:
            model.load_weights(str(ckpt_path))
            print("Loaded weights:", ckpt_path.name, flush=True)
        except Exception as e:
            print("Train fresh (weight load failed):", e, flush=True)

    cb_ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    _pat_1d = int(os.environ.get("PDM_ES_PATIENCE_1DCNN", "12"))
    _md_1d = float(os.environ.get("PDM_ES_MIN_DELTA_1DCNN", "1e-5"))
    cb_es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=_pat_1d,
        min_delta=_md_1d,
        restore_best_weights=True,
        verbose=1,
    )
    cb_rlr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=float(os.environ.get("PDM_RLRP_FACTOR_1DCNN", "0.5")),
        patience=max(2, _pat_1d // 3),
        min_lr=float(os.environ.get("PDM_MIN_LR_1DCNN", "1e-7")),
        verbose=1,
    )

    model.fit(
        X_tr,
        y_train,
        validation_data=(X_va, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[cb_ckpt, cb_rlr, cb_es],
    )

    probs = model.predict(X_va, verbose=0).ravel()
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "architecture": arch_name,
        "accuracy": float(accuracy_score(y_val, pred)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, probs)),
        "pr_auc": float(average_precision_score(y_val, probs)),
    }
    cm = confusion_matrix(y_val, pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp + 1e-12))
    else:
        metrics["specificity"] = float("nan")
    print(
        f"      -> {arch_name} ({scaler_kind}) val_f1={metrics['f1']:.4f} "
        f"val_acc={metrics['accuracy']:.4f} roc_auc={metrics['roc_auc']:.4f}",
        flush=True,
    )
    return metrics


# ---------- Shallow ML (one balanced draw, 5-fold), resume-aware ----------
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(SEED)
neg_pick = rng.choice(ok_idx, size=min(N_NEG_BALANCE, len(ok_idx)), replace=False)
idx_bal = np.concatenate([fail_idx, neg_pick])
rng.shuffle(idx_bal)
X_bal_s = X_full.iloc[idx_bal].reset_index(drop=True)
y_bal_s = y_full[idx_bal]

skf_shallow = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
shallow_rows = []

for fold, (tr, va) in enumerate(skf_shallow.split(X_bal_s, y_bal_s)):
    Xtr, Xva = X_bal_s.iloc[tr].values, X_bal_s.iloc[va].values
    ytr, yva = y_bal_s[tr], y_bal_s[va]
    scaler_s = StandardScaler()
    Xtrs = scaler_s.fit_transform(Xtr)
    Xvas = scaler_s.transform(Xva)

    clfs = [
        ("dt", DecisionTreeClassifier(random_state=SEED)),
        ("svc_rbf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
        ("knn_k5", KNeighborsClassifier(n_neighbors=5)),
    ]

    for sname, clf in clfs:
        key = f"shallow|{PIPELINE}|{sname}|fold{fold}"
        if tracker.is_done(key):
            print("SKIP", key, flush=True)
            shallow_rows.append({k: v for k, v in tracker.done[key].items() if k != "status"})
            continue

        print(f"Shallow fit | {sname} | fold {fold + 1}/{N_SPLITS}", flush=True)
        clf.fit(Xtrs, ytr)
        pr = clf.predict_proba(Xvas)[:, 1]
        pred = (pr >= 0.5).astype(int)
        m = {
            "key": key,
            "model": sname,
            "family": "shallow",
            "accuracy": float(accuracy_score(yva, pred)),
            "precision": float(precision_score(yva, pred, zero_division=0)),
            "recall": float(recall_score(yva, pred, zero_division=0)),
            "f1": float(f1_score(yva, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(yva, pr)),
            "pr_auc": float(average_precision_score(yva, pr)),
        }
        cm = confusion_matrix(yva, pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            m["specificity"] = float(tn / (tn + fp + 1e-12))
        else:
            m["specificity"] = float("nan")
        tracker.mark_done(key, m)
        shallow_rows.append(m)

pd.DataFrame(shallow_rows).to_csv(ARTIFACTS / "tables" / "shallow_cv_runs.csv", index=False)
tracker.save()
print("Shallow CV saved:", ARTIFACTS / "tables" / "shallow_cv_runs.csv", flush=True)


# ---------- 1D-CNN family (balanced reps × norm × fold) ----------
rows = []
rng = np.random.default_rng(SEED)

for arch in ARCHS:
    if arch not in BUILDERS:
        print("Unknown architecture, skip:", arch, flush=True)
        continue
    builder = BUILDERS[arch]
    print("\n" + "=" * 72, flush=True)
    print(f"1D-CNN architecture: {arch}", flush=True)
    print("=" * 72, flush=True)

    for rep in range(N_BALANCE_REPS):
        print(f"\n--- Balance repetition {rep + 1} / {N_BALANCE_REPS} ---", flush=True)
        neg_pick = rng.choice(ok_idx, size=min(N_NEG_BALANCE, len(ok_idx)), replace=False)
        idx = np.concatenate([fail_idx, neg_pick])
        rng.shuffle(idx)
        X_bal = X_full.iloc[idx].reset_index(drop=True)
        y_bal = y_full[idx]

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)

        for norm_kind in ("minmax", "standard"):
            print(f"  Scaler: {norm_kind}", flush=True)
            for fold, (tr, va) in enumerate(skf.split(X_bal, y_bal)):
                key = f"{arch}|{PIPELINE}|rep{rep}|{norm_kind}|fold{fold}"
                if tracker.is_done(key):
                    print("SKIP", key, flush=True)
                    rows.append({k: v for k, v in tracker.done[key].items() if k != "status"})
                    continue

                print(
                    f"    Training | fold {fold + 1}/{N_SPLITS} | train={len(tr)} val={len(va)}",
                    flush=True,
                )
                ckpt = (
                    ARTIFACTS
                    / "checkpoints"
                    / f"{arch}_{PIPELINE}_rep{rep}_{norm_kind}_fold{fold}.weights.h5"
                )
                m = train_1dcnn_fold(
                    builder,
                    arch,
                    X_bal.iloc[tr].values,
                    y_bal[tr],
                    X_bal.iloc[va].values,
                    y_bal[va],
                    norm_kind,
                    X_bal.shape[1],
                    ckpt,
                    EPOCHS_1DCNN,
                )
                row = {"key": key, "family": "1dcnn", **m}
                tracker.mark_done(key, row)
                rows.append(row)

tracker.save()

df_1dcnn_runs = pd.DataFrame(rows)
df_1dcnn_runs.to_csv(ARTIFACTS / "tables" / "1dcnn_cv_runs.csv", index=False)
if len(df_1dcnn_runs):
    _cv_mean = df_1dcnn_runs.groupby("architecture")[["f1", "accuracy", "recall"]].mean()
    _show_df("1D-CNN CV mean by architecture (f1, accuracy, recall):", _cv_mean)

print("Saved:", ARTIFACTS / "tables" / "1dcnn_cv_runs.csv", flush=True)

# Legacy alias for downstream cells expecting lenet-only CSV
df_lenet_runs = df_1dcnn_runs[df_1dcnn_runs["architecture"] == "lenet"].copy()
df_lenet_runs.to_csv(ARTIFACTS / "tables" / "lenet_cv_runs.csv", index=False)

# --- Summary vs Ileri et al. (2024): balanced 330+402, 5-fold, max(minmax, z-score) per rep ---
def _parse_balance_rep_norm(key: str):
    parts = str(key).split("|")
    rep_id = int(parts[2][3:]) if len(parts) > 2 and parts[2].startswith("rep") else -1
    norm_kind = parts[3] if len(parts) > 3 else ""
    return rep_id, norm_kind


_ileri_rows = []
_ileri_rows.append(
    {
        "row_kind": "literature_reference",
        "citation": "Ileri et al. Appl. Sci. 2024 Table 4 (1D-LeNet, MF, balanced + normalized)",
        "protocol_note": "max over min-max vs z-score; max over 100 random balanced subsets",
        "f1_pct": 98.32,
        "accuracy_pct": 98.50,
        "our_f1_mean_pct": None,
        "our_f1_std_pct": None,
        "our_f1_max_pct": None,
        "our_acc_mean_pct": None,
        "our_acc_std_pct": None,
        "our_acc_max_pct": None,
        "n_balance_reps_configured": int(N_BALANCE_REPS),
        "pipeline": str(PIPELINE),
    }
)

if len(df_1dcnn_runs):
    _tmp = df_1dcnn_runs.copy()
    _tmp[["bal_rep", "norm"]] = _tmp["key"].apply(lambda k: pd.Series(_parse_balance_rep_norm(k)))
    _fold_means = (
        _tmp.groupby(["architecture", "bal_rep", "norm"], as_index=False)[["f1", "accuracy"]]
        .mean()
        .rename(columns={"f1": "mean_f1_fold", "accuracy": "mean_acc_fold"})
    )
    _rep_best = _fold_means.groupby(["architecture", "bal_rep"], as_index=False).agg(
        f1_rep=("mean_f1_fold", "max"),
        acc_rep=("mean_acc_fold", "max"),
    )
    _agg = _rep_best.groupby("architecture", as_index=False).agg(
        f1_mean=("f1_rep", "mean"),
        f1_std=("f1_rep", "std"),
        f1_max=("f1_rep", "max"),
        acc_mean=("acc_rep", "mean"),
        acc_std=("acc_rep", "std"),
        acc_max=("acc_rep", "max"),
    )
    for _, r in _agg.iterrows():
        _ileri_rows.append(
            {
                "row_kind": "this_notebook_aggregate",
                "citation": "",
                "protocol_note": "Per rep: mean val metric over 5 folds; then max(minmax, standard). Across reps: mean/std/max.",
                "f1_pct": None,
                "accuracy_pct": None,
                "our_f1_mean_pct": round(float(r["f1_mean"]) * 100.0, 4),
                "our_f1_std_pct": round(float(r["f1_std"]) * 100.0, 4) if pd.notna(r["f1_std"]) else None,
                "our_f1_max_pct": round(float(r["f1_max"]) * 100.0, 4),
                "our_acc_mean_pct": round(float(r["acc_mean"]) * 100.0, 4),
                "our_acc_std_pct": round(float(r["acc_std"]) * 100.0, 4) if pd.notna(r["acc_std"]) else None,
                "our_acc_max_pct": round(float(r["acc_max"]) * 100.0, 4),
                "n_balance_reps_configured": int(N_BALANCE_REPS),
                "pipeline": str(PIPELINE),
                "architecture": r["architecture"],
            }
        )

df_ileri_cmp = pd.DataFrame(_ileri_rows)
df_ileri_cmp.to_csv(ARTIFACTS / "tables" / "ileri_2024_comparison_summary.csv", index=False)
print("Wrote protocol comparison:", ARTIFACTS / "tables" / "ileri_2024_comparison_summary.csv", flush=True)
"""
    )
)

cells.append(
    md(
        r"""
## Step 4 — Training: CNN–LSTM proposal (windows + imbalance-aware loss)

**Fine-tuning:** `ModelCheckpoint` + **EarlyStopping** (monitor **`val_auc` by default**, or `val_loss` via env) + **ReduceLROnPlateau**; optional **focal loss** / **class weights** via env vars. Live **`[fit-check]`** lines summarize train–val gap + rolling averages (overfit / underfit hints).

Trains on **B2** features by default. **Class weights** follow `PDM_USE_CLASS_WEIGHT` / loss choice (default: off under focal, on under BCE).

Checkpoint: `ARTIFACTS/models/cnn_lstm_best.keras`
"""
    )
)

cells.append(
    code(
        r"""
WINDOW = 10
STRIDE = 1


def make_windows(X: pd.DataFrame, y: np.ndarray, window: int, stride: int):
    Xv = X.values.astype(np.float32)
    ys = []
    Xs = []
    for i in range(0, len(Xv) - window + 1, stride):
        Xs.append(Xv[i : i + window])
        ys.append(y[i + window - 1])
    return np.stack(Xs), np.array(ys)


Xw, yw = make_windows(X_b2, y_b2, WINDOW, STRIDE)
print("Window tensor:", Xw.shape, "Labels:", yw.shape, flush=True)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    Xw, yw, test_size=0.2, stratify=yw, random_state=SEED
)

# Dedicated validation split for threshold tuning (avoids post-hoc thresholding on test set)
X_train, X_val_tune, y_train, y_val_tune = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.15,
    stratify=y_train_full,
    random_state=SEED,
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
cw = {0: 1.0, 1: float(neg / max(pos, 1))}
print("Train class weights (~):", cw, "Train:", X_train.shape, "Val:", X_val_tune.shape, "Test:", X_test.shape, flush=True)

REUSE_RESULTS = os.environ.get("PDM_REUSE_RESULTS", "1") == "1"
FORCE_CNN_RETRAIN = os.environ.get("PDM_FORCE_CNN_RETRAIN", "0") == "1"

if RUN_MODE == "full":
    REUSE_RESULTS = False
    FORCE_CNN_RETRAIN = True
elif RUN_MODE in {"fast", "resume"}:
    REUSE_RESULTS = True
    if "PDM_FORCE_CNN_RETRAIN" not in os.environ:
        FORCE_CNN_RETRAIN = False


def make_focal_loss(gamma: float, alpha: float):
    def focal(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        mod = tf.pow(1.0 - p_t, gamma)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * mod * ce)

    return focal


def recall_at_max_fpr(y_true, y_score, max_fpr: float):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = fpr <= max_fpr + 1e-12
    if not np.any(ok):
        return float("nan"), float("nan"), float("nan")
    idx = np.where(ok)[0]
    j = idx[int(np.argmax(tpr[idx]))]
    return float(tpr[j]), float(fpr[j]), float(thr[j])


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        ece += (cnt / max(n, 1)) * abs(acc - conf)
    return float(ece)


def mc_dropout_predict(model, X_np: np.ndarray, n_samples: int, batch_size: int = 512):
    n = len(X_np)
    stacks = []
    for _ in range(n_samples):
        parts = []
        for start in range(0, n, batch_size):
            xb = X_np[start : start + batch_size].astype(np.float32)
            pb = model(xb, training=True).numpy().ravel()
            parts.append(pb)
        stacks.append(np.concatenate(parts))
    stacked = np.stack(stacks, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def build_cnn_lstm(window: int, n_feat: int) -> Model:
    inp = layers.Input(shape=(window, n_feat))
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(inp)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out)


cnn_ckpt = ARTIFACTS / "models" / "cnn_lstm_best.keras"
cnn_hist_csv = ARTIFACTS / "tables" / "cnn_lstm_history.csv"

model_cnn = build_cnn_lstm(WINDOW, Xw.shape[2])

LOSS_KIND = os.environ.get("PDM_CNN_LOSS", "focal").strip().lower()
_cw_flag = os.environ.get("PDM_USE_CLASS_WEIGHT", "").strip()
if _cw_flag == "1":
    fit_cw = cw
elif _cw_flag == "0":
    fit_cw = None
else:
    fit_cw = None if LOSS_KIND == "focal" else cw

if LOSS_KIND == "focal":
    gamma = float(os.environ.get("PDM_FOCAL_GAMMA", "2"))
    alpha = float(os.environ.get("PDM_FOCAL_ALPHA", "0.75"))
    cnn_loss = make_focal_loss(gamma, alpha)
    print(
        "CNN-LSTM loss=focal gamma=",
        gamma,
        "alpha=",
        alpha,
        "class_weight=",
        fit_cw,
        flush=True,
    )
else:
    cnn_loss = tf.keras.losses.BinaryCrossentropy()
    print("CNN-LSTM loss=BCE class_weight=", fit_cw, flush=True)

model_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=cnn_loss,
    metrics=[tf.keras.metrics.AUC(name="auc")],
)


class FitDiagnosticsCallback(tf.keras.callbacks.Callback):
    # Rolling train–val gap + AUC to reduce single-epoch noise; hints for over/underfitting.
    def __init__(self, gap_window: int = 3, min_epochs_signal: int = 5):
        super().__init__()
        self.gap_window = max(1, int(gap_window))
        self.min_epochs_signal = int(min_epochs_signal)
        self._gaps: list[float] = []
        self._val_aucs: list[float] = []
        self._val_losses: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        auc = logs.get("auc")
        val_auc = logs.get("val_auc")
        if loss is None or val_loss is None:
            return
        gap = float(val_loss - loss)
        self._gaps.append(gap)
        self._val_losses.append(float(val_loss))
        self._val_aucs.append(float(val_auc) if val_auc is not None else float("nan"))
        tail = self._gaps[-self.gap_window :]
        gap_ma = float(sum(tail) / len(tail))
        val_auc_ma = float("nan")
        if self._val_aucs:
            vv = [v for v in self._val_aucs[-self.gap_window :] if np.isfinite(v)]
            if vv:
                val_auc_ma = float(sum(vv) / len(vv))
        signal = "ok"
        if epoch + 1 >= self.min_epochs_signal:
            if gap_ma > 0.07:
                signal = "watch_overfit"
            elif gap_ma > 0.04 and val_auc_ma < 0.62:
                signal = "watch_overfit_mild"
            elif np.isfinite(val_auc_ma) and val_auc_ma < 0.56 and gap_ma < 0.035:
                signal = "watch_underfit"
            elif len(self._val_losses) >= 5:
                vl = np.asarray(self._val_losses[-5:], dtype=np.float64)
                if float(np.std(vl)) < 1e-5 and gap_ma < 0.02:
                    signal = "watch_val_loss_stalled"
        auc_v = float(auc) if auc is not None else float("nan")
        val_auc_v = float(val_auc) if val_auc is not None else float("nan")
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr = float("nan")
        print(
            f"[fit-check] epoch={epoch + 1} loss={float(loss):.4f} "
            f"val_loss={float(val_loss):.4f} gap={gap:.4f} gap_ma{self.gap_window}={gap_ma:.4f} "
            f"auc={auc_v:.4f} val_auc={val_auc_v:.4f} val_auc_ma{self.gap_window}={val_auc_ma:.4f} "
            f"lr={lr:.2e} status={signal}",
            flush=True,
        )

if REUSE_RESULTS and cnn_ckpt.exists() and (not FORCE_CNN_RETRAIN):
    model_cnn.load_weights(str(cnn_ckpt))
    print("Loaded existing CNN-LSTM checkpoint:", cnn_ckpt, flush=True)
    if cnn_hist_csv.exists():
        history = type("obj", (), {"history": pd.read_csv(cnn_hist_csv).to_dict(orient="list")})()
    else:
        history = type("obj", (), {"history": {"loss": [], "val_loss": []}})()
else:
    _es_mon = os.environ.get("PDM_ES_MONITOR", "val_auc").strip().lower()
    if _es_mon not in {"val_loss", "val_auc"}:
        print("PDM_ES_MONITOR must be val_loss or val_auc; got:", _es_mon, "→ using val_auc", flush=True)
        _es_mon = "val_auc"
    _es_pat = int(os.environ.get("PDM_ES_PATIENCE", "15"))
    if _es_mon == "val_auc":
        _es_md = float(os.environ.get("PDM_ES_MIN_DELTA", "0.002"))
        _es_mode = "max"
    else:
        _es_md = float(os.environ.get("PDM_ES_MIN_DELTA", "1e-4"))
        _es_mode = "min"
    _rlrp_pat = int(os.environ.get("PDM_RLRP_PATIENCE", str(max(3, _es_pat // 3))))
    _rlrp_fact = float(os.environ.get("PDM_RLRP_FACTOR", "0.5"))
    _min_lr = float(os.environ.get("PDM_MIN_LR", "1e-6"))
    print(
        "\n>>> CNN-LSTM: training "
        + str(int(os.environ.get("PDM_EPOCHS_CNN", "80")))
        + " epochs max | EarlyStopping monitor="
        + _es_mon
        + " patience="
        + str(_es_pat)
        + " | ReduceLROnPlateau patience="
        + str(_rlrp_pat)
        + "\n",
        flush=True,
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(cnn_ckpt),
            monitor=_es_mon,
            mode=_es_mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=_es_mon,
            mode=_es_mode,
            patience=_es_pat,
            min_delta=_es_md,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=_es_mon,
            mode=_es_mode,
            factor=_rlrp_fact,
            patience=_rlrp_pat,
            min_lr=_min_lr,
            verbose=1,
        ),
        FitDiagnosticsCallback(
            gap_window=int(os.environ.get("PDM_FITCHECK_GAP_WINDOW", "3")),
            min_epochs_signal=int(os.environ.get("PDM_FITCHECK_MIN_EPOCHS", "5")),
        ),
    ]

    history = model_cnn.fit(
        X_train,
        y_train,
        validation_data=(X_val_tune, y_val_tune),
        epochs=int(os.environ.get("PDM_EPOCHS_CNN", "80")),
        batch_size=64,
        class_weight=fit_cw,
        callbacks=callbacks,
        verbose=1,
    )

    pd.DataFrame(history.history).to_csv(cnn_hist_csv, index=False)

# Automated training diagnosis: underfit / overfit / stable (uses full history).
fit_diag = {"status": "not_available", "note": "history was empty or missing key columns"}
if hasattr(history, "history") and isinstance(history.history, dict):
    _hist_df = pd.DataFrame(history.history)
    if len(_hist_df) and {"loss", "val_loss"}.issubset(_hist_df.columns):
        _best_epoch_loss = int(_hist_df["val_loss"].astype(float).idxmin()) + 1
        _final_loss = float(_hist_df["loss"].iloc[-1])
        _final_val_loss = float(_hist_df["val_loss"].iloc[-1])
        _gap = _final_val_loss - _final_loss
        _gaps_series = (pd.to_numeric(_hist_df["val_loss"], errors="coerce") - pd.to_numeric(_hist_df["loss"], errors="coerce")).astype(float)
        _gap_last3 = float(_gaps_series.iloc[-3:].mean()) if len(_gaps_series) >= 3 else float(_gap)
        _final_auc = float(_hist_df["auc"].iloc[-1]) if "auc" in _hist_df else float("nan")
        _final_val_auc = float(_hist_df["val_auc"].iloc[-1]) if "val_auc" in _hist_df else float("nan")
        _best_val_auc_ep = None
        _max_val_auc = float("nan")
        _val_auc_trend = float("nan")
        if "val_auc" in _hist_df.columns:
            _va = pd.to_numeric(_hist_df["val_auc"], errors="coerce").astype(float)
            _max_val_auc = float(_va.max())
            _best_val_auc_ep = int(_va.idxmax()) + 1
            if len(_va) >= 6:
                _val_auc_trend = float(_va.iloc[-3:].mean() - _va.iloc[:3].mean())
        _stall_val_loss = False
        if len(_hist_df) >= 6:
            _vl_tail = pd.to_numeric(_hist_df["val_loss"].iloc[-5:], errors="coerce").astype(float)
            _stall_val_loss = bool(float(_vl_tail.std()) < 1e-5)
        if np.isfinite(_final_val_auc) and _final_val_auc < 0.56 and _gap_last3 < 0.04:
            _status = "possible_underfitting"
            _recommendation = (
                "Low val_auc with small train–val gap: try BCE+class weights (`PDM_CNN_LOSS=bce`, `PDM_USE_CLASS_WEIGHT=1`), "
                "tune focal alpha/gamma, larger window or stronger tabular baseline; ReduceLR already lowers LR on plateau."
            )
        elif _gap_last3 > 0.10 or (_gap > 0.12 and _best_epoch_loss < len(_hist_df) - 2):
            _status = "possible_overfitting"
            _recommendation = (
                "Val loss ≫ train (rolling gap high): increase dropout, add L2, shorter ES patience, or monitor `val_loss` "
                "(`PDM_ES_MONITOR=val_loss`) if AUC is noisy; EarlyStopping restores best weights."
            )
        elif _stall_val_loss and np.isfinite(_final_val_auc) and _final_val_auc < 0.60:
            _status = "possible_underfitting_plateau"
            _recommendation = (
                "Validation loss nearly flat — model may lack capacity or LR fell too far; try higher initial LR, "
                "fewer RLRP reductions (`PDM_RLRP_PATIENCE`), or `PDM_FORCE_CNN_RETRAIN=1` after hyperparameter change."
            )
        else:
            _status = "no_clear_overfit_underfit"
            _recommendation = "Proceed to test metrics; judge by PR-AUC/recall/F1 on held-out data."
        fit_diag = {
            "status": _status,
            "epochs_run": int(len(_hist_df)),
            "best_val_loss_epoch": _best_epoch_loss,
            "best_val_auc_epoch": _best_val_auc_ep,
            "max_val_auc": _max_val_auc,
            "val_auc_trend_last_vs_first": _val_auc_trend,
            "train_val_loss_gap_final": float(_gap),
            "train_val_loss_gap_ma_last3": float(_gap_last3),
            "val_loss_stalled_tail": bool(_stall_val_loss),
            "final_loss": _final_loss,
            "final_val_loss": _final_val_loss,
            "final_val_minus_train_loss": float(_gap),
            "final_auc": _final_auc,
            "final_val_auc": _final_val_auc,
            "early_stop_monitor": os.environ.get("PDM_ES_MONITOR", "val_auc"),
            "recommendation": _recommendation,
        }
        (ARTIFACTS / "tables" / "cnn_lstm_fit_diagnostics.csv").parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([fit_diag]).to_csv(ARTIFACTS / "tables" / "cnn_lstm_fit_diagnostics.csv", index=False)
        (ARTIFACTS / "paper_bundle" / "cnn_lstm_fit_diagnostics.json").write_text(
            json.dumps(fit_diag, indent=2), encoding="utf-8"
        )
        print("CNN-LSTM fit diagnosis:", fit_diag, flush=True)

# Threshold tuning on validation (F1 + cost-sensitive)
val_probs = model_cnn.predict(X_val_tune, verbose=0).ravel()
thr_grid = np.linspace(0.05, 0.95, 91)
f1_grid = [f1_score(y_val_tune, (val_probs >= t).astype(int), zero_division=0) for t in thr_grid]
best_idx = int(np.argmax(f1_grid))
best_thr = float(thr_grid[best_idx])
print(
    "Best validation threshold (max F1):",
    best_thr,
    "val_f1:",
    float(f1_grid[best_idx]),
    flush=True,
)

FN_COST = float(os.environ.get("PDM_FN_COST", "10"))
FP_COST = float(os.environ.get("PDM_FP_COST", "1"))
costs_val = []
for t in thr_grid:
    pred_v = (val_probs >= t).astype(int)
    fn = int(np.sum((y_val_tune == 1) & (pred_v == 0)))
    fp = int(np.sum((y_val_tune == 0) & (pred_v == 1)))
    costs_val.append(fn * FN_COST + fp * FP_COST)
best_cost_idx = int(np.argmin(costs_val))
best_thr_cost = float(thr_grid[best_cost_idx])
print(
    "Best validation threshold (min cost, FN*",
    FN_COST,
    "+ FP*",
    FP_COST,
    "):",
    best_thr_cost,
    "val_cost:",
    float(costs_val[best_cost_idx]),
    flush=True,
)

probs = model_cnn.predict(X_test, verbose=0).ravel()
pred_05 = (probs >= 0.5).astype(int)
pred_tuned = (probs >= best_thr).astype(int)
pred_cost = (probs >= best_thr_cost).astype(int)

r05, f05, _t05 = recall_at_max_fpr(y_test, probs, 0.05)
r10, f10, _t10 = recall_at_max_fpr(y_test, probs, 0.10)

cnn_metrics = {
    "accuracy@0.5": float(accuracy_score(y_test, pred_05)),
    "precision@0.5": float(precision_score(y_test, pred_05, zero_division=0)),
    "recall@0.5": float(recall_score(y_test, pred_05, zero_division=0)),
    "f1@0.5": float(f1_score(y_test, pred_05, zero_division=0)),
    "accuracy@tuned": float(accuracy_score(y_test, pred_tuned)),
    "precision@tuned": float(precision_score(y_test, pred_tuned, zero_division=0)),
    "recall@tuned": float(recall_score(y_test, pred_tuned, zero_division=0)),
    "f1@tuned": float(f1_score(y_test, pred_tuned, zero_division=0)),
    "best_threshold_f1": best_thr,
    "best_threshold": best_thr,
    "accuracy@cost": float(accuracy_score(y_test, pred_cost)),
    "precision@cost": float(precision_score(y_test, pred_cost, zero_division=0)),
    "recall@cost": float(recall_score(y_test, pred_cost, zero_division=0)),
    "f1@cost": float(f1_score(y_test, pred_cost, zero_division=0)),
    "best_threshold_cost": best_thr_cost,
    "fn_cost": FN_COST,
    "fp_cost": FP_COST,
    "roc_auc": float(roc_auc_score(y_test, probs)),
    "pr_auc": float(average_precision_score(y_test, probs)),
    "brier_score": float(brier_score_loss(y_test, probs)),
    "ece_10bins": compute_ece(y_test, probs, 10),
    "recall_at_fpr_le_0.05": r05,
    "operating_fpr_at_recall_choice_0.05": f05,
    "recall_at_fpr_le_0.10": r10,
    "operating_fpr_at_recall_choice_0.10": f10,
    "cnn_loss": LOSS_KIND,
}

mc_n = int(os.environ.get("PDM_MC_SAMPLES", "20"))
if mc_n > 0:
    print("MC Dropout uncertainty:", mc_n, "forward passes on test set...", flush=True)
    _mc_mean, mc_std = mc_dropout_predict(model_cnn, X_test, mc_n, batch_size=512)
    cnn_metrics["mc_dropout_mean_abs_std"] = float(np.mean(np.abs(mc_std)))
    cnn_metrics["mc_dropout_n"] = mc_n
else:
    cnn_metrics["mc_dropout_mean_abs_std"] = float("nan")
    cnn_metrics["mc_dropout_n"] = 0

print(cnn_metrics, flush=True)


def _json_metric_val(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


cnn_metrics_json = {k: _json_metric_val(v) for k, v in cnn_metrics.items()}

with open(ARTIFACTS / "paper_bundle" / "cnn_lstm_metrics.json", "w", encoding="utf-8") as f:
    json.dump(cnn_metrics_json, f, indent=2)
pd.Series(cnn_metrics).to_csv(ARTIFACTS / "tables" / "cnn_lstm_metrics_flat.csv")

# --- Step 5a: Side-by-side summary (honest comparison: balanced CV vs held-out test)
thr_side = float(best_thr)
prob_side = model_cnn.predict(X_test, verbose=0).ravel()
pred_side = (prob_side >= thr_side).astype(int)

_compare = {}
try:
    _lenet = df_1dcnn_runs[df_1dcnn_runs["architecture"] == "lenet"]
    _compare["F1"] = {
        "A: 1D-LeNet (5-fold CV mean ± std, balanced B2)": f"{_lenet['f1'].mean():.4f} ± {_lenet['f1'].std():.4f}",
        "B: CNN-LSTM (held-out test, F1-tuned thr)": f"{f1_score(y_test, pred_side, zero_division=0):.4f}",
    }
    _compare["ROC-AUC"] = {
        "A: 1D-LeNet (CV mean)": f"{_lenet['roc_auc'].mean():.4f}",
        "B: CNN-LSTM (test)": f"{roc_auc_score(y_test, prob_side):.4f}",
    }
    _compare["PR-AUC"] = {
        "A: 1D-LeNet (CV mean)": f"{_lenet['pr_auc'].mean():.4f}",
        "B: CNN-LSTM (test)": f"{average_precision_score(y_test, prob_side):.4f}",
    }
except Exception as _se:
    print("Side-by-side: need df_1dcnn_runs from Step 3 cell —", _se, flush=True)
    _compare["F1"] = {
        "A: 1D-LeNet": "n/a (run Step 3 first)",
        "B: CNN-LSTM (test, tuned)": f"{f1_score(y_test, pred_side, zero_division=0):.4f}",
    }

df_side_metrics = pd.DataFrame(_compare).T
_show_df(
    "Side-by-side metrics (A vs B — different protocols; describe clearly in your paper):",
    df_side_metrics,
)
df_side_metrics.to_csv(ARTIFACTS / "tables" / "side_by_side_ab_metrics.csv")
print("Saved", ARTIFACTS / "tables" / "side_by_side_ab_metrics.csv", flush=True)
"""
    )
)

cells.append(
    md(
        r"""
## Step 5b — Augmentation ablation (training split only): SMOTE vs CTGAN on flattened windows

**CTGAN** may be slow; it runs inside `train_test_split` train portion only. Saves synthetic evaluation CSV.
"""
    )
)

cells.append(
    code(
        r"""
from imblearn.over_sampling import SMOTE

flat_train = X_train.reshape(X_train.shape[0], -1)
flat_test = X_test.reshape(X_test.shape[0], -1)

REUSE_RESULTS_SM = os.environ.get("PDM_REUSE_RESULTS", "1") == "1"
FORCE_SMOTE_MLP_REFIT = os.environ.get("PDM_FORCE_SMOTE_MLP_REFIT", "0") == "1"
if RUN_MODE == "full":
    REUSE_RESULTS_SM = False
    FORCE_SMOTE_MLP_REFIT = True
elif RUN_MODE in {"fast", "resume"}:
    REUSE_RESULTS_SM = True
    if "PDM_FORCE_SMOTE_MLP_REFIT" not in os.environ:
        FORCE_SMOTE_MLP_REFIT = False

_npz_sm = ARTIFACTS / "tables" / "_smote_flat_train.npz"
_js_sm = ARTIFACTS / "paper_bundle" / "mlp_smote_metrics.json"

if REUSE_RESULTS_SM and (not FORCE_SMOTE_MLP_REFIT) and _js_sm.exists() and _npz_sm.exists():
    smote_metrics = json.loads(_js_sm.read_text(encoding="utf-8"))
    _zsm = np.load(_npz_sm)
    X_rs, y_rs = _zsm["X"], _zsm["y"]
    print("Loaded SMOTE arrays + MLP metrics from disk (skip SMOTE+MLP refit).", flush=True)
else:
    sm = SMOTE(random_state=SEED)
    print("Running SMOTE on flattened train windows...", flush=True)
    X_rs, y_rs = sm.fit_resample(flat_train, y_train)
    print("After SMOTE:", X_rs.shape, np.bincount(y_rs.astype(int)), flush=True)

    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED, verbose=True)
    mlp.fit(X_rs, y_rs)
    p_smote = mlp.predict_proba(flat_test)[:, 1]
    pred_smote = (p_smote >= 0.5).astype(int)

    smote_metrics = {
        "accuracy": float(accuracy_score(y_test, pred_smote)),
        "precision": float(precision_score(y_test, pred_smote, zero_division=0)),
        "recall": float(recall_score(y_test, pred_smote, zero_division=0)),
        "f1": float(f1_score(y_test, pred_smote, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, p_smote)),
        "pr_auc": float(average_precision_score(y_test, p_smote)),
    }
    print("MLP + SMOTE (flattened windows):", smote_metrics, flush=True)

    pd.Series(smote_metrics).to_csv(ARTIFACTS / "tables" / "mlp_smote_metrics.csv")
    _js_sm.parent.mkdir(parents=True, exist_ok=True)
    _js_sm.write_text(json.dumps(smote_metrics, indent=2), encoding="utf-8")
    np.savez_compressed(_npz_sm, X=X_rs, y=y_rs)

# --- CTGAN attempt (optional heavy)
FORCE_CTGAN_REFIT = os.environ.get("PDM_FORCE_CTGAN_REFIT", "0") == "1"
REUSE_CTGAN = REUSE_RESULTS_SM
if RUN_MODE == "full":
    REUSE_CTGAN = False
    FORCE_CTGAN_REFIT = True
elif RUN_MODE in {"fast", "resume"}:
    REUSE_CTGAN = True
    if "PDM_FORCE_CTGAN_REFIT" not in os.environ:
        FORCE_CTGAN_REFIT = False
ctgan_metrics = {"note": "skipped"}
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from scipy.stats import ks_2samp

    # Subsample for demo speed on Colab
    max_rows = int(os.environ.get("PDM_CTGAN_MAX", "4000"))
    idx_sub = np.arange(len(X_rs))
    if len(idx_sub) > max_rows:
        rng = np.random.default_rng(SEED)
        idx_sub = rng.choice(idx_sub, size=max_rows, replace=False)

    df_ct = pd.DataFrame(X_rs[idx_sub])
    df_ct["target"] = y_rs[idx_sub]

    md_meta = SingleTableMetadata()
    md_meta.detect_from_dataframe(df_ct)

    print(
        "\n>>> CTGAN: fitting on "
        + str(len(df_ct))
        + " rows (epochs="
        + os.environ.get("PDM_CTGAN_EPOCHS", "50")
        + ") — SDV progress may take several minutes...\n",
        flush=True,
    )
    synthesizer = CTGANSynthesizer(md_meta, epochs=int(os.environ.get("PDM_CTGAN_EPOCHS", "50")))
    ck_ct = ARTIFACTS / "models" / "ctgan_synthesizer.pkl"
    if REUSE_CTGAN and ck_ct.exists() and (not FORCE_CTGAN_REFIT):
        synthesizer = CTGANSynthesizer.load(str(ck_ct))
        print("Loaded CTGAN:", ck_ct, flush=True)
    else:
        synthesizer.fit(df_ct)
        synthesizer.save(str(ck_ct))
        print("CTGAN fit complete; saved:", ck_ct, flush=True)

    syn = synthesizer.sample(num_rows=min(len(df_ct), 5000))
    print("Synthetic sample:", syn.shape, flush=True)

    # CTGAN fidelity: KS test per feature (real vs synthetic), plus overlay plots
    feat_cols = [c for c in df_ct.columns if c != "target"]
    ks_rows = []
    for c in feat_cols:
        try:
            stat, pval = ks_2samp(df_ct[c].astype(float).values, syn[c].astype(float).values)
            ks_rows.append({"feature": str(c), "ks_stat": float(stat), "ks_pvalue": float(pval)})
        except Exception as _e:
            ks_rows.append({"feature": str(c), "ks_stat": float("nan"), "ks_pvalue": float("nan")})
    df_ks = pd.DataFrame(ks_rows)
    df_ks.to_csv(ARTIFACTS / "tables" / "ctgan_ks.csv", index=False)

    # Save up to 12 overlays to avoid huge plotting cost
    n_plot = min(12, len(feat_cols))
    for i, c in enumerate(feat_cols[:n_plot]):
        if i == 0 or (i + 1) % 4 == 0:
            print(f"CTGAN plots: {i + 1}/{n_plot} features...", flush=True)
        plt.figure(figsize=(5, 3))
        sns.kdeplot(df_ct[c].astype(float), label="real", fill=True)
        sns.kdeplot(syn[c].astype(float), label="synthetic", fill=True)
        plt.title(f"CTGAN overlay feature {c}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "figures" / f"ctgan_overlay_{i:02d}_{c}.png", dpi=300)
        plt.close()

    ctgan_metrics = {
        "status": "fitted",
        "synthetic_rows": int(syn.shape[0]),
        "ks_mean_pvalue": float(df_ks["ks_pvalue"].mean(skipna=True)),
        "ks_min_pvalue": float(df_ks["ks_pvalue"].min(skipna=True)),
        "ks_pass_rate_p_gt_0.05": float((df_ks["ks_pvalue"] > 0.05).mean()),
    }
except Exception as e:
    ctgan_metrics = {"error": str(e)}
    print("CTGAN skipped/failed:", e, flush=True)

with open(ARTIFACTS / "paper_bundle" / "ctgan_status.json", "w", encoding="utf-8") as f:
    json.dump(ctgan_metrics, f, indent=2)
"""
    )
)

cells.append(
    md(
        r"""
## Step 5c — Classical baselines on **last timestep** of each window (sanity check)

Matches approximate input dimensionality with flattened-window models.
"""
    )
)

cells.append(
    code(
        r"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

X_vec_train = X_train[:, -1, :]
X_vec_test = X_test[:, -1, :]

_reuse_tab = os.environ.get("PDM_REUSE_RESULTS", "1") == "1"
_force_tab = os.environ.get("PDM_FORCE_TABULAR_REFIT", "0") == "1"
if RUN_MODE == "full":
    _reuse_tab = False
    _force_tab = True
elif RUN_MODE in {"fast", "resume"}:
    _reuse_tab = True
    if "PDM_FORCE_TABULAR_REFIT" not in os.environ:
        _force_tab = False

_p_bl = ARTIFACTS / "tables" / "baselines_last_timestep.csv"
if _reuse_tab and (not _force_tab) and _p_bl.exists():
    df_bl = pd.read_csv(_p_bl)
    print("Loaded tabular baselines from disk:", _p_bl, flush=True)
    _show_df("Baselines on last timestep (loaded, no refit):", df_bl)
else:
    baselines = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "rf": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
        ),
    }

    try:
        baselines["hgb"] = HistGradientBoostingClassifier(
            max_depth=9,
            max_iter=350,
            learning_rate=0.06,
            random_state=SEED,
            class_weight="balanced",
        )
    except TypeError:
        baselines["hgb"] = HistGradientBoostingClassifier(
            max_depth=9,
            max_iter=350,
            learning_rate=0.06,
            random_state=SEED,
        )

    try:
        import xgboost as xgb

        baselines["xgb"] = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=SEED,
            tree_method="hist",
            eval_metric="logloss",
        )
    except Exception as e:
        print("XGBoost unavailable:", e, flush=True)

    rows_bl = []
    for name, clf in baselines.items():
        print(f"Baseline fit | {name}", flush=True)
        clf.fit(X_vec_train, y_train)
        pr = clf.predict_proba(X_vec_test)[:, 1]
        pred = (pr >= 0.5).astype(int)
        rows_bl.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, pr),
                "pr_auc": average_precision_score(y_test, pr),
            }
        )

    df_bl = pd.DataFrame(rows_bl)
    df_bl.to_csv(ARTIFACTS / "tables" / "baselines_last_timestep.csv", index=False)
    _show_df("Baselines on last timestep (test metrics):", df_bl)
"""
    )
)

cells.append(
    md(
        r"""
## Step 5d — Best baseline vs best contribution (auto pick)

We train **many models on purpose** (zoo + proposals). For writing:

- **Baseline track:** keep **all** shallow + 1D-CNN models in tables; for the sentence that mirrors Ileri et al., cite **LeNet** + `ileri_2024_comparison_summary.csv`. Optionally highlight the **best mean CV F1** architecture as “strongest baseline in our zoo.”
- **Contribution track:** pick **one** winner on **held-out test** with `PDM_BEST_MODEL_METRIC` = `f1` (default), `pr_auc`, or `roc_auc` among CNN–LSTM, MLP+SMOTE, LR/RF/XGB/HGB.

Files: `tables/best_models_summary.csv`, `paper_bundle/best_models_recommendation.json`.
"""
    )
)

cells.append(
    code(
        r"""
_pick = os.environ.get("PDM_BEST_MODEL_METRIC", "f1").strip().lower()
if _pick not in {"f1", "pr_auc", "roc_auc"}:
    _pick = "f1"

_contrib_rows = []
if "cnn_metrics" in globals():
    _cf1 = cnn_metrics.get("f1@tuned", np.nan)
    if not np.isfinite(float(_cf1)):
        _cf1 = cnn_metrics.get("f1@0.5", np.nan)
    _contrib_rows.append(
        {
            "track": "contribution",
            "model": "cnn_lstm",
            "family": "proposal_dl",
            "f1": float(_cf1) if np.isfinite(float(_cf1)) else np.nan,
            "pr_auc": float(cnn_metrics.get("pr_auc", np.nan)),
            "roc_auc": float(cnn_metrics.get("roc_auc", np.nan)),
        }
    )
if "smote_metrics" in globals():
    _sm = smote_metrics
    _contrib_rows.append(
        {
            "track": "contribution",
            "model": "mlp_smote_flattened_windows",
            "family": "proposal_augment",
            "f1": float(_sm.get("f1", np.nan)),
            "pr_auc": float(_sm.get("pr_auc", np.nan)),
            "roc_auc": float(_sm.get("roc_auc", np.nan)),
        }
    )
if Path(ARTIFACTS / "tables" / "baselines_last_timestep.csv").exists():
    for _, r in pd.read_csv(ARTIFACTS / "tables" / "baselines_last_timestep.csv").iterrows():
        _contrib_rows.append(
            {
                "track": "contribution",
                "model": str(r["model"]),
                "family": "proposal_sklearn",
                "f1": float(r.get("f1", np.nan)),
                "pr_auc": float(r.get("pr_auc", np.nan)),
                "roc_auc": float(r.get("roc_auc", np.nan)),
            }
        )

df_contrib_pick = pd.DataFrame(_contrib_rows)
_best_contrib = None
if len(df_contrib_pick):
    _col = _pick if _pick in df_contrib_pick.columns else "f1"
    _idx = df_contrib_pick[_col].astype(float).idxmax()
    _best_contrib = df_contrib_pick.loc[_idx].to_dict()

_base_rows = []
_p_sh = ARTIFACTS / "tables" / "shallow_cv_runs.csv"
if Path(_p_sh).exists():
    _ds = pd.read_csv(_p_sh)
    for mname, g in _ds.groupby("model"):
        _base_rows.append(
            {
                "track": "baseline_cv",
                "model": f"shallow:{mname}",
                "family": "base_shallow",
                "f1_mean": float(g["f1"].mean()),
                "f1_std": float(g["f1"].std()) if len(g) > 1 else 0.0,
                "roc_auc_mean": float(g["roc_auc"].mean()),
                "pr_auc_mean": float(g["pr_auc"].mean()),
            }
        )

if "df_1dcnn_runs" in globals() and len(df_1dcnn_runs):
    for arch, g in df_1dcnn_runs.groupby("architecture"):
        _base_rows.append(
            {
                "track": "baseline_cv",
                "model": f"1dcnn:{arch}",
                "family": "base_1dcnn",
                "f1_mean": float(g["f1"].mean()),
                "f1_std": float(g["f1"].std()) if len(g) > 1 else 0.0,
                "roc_auc_mean": float(g["roc_auc"].mean()),
                "pr_auc_mean": float(g["pr_auc"].mean()),
            }
        )

df_base_pick = pd.DataFrame(_base_rows)
_best_1dcnn = None
_best_shallow = None
if len(df_base_pick):
    _dcnn_only = df_base_pick[df_base_pick["family"] == "base_1dcnn"]
    if len(_dcnn_only):
        _j = _dcnn_only["f1_mean"].astype(float).idxmax()
        _best_1dcnn = _dcnn_only.loc[_j].to_dict()
    _sh_only = df_base_pick[df_base_pick["family"] == "base_shallow"]
    if len(_sh_only):
        _j = _sh_only["f1_mean"].astype(float).idxmax()
        _best_shallow = _sh_only.loc[_j].to_dict()

_primary_bl = os.environ.get("PDM_PRIMARY_BASELINE", "lenet").strip().lower()
_paper_lenet_note = (
    "Compare LeNet headline to Ileri Table 4 via tables/ileri_2024_comparison_summary.csv "
    "(your max-over-reps vs their max-over-100)."
)

rec = {
    "selection_metric_contribution": _pick,
    "manuscript_tip": (
        "Full zoo in tables; abstract highlights paper-aligned LeNet (baseline) "
        "and best contribution model under selection_metric_contribution."
    ),
    "best_contribution_test_split": _best_contrib,
    "best_1dcnn_by_mean_cv_f1": _best_1dcnn,
    "best_shallow_by_mean_cv_f1": _best_shallow,
    "primary_baseline_for_story": _primary_bl,
    "paper_comparison_note": _paper_lenet_note,
}


def _clean_pick(d):
    if not isinstance(d, dict):
        return None
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            try:
                v = v.item()
            except Exception:
                v = float(v) if isinstance(v, (float, np.floating)) else str(v)
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            v = None
        out[str(k)] = v
    return out


rec["best_contribution_test_split"] = _clean_pick(rec["best_contribution_test_split"])
rec["best_1dcnn_by_mean_cv_f1"] = _clean_pick(rec["best_1dcnn_by_mean_cv_f1"])
rec["best_shallow_by_mean_cv_f1"] = _clean_pick(rec["best_shallow_by_mean_cv_f1"])

df_out = pd.concat(
    [
        df_base_pick.assign(kind="baseline_balanced_cv"),
        df_contrib_pick.assign(kind="contribution_held_out_test"),
    ],
    ignore_index=True,
)
_out_csv = ARTIFACTS / "tables" / "best_models_summary.csv"
df_out.to_csv(_out_csv, index=False)
_out_js = ARTIFACTS / "paper_bundle" / "best_models_recommendation.json"
Path(_out_js).parent.mkdir(parents=True, exist_ok=True)
with open(_out_js, "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=2)

print("Best-model summary:", _out_csv, flush=True)
print(json.dumps(rec, indent=2), flush=True)
"""
    )
)

cells.append(
    md(
        r"""
## Step 6 — Testing plots & calibration (CNN–LSTM)

ROC / PR / confusion matrix (**F1-tuned threshold from validation**), training loss, and **reliability (calibration) curve**.
"""
    )
)

cells.append(
    code(
        r"""
probs = model_cnn.predict(X_test, verbose=0).ravel()
_thr_plot = float(globals().get("best_thr", 0.5))
pred = (probs >= _thr_plot).astype(int)

fpr, tpr, _ = roc_curve(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, probs):.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC — CNN-LSTM")
plt.legend()
plt.tight_layout()
plt.savefig(ARTIFACTS / "figures" / "roc_cnn_lstm.png", dpi=300)

plt.figure(figsize=(5, 4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR — CNN-LSTM")
plt.tight_layout()
plt.savefig(ARTIFACTS / "figures" / "pr_cnn_lstm.png", dpi=300)

cm = confusion_matrix(y_test, pred, labels=[0, 1])
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion — CNN-LSTM @ thr={_thr_plot:.3f} (val-F1 tuned)")
plt.tight_layout()
plt.savefig(ARTIFACTS / "figures" / "confusion_cnn_lstm.png", dpi=300)

from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    y_test, probs, n_bins=10, strategy="uniform"
)
plt.figure(figsize=(5, 4))
plt.plot(prob_pred, prob_true, "s-", label="CNN-LSTM")
plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability diagram (calibration) — CNN-LSTM")
plt.legend()
plt.tight_layout()
plt.savefig(ARTIFACTS / "figures" / "calibration_reliability.png", dpi=300)
plt.show()

if hasattr(history, "history") and isinstance(history.history, dict) and ("loss" in history.history):
    _hdf = pd.DataFrame(history.history)
    if all(c in _hdf.columns for c in ["loss", "val_loss"]):
        plt.figure(figsize=(6, 4))
        _hdf[["loss", "val_loss"]].plot()
        plt.title("CNN-LSTM loss")
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "figures" / "loss_cnn_lstm.png", dpi=300)
        plt.show()
"""
    )
)

cells.append(
    md(
        r"""
## Step 7 — Explainability: SHAP GradientExplainer (subset for runtime)

Uses TensorFlow gradient explainer — may be memory-heavy; reduce `MAX_BG` / `MAX_TEST` if needed.
"""
    )
)

cells.append(
    code(
        r"""
import shap

# NOTE: If error points to plt.barh(feat_names, imp) you are running an OLD notebook on Colab — re-upload this file.

MAX_BG = int(os.environ.get("PDM_SHAP_BG", "64"))
MAX_TEST = int(os.environ.get("PDM_SHAP_TEST", "128"))

shap_vals = None
test_explain = X_test[:MAX_TEST].astype(np.float32)
_cache = ARTIFACTS / "tables" / "shap_cache.npz"

# If SHAP/plot errors after a library upgrade, delete cache once and recompute:
# import os; os.remove(str(_cache)) if _cache.exists() else None

try:
    if _cache.exists():
        z = np.load(_cache)
        shap_vals = np.asarray(z["shap"])
        print("Loaded SHAP cache:", _cache)
    else:
        bg = X_train[:MAX_BG].astype(np.float32)
        explainer = shap.GradientExplainer(model_cnn, bg)
        sv = explainer.shap_values(test_explain)
        if isinstance(sv, list):
            shap_vals = np.asarray(sv[0])
        else:
            shap_vals = np.asarray(sv)
        np.savez_compressed(_cache, shap=shap_vals, x=test_explain, y=y_test[:MAX_TEST])
        print("Saved SHAP cache:", _cache)
except Exception as e:
    print("SHAP skipped/failed:", e)

if shap_vals is not None:
    shap_arr = np.asarray(shap_vals)
    if shap_arr.dtype == object:
        shap_arr = np.asarray(shap_arr.tolist(), dtype=np.float64)
    else:
        shap_arr = shap_arr.astype(np.float64, copy=False)
    # Drop leading singleton dims (some caches store (1, N, T, F))
    while shap_arr.ndim > 3 and shap_arr.shape[0] == 1:
        shap_arr = shap_arr[0]
    if shap_arr.ndim == 4:
        shap_arr = shap_arr[..., 0]
    shap_arr = np.squeeze(shap_arr)

    if shap_arr.ndim == 3:
        imp = np.mean(np.abs(shap_arr), axis=(0, 1)).ravel()
    elif shap_arr.ndim == 2:
        imp = np.mean(np.abs(shap_arr), axis=0).ravel()
    else:
        imp = np.array([])
        print("Unexpected SHAP shape after normalize:", getattr(shap_arr, "shape", None))

    if imp.size > 0:
        # Plain Python floats — avoids Matplotlib interpreting arrays as linewidth etc.
        widths = [float(np.asarray(v).squeeze()) for v in imp]
        n_feat = len(widths)
        labels = [f"f{i}" for i in range(n_feat)]
        y_idx = list(range(n_feat))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(y_idx, widths)
        ax.set_yticks(y_idx)
        ax.set_yticklabels(labels)
        ax.set_title("Mean |SHAP| aggregated over time & samples (CNN-LSTM)")
        ax.set_xlabel("Mean |SHAP|")
        fig.tight_layout()
        fig.savefig(ARTIFACTS / "figures" / "shap_mean_abs_bar.png", dpi=300)
        plt.show()
    else:
        print("SHAP present but importance vector empty — try deleting cache and recompute.")
else:
    print("No SHAP values to plot.")
"""
    )
)

cells.append(
    md(
        r"""
## Human-centric LLM maintenance reports (Regolo / OpenAI-compatible)

This cell turns model outputs into **human-facing decision support** using an **OpenAI-compatible** HTTP API.

### Regolo (recommended for this project)

1. Create a key in the [Regolo dashboard](https://dashboard.regolo.ai) (Virtual Keys).
2. In Colab: **Secrets** (left sidebar) → add `REGOLO_API_KEY`, or set env in a **previous** cell **without committing the notebook**:
   ```python
   import os
   os.environ["REGOLO_API_KEY"] = userdata.get("REGOLO_API_KEY")  # Colab Secrets
   os.environ["PDM_LLM_BASE_URL"] = "https://api.regolo.ai/v1"     # optional if REGOLO_API_KEY is set (auto-default)
   os.environ["PDM_LLM_MODEL"] = "Llama-3.3-70B-Instruct"          # or another Regolo chat model name
   ```
3. If **`REGOLO_API_KEY`** is set, the notebook **defaults** `base_url` to **`https://api.regolo.ai/v1`** unless you override `PDM_LLM_BASE_URL` / `OPENAI_BASE_URL`.

### OpenAI

```python
import os
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
# Optional explicit base (otherwise official OpenAI endpoint):
# os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
os.environ["PDM_LLM_MODEL"] = "gpt-4o-mini"
```

**Never** paste API keys into notebook source that you save to GitHub. Rotate any key that appeared in chat/logs.

Run the **next code cell** on Colab to copy Secrets into `os.environ` (no key literals). If no key is present, the notebook writes a deterministic stub so the paper pipeline still runs.
"""
    )
)

cells.append(
    code(
        r"""
# --- Colab Secrets → env (safe for saved notebooks). Local Jupyter: set REGOLO_API_KEY in the shell instead.
try:
    from google.colab import userdata  # type: ignore
except Exception:
    userdata = None


def _colab_secret(name: str) -> str:
    if userdata is None:
        return ""
    try:
        v = userdata.get(name)
        return (v or "").strip() if isinstance(v, str) else ""
    except Exception:
        return ""


_reg = _colab_secret("REGOLO_API_KEY")
_oai = _colab_secret("OPENAI_API_KEY")
if _reg:
    os.environ["REGOLO_API_KEY"] = _reg
if _oai:
    os.environ["OPENAI_API_KEY"] = _oai
if _reg:
    os.environ.setdefault("PDM_LLM_BASE_URL", "https://api.regolo.ai/v1")
    os.environ.setdefault("PDM_LLM_MODEL", "Llama-3.3-70B-Instruct")
elif _oai:
    os.environ.setdefault("PDM_LLM_MODEL", "gpt-4o-mini")

if userdata is None:
    print(
        "Colab userdata not available — using existing process env for REGOLO_API_KEY / OPENAI_API_KEY.",
        flush=True,
    )
elif not (_reg or _oai):
    print(
        "No REGOLO_API_KEY / OPENAI_API_KEY in Colab Secrets — LLM cell will use stub unless env already set.",
        flush=True,
    )
else:
    print("Loaded API key(s) from Colab Secrets into os.environ.", flush=True)
"""
    )
)

cells.append(
    code(
        r"""
import textwrap

TOPK = int(os.environ.get("PDM_LLM_TOPK", "5"))
example_idx = 0
prob = float(model_cnn.predict(X_test[example_idx : example_idx + 1], verbose=0).ravel()[0])
decision_threshold = float(globals().get("best_thr", 0.5))
risk_level = "high" if prob >= decision_threshold else ("medium" if prob >= 0.5 * decision_threshold else "low")
feature_names = list(X_b2.columns)

if shap_vals is not None:
    _ex = np.asarray(shap_vals[example_idx])
    if _ex.ndim == 4:
        _ex = _ex[..., 0]
    _ex = np.squeeze(_ex)
    if _ex.ndim != 2:
        contrib = np.abs(_ex).ravel()
    else:
        contrib = np.mean(np.abs(_ex), axis=0).ravel()
    contrib = np.asarray(contrib, dtype=np.float64).ravel()
    order = np.argsort(-contrib)[:TOPK]
    readable = []
    for j in order:
        j_int = int(j)
        readable.append(
            {
                "feature_index": j_int,
                "feature_name": feature_names[j_int] if j_int < len(feature_names) else f"f{j_int}",
                "last_step_value": float(X_test[example_idx, -1, j_int]) if j_int < X_test.shape[2] else None,
                "mean_abs_shap_timestep_avg": float(contrib[j_int]),
            }
        )
else:
    n_feat = X_test.shape[2]
    readable = [
        {
            "feature_index": int(j),
            "feature_name": feature_names[j] if j < len(feature_names) else f"f{j}",
            "last_step_value": float(X_test[example_idx, -1, j]),
            "mean_abs_shap_timestep_avg": 0.0,
        }
        for j in range(min(TOPK, n_feat))
    ]

payload = {
    "fault_probability": prob,
    "decision_threshold": decision_threshold,
    "risk_level": risk_level,
    "top_features": readable,
    "model": "XAI-PdMNet (B2 engineered features + CNN-LSTM)",
    "fit_diagnostics": globals().get("fit_diag", {}),
    "metrics_context": globals().get("cnn_metrics", {}),
    "human_centric_constraints": {
        "avoid_overclaiming": True,
        "no_causal_claims_from_shap_only": True,
        "ask_for_human_confirmation_before_shutdown": True,
        "plain_language": True,
    },
}

(Path(ARTIFACTS / "paper_bundle")).mkdir(parents=True, exist_ok=True)
(Path(ARTIFACTS / "paper_bundle") / "llm_payload_example.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)

_provider = os.environ.get("PDM_LLM_PROVIDER", "auto").strip().lower()
if _provider not in {"auto", "regolo", "openai"}:
    _provider = "auto"
_rk = (os.environ.get("REGOLO_API_KEY") or "").strip()
_ok = (os.environ.get("OPENAI_API_KEY") or "").strip()
if _provider == "openai":
    key = _ok or _rk
else:
    key = _rk or _ok

if key:
    try:
        from openai import OpenAI

        base_url = (os.environ.get("PDM_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")

        if not base_url:
            if _provider == "regolo" or (_provider == "auto" and _rk):
                base_url = (
                    os.environ.get("PDM_REGOLO_BASE_URL", "https://api.regolo.ai/v1").strip().rstrip("/")
                )

        model_name = os.environ.get("PDM_LLM_MODEL") or os.environ.get("PDM_OPENAI_MODEL")
        if not model_name:
            if base_url and "regolo.ai" in base_url:
                model_name = "Llama-3.3-70B-Instruct"
            else:
                model_name = "gpt-4o-mini"

        client_kw = {"api_key": key.strip(), "timeout": float(os.environ.get("PDM_LLM_TIMEOUT_SEC", "120"))}
        if base_url:
            client_kw["base_url"] = base_url
        client = OpenAI(**client_kw)

        prompt = textwrap.dedent(
            f'''
            You are a human-centric predictive-maintenance assistant for an industrial operator.
            Use the JSON payload below to produce ONLY valid JSON with these keys:
            operator_brief, technician_work_order, safety_escalation_note, manager_summary,
            human_confirmation_questions, trust_and_limitations, next_30_min_actions.

            Requirements:
            - Use plain language, not research jargon.
            - Be specific about top sensor/proxy names and values.
            - Treat SHAP as association, not proof of physical cause.
            - Include clear escalation triggers and ask for human confirmation.
            - Mention model reliability issues if fit_diagnostics indicates underfitting/overfitting.
            - Do not recommend shutdown unless the risk is high and human checks confirm abnormal readings.

            Payload:
            {json.dumps(payload, indent=2)}
            '''
        ).strip()

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return valid JSON for human-centric industrial maintenance decision support."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        report = resp.choices[0].message.content
        report_json = None
        try:
            report_json = json.loads(report)
        except Exception:
            report_json = {"raw_response": report}

        (ARTIFACTS / "paper_bundle" / "human_centric_llm_report.json").write_text(
            json.dumps(report_json, indent=2), encoding="utf-8"
        )
        report_md = "# Human-Centric Maintenance Report\n\n"
        for k, v in report_json.items():
            report_md += f"## {k.replace('_', ' ').title()}\n{v}\n\n"
        (ARTIFACTS / "paper_bundle" / "human_centric_llm_report.md").write_text(
            report_md, encoding="utf-8"
        )
        print("LLM model:", model_name, "| base_url:", base_url or "(OpenAI default)", flush=True)
        print(report_md)
    except Exception as e:
        print("LLM call failed:", repr(e), flush=True)
        print(
            "Hints: set REGOLO_API_KEY + optional PDM_LLM_BASE_URL=https://api.regolo.ai/v1 "
            "and PDM_LLM_MODEL to a Regolo chat model id; or use OPENAI_API_KEY for OpenAI.",
            flush=True,
        )
else:
    report_json = {
        "operator_brief": f"Current predicted failure risk is {prob:.1%} ({risk_level}) against threshold {decision_threshold:.3f}.",
        "technician_work_order": "Check the listed sensor/proxy readings, compare with machine logs, and inspect wear/load/thermal conditions before any intervention.",
        "safety_escalation_note": "Escalate to supervisor if risk remains high on repeated readings or if physical inspection confirms abnormal vibration, heat, or tool wear.",
        "manager_summary": "The model generated an advisory maintenance alert; this is decision support and requires human confirmation.",
        "human_confirmation_questions": [
            "Are the top contributor sensors physically plausible for this machine state?",
            "Did recent maintenance or tooling changes occur?",
            "Do operator observations confirm abnormal heat, torque, speed, or wear?",
        ],
        "trust_and_limitations": "No API key was set; this deterministic report is generated from model probability and SHAP-style feature rankings. SHAP is associative, not causal.",
        "next_30_min_actions": [
            "Repeat measurement or rerun inference after a short interval.",
            "Inspect top contributor channels and tool wear.",
            "Log final human decision and corrective action.",
        ],
        "top_features": readable,
        "fit_diagnostics": globals().get("fit_diag", {}),
    }
    (ARTIFACTS / "paper_bundle" / "human_centric_llm_report_stub.json").write_text(
        json.dumps(report_json, indent=2), encoding="utf-8"
    )
    report_md = "# Human-Centric Maintenance Report (Stub)\n\n"
    for k, v in report_json.items():
        report_md += f"## {k.replace('_', ' ').title()}\n{v}\n\n"
    (ARTIFACTS / "paper_bundle" / "human_centric_llm_report_stub.md").write_text(report_md, encoding="utf-8")
    print(report_md)
"""
    )
)

cells.append(
    md(
        r"""
## Simple operator simulation — replay test windows with Plotly sliders

Exports HTML under `artifacts/paper_bundle/plotly_replay.html`.
"""
    )
)

cells.append(
    code(
        r"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n_show = min(200, len(X_test))
probs_show = model_cnn.predict(X_test[:n_show], verbose=0).ravel()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35])
feat_idx = 0
series = X_test[:n_show, -1, feat_idx]

fig.add_trace(go.Scatter(y=series, mode="lines", name="Last-step sensor (dim 0)"), row=1, col=1)
fig.add_trace(go.Scatter(y=probs_show, mode="lines", name="Fault probability"), row=2, col=1)
fig.update_layout(title="Digital twin style replay (approx)", height=520)
out_html = ARTIFACTS / "paper_bundle" / "plotly_replay.html"
fig.write_html(str(out_html))
print("Wrote", out_html, flush=True)
fig.show()
"""
    )
)

cells.append(
    md(
        r"""
## Step 8 — Export summary workbook for paper drafting
"""
    )
)

cells.append(
    code(
        r"""
summary_rows = []

if "cnn_metrics" in globals():
    summary_rows.append({"model": "cnn_lstm", "family": "proposal_dl", **cnn_metrics})
if "smote_metrics" in globals():
    summary_rows.append({"model": "mlp_smote_flattened_windows", "family": "proposal_augment", **smote_metrics})

if Path(ARTIFACTS / "tables" / "baselines_last_timestep.csv").exists():
    summary_rows.extend(
        pd.read_csv(ARTIFACTS / "tables" / "baselines_last_timestep.csv")
        .assign(family="proposal_sklearn")
        .to_dict("records")
    )

df_summary = pd.DataFrame(summary_rows)

_p_shallow = ARTIFACTS / "tables" / "shallow_cv_runs.csv"
_p_1dcnn = ARTIFACTS / "tables" / "1dcnn_cv_runs.csv"

mean_blocks = []
if _p_shallow.exists():
    _ds = pd.read_csv(_p_shallow)
    mean_blocks.append(
        _ds.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity"]]
        .mean()
        .reset_index()
        .assign(family="base_shallow_cv_mean")
    )
if _p_1dcnn.exists():
    _dc = pd.read_csv(_p_1dcnn)
    mean_blocks.append(
        _dc.groupby("architecture")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity"]]
        .mean()
        .reset_index()
        .rename(columns={"architecture": "model"})
        .assign(family="base_1dcnn_cv_mean")
    )

df_cv_mean = pd.concat(mean_blocks, ignore_index=True) if mean_blocks else pd.DataFrame()

out_xlsx = ARTIFACTS / "paper_bundle" / "results_summary.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="proposal_single_split", index=False)
    df_cv_mean.to_excel(writer, sheet_name="base_cv_means", index=False)
    if _p_shallow.exists():
        pd.read_csv(_p_shallow).to_excel(writer, sheet_name="shallow_cv_folds", index=False)
    if _p_1dcnn.exists():
        pd.read_csv(_p_1dcnn).to_excel(writer, sheet_name="1dcnn_cv_folds", index=False)
    if Path(ARTIFACTS / "tables" / "lenet_cv_runs.csv").exists():
        pd.read_csv(ARTIFACTS / "tables" / "lenet_cv_runs.csv").to_excel(writer, sheet_name="lenet_only_cv", index=False)
    _p_side = ARTIFACTS / "tables" / "side_by_side_ab_metrics.csv"
    if Path(_p_side).exists():
        pd.read_csv(_p_side).to_excel(writer, sheet_name="side_by_side_AB", index=False)
    _p_best = ARTIFACTS / "tables" / "best_models_summary.csv"
    if Path(_p_best).exists():
        pd.read_csv(_p_best).to_excel(writer, sheet_name="best_models_zoo", index=False)

if len(df_cv_mean):
    df_cv_mean.to_csv(ARTIFACTS / "paper_bundle" / "model_comparison_cv_means.csv", index=False)

print("Wrote", out_xlsx, flush=True)
"""
    )
)

cells.append(
    md(
        r"""
## Step 9 — Reviewer readiness checklist (auto-generated)

This cell helps decide if your current run is manuscript-ready.
"""
    )
)

cells.append(
    code(
        r"""
review = []

def _exists(p: Path) -> bool:
    return p.exists()

required = [
    ARTIFACTS / "tables" / "shallow_cv_runs.csv",
    ARTIFACTS / "tables" / "1dcnn_cv_runs.csv",
    ARTIFACTS / "tables" / "cnn_lstm_fit_diagnostics.csv",
    ARTIFACTS / "paper_bundle" / "cnn_lstm_metrics.json",
    ARTIFACTS / "paper_bundle" / "cnn_lstm_fit_diagnostics.json",
    ARTIFACTS / "paper_bundle" / "results_summary.xlsx",
    ARTIFACTS / "figures" / "roc_cnn_lstm.png",
    ARTIFACTS / "figures" / "pr_cnn_lstm.png",
    ARTIFACTS / "figures" / "confusion_cnn_lstm.png",
    ARTIFACTS / "figures" / "loss_cnn_lstm.png",
    ARTIFACTS / "figures" / "calibration_reliability.png",
    ARTIFACTS / "figures" / "shap_mean_abs_bar.png",
]

for p in required:
    review.append({"item": p.name, "present": _exists(p)})

_ctgan_ks = ARTIFACTS / "tables" / "ctgan_ks.csv"
review.append({"item": "ctgan_ks.csv", "present": _exists(_ctgan_ks)})

_llm_report = ARTIFACTS / "paper_bundle" / "human_centric_llm_report.json"
_llm_stub = ARTIFACTS / "paper_bundle" / "human_centric_llm_report_stub.json"
review.append({"item": "human_centric_report_real_or_stub", "present": _exists(_llm_report) or _exists(_llm_stub)})

if _ctgan_ks.exists():
    _k = pd.read_csv(_ctgan_ks)
    review.append({"item": "ctgan_ks_rows>0", "present": bool(len(_k) > 0)})
    if "ks_pvalue" in _k.columns:
        review.append({"item": "ctgan_ks_pass_rate_gt_0.3", "present": bool((_k["ks_pvalue"] > 0.05).mean() > 0.3)})

_m = ARTIFACTS / "paper_bundle" / "cnn_lstm_metrics.json"
if _m.exists():
    mj = json.loads(_m.read_text(encoding="utf-8"))
    review.append({"item": "cnn_recall_tuned_gt_0.2", "present": bool(mj.get("recall@tuned", 0) > 0.2)})
    review.append({"item": "cnn_f1_tuned_gt_0.2", "present": bool(mj.get("f1@tuned", 0) > 0.2)})
    review.append({"item": "cnn_pr_auc_gt_0.1", "present": bool(mj.get("pr_auc", 0) > 0.1)})

df_review = pd.DataFrame(review)
_show_df("Reviewer readiness checklist:", df_review)
df_review.to_csv(ARTIFACTS / "paper_bundle" / "reviewer_readiness_checklist.csv", index=False)
print("Wrote", ARTIFACTS / "paper_bundle" / "reviewer_readiness_checklist.csv", flush=True)
"""
    )
)

cells.append(
    md(
        r"""
## One-cell results extraction — master table + dashboard diagram

Run after metrics/plots exist. Produces:

- `tables/master_results_summary.csv`
- `figures/results_dashboard_all.png`

Shows **one consolidated metric table** and a **multi-panel figure** (bars + pipeline schematic).
"""
    )
)

cells.append(
    code(
        r"""
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

master_rows = []


def _row(branch: str, protocol: str, f1_r=None, roc_r=None, pr_r=None, acc=None, recall=None):
    master_rows.append(
        {
            "branch": branch[:80],
            "protocol": protocol[:60],
            "f1": f1_r,
            "roc_auc": roc_r,
            "pr_auc": pr_r,
            "accuracy": acc,
            "recall": recall,
        }
    )


_p_shallow = ARTIFACTS / "tables" / "shallow_cv_runs.csv"
if _p_shallow.exists():
    _ds = pd.read_csv(_p_shallow)
    for mname in sorted(_ds["model"].dropna().unique()):
        g = _ds[_ds["model"] == mname]
        _row(
            f"shallow:{mname}",
            "5-fold CV balanced subset",
            float(g["f1"].mean()),
            float(g["roc_auc"].mean()),
            float(g["pr_auc"].mean()),
            float(g["accuracy"].mean()) if "accuracy" in g else float("nan"),
            float(g["recall"].mean()) if "recall" in g else float("nan"),
        )

_p_1d = ARTIFACTS / "tables" / "1dcnn_cv_runs.csv"
if _p_1d.exists():
    _dc = pd.read_csv(_p_1d)
    for arch in sorted(_dc["architecture"].dropna().unique()):
        g = _dc[_dc["architecture"] == arch]
        _row(
            f"1dcnn:{arch}",
            "5-fold CV balanced subset",
            float(g["f1"].mean()),
            float(g["roc_auc"].mean()),
            float(g["pr_auc"].mean()),
            float(g["accuracy"].mean()) if "accuracy" in g else float("nan"),
            float(g["recall"].mean()) if "recall" in g else float("nan"),
        )

_p_bl = ARTIFACTS / "tables" / "baselines_last_timestep.csv"
if _p_bl.exists():
    _db = pd.read_csv(_p_bl)
    for _, r in _db.iterrows():
        _row(
            f"baseline:{r['model']}",
            "held-out test last timestep",
            float(r.get("f1", float("nan"))),
            float(r.get("roc_auc", float("nan"))),
            float(r.get("pr_auc", float("nan"))),
            float(r.get("accuracy", float("nan"))),
            float(r.get("recall", float("nan"))),
        )

_p_smote = ARTIFACTS / "tables" / "mlp_smote_metrics.csv"
if _p_smote.exists():
    try:
        sdict = pd.read_csv(_p_smote, index_col=0).squeeze().to_dict()
        _row(
            "mlp_smote_flattened_windows",
            "held-out test after SMOTE train",
            float(sdict.get("f1", float("nan"))),
            float(sdict.get("roc_auc", float("nan"))),
            float(sdict.get("pr_auc", float("nan"))),
            float(sdict.get("accuracy", float("nan"))),
            float(sdict.get("recall", float("nan"))),
        )
    except Exception:
        pass

mj = {}
_mjson = ARTIFACTS / "paper_bundle" / "cnn_lstm_metrics.json"
if _mjson.exists():
    mj = json.loads(_mjson.read_text(encoding="utf-8"))
elif "cnn_metrics" in globals():
    mj = dict(cnn_metrics)

if mj:
    _row(
        "cnn_lstm:XAI-PdMNet (tuned thr)",
        "held-out test windows",
        float(mj.get("f1@tuned", float("nan"))),
        float(mj.get("roc_auc", float("nan"))),
        float(mj.get("pr_auc", float("nan"))),
        float(mj.get("accuracy@tuned", float("nan"))),
        float(mj.get("recall@tuned", float("nan"))),
    )
    _row(
        "cnn_lstm:@0.5",
        "held-out test windows",
        float(mj.get("f1@0.5", float("nan"))),
        float(mj.get("roc_auc", float("nan"))),
        float(mj.get("pr_auc", float("nan"))),
        float(mj.get("accuracy@0.5", float("nan"))),
        float(mj.get("recall@0.5", float("nan"))),
    )

df_master = pd.DataFrame(master_rows)
out_csv = ARTIFACTS / "tables" / "master_results_summary.csv"
if len(df_master):
    df_master.to_csv(out_csv, index=False)
    print("Saved", out_csv, flush=True)
    _show_df("MASTER RESULTS TABLE (all branches)", df_master)
else:
    print("No rows assembled yet — run training/evaluation cells first.", flush=True)

# ---------- Dashboard figure ----------
fig = plt.figure(figsize=(14, 16))
gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.15, 1.15, 1.25], hspace=0.35, wspace=0.28)

if len(df_master):
    dfp = df_master.copy()
    dfp["plot_label"] = dfp["branch"].str.slice(0, 42)

    ax0 = fig.add_subplot(gs[0, :])
    z = dfp.sort_values("f1")
    ax0.barh(z["plot_label"], z["f1"].fillna(0), color="#3949ab")
    ax0.set_xlabel("F1")
    ax0.set_title("F1 across branches (NaN shown as 0 in bar only)")
    ax0.grid(axis="x", alpha=0.3)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.barh(z["plot_label"], z["roc_auc"].fillna(0), color="#00897b")
    ax1.set_xlabel("ROC-AUC")
    ax1.set_title("ROC-AUC")
    ax1.grid(axis="x", alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.barh(z["plot_label"], z["pr_auc"].fillna(0), color="#f4511e")
    ax2.set_xlabel("PR-AUC")
    ax2.set_title("PR-AUC (minority ranking)")
    ax2.grid(axis="x", alpha=0.3)

    ax3 = fig.add_subplot(gs[2, :])
    zz = z[["plot_label", "accuracy", "recall"]].set_index("plot_label")
    xx = np.arange(len(zz))
    w = 0.35
    ax3.bar(xx - w / 2, zz["accuracy"].fillna(0), width=w, label="accuracy", color="#6d4c41")
    ax3.bar(xx + w / 2, zz["recall"].fillna(0), width=w, label="recall", color="#fbc02d")
    ax3.set_xticks(xx)
    ax3.set_xticklabels(zz.index, rotation=25, ha="right")
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.set_title("Accuracy vs Recall (same branches)")
    ax3.grid(axis="y", alpha=0.3)

# Pipeline schematic (conceptual)
axf = fig.add_subplot(gs[3, :])
axf.axis("off")
axf.set_xlim(0, 10)
axf.set_ylim(0, 3)
steps = [
    ("1\nRaw AI4I", 0.6, 1.6),
    ("2\nClean\n(Ileri)", 2.0, 1.6),
    ("3\nB2 features\n+ windows", 3.5, 1.6),
    ("4\nBaseline zoo\nCV / sklearn", 5.3, 2.35),
    ("5\nXAI-PdMNet\nCNN-LSTM", 5.3, 0.85),
    ("6\nCalib\nROC/PR/ECE", 7.2, 1.6),
    ("7\nSHAP + LLM\n+ Twin UI", 8.8, 1.6),
]
for txt, x, y in steps:
    box = FancyBboxPatch(
        (x - 0.55, y - 0.38),
        1.1,
        0.76,
        boxstyle="round,pad=0.03",
        linewidth=1.2,
        edgecolor="#37474f",
        facecolor="#eceff1",
    )
    axf.add_patch(box)
    axf.text(x, y, txt, ha="center", va="center", fontsize=9)

pairs = [(0.6, 2.0), (2.0, 3.5), (3.5, 5.3), (5.3, 7.2), (7.2, 8.8)]
for xa, xb in pairs:
    axf.annotate(
        "",
        xy=(xb - 0.55, 1.6),
        xytext=(xa + 0.55, 1.6),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#455a64"),
    )
axf.annotate(
    "",
    xy=(5.3, 1.22),
    xytext=(5.3, 2.0),
    arrowprops=dict(arrowstyle="->", lw=1.2, color="#6a1b9a"),
)
axf.annotate(
    "",
    xy=(5.3, 1.22),
    xytext=(3.5 + 0.55, 1.6),
    arrowprops=dict(arrowstyle="->", lw=1.2, color="#6a1b9a"),
)
axf.text(
    5.0,
    2.55,
    "Industry 5.0 human-centric layer",
    fontsize=10,
    fontweight="bold",
    color="#4a148c",
)

handles = [
    mpatches.Patch(color="#3949ab", label="F1 bars"),
    mpatches.Patch(color="#00897b", label="ROC-AUC"),
    mpatches.Patch(color="#f4511e", label="PR-AUC"),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99))
fig.suptitle(
    "PdM Research Dashboard — metrics + pipeline (protocols differ; cite honestly)",
    fontsize=13,
    fontweight="bold",
    y=0.995,
)

dash_path = ARTIFACTS / "figures" / "results_dashboard_all.png"
fig.savefig(dash_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved dashboard:", dash_path, flush=True)
"""
    )
)

cells.append(
    md(
        r"""
## Step 10 — Final round bundle + expert logic audit

Run this after the full notebook. It creates **one clean folder** under `ARTIFACTS`:

`FINAL_ROUND_BUNDLE/`

with:

- `code/` — available notebook/code/config snapshots and feature schema
- `models/` — `.keras`, `.pkl`, `.h5` final models
- `checkpoints/` — CV fold weights
- `results_csv/` — all CSV/JSON/NPZ result tables
- `results_excel/` — paper workbook(s)
- `figures/` — publication plots
- `reports/` — SHAP/LLM/human-centric reports and paper bundle files
- `configs/` — run configuration, versions, manifest
- `logs/` — expert audit CSV/JSON

Also created:

- **`RESEARCH_EXPORT_PACK/`** (under `ARTIFACTS`, name via `PDM_RESEARCH_PACK_NAME`) — **`csv/`** + **`figures/`** only, for quick manuscript / thesis packaging
- **`FINAL_ROUND_BUNDLE/research_pack_flat/`** — same CSV + figures copied again so **one zip** of the final bundle includes plots and tables

This is the folder to download/share for paper writing and supervision review.

### Research export pack (CSV + figures only)

Step 10 also builds **`ARTIFACTS/<PDM_RESEARCH_PACK_NAME>/`** (default name **`RESEARCH_EXPORT_PACK`**):

- **`csv/`** — every `*.csv` from `tables/` and `paper_bundle/` (paper_bundle copies prefixed if names collide)
- **`figures/`** — every plot from `figures/` (`png`, `jpg`, `svg`, `pdf`)

Zip that single folder for collaborators or appendix uploads.
"""
    )
)

cells.append(
    code(
        r"""
import hashlib
import shutil
import time


def _safe_copy_file(src: Path, dst: Path) -> bool:
    try:
        if not src.exists() or not src.is_file():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print("Copy skipped:", src, "->", dst, "|", e, flush=True)
        return False


def _copy_matching(src_dir: Path, dst_dir: Path, suffixes: tuple[str, ...]) -> list[dict]:
    copied = []
    if not src_dir.exists():
        return copied
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        if suffixes and p.suffix.lower() not in suffixes and not p.name.endswith(".weights.h5"):
            continue
        rel = p.relative_to(src_dir)
        dst = dst_dir / rel
        if _safe_copy_file(p, dst):
            copied.append({"source": str(p), "dest": str(dst), "bytes": int(dst.stat().st_size)})
    return copied


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


print("Creating final bundle:", FINAL_BUNDLE, flush=True)

_research_pack_name = os.environ.get("PDM_RESEARCH_PACK_NAME", "RESEARCH_EXPORT_PACK").strip() or "RESEARCH_EXPORT_PACK"
RESEARCH_PACK = (ARTIFACTS / _research_pack_name).resolve()
(RESEARCH_PACK / "csv").mkdir(parents=True, exist_ok=True)
(RESEARCH_PACK / "figures").mkdir(parents=True, exist_ok=True)

for sub in (
    "code",
    "models",
    "checkpoints",
    "results_csv",
    "results_excel",
    "figures",
    "reports",
    "configs",
    "logs",
):
    (FINAL_BUNDLE / sub).mkdir(parents=True, exist_ok=True)

manifest = {
    "created_at_unix": int(time.time()),
    "experiment_name": str(_exp_name),
    "artifacts": str(ARTIFACTS),
    "final_bundle": str(FINAL_BUNDLE),
    "research_pack": str(RESEARCH_PACK),
    "run_mode": RUN_MODE,
    "seed": SEED,
    "copies": [],
    "research_pack_files": [],
    "expert_audit": [],
}

# Code/config snapshots available inside Colab.
for candidate in [
    Path("predictive_maintenance_ai4i2020.ipynb"),
    Path("gen_notebook.py"),
    Path("requirements.txt"),
    Path("colab/predictive_maintenance_ai4i2020.ipynb"),
    Path("colab/gen_notebook.py"),
    Path("colab/requirements.txt"),
]:
    if candidate.exists():
        if _safe_copy_file(candidate, FINAL_BUNDLE / "code" / candidate.name):
            manifest["copies"].append({"kind": "code", "source": str(candidate)})

# Always write the reproducibility essentials, even if notebook source file is unavailable in Colab.
pd.DataFrame({"feature": list(X_b2.columns)}).to_csv(
    FINAL_BUNDLE / "code" / "b2_feature_schema.csv", index=False
)
(FINAL_BUNDLE / "code" / "runtime_reproducibility_notes.md").write_text(
    "\n".join(
        [
            "# Runtime Reproducibility Notes",
            "",
            f"- ARTIFACTS: `{ARTIFACTS}`",
            f"- FINAL_BUNDLE: `{FINAL_BUNDLE}`",
            f"- RUN_MODE: `{RUN_MODE}`",
            f"- SEED: `{SEED}`",
            f"- B2 features: `{len(X_b2.columns)}`",
            "",
            "This folder was created by Step 10 after data cleaning, preprocessing, training, testing, calibration, XAI, and reporting cells.",
        ]
    ),
    encoding="utf-8",
)

# Models/checkpoints/results/figures/reports.
manifest["copies"].extend(_copy_matching(ARTIFACTS / "models", FINAL_BUNDLE / "models", (".keras", ".pkl", ".h5")))
manifest["copies"].extend(_copy_matching(ARTIFACTS / "checkpoints", FINAL_BUNDLE / "checkpoints", (".h5", ".weights")))
manifest["copies"].extend(
    _copy_matching(ARTIFACTS / "tables", FINAL_BUNDLE / "results_csv", (".csv", ".json", ".npz"))
)
manifest["copies"].extend(_copy_matching(ARTIFACTS / "figures", FINAL_BUNDLE / "figures", (".png", ".jpg", ".jpeg", ".svg", ".pdf")))
manifest["copies"].extend(
    _copy_matching(ARTIFACTS / "paper_bundle", FINAL_BUNDLE / "reports", (".json", ".txt", ".md", ".html", ".csv", ".png"))
)
manifest["copies"].extend(_copy_matching(ARTIFACTS / "paper_bundle", FINAL_BUNDLE / "results_excel", (".xlsx",)))

# Flat research pack: all CSV tables + all figures (duplicate paths for manuscripts / zip sharing).
_fig_suffixes = (".png", ".jpg", ".jpeg", ".svg", ".pdf")
for p in sorted((ARTIFACTS / "figures").glob("*")):
    if p.is_file() and p.suffix.lower() in _fig_suffixes:
        dst = RESEARCH_PACK / "figures" / p.name
        if _safe_copy_file(p, dst):
            manifest["research_pack_files"].append({"kind": "figure", "path": str(dst.relative_to(RESEARCH_PACK))})

_tables_csv = ARTIFACTS / "tables"
if _tables_csv.exists():
    for p in sorted(_tables_csv.glob("*.csv")):
        dst = RESEARCH_PACK / "csv" / p.name
        if _safe_copy_file(p, dst):
            manifest["research_pack_files"].append({"kind": "csv_tables", "path": str(dst.relative_to(RESEARCH_PACK))})

_pb = ARTIFACTS / "paper_bundle"
if _pb.exists():
    for p in sorted(_pb.glob("*.csv")):
        dst = RESEARCH_PACK / "csv" / p.name
        if dst.exists():
            dst = RESEARCH_PACK / "csv" / ("paper_bundle__" + p.name)
        if _safe_copy_file(p, dst):
            manifest["research_pack_files"].append({"kind": "csv_paper_bundle", "path": str(dst.relative_to(RESEARCH_PACK))})

_rp_readme = "\n".join(
    [
        "# Research export pack",
        "",
        f"Experiment: `{_exp_name}`",
        f"ARTIFACTS: `{ARTIFACTS}`",
        "",
        "## Contents",
        "",
        "- **`csv/`** — All pipeline CSV exports from `artifacts/tables/` plus CSV files from `artifacts/paper_bundle/` (prefixed `paper_bundle__` if the filename already existed).",
        "- **`figures/`** — All plots saved under `artifacts/figures/` (PNG/JPG/SVG/PDF).",
        "",
        "Use this folder for manuscripts, thesis figures, and supervisor handoffs. Full checkpoints and code live under `FINAL_ROUND_BUNDLE/`.",
        "",
        f"Files copied this run: **{len(manifest['research_pack_files'])}**",
    ]
)
(RESEARCH_PACK / "README_RESEARCH_PACK.md").write_text(_rp_readme, encoding="utf-8")
print("Research pack (CSV + figures):", RESEARCH_PACK, flush=True)

# Same layout inside final bundle so one zip download has tables + plots alongside models/code.
_rsnap = FINAL_BUNDLE / "research_pack_flat"
(_rsnap / "csv").mkdir(parents=True, exist_ok=True)
(_rsnap / "figures").mkdir(parents=True, exist_ok=True)
manifest["copies"].extend(_copy_matching(RESEARCH_PACK / "csv", _rsnap / "csv", (".csv",)))
manifest["copies"].extend(_copy_matching(RESEARCH_PACK / "figures", _rsnap / "figures", _fig_suffixes))
_safe_copy_file(RESEARCH_PACK / "README_RESEARCH_PACK.md", _rsnap / "README_RESEARCH_PACK.md")

# Configs.
for cfg in [CONFIG_PATH, ARTIFACTS / "paper_bundle" / "versions.csv", PROGRESS_PATH]:
    if cfg.exists():
        _safe_copy_file(cfg, FINAL_BUNDLE / "configs" / cfg.name)

# Expert logic audit.
def _audit(name: str, ok: bool, detail: str, severity: str = "info") -> None:
    manifest["expert_audit"].append(
        {"check": name, "ok": bool(ok), "severity": severity, "detail": detail}
    )


_audit(
    "b2_no_fault_columns",
    not any(c in X_b2.columns for c in FAULT_COLS),
    "B2 must not include TWF/HDF/PWF/OSF/RNF leakage columns.",
    "critical",
)
_audit(
    "b2_no_id_columns",
    not any(c in X_b2.columns for c in ["UID", "UDI", "Product ID"]),
    "B2 should not include identifiers.",
    "high",
)
_audit(
    "cleaned_has_failures",
    int(df_clean["Machine failure"].sum()) > 0,
    f"Cleaned positives={int(df_clean['Machine failure'].sum())}",
    "critical",
)
_audit(
    "b2_feature_count_reasonable",
    X_b2.shape[1] >= 20,
    f"B2 shape={X_b2.shape}; expected raw + engineered + Type one-hot.",
    "medium",
)
_audit(
    "no_nan_in_b2",
    not bool(pd.isna(X_b2.to_numpy()).any()),
    "B2 numeric matrix should be finite after coercion/fill.",
    "critical",
)
if "X_train" in globals() and "X_test" in globals():
    _audit("train_test_available", True, f"Train={X_train.shape}, Test={X_test.shape}", "critical")
    _audit(
        "test_has_positive_class",
        int(np.sum(y_test == 1)) > 0,
        f"Test positives={int(np.sum(y_test == 1))}, negatives={int(np.sum(y_test == 0))}",
        "critical",
    )
else:
    _audit("train_test_available", False, "Run Step 4 before Step 10.", "critical")

for required_path, severity in [
    (ARTIFACTS / "models" / "cnn_lstm_best.keras", "high"),
    (ARTIFACTS / "tables" / "shallow_cv_runs.csv", "medium"),
    (ARTIFACTS / "tables" / "1dcnn_cv_runs.csv", "medium"),
    (ARTIFACTS / "paper_bundle" / "results_summary.xlsx", "high"),
    (ARTIFACTS / "paper_bundle" / "cnn_lstm_metrics.json", "high"),
    (ARTIFACTS / "paper_bundle" / "reviewer_readiness_checklist.csv", "high"),
    (ARTIFACTS / "figures" / "calibration_reliability.png", "medium"),
]:
    _audit(
        "artifact_present_" + required_path.name,
        required_path.exists(),
        str(required_path),
        severity,
    )

if "cnn_metrics" in globals():
    _audit("cnn_pr_auc_recorded", "pr_auc" in cnn_metrics, str(cnn_metrics.get("pr_auc")), "high")
    _audit(
        "cnn_f1_not_nan",
        np.isfinite(float(cnn_metrics.get("f1@tuned", np.nan))),
        str(cnn_metrics.get("f1@tuned")),
        "high",
    )
if "fit_diag" in globals():
    _audit(
        "fit_diagnostics_available",
        bool(fit_diag),
        json.dumps(fit_diag)[:500],
        "high",
    )

df_audit = pd.DataFrame(manifest["expert_audit"])
_show_df("Final expert logic audit:", df_audit)
df_audit.to_csv(FINAL_BUNDLE / "logs" / "final_expert_logic_audit.csv", index=False)

# Hash index for bundle integrity.
hash_rows = []
for p in FINAL_BUNDLE.rglob("*"):
    if p.is_file():
        try:
            hash_rows.append(
                {
                    "relative_path": str(p.relative_to(FINAL_BUNDLE)),
                    "bytes": int(p.stat().st_size),
                    "sha256": _sha256_file(p),
                }
            )
        except Exception as e:
            hash_rows.append({"relative_path": str(p), "bytes": None, "sha256": f"ERROR: {e}"})
df_hash = pd.DataFrame(hash_rows).sort_values("relative_path")
df_hash.to_csv(FINAL_BUNDLE / "configs" / "bundle_file_hashes.csv", index=False)

manifest["file_count"] = int(len(df_hash))
manifest["audit_passed_critical"] = bool(
    not any((not r["ok"]) and r["severity"] == "critical" for r in manifest["expert_audit"])
)
(FINAL_BUNDLE / "configs" / "final_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

readme = f'''# XAI-PdM Final Round Bundle

Experiment: `{_exp_name}`

Artifacts root: `{ARTIFACTS}`

Final bundle: `{FINAL_BUNDLE}`

## Folder Map

- `code/`: notebook/code snapshots if available, feature schema, reproducibility notes
- `models/`: trained final models
- `checkpoints/`: fold checkpoints / resumable weights
- `results_csv/`: CSV/JSON/NPZ result tables
- `results_excel/`: paper workbook(s)
- `figures/`: saved plots
- `reports/`: LLM, SHAP, digital twin, paper bundle files
- `configs/`: versions, config, manifest, file hashes
- `logs/`: final expert logic audit

## Expert Audit

Critical checks passed: `{manifest["audit_passed_critical"]}`

Open `logs/final_expert_logic_audit.csv` for all checks.

## Research export pack

Sibling folder: **`{RESEARCH_PACK.name}`** under `ARTIFACTS` — flat **`csv/`** + **`figures/`** for papers (see `README_RESEARCH_PACK.md` there).
'''
(FINAL_BUNDLE / "README_FINAL_BUNDLE.md").write_text(readme, encoding="utf-8")

print("FINAL ROUND BUNDLE READY:", FINAL_BUNDLE, flush=True)
print("Research pack ready:", RESEARCH_PACK, flush=True)
print("Files in bundle:", manifest["file_count"], flush=True)
print("Critical audit passed:", manifest["audit_passed_critical"], flush=True)
"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", OUT, flush=True)
