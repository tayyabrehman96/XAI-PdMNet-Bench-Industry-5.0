# Beyond Data Leakage: Explainable Predictive Maintenance for Industry 5.0

Official code and notebook companion for the research framework **XAI-PdMNet-Bench**  
(AI4I 2020, leakage-safe features, CTGAN, XGBoost/CNN–LSTM, SHAP, template-based reporting).

**Public repository:**  
[github.com/tayyabrehman96/Beyond-Data-Leakage-An-Explainable-Predictive-Maintenance-Benchmarking-Framework-for-Industry-5.0](https://github.com/tayyabrehman96/Beyond-Data-Leakage-An-Explainable-Predictive-Maintenance-Benchmarking-Framework-for-Industry-5.0)

---

## Dataset and CSV (primary source)

| Item | Detail |
|------|--------|
| **Name** | AI4I 2020 Predictive Maintenance |
| **Registry** | [UCI Machine Learning Repository — dataset ID 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| **DOI** | [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C) |
| **Licence** | CC BY 4.0 |
| **Direct CSV (HTTPS)** | [`https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv`](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv) |

**Do not redistribute the CSV as your own asset**; cite UCI + DOI. Locally, set `PDM_AI4I_CSV=/path/to/ai4i2020.csv` after download if you work offline.

**Leakage note:** columns `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are logically derived from `Machine failure`. For valid deployment-style experiments, use **track B2** (these columns dropped as inputs). Track **B1** may retain them only for controlled reproduction comparisons.

---

## What this repository contains (curated, not a full dump)

| Path | Purpose |
|------|---------|
| [`colab/`](colab/) | Google Colab workflow: main notebook, generator, `requirements.txt`, Colab-specific README |
| [`src/xai_pdmbench/`](src/xai_pdmbench/) | Small **reusable** library: UCI constants, Ileri-style cleaning, **B1/B2** feature builders (matches notebook logic) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | End-to-end pipeline + **Mermaid** architecture (renders on GitHub) |
| [`scripts/export_notebook_cells.py`](scripts/export_notebook_cells.py) | Splits the `.ipynb` into many `.py`/`.md` files under `colab/notebook_export/` for review and partial reuse |
| [`generate_figures.py`](generate_figures.py) | Local script to regenerate manuscript-style figures (optional) |
| [`docs/REPO_UPLOAD_CHECKLIST.md`](docs/REPO_UPLOAD_CHECKLIST.md) | Checklist of what belongs on GitHub vs what to omit |

**Intentionally omitted from Git** (see [`.gitignore`](.gitignore)): trained weights, `ARTIFACTS/`, checkpoints, cached CSV copies, API keys, scratch notebooks, and bulky export packs. Upload only what your paper needs; use Releases or Zenodo for frozen artifact bundles.

---

## Quick start

### 1) Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r colab/requirements.txt
pip install -e .         # optional editable install of xai_pdmbench
```

Editable install uses [`pyproject.toml`](pyproject.toml) (see below).

### 2) Regenerate the main Colab notebook (optional)

After editing cell sources in [`colab/gen_notebook.py`](colab/gen_notebook.py):

```bash
python colab/gen_notebook.py
```

### 3) Export notebook into multiple files (for navigation / code review)

```bash
python scripts/export_notebook_cells.py
```

Output: `colab/notebook_export/` — one file per cell (`cell_000_intro.md`, `cell_005_code.py`, …). **Not** a substitute for the full notebook in Colab; use the `.ipynb` for execution.

### 4) Use the Python modules in your own script

```python
import pandas as pd
from xai_pdmbench.constants import AI4I_CSV_URL, FAULT_COLUMNS
from xai_pdmbench.data import normalize_columns, clean_ai4i_basepaper
from xai_pdmbench.features import build_features_b2

df = pd.read_csv(AI4I_CSV_URL)
df = normalize_columns(df)
df = clean_ai4i_basepaper(df)
X, y = build_features_b2(df)
```

---

## Architecture diagram

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the six-stage pipeline, data flow, and a Mermaid diagram suitable for GitHub. Add your own `docs/assets/pipeline.png` if you want a bitmap figure in the README (not required).

---

## Related references

- Base CNN benchmark (leakage-prone setting in many reproductions): Ileri, Altun, Narin (2024), *Appl. Sci.* — [DOI 10.3390/app14114899](https://doi.org/10.3390/app14114899)
- CTGAN: Xu et al., *NeurIPS* DGM workshop — SDV implementation in notebook
- SHAP: Lundberg & Lee — [TreeExplainer / GradientExplainer](https://github.com/slundberg/shap)

---

## Citation

Use the citation block from your forthcoming *Sensors* / MDPI manuscript when available. Until then, cite the repository URL and the UCI dataset DOI above.

---

## Licence

Code in this repository is provided for research reproduction. The **AI4I 2020** data remain under **CC BY 4.0** (UCI).

