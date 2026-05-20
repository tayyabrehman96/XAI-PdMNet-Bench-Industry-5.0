# AI4I 2020 — Colab research notebook

**Repository overview, UCI CSV links, and architecture:** see the project root [`README.md`](../README.md) and [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

**Modular Python (cleaning + B2 features):** [`src/xai_pdmbench/`](../src/xai_pdmbench/) — use `pip install -e .` from the repo root.

**Split notebook into many small files (optional):** `python scripts/export_notebook_cells.py` → `colab/notebook_export/` (ignored by Git by default; remove the ignore rule to commit).

## Files

| File | Purpose |
|------|---------|
| `predictive_maintenance_ai4i2020.ipynb` | Main pipeline (generated — run `gen_notebook.py` after editing). |
| `gen_notebook.py` | Rebuilds the `.ipynb` from embedded cell sources (fixes delimiter issues). |
| `requirements.txt` | Optional local `pip install -r`. |

## Google Colab

1. Upload `predictive_maintenance_ai4i2020.ipynb` or open from Drive.
2. **Enable GPU** (optional) via Runtime → Change runtime type.
3. Uncomment **Drive mount** in the pip cell and set `PDM_ARTIFACTS` so checkpoints survive disconnects, e.g. `export PDM_ARTIFACTS=/content/drive/MyDrive/XAI_PdM_artifacts`.
4. Environment knobs (optional): `PDM_SEED`, `PDM_EXPERIMENT_NAME` (creates a dedicated folder per run), `PDM_AI4I_CSV`, `PDM_PIPELINE` (`B1`|`B2`), `PDM_N_BALANCE_REPS`, `PDM_1DCNN_MODELS`, `PDM_EPOCHS_1DCNN`, `PDM_EPOCHS_CNN`, `PDM_CTGAN_EPOCHS`, reuse flags (`PDM_RUN_MODE`, etc.). LLM: **`REGOLO_API_KEY`** (Regolo defaults API base to `https://api.regolo.ai/v1`) or **`OPENAI_API_KEY`**, plus `PDM_LLM_MODEL` (e.g. `Llama-3.3-70B-Instruct` on Regolo).
5. No-reupload / reuse knobs: `PDM_RUN_MODE=fast|resume|full` (recommended one-switch control). Optional manual overrides: `PDM_REUSE_RESULTS=1`, `PDM_FORCE_REFRESH_DATA=1`, `PDM_FORCE_CNN_RETRAIN=1`, `PDM_FORCE_CTGAN_REFIT=1`.

Artifacts are written under `ARTIFACTS/` (`figures`, `tables`, `models`, `checkpoints`, `paper_bundle`). With `PDM_EXPERIMENT_NAME`, the notebook writes to `<base_artifacts>/<experiment_name>/...` so each run has isolated models/results. Resume: re-run cells — CV skips completed keys in `tables/cv_progress.json`; SHAP loads `tables/shap_cache.npz`; CNN-LSTM and CTGAN can load saved models unless forced to retrain/refit.

The notebook begins with **Sections 1–4** (Markdown): **Dataset**, **Gaps vs base paper**, **Contributions**, **System architecture** (pipeline + CNN–LSTM table).

## Dataset

[AI4I 2020 — UCI 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
