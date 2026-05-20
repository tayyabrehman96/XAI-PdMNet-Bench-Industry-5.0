# What to push to GitHub (curated checklist)

Aligned with [`.gitignore`](../.gitignore): **small, reproducible** assets only.

## Always include

- [ ] [`README.md`](../README.md) (root technical overview + UCI / CSV links)
- [ ] [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) (+ optional `docs/assets/pipeline.png`)
- [ ] [`LICENSE`](../LICENSE)
- [ ] [`pyproject.toml`](../pyproject.toml)
- [ ] [`src/xai_pdmbench/`](../src/xai_pdmbench/)
- [ ] [`colab/README.md`](../colab/README.md)
- [ ] [`colab/requirements.txt`](../colab/requirements.txt)
- [ ] [`colab/gen_notebook.py`](../colab/gen_notebook.py)
- [ ] [`colab/predictive_maintenance_ai4i2020.ipynb`](../colab/predictive_maintenance_ai4i2020.ipynb)
- [ ] [`scripts/export_notebook_cells.py`](../scripts/export_notebook_cells.py)
- [ ] [`generate_figures.py`](../generate_figures.py) (optional but useful)

## Usually exclude

- `ARTIFACTS/`, checkpoints, `.h5`, cached `ai4i2020_cached.csv`
- Local scratch notebooks (`After collab run.ipynb`, …)
- API keys / `.env`
- `colab/notebook_export/` (generated; remove `.gitignore` line if you want it public)

## After clone

```bash
pip install -r colab/requirements.txt
pip install -e .
python scripts/export_notebook_cells.py   # optional
python colab/gen_notebook.py             # regenerate .ipynb if you edited cells in gen_notebook.py
```
