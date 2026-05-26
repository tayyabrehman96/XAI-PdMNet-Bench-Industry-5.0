# XAI-PdMNet-Bench-Industry-5.0

Code-first reproducibility package for the AI4I 2020 and PHM 2010 predictive-maintenance benchmarks.

## Repository layout

- `data/` input CSVs and PHM source arrays used by the scripts
- `scripts/` dataset preparation, training, and result-bundle builders
- `src/` shared preprocessing and feature code
- `trained_models/` exported machine-learning and deep-learning artifacts
- `results/` generated metrics, inventories, and the Excel results bundle

## Reproduce the package

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow:

```bash
python scripts/run_all_training.py
```

This regenerates the PHM derived CSVs, trains the AI4I and PHM models, refreshes the model inventories, and writes the result tables in `results/`.

## Main outputs

- `trained_models/ai4i/`
- `trained_models/phm2010/`
- `trained_models/ALL_TRAINED_MODELS_INVENTORY.csv`
- `results/model_inventory.csv`
- `results/reproduction_results_bundle.xlsx`

## Notes

- The repository keeps generated trained-model artifacts in version control so a reviewer can inspect the exact exported files.
- Result tables are generated from local data and local model artifacts. They are not copied from an article PDF.
