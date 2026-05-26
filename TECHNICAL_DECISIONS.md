# Technical Decisions

## Goal

This repository is organized as a reproducibility package for code review and rerun, not as a publication archive.

## Decisions

1. Script-first layout

The main workflow is driven by Python scripts in `scripts/` so a reviewer can rerun the package without opening notebooks.

2. Code-first results

All CSV summaries and the Excel bundle in `results/` are generated from repository data, trained artifacts, or rerun code. PDF tables are not copied into the outputs.

3. Two dataset tracks kept together

Both benchmarks are kept in one repo:

- `data/ai4i/ai4i2020.csv`
- `data/phm2010/` with source arrays and generated release tables

This keeps the transfer-learning and PHM validation flow runnable from one checkout.

4. Trained artifacts are versioned

The repository stores the exported model artifacts under `trained_models/` so reviewers can confirm that the files referenced by the code and result inventories are present.

5. Deep-learning exports include full model files and weight files

PyTorch models are stored as both:

- full serialized model files such as `*.pt`
- state-dict weight files such as `*_weights.pt`

This supports both direct loading and architecture-plus-weights inspection.

6. Inventory files are part of the package

The generated inventory files in `trained_models/` and `results/` make it easy to verify dataset presence, model presence, and expected outputs after a rerun.

7. Publication-only assets are excluded

Publication-only folders, diagrams, and PDF-focused helper files are intentionally omitted so the repo stays focused on executable reproducibility.
