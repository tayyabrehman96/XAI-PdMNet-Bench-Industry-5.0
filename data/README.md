# Data directory — public sources & bundled files

## Files in **this repository** (convenience / regenerated)

| Path | Purpose | GitHub | Raw HTTPS |
|------|---------|--------|-----------|
| `ai4i/ai4i2020.csv` | UCI mirror for offline training | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/data/ai4i/ai4i2020.csv) | [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/data/ai4i/ai4i2020.csv) |
| `phm2010/phm2010_windows_6x500.csv` | Derived PHM sliding windows `(6 × 500)` per cut | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/data/phm2010/phm2010_windows_6x500.csv) | [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/data/phm2010/phm2010_windows_6x500.csv) |
| `phm2010/phm2010_feature_table.csv` | Derived PHM handcrafted features per cut | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/data/phm2010/phm2010_feature_table.csv) | [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/data/phm2010/phm2010_feature_table.csv) |

**Repository root:** [https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0)

---

## AI4I 2020 — **official download** (always cite)

- Landing: [UCI dataset 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- CSV: [uci…/00601/ai4i2020.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)
- DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C) — **CC BY 4.0**

---

## PHM 2010 CNC milling — **official / society sources**

Use these to obtain the **original** challenge archives; place or convert them so `scripts/prepare_phm2010_release_dataset.py` can read `data/phm2010/source/originfeature/data_x{1,4,6}.npy` and `data_y{1,4,6}.npy`.

- [PHM Society — 2010 Data Challenge](https://www.phmsociety.org/competition/phm/10)
- [PHM Society — NASA prognostics index (mirror)](https://data.phmsociety.org/nasa/)
- [IEEE DataPort — 2010 PHM Society Data Challenge](https://ieee-dataport.org/documents/2010-phm-society-conference-data-challenge)

---

## Regenerating PHM CSVs locally

```bash
python scripts/prepare_phm2010_release_dataset.py
```

Requires the `.npy` source arrays under `phm2010/source/originfeature/` as documented in the script header.
