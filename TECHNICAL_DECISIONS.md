# Technical decisions (deep dive)

Companion to the root **[README.md](README.md)**. This file records **engineering intent** beyond what fits in onboarding docs.

---

## 1. Why two benchmarks in one tree?

We pair **tabular AI4I (weak temporal semantics)** against **explicitly sequential PHM windows** plus **cutter-wise distribution shift**:

- Lets reviewers compare classical GBDT/LR behaviour vs conv-recurrent hybrids under **matching reporting conventions** (CSV + Excel artefacts).
- The **cross-domain encoder transfer** pipeline becomes a reproducible artefact (`transfer_ai4i_to_phm`) rather than a narrative claim.

Cost: heavier checkout. Benefit: contiguous scientific story for Industry 5.0 maintenance AI.

---

## 2. Leakage policy (critical for honest publication)

Industrial papers often regress `Machine failure` **while still feeding symptom flags derived from failures** (`TWF…RNF`). We treat that as structural **evaluation leakage**:

> **Production sensors do not magically reveal “which subsystem already failed”; they expose stress/temperature-derived proxies only.**

Hence default training uses **`build_features_b2`**: drop `TWF, HDF, PWF, OSF, RNF` but keep `Machine failure` supervision.

Residual risk: correlations between wear proxies and imminent failure persist — this is unavoidable physics, not tautology.

---

## 3. Why keep contradictory AI4I rows absent?

Filtering rules:

```
bad_rnf := (RNF == 1) ∧ (Machine failure == 0)
bad_mf  := (Machine failure == 1) ∧ (Σ fault flags == 0)
KEEP    := ¬(bad_rnf ∨ bad_mf)
```

Interpretation: aligns multi-label symptom logic with aggregated failure bookkeeping. Drops are **few** (~dozens) but prevent optimisers from exploiting annotation noise.

Implementation: `clean_ai4i_benchmark_rows()` in **`src/xai_pdmbench/data.py`**.

---

## 4. PHM aggregation choices

### 4.1 Window length reduction

Starting length **5000** per channel is decimated ×10 → **500** via non-overlapping averaging.

Rationale:

- Bounds CSV footprint (three cutters × ≈ 315 cuts × `(6 × 500)` floats plus metadata) while remaining spreadsheet-friendly where needed.
- Emulates instrumentation downsampling pipelines used on edge devices prior to neural ingestion.

Sensitivity: sharper events may blur — acceptable for package-level benchmarking, not microscopic wear physics.

### 4.2 Scalar feature extraction

Beyond raw windows we emit **skew + kurtosis** per vibration channel capturing impulsive chatter vs smooth wear regimes.

Dimensional output: **`6 × 9 = 54`** interpretable summaries per observation.

---

## 5. Training / evaluation quirks (by design)

| Topic | Behaviour | Why |
|-------|-----------|-----|
| **Stratification** | 80 / 20 class preserving per script | Mirrors classroom rigour despite tiny failure counts |
| **`pos_weight`** in Torch | Ratio negatives / positives | Cheap class rebalancing avoiding SMOTE hallucination artefacts in core trainers |
| **Threshold 0.31** CNN-LSTM naming | Filename encodes classifier operating point | Prevents accidental apple-oranges KPI merges |
| **PHM unseen cutter split** | Train C1 + C4 → test **C6** | Stress-tests domain shift & transfer learning claims |

---

## 6. Versioning artefacts in Git LFS-less mode

Exported trees may exceed comfortable diff sizes yet remain **conceptually textual/binary reproducible**:

- `.joblib/.json/.npz/.pt/_weights.pt` co-located for mechanical audit.
- `organize_trained_models.py` cross-check presence so partial reruns expose missing artefacts early.

Alternative (not applied here): **Git LFS** or **Zenodo release tarballs**.

---

## 7. Naming artifacts with legacy suffixes (`_ctgan`)

Some filenames include **`_ctgan`** even when **CTGAN is not instantiated** inside the script path.

Reason: Harmonisation with richer experimental variants (generative augmentation) without breaking downstream inventory parsers.

Readers should infer “**release bundle identifier**”, not literal algorithm presence unless documented elsewhere.

---

## 8. Script responsibility matrix

| Script | Responsibility |
|--------|----------------|
| `prepare_phm2010_release_dataset.py` | Materialise reproducible CSVs |
| `train_fast_release_models.py` | Bare-metal numeric baselines |
| `train_ai4i_release_models.py` | sklearn / XGBoost exports |
| `train_torch_ai4i_deep_models.py` | Modern DL ablations on AI4I |
| `train_phm_release_models.py` | PHM multi-modal + transfer |
| `organize_trained_models.py` | Inventory audit |
| `build_results_bundle.py` | Spreadsheet synthesis |
| `run_all_training.py` | Orchestration + `PYTHONPATH` hygiene |

---

## 9. Non-goals (explicit)

- **No automatic hyperparameter search** — clarity > marginal leader-board delta.
- **No SHAP/LLM reporting in this package** — handled in extended article companion repositories.
- **No container image** — requirements are intentionally minimal to reduce supply-chain surface.

---

## 10. Future cleanups (maintainer backlog)

1. Add optional **`pyproject.toml`** for formal `pip install -e .` packaging.
2. Harmonise split utilities into one module to avoid duplicated `stratified_split` bodies.
3. Emit JSON metadata with **Git commit hash** on each training run for forensic provenance.

---

Questions or suggested extensions? Open a GitHub issue referencing the script + expected metric delta hypothesis.
