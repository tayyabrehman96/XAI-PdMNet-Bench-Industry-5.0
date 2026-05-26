# Trained models — download links

Repository: **[XAI-PdMNet-Bench-Industry-5.0](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0)**

Use **blob** links to preview on GitHub; use **raw** links for direct HTTPS download (`curl -L -O <url>`). Paths assume branch **`main`**.

Inventory after a full run (which files exist, byte sizes):  
[`trained_models/ALL_TRAINED_MODELS_INVENTORY.csv`](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ALL_TRAINED_MODELS_INVENTORY.csv)

---

## AI4I (`trained_models/ai4i/`)

| Model | Produced by script | GitHub · Raw |
|-------|---------------------|---------------|
| `logistic_regression.joblib` | `scripts/train_ai4i_release_models.py` | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/logistic_regression.joblib) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/logistic_regression.joblib) |
| `random_forest_ctgan.joblib` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/random_forest_ctgan.joblib) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/random_forest_ctgan.joblib) |
| `hist_gradient_boosting_ctgan.joblib` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/hist_gradient_boosting_ctgan.joblib) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/hist_gradient_boosting_ctgan.joblib) |
| `xgboost_ctgan.json` | same *(optional if XGBoost import fails)* | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/xgboost_ctgan.json) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/xgboost_ctgan.json) |
| `numpy_logistic_weighted.npz` | `scripts/train_fast_release_models.py` | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/numpy_logistic_weighted.npz) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/numpy_logistic_weighted.npz) |
| `numpy_gaussian_nb.npz` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/numpy_gaussian_nb.npz) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/numpy_gaussian_nb.npz) |
| `numpy_knn_k5.npz` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/numpy_knn_k5.npz) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/numpy_knn_k5.npz) |
| `alexnet1d_safe.pt` | `scripts/train_torch_ai4i_deep_models.py` | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/alexnet1d_safe.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/alexnet1d_safe.pt) |
| `alexnet1d_safe_weights.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/alexnet1d_safe_weights.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/alexnet1d_safe_weights.pt) |
| `tabtransformer.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/tabtransformer.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/tabtransformer.pt) |
| `tabtransformer_weights.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/tabtransformer_weights.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/tabtransformer_weights.pt) |
| `cnn_lstm_thr031.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/cnn_lstm_thr031.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/cnn_lstm_thr031.pt) |
| `cnn_lstm_thr031_weights.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/cnn_lstm_thr031_weights.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/cnn_lstm_thr031_weights.pt) |
| **`ai4i_feature_columns.json`** | sklearn / XGB export | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/ai4i_feature_columns.json) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/ai4i_feature_columns.json) |
| **`fast_feature_columns.json`** | NumPy baseline export | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/ai4i/fast_feature_columns.json) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/ai4i/fast_feature_columns.json) |

---

## PHM 2010 (`trained_models/phm2010/`)

| Model | Produced by script | GitHub · Raw |
|-------|---------------------|---------------|
| `xgboost_phm.json` | `scripts/train_phm_release_models.py` | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/phm2010/xgboost_phm.json) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/phm2010/xgboost_phm.json) |
| `cnn1d_phm.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/phm2010/cnn1d_phm.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/phm2010/cnn1d_phm.pt) |
| `cnn1d_phm_weights.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/phm2010/cnn1d_phm_weights.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/phm2010/cnn1d_phm_weights.pt) |
| `rcnn_phm.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/phm2010/rcnn_phm.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/phm2010/rcnn_phm.pt) |
| `rcnn_phm_weights.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/phm2010/rcnn_phm_weights.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/phm2010/rcnn_phm_weights.pt) |
| `transfer_ai4i_to_phm.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/phm2010/transfer_ai4i_to_phm.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/phm2010/transfer_ai4i_to_phm.pt) |
| `transfer_ai4i_to_phm_weights.pt` | same | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/phm2010/transfer_ai4i_to_phm_weights.pt) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/phm2010/transfer_ai4i_to_phm_weights.pt) |
| **`phm_feature_columns.json`** | PHM tabular exports | [view](https://github.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/blob/main/trained_models/phm2010/phm_feature_columns.json) · [download](https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/trained_models/phm2010/phm_feature_columns.json) |

---

## Raw URL template

Replace `<path-under-repo>` with the POSIX path **after** the repository root:

`https://raw.githubusercontent.com/tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/main/<path-under-repo>`

Example: `<path-under-repo>` = `trained_models/ai4i/xgboost_ctgan.json`.

Some files may be **missing** until you run training (Git shows them only if committed). Regenerate everything with **`python scripts/run_all_training.py`**.
