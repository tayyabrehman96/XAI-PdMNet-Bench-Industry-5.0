#!/usr/bin/env python3
"""Build reproducible PHM 2010 CSV datasets from bundled source arrays.

Inputs (real data-derived source arrays):
  data/phm2010/source/originfeature/data_x{1,4,6}.npy  # shape: (315, 6, 5000)
  data/phm2010/source/originfeature/data_y{1,4,6}.npy  # shape: (315,)

Outputs:
  data/phm2010/phm2010_windows_6x500.csv
  data/phm2010/phm2010_feature_table.csv
  data/phm2010/phm2010_dataset_status.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "phm2010" / "source" / "originfeature"
OUT_DIR = ROOT / "data" / "phm2010"
WINDOWS_OUT = OUT_DIR / "phm2010_windows_6x500.csv"
FEATURES_OUT = OUT_DIR / "phm2010_feature_table.csv"
STATUS_OUT = OUT_DIR / "phm2010_dataset_status.csv"
FAIL_THRESHOLD = 152.0
TARGET_WINDOW = 500
CHANNEL_NAMES = ("force_x", "force_y", "force_z", "vib_x", "vib_y", "vib_z")


def load_inputs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapping = {
        "C1": ("data_x1.npy", "data_y1.npy"),
        "C4": ("data_x4.npy", "data_y4.npy"),
        "C6": ("data_x6.npy", "data_y6.npy"),
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cutter, (x_name, y_name) in mapping.items():
        x_path = SRC / x_name
        y_path = SRC / y_name
        if not x_path.exists() or not y_path.exists():
            raise SystemExit(
                f"Missing PHM source arrays for {cutter}: {x_path.name}, {y_path.name}"
            )
        x = np.load(x_path)
        y = np.load(y_path)
        if x.ndim != 3 or x.shape[1] != 6:
            raise SystemExit(f"Unexpected shape for {x_path}: {x.shape}")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise SystemExit(f"Unexpected shape mismatch for {y_path}: {y.shape} vs {x.shape}")
        out[cutter] = (x.astype(np.float32), y.astype(np.float32))
    return out


def downsample_windows(x: np.ndarray, target_window: int = TARGET_WINDOW) -> np.ndarray:
    n, c, t = x.shape
    if t % target_window != 0:
        raise SystemExit(f"Cannot downsample window length {t} to {target_window}")
    factor = t // target_window
    return x.reshape(n, c, target_window, factor).mean(axis=3)


def skew_kurtosis(v: np.ndarray) -> tuple[float, float]:
    m = float(v.mean())
    centered = v - m
    m2 = float(np.mean(centered**2))
    if m2 < 1e-12:
        return 0.0, 0.0
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = m3 / (m2 ** 1.5 + 1e-12)
    kurt = m4 / (m2**2 + 1e-12)
    return skew, kurt


def per_channel_features(signal: np.ndarray) -> dict[str, float]:
    feats: dict[str, float] = {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        s = signal[ch_idx].astype(np.float64)
        mean = float(s.mean())
        std = float(s.std())
        rms = float(np.sqrt(np.mean(s**2)))
        mn = float(s.min())
        mx = float(s.max())
        p2p = mx - mn
        skew, kurt = skew_kurtosis(s)
        feats[f"{ch_name}_mean"] = mean
        feats[f"{ch_name}_std"] = std
        feats[f"{ch_name}_rms"] = rms
        feats[f"{ch_name}_min"] = mn
        feats[f"{ch_name}_max"] = mx
        feats[f"{ch_name}_p2p"] = p2p
        feats[f"{ch_name}_skew"] = skew
        feats[f"{ch_name}_kurtosis"] = kurt
    return feats


def build_rows(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows: list[dict[str, float | int | str]] = []
    feature_rows: list[dict[str, float | int | str]] = []
    flat_cols = [f"ch{ch}_{t:03d}" for ch in range(6) for t in range(TARGET_WINDOW)]

    for cutter, (x, y) in arrays.items():
        x_small = downsample_windows(x, TARGET_WINDOW)
        n_cuts = x.shape[0]
        for cut_idx in range(n_cuts):
            wear = float(y[cut_idx])
            fail = int(wear >= FAIL_THRESHOLD)

            flat = x_small[cut_idx].reshape(-1)
            w_row: dict[str, float | int | str] = {
                "cutter": cutter,
                "cut_index": cut_idx + 1,
                "wear": wear,
                "failure_label": fail,
            }
            for name, value in zip(flat_cols, flat, strict=True):
                w_row[name] = float(value)
            window_rows.append(w_row)

            f_row: dict[str, float | int | str] = {
                "cutter": cutter,
                "cut_index": cut_idx + 1,
                "wear": wear,
                "failure_label": fail,
            }
            f_row.update(per_channel_features(x[cut_idx]))
            feature_rows.append(f_row)

    windows_df = pd.DataFrame(window_rows)
    features_df = pd.DataFrame(feature_rows)
    return windows_df, features_df


def write_status(arrays: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    rows = []
    total_cuts = 0
    total_fail = 0
    for cutter, (_, y) in arrays.items():
        count = int(y.shape[0])
        fail = int((y >= FAIL_THRESHOLD).sum())
        rows.append(
            {
                "cutter": cutter,
                "cuts": count,
                "failure_threshold": FAIL_THRESHOLD,
                "failure_cuts": fail,
                "normal_cuts": count - fail,
            }
        )
        total_cuts += count
        total_fail += fail
    rows.append(
        {
            "cutter": "ALL",
            "cuts": total_cuts,
            "failure_threshold": FAIL_THRESHOLD,
            "failure_cuts": total_fail,
            "normal_cuts": total_cuts - total_fail,
        }
    )
    with STATUS_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cutter", "cuts", "failure_threshold", "failure_cuts", "normal_cuts"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = load_inputs()
    windows_df, features_df = build_rows(arrays)
    windows_df.to_csv(WINDOWS_OUT, index=False)
    features_df.to_csv(FEATURES_OUT, index=False)
    write_status(arrays)

    manifest = {
        "source_dir": str(SRC.relative_to(ROOT)).replace("\\", "/"),
        "failure_threshold": FAIL_THRESHOLD,
        "target_window": TARGET_WINDOW,
        "windows_csv": str(WINDOWS_OUT.relative_to(ROOT)).replace("\\", "/"),
        "feature_csv": str(FEATURES_OUT.relative_to(ROOT)).replace("\\", "/"),
        "rows_windows": int(len(windows_df)),
        "rows_features": int(len(features_df)),
    }
    (OUT_DIR / "phm2010_release_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Wrote", WINDOWS_OUT)
    print("Wrote", FEATURES_OUT)
    print("Wrote", STATUS_OUT)


if __name__ == "__main__":
    main()

