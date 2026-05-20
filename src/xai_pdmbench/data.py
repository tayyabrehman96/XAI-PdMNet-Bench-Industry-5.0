"""Load, normalize column names, and apply Ileri-style cleaning rules."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from xai_pdmbench.constants import AI4I_CSV_URL, FAULT_COLUMNS


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize common UCI column spellings (strip whitespace)."""
    return df.rename(columns={c: c.strip() for c in df.columns})


def clean_ai4i_basepaper(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Remove inconsistent Machine failure vs fault-flag rows (Ileri et al. narrative).

    Drops rows where: (RNF==1 and MF==0) OR (MF==1 and no fault flag set).
    """
    d = frame.copy()
    fault_cols = [c for c in FAULT_COLUMNS if c in d.columns]
    if len(fault_cols) != len(FAULT_COLUMNS):
        missing = set(FAULT_COLUMNS) - set(fault_cols)
        raise ValueError(f"Dataframe missing fault columns: {sorted(missing)}")

    mf = d["Machine failure"].astype(int)
    fc = list(FAULT_COLUMNS)
    faults_present = d[fc].astype(int).sum(axis=1) >= 1
    bad_rnf = (d["RNF"].astype(int) == 1) & (mf == 0)
    bad_mf = (mf == 1) & (~faults_present)
    mask = bad_rnf | bad_mf
    removed = int(mask.sum())
    out = d[~mask].reset_index(drop=True)
    print(f"clean_ai4i_basepaper: dropped {removed} inconsistent rows; remaining {len(out)}")
    return out


def read_ai4i_csv(path_or_url: str | Path | None = None) -> pd.DataFrame:
    """
    Read AI4I from a local path or from the official UCI HTTPS CSV.

    Parameters
    ----------
    path_or_url :
        If None, uses :data:`xai_pdmbench.constants.AI4I_CSV_URL`.
    """
    src = str(path_or_url) if path_or_url else AI4I_CSV_URL
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src) as resp:  # noqa: S310 — intentional UCI URL
            return normalize_columns(pd.read_csv(resp))
    return normalize_columns(pd.read_csv(src))
