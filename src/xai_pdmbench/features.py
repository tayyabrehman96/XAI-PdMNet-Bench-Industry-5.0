"""Feature builders for track B1 (repro) and B2 (leakage-safe contribution)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xai_pdmbench.constants import FAULT_COLUMNS


def build_features_b1(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Reproduction-oriented track: may retain TWF…RNF (potential label leakage).

    Returns
    -------
    X, y
        Float32 feature matrix and binary failure labels.
    """
    dd = d.copy()
    drop_ids = [c for c in ("UID", "UDI", "Product ID") if c in dd.columns]
    dd = dd.drop(columns=drop_ids, errors="ignore")
    y = dd["Machine failure"].astype(int).values
    dd = dd.drop(columns=["Machine failure"])
    cat_cols = [c for c in dd.columns if dd[c].dtype == object]
    X_num = dd.drop(columns=cat_cols, errors="ignore")
    if cat_cols:
        X_cat = pd.get_dummies(dd[cat_cols], drop_first=False)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = X_num.reset_index(drop=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.astype(np.float32), y


def build_features_b2(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Leakage-safe track: drops TWF…RNF and adds 17 process-informed scalars + Type OHE.

    Mirrors the release feature engineering used by the training scripts.
    """
    dd = d.copy()
    dd = dd.drop(columns=[c for c in ("UID", "UDI", "Product ID") if c in dd.columns], errors="ignore")
    dd = dd.drop(columns=[c for c in FAULT_COLUMNS if c in dd.columns], errors="ignore")
    y = dd["Machine failure"].astype(int).values
    dd = dd.drop(columns=["Machine failure"])

    torque_col = next((c for c in dd.columns if "Torque" in c), None)
    rpm_col = next((c for c in dd.columns if "Rotational speed" in c or "speed" in c.lower()), None)
    air_col = next((c for c in dd.columns if "air temperature" in c.lower()), None)
    proc_col = next((c for c in dd.columns if "process temperature" in c.lower()), None)
    wear_col = next((c for c in dd.columns if "tool wear" in c.lower()), None)

    t_num = pd.to_numeric(dd[torque_col], errors="coerce") if torque_col else None
    r_num = pd.to_numeric(dd[rpm_col], errors="coerce") if rpm_col else None
    a = pd.to_numeric(dd[air_col], errors="coerce") if air_col else None
    p = pd.to_numeric(dd[proc_col], errors="coerce") if proc_col else None
    w = pd.to_numeric(dd[wear_col], errors="coerce") if wear_col else None

    if t_num is not None and r_num is not None:
        power = t_num * r_num
        dd["Power_proxy"] = power
        dd["RPM_over_torque"] = r_num / t_num.clip(lower=1e-6)
        dd["Log1p_torque"] = np.log1p(np.maximum(t_num.fillna(0.0), 0.0))
        dd["Log1p_rpm"] = np.log1p(np.maximum(r_num.fillna(0.0), 0.0))
        dd["Log_power_proxy"] = np.log1p(np.maximum(power.fillna(0.0), 0.0))

    if w is not None:
        w_raw = w
        mx = float(np.nanmax(w_raw.to_numpy())) if w_raw.notna().any() else 1.0
        if not np.isfinite(mx) or mx <= 0:
            mx = 1.0
        w_safe = w_raw.fillna(0.0)
        dd["Wear_norm"] = w_safe / max(mx, 1e-6)
        dd["Log1p_wear"] = np.log1p(np.maximum(w_safe, 0.0))
        if t_num is not None and r_num is not None:
            dd["Torque_x_wear"] = t_num.fillna(0.0) * w_safe
            dd["Rpm_x_wear"] = r_num.fillna(0.0) * w_safe
            dd["Power_per_wear"] = (t_num * r_num) / (np.maximum(w_safe, 0.0) + 1.0)

    if a is not None and p is not None:
        aa = a.clip(lower=1.0)
        dtemp = p - a
        if t_num is not None:
            dd["Proc_x_torque"] = p * t_num
            dd["Temp_x_torque"] = dtemp * t_num
        dd["Delta_temp_K"] = dtemp
        dd["Thermal_ratio"] = dtemp / aa
        dd["Temp_product_K2"] = a * p
        dd["Inv_air_temp"] = 1.0 / aa
        dd["Delta_temp_sq"] = dtemp**2
        dd["Process_over_air"] = p / aa
        if w is not None:
            w_d = w.fillna(0.0).clip(lower=0.0)
            dd["Thermal_per_wear"] = dtemp / (w_d + 1.0)

    cat_cols = [c for c in dd.columns if dd[c].dtype == object]
    X_num = dd.drop(columns=cat_cols, errors="ignore")
    if cat_cols:
        X_cat = pd.get_dummies(dd[cat_cols], drop_first=False)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = X_num.reset_index(drop=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.astype(np.float32), y
