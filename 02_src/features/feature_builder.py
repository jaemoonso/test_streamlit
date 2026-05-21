from __future__ import annotations

from typing import Iterable

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    base_cols: Iterable[str],
    lags: Iterable[int] = (1, 3, 6),
) -> pd.DataFrame:
    out = df.copy()
    for col in base_cols:
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    base_cols: Iterable[str],
    windows: Iterable[int] = (3, 6),
) -> pd.DataFrame:
    out = df.copy()
    for col in base_cols:
        for w in windows:
            out[f"{col}_roll_mean_{w}"] = out[col].rolling(window=w).mean()
            out[f"{col}_roll_std_{w}"] = out[col].rolling(window=w).std()
    return out


def create_model_frame(
    df: pd.DataFrame,
    target_col: str = "% Silica Concentrate",
) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in {"date", target_col}]
    out = add_lag_features(df, feature_cols, lags=(1, 3))
    out = add_rolling_features(out, feature_cols, windows=(3,))
    out = out.dropna().reset_index(drop=True)
    return out
