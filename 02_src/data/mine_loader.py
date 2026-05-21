from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


DATA_PATH = Path("00_data/quality_prediction_mining_process/MiningProcess_Flotation_Plant_Database.csv")


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def parse_mining_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for col in out.columns:
        if col == "date":
            continue
        out[col] = (
            out[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
            .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
        )
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["date"].dt.hour
    out["dayofweek"] = out["date"].dt.dayofweek
    return out


def build_training_table(
    df: pd.DataFrame,
    target_col: str = "% Silica Concentrate",
) -> pd.DataFrame:
    out = add_time_features(df)
    keep_cols = [c for c in out.columns if c != "% Iron Concentrate" or c == target_col]
    out = out[keep_cols]
    out = out.dropna(subset=[target_col]).reset_index(drop=True)
    return out


def timeseries_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()
    return train_df, valid_df, test_df
