from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "02_src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.mine_loader import (
    build_training_table,
    load_raw_data,
    parse_mining_dataframe,
    timeseries_split,
)
from features.feature_builder import create_model_frame


TARGET_COL = "% Silica Concentrate"
MODEL_DIR = ROOT_DIR / "03_models"
REPORT_DIR = ROOT_DIR / "04_reports"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(sample_rows: int | None = 150_000, rf_estimators: int = 120, rf_max_depth: int = 12) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    raw = load_raw_data()
    parsed = parse_mining_dataframe(raw)
    base = build_training_table(parsed, target_col=TARGET_COL)
    frame = create_model_frame(base, target_col=TARGET_COL)
    if sample_rows is not None and sample_rows > 0 and len(frame) > sample_rows:
        frame = frame.iloc[-sample_rows:].reset_index(drop=True)
        print(f"Using sampled tail rows for faster training: {len(frame):,}")
    else:
        print(f"Using full rows: {len(frame):,}")

    train_df, valid_df, test_df = timeseries_split(frame)

    features = [c for c in frame.columns if c not in {"date", TARGET_COL}]
    x_train, y_train = train_df[features], train_df[TARGET_COL]
    x_valid, y_valid = valid_df[features], valid_df[TARGET_COL]
    x_test, y_test = test_df[features], test_df[TARGET_COL]

    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=rf_estimators,
        max_depth=rf_max_depth,
        random_state=42,
        n_jobs=1,
    )

    print("Training Ridge...")
    ridge.fit(x_train, y_train)
    print("Training RandomForest...")
    rf.fit(x_train, y_train)

    models = {"ridge": ridge, "random_forest": rf}
    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for name, model in models.items():
        for split_name, x, y, src in [
            ("valid", x_valid, y_valid, valid_df),
            ("test", x_test, y_test, test_df),
        ]:
            pred = model.predict(x)
            m = regression_metrics(y.values, pred)
            rows.append(
                {
                    "model": name,
                    "split": split_name,
                    **m,
                }
            )

            split_pred = pd.DataFrame(
                {
                    "date": src["date"].values,
                    "actual": y.values,
                    "pred": pred,
                    "residual": y.values - pred,
                    "model": name,
                    "split": split_name,
                }
            )
            pred_rows.append(split_pred)

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    metrics_df = pd.DataFrame(rows).sort_values(["split", "rmse"]).reset_index(drop=True)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df["abs_error"] = pred_df["residual"].abs()

    best_test = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").iloc[0]
    summary = {
        "target": TARGET_COL,
        "best_model_test": best_test["model"],
        "best_model_test_rmse": float(best_test["rmse"]),
    }

    metrics_df.to_csv(REPORT_DIR / "metrics.csv", index=False)
    pred_df.to_csv(REPORT_DIR / "predictions.csv", index=False)
    with open(REPORT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training complete")
    print(metrics_df)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression models for mining quality prediction.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=150_000,
        help="Use latest N rows for faster training. Set 0 or negative for full data.",
    )
    parser.add_argument("--rf-estimators", type=int, default=120, help="RandomForest n_estimators.")
    parser.add_argument("--rf-max-depth", type=int, default=12, help="RandomForest max_depth.")
    args = parser.parse_args()
    sample_rows = None if args.sample_rows <= 0 else args.sample_rows
    train(
        sample_rows=sample_rows,
        rf_estimators=args.rf_estimators,
        rf_max_depth=args.rf_max_depth,
    )
