from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "02_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.mine_loader import build_training_table, load_raw_data, parse_mining_dataframe
from features.feature_builder import create_model_frame

TARGET = "% Silica Concentrate"
DATA_PATH = ROOT / "00_data" / "quality_prediction_mining_process" / "MiningProcess_Flotation_Plant_Database.csv"
EXP_DIR = ROOT / "04_reports" / "experiments"
OUT_DIR = ROOT / "04_reports" / "deploy"


def _auto_cast(v: str):
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d*", v):
        return float(v)
    return v


def parse_exp_id(exp_id: str) -> dict:
    out = {
        "model": "ridge",
        "params": {},
        "impute": "median",
        "outlier": "none",
        "sampling": "none",
        "scale": False,
        "target_transform": "none",
    }
    for part in exp_id.split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key == "model":
            out["model"] = value
        elif key == "params":
            params = {}
            for kv in value.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    params[k] = _auto_cast(v)
            out["params"] = params
        elif key == "impute":
            out["impute"] = value
        elif key == "scale":
            out["scale"] = value.lower() == "true"
        elif key == "target_transform":
            out["target_transform"] = value
    return out


def transform_target(y: pd.Series, mode: str) -> pd.Series:
    if mode == "log1p":
        return np.log1p(np.clip(y, a_min=0, a_max=None))
    return y


def inverse_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.expm1(y)
    return y


def build_model(cfg: dict) -> Pipeline:
    m = cfg["model"]
    p = cfg.get("params", {})
    if m == "random_forest":
        model = RandomForestRegressor(random_state=42, n_jobs=1, **p)
    elif m == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(random_state=42, n_jobs=1, objective="reg:squarederror", **p)
    elif m == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(random_state=42, n_jobs=1, **p)
    else:
        model = Ridge(**p)

    steps = [("imputer", SimpleImputer(strategy=cfg.get("impute", "median")))]
    if cfg.get("scale", False):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for p in EXP_DIR.glob("**/experiment_summary.csv"):
        run_name = p.parent.name if p.parent != EXP_DIR else "root"
        df = pd.read_csv(p)
        df["run_name"] = run_name
        all_summaries.append(df)
    summary = pd.concat(all_summaries, ignore_index=True).sort_values("test_rmse")
    best = summary.iloc[0]
    best_exp_id = str(best["exp_id"])
    best_cfg = parse_exp_id(best_exp_id)

    raw = load_raw_data(DATA_PATH)
    parsed = parse_mining_dataframe(raw)
    base = build_training_table(parsed, target_col=TARGET)
    frame = create_model_frame(base, target_col=TARGET)
    frame = frame.iloc[-50000:].reset_index(drop=True)

    split = int(len(frame) * 0.8)
    train_df = frame.iloc[:split].reset_index(drop=True)
    test_df = frame.iloc[split:].reset_index(drop=True)
    feats = [c for c in frame.columns if c not in {"date", TARGET}]
    x_train, y_train = train_df[feats], train_df[TARGET]
    x_test, y_test = test_df[feats], test_df[TARGET]

    model = build_model(best_cfg)
    y_train_model = transform_target(y_train, best_cfg.get("target_transform", "none"))
    model.fit(x_train, y_train_model)

    pred_model = model.predict(x_test)
    pred = inverse_target(pred_model, best_cfg.get("target_transform", "none"))
    err = pd.DataFrame({"actual": y_test.values, "pred": pred})
    err["abs_error"] = (err["actual"] - err["pred"]).abs()
    err["is_threshold_miss"] = (err["actual"] >= 3.0) != (err["pred"] >= 3.0)
    err["near_threshold_miss"] = err["is_threshold_miss"] & ((err["actual"] - 3.0).abs() < 0.2)
    err["large_margin_error"] = err["abs_error"] >= err["abs_error"].quantile(0.95)
    err.to_csv(OUT_DIR / "error_cases_enriched.csv", index=False)

    n_sample = min(5000, len(x_test))
    x_eval = x_test.sample(n_sample, random_state=42) if len(x_test) > n_sample else x_test
    y_eval = y_test.loc[x_eval.index]
    p = permutation_importance(model, x_eval, y_eval, n_repeats=5, random_state=42, scoring="neg_root_mean_squared_error")
    fi = pd.DataFrame(
        {"feature": x_eval.columns, "importance_mean": p.importances_mean, "importance_std": p.importances_std}
    ).sort_values("importance_mean", ascending=False)
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    top_feats = fi["feature"].head(10).tolist()
    low_feats = fi["feature"].tail(10).tolist()
    q = frame[TARGET].quantile(0.75)
    hi = frame[TARGET] >= q
    lo = frame[TARGET] < q
    rows = []
    for group, cols in [("important", top_feats), ("less_important", low_feats)]:
        for c in cols:
            a = frame.loc[hi, c].dropna()
            b = frame.loc[lo, c].dropna()
            if len(a) < 10 or len(b) < 10:
                continue
            pv = mannwhitneyu(a, b, alternative="two-sided").pvalue
            rows.append(
                {
                    "group": group,
                    "feature": c,
                    "pvalue": float(pv),
                    "median_diff_high_vs_low_target": float(a.median() - b.median()),
                }
            )
    pd.DataFrame(rows).sort_values(["group", "pvalue"]).to_csv(OUT_DIR / "important_vs_unimportant_stats.csv", index=False)

    with open(OUT_DIR / "deploy_model_context.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_exp_id": best_exp_id,
                "best_run_name": str(best["run_name"]),
                "best_test_rmse": float(best["test_rmse"]),
                "best_cv_rmse": float(best["rmse_mean"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Deploy artifacts generated in: {OUT_DIR}")


if __name__ == "__main__":
    main()
