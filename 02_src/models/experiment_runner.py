from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "02_src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.mine_loader import build_training_table, load_raw_data, parse_mining_dataframe
from features.feature_builder import create_model_frame

TARGET_COL = "% Silica Concentrate"
BASE_REPORT_DIR = ROOT_DIR / "04_reports" / "experiments"


@dataclass
class ExperimentConfig:
    model_name: str
    model_params: dict[str, Any]
    impute_strategy: str
    outlier_strategy: str
    sampling_strategy: str
    scale: bool
    target_transform: str

    @property
    def exp_id(self) -> str:
        params = ",".join([f"{k}={v}" for k, v in self.model_params.items()])
        return (
            f"model={self.model_name}|params={params}|impute={self.impute_strategy}"
            f"|outlier={self.outlier_strategy}|sampling={self.sampling_strategy}|scale={self.scale}"
            f"|target_transform={self.target_transform}"
        )


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _target_bins(y: pd.Series, n_bins: int = 10) -> pd.Series:
    n_unique = y.nunique()
    bins = min(max(3, n_bins), int(n_unique)) if n_unique > 1 else 2
    return pd.qcut(y, q=bins, duplicates="drop", labels=False)


def _clip_outliers(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strategy == "none":
        return x_train, x_valid

    x_train_out = x_train.copy()
    x_valid_out = x_valid.copy()
    numeric_cols = x_train_out.columns.tolist()

    for col in numeric_cols:
        s = x_train_out[col]
        if strategy == "iqr_clip":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
        elif strategy == "quantile_1_99":
            low = s.quantile(0.01)
            high = s.quantile(0.99)
        else:
            raise ValueError(f"Unknown outlier strategy: {strategy}")

        x_train_out[col] = x_train_out[col].clip(low, high)
        x_valid_out[col] = x_valid_out[col].clip(low, high)

    return x_train_out, x_valid_out


def _apply_sampling(
    x: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    random_state: int,
    threshold_value: float,
    threshold_weight: float,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None]:
    if strategy == "none":
        return x, y, None

    y_bins = _target_bins(y, n_bins=10)
    if strategy == "target_bin_weighted":
        freq = y_bins.value_counts().to_dict()
        weights = y_bins.map(lambda b: 1.0 / freq.get(b, 1)).values
        weights = weights / np.mean(weights)
        return x, y, weights
    if strategy == "threshold_weighted":
        # Put more emphasis on decision-critical target region.
        near = (y - threshold_value).abs() <= 0.2
        high_risk = y >= threshold_value
        weights = np.ones(len(y), dtype=float)
        weights[near.values] *= threshold_weight
        weights[high_risk.values] *= (threshold_weight * 0.7)
        weights = weights / np.mean(weights)
        return x, y, weights

    if strategy == "undersample_major_bins":
        rng = np.random.default_rng(random_state)
        idx_keep: list[int] = []
        min_size = y_bins.value_counts().min()
        for b, idx in y_bins.groupby(y_bins).groups.items():
            idx_arr = np.array(list(idx))
            choose = min(min_size * 2, len(idx_arr))
            picked = rng.choice(idx_arr, size=choose, replace=False)
            idx_keep.extend(picked.tolist())
        idx_keep = sorted(idx_keep)
        return x.loc[idx_keep].reset_index(drop=True), y.loc[idx_keep].reset_index(drop=True), None

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _make_model(name: str, params: dict[str, Any]):
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(random_state=42, n_jobs=1, **params)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise RuntimeError("xgboost is not installed") from e
        return XGBRegressor(random_state=42, n_jobs=1, objective="reg:squarederror", **params)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise RuntimeError("lightgbm is not installed") from e
        return LGBMRegressor(random_state=42, n_jobs=1, **params)
    raise ValueError(f"Unknown model: {name}")


def _make_imputer(strategy: str):
    if strategy == "median":
        return SimpleImputer(strategy="median")
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    if strategy == "knn":
        return KNNImputer(n_neighbors=5)
    raise ValueError(f"Unknown imputer strategy: {strategy}")


def _train_one_fold(
    cfg: ExperimentConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int,
    threshold_value: float,
    threshold_weight: float,
) -> tuple[dict[str, float], np.ndarray]:
    x_train2, y_train2, sample_weight = _apply_sampling(
        x_train,
        y_train,
        cfg.sampling_strategy,
        random_state=random_state,
        threshold_value=threshold_value,
        threshold_weight=threshold_weight,
    )
    x_train2, x_valid2 = _clip_outliers(x_train2, x_valid, cfg.outlier_strategy)

    steps: list[tuple[str, Any]] = [("imputer", _make_imputer(cfg.impute_strategy))]
    if cfg.scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", _make_model(cfg.model_name, cfg.model_params)))
    pipe = Pipeline(steps=steps)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    y_train_model = _transform_target(y_train2, cfg.target_transform)
    pipe.fit(x_train2, y_train_model, **fit_kwargs)
    pred_model = pipe.predict(x_valid2)
    pred = _inverse_transform_target(pred_model, cfg.target_transform)
    return _metric_dict(y_valid.values, pred), pred


def _transform_target(y: pd.Series, mode: str) -> pd.Series:
    if mode == "none":
        return y
    if mode == "log1p":
        return np.log1p(np.clip(y, a_min=0, a_max=None))
    raise ValueError(f"Unknown target transform mode: {mode}")


def _inverse_transform_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "log1p":
        return np.expm1(y)
    raise ValueError(f"Unknown target transform mode: {mode}")


def run_experiments(
    sample_rows: int = 120_000,
    seed: int = 42,
    max_experiments: int | None = None,
    models: list[str] | None = None,
    run_name: str = "default",
    quick_mode: bool = False,
    cv_mode: str = "stratified",
    target_transform: str = "none",
    threshold_value: float = 3.0,
    threshold_weight: float = 3.0,
    sampling_modes: list[str] | None = None,
) -> None:
    report_dir = BASE_REPORT_DIR / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    parsed = parse_mining_dataframe(raw)
    base = build_training_table(parsed, target_col=TARGET_COL)
    frame = create_model_frame(base, target_col=TARGET_COL)
    if sample_rows > 0 and len(frame) > sample_rows:
        frame = frame.iloc[-sample_rows:].reset_index(drop=True)

    # Holdout test for final "OOF-style test" aggregation
    split = int(len(frame) * 0.8)
    train_df = frame.iloc[:split].reset_index(drop=True)
    test_df = frame.iloc[split:].reset_index(drop=True)

    feature_cols = [c for c in frame.columns if c not in {"date", TARGET_COL}]
    x_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    x_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    if cv_mode == "stratified":
        y_bins = _target_bins(y_train, n_bins=10)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_splits = list(cv.split(x_train, y_bins))
    elif cv_mode == "timeseries":
        cv = TimeSeriesSplit(n_splits=5)
        cv_splits = list(cv.split(x_train))
    else:
        raise ValueError("cv_mode must be one of: stratified, timeseries")

    if quick_mode:
        model_space_all: dict[str, list[dict[str, Any]]] = {
            "ridge": [{"alpha": a} for a in [0.1, 1.0]],
            "random_forest": [
                {"n_estimators": 40, "max_depth": 8, "min_samples_leaf": 2},
            ],
            "xgboost": [
                {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9},
            ],
            "lightgbm": [
                {"n_estimators": 80, "num_leaves": 31, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9},
            ],
        }
    else:
        model_space_all = {
            "ridge": [{"alpha": a} for a in [0.1, 1.0, 5.0]],
            "random_forest": [
                {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 1},
                {"n_estimators": 200, "max_depth": 14, "min_samples_leaf": 2},
            ],
            "xgboost": [
                {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
                {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
            ],
            "lightgbm": [
                {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
                {"n_estimators": 400, "num_leaves": 63, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
            ],
        }
    if models is None or len(models) == 0:
        model_space = model_space_all
    else:
        model_space = {k: v for k, v in model_space_all.items() if k in set(models)}
        if not model_space:
            raise ValueError(f"No valid models selected: {models}")

    impute_space = ["median", "knn"]
    outlier_space = ["none", "iqr_clip", "quantile_1_99"]
    sampling_space = sampling_modes or ["none", "target_bin_weighted", "undersample_major_bins"]
    scale_space = [False, True]

    baseline = ExperimentConfig(
        model_name="ridge",
        model_params={"alpha": 1.0},
        impute_strategy="median",
        outlier_strategy="none",
        sampling_strategy="none",
        scale=True,
        target_transform=target_transform,
    )

    configs: list[ExperimentConfig] = [baseline]
    for model_name, param_list in model_space.items():
        for params, impute, outlier, sampling, scale in itertools.product(
            param_list, impute_space, outlier_space, sampling_space, scale_space
        ):
            cfg = ExperimentConfig(
                model_name=model_name,
                model_params=params,
                impute_strategy=impute,
                outlier_strategy=outlier,
                sampling_strategy=sampling,
                scale=scale,
                target_transform=target_transform,
            )
            if cfg.exp_id != baseline.exp_id:
                configs.append(cfg)
    if max_experiments is not None and max_experiments > 0:
        configs = configs[:max_experiments]

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    test_pred_rows: list[pd.DataFrame] = []

    for exp_idx, cfg in enumerate(configs, start=1):
        try:
            print(f"[{exp_idx}/{len(configs)}] {cfg.exp_id}")
            fold_test_preds = []
            for fold, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
                x_tr = x_train.iloc[tr_idx].reset_index(drop=True)
                y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
                x_va = x_train.iloc[va_idx].reset_index(drop=True)
                y_va = y_train.iloc[va_idx].reset_index(drop=True)

                metrics, pred_va = _train_one_fold(
                    cfg,
                    x_tr,
                    y_tr,
                    x_va,
                    y_va,
                    random_state=seed + fold,
                    threshold_value=threshold_value,
                    threshold_weight=threshold_weight,
                )

                fold_rows.append(
                    {
                        "exp_id": cfg.exp_id,
                        "fold": fold,
                        **metrics,
                    }
                )
                oof_rows.append(
                    pd.DataFrame(
                        {
                            "exp_id": cfg.exp_id,
                            "row_index": va_idx,
                            "actual": y_va.values,
                            "pred": pred_va,
                            "fold": fold,
                        }
                    )
                )

                # Fold model for test inference (OOF-like ensemble)
                x_tr2, y_tr2, sample_weight = _apply_sampling(
                    x_tr,
                    y_tr,
                    cfg.sampling_strategy,
                    random_state=seed + fold,
                    threshold_value=threshold_value,
                    threshold_weight=threshold_weight,
                )
                x_tr2, x_te2 = _clip_outliers(x_tr2, x_test, cfg.outlier_strategy)
                steps: list[tuple[str, Any]] = [("imputer", _make_imputer(cfg.impute_strategy))]
                if cfg.scale:
                    steps.append(("scaler", StandardScaler()))
                steps.append(("model", _make_model(cfg.model_name, cfg.model_params)))
                model = Pipeline(steps=steps)
                fit_kwargs = {}
                if sample_weight is not None:
                    fit_kwargs["model__sample_weight"] = sample_weight
                y_tr_model = _transform_target(y_tr2, cfg.target_transform)
                model.fit(x_tr2, y_tr_model, **fit_kwargs)
                pred_test_model = model.predict(x_te2)
                fold_test_preds.append(_inverse_transform_target(pred_test_model, cfg.target_transform))

            test_pred_mean = np.mean(np.vstack(fold_test_preds), axis=0)
            test_m = _metric_dict(y_test.values, test_pred_mean)
            test_pred_rows.append(
                pd.DataFrame(
                    {
                        "exp_id": cfg.exp_id,
                        "actual": y_test.values,
                        "pred": test_pred_mean,
                    }
                )
            )
            fold_rows.append(
                {
                    "exp_id": cfg.exp_id,
                    "fold": 0,
                    "rmse": test_m["rmse"],
                    "mae": test_m["mae"],
                    "r2": test_m["r2"],
                }
            )
        except Exception as e:
            fold_rows.append(
                {
                    "exp_id": cfg.exp_id,
                    "fold": -1,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "r2": np.nan,
                    "error": str(e),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    test_pred_df = pd.concat(test_pred_rows, ignore_index=True) if test_pred_rows else pd.DataFrame()
    fold_df.to_csv(report_dir / "cv_fold_metrics.csv", index=False)
    oof_df.to_csv(report_dir / "oof_predictions.csv", index=False)
    test_pred_df.to_csv(report_dir / "test_predictions.csv", index=False)

    cv_only = fold_df[fold_df["fold"] > 0].copy()
    summary = (
        cv_only.groupby("exp_id")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["exp_id", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std"]
    holdout = fold_df[fold_df["fold"] == 0][["exp_id", "rmse", "mae", "r2"]].rename(
        columns={"rmse": "test_rmse", "mae": "test_mae", "r2": "test_r2"}
    )
    summary = summary.merge(holdout, on="exp_id", how="left").sort_values("rmse_mean")
    summary.to_csv(report_dir / "experiment_summary.csv", index=False)

    # Statistical significance vs baseline on fold RMSE
    base_scores = cv_only[cv_only["exp_id"] == baseline.exp_id].sort_values("fold")["rmse"].values
    sig_rows = []
    for exp_id in summary["exp_id"]:
        this_scores = cv_only[cv_only["exp_id"] == exp_id].sort_values("fold")["rmse"].values
        if len(this_scores) != len(base_scores) or len(this_scores) == 0:
            continue
        if exp_id == baseline.exp_id:
            sig_rows.append(
                {
                    "exp_id": exp_id,
                    "mean_diff_rmse_vs_baseline": 0.0,
                    "ttest_pvalue": 1.0,
                    "wilcoxon_pvalue": 1.0,
                    "is_significant_0_05": False,
                }
            )
            continue

        t_p = ttest_rel(this_scores, base_scores, alternative="less").pvalue
        try:
            w_p = wilcoxon(this_scores, base_scores, alternative="less").pvalue
        except Exception:
            w_p = np.nan
        diff = float(np.mean(this_scores - base_scores))
        sig_rows.append(
            {
                "exp_id": exp_id,
                "mean_diff_rmse_vs_baseline": diff,
                "ttest_pvalue": float(t_p),
                "wilcoxon_pvalue": float(w_p) if not np.isnan(w_p) else np.nan,
                "is_significant_0_05": bool((t_p < 0.05) or (not np.isnan(w_p) and w_p < 0.05)),
            }
        )

    sig_df = pd.DataFrame(sig_rows).sort_values(["is_significant_0_05", "mean_diff_rmse_vs_baseline"], ascending=[False, True])
    sig_df.to_csv(report_dir / "significance_vs_baseline.csv", index=False)

    best = summary.iloc[0].to_dict() if len(summary) else {}
    with open(report_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved in: {report_dir}")
    if best:
        print(f"Best exp_id: {best['exp_id']}")
        print(f"CV RMSE mean: {best['rmse_mean']:.4f} | Test RMSE: {best['test_rmse']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robust comparative ML experiments with statistical tests.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=120000,
        help="Use latest N rows for faster iteration. Use <=0 for full data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=0,
        help="Limit number of experiments for quick smoke run. 0 means all.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Subset models to run. Choices: ridge random_forest xgboost lightgbm",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Output subdirectory name under 04_reports/experiments",
    )
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Use smaller model parameter sets for faster benchmarking.",
    )
    parser.add_argument(
        "--cv-mode",
        type=str,
        default="stratified",
        choices=["stratified", "timeseries"],
        help="Cross-validation mode.",
    )
    parser.add_argument(
        "--target-transform",
        type=str,
        default="none",
        choices=["none", "log1p"],
        help="Target transform for training/inference.",
    )
    parser.add_argument("--threshold-value", type=float, default=3.0, help="Target threshold for weighted training.")
    parser.add_argument("--threshold-weight", type=float, default=3.0, help="Weight multiplier near threshold.")
    parser.add_argument(
        "--sampling-modes",
        nargs="+",
        default=[],
        help="Sampling strategies to use. Example: none threshold_weighted",
    )
    args = parser.parse_args()
    run_experiments(
        sample_rows=args.sample_rows,
        seed=args.seed,
        max_experiments=(args.max_experiments if args.max_experiments > 0 else None),
        models=args.models,
        run_name=args.run_name,
        quick_mode=args.quick_mode,
        cv_mode=args.cv_mode,
        target_transform=args.target_transform,
        threshold_value=args.threshold_value,
        threshold_weight=args.threshold_weight,
        sampling_modes=(args.sampling_modes if len(args.sampling_modes) > 0 else None),
    )
