from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "04_reports"


def _ensure_shift_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if "shift" not in out.columns and date_col in out.columns:
        hours = out[date_col].dt.hour
        out["shift"] = pd.cut(
            hours,
            bins=[-1, 7, 15, 23],
            labels=["night", "day", "evening"],
        ).astype(str)
    return out


def _summarize_and_save(high_error: pd.DataFrame, prefix: str) -> None:
    hourly = high_error.groupby(high_error["date"].dt.hour)["abs_error"].mean().reset_index()
    daily = high_error.groupby(high_error["date"].dt.dayofweek)["abs_error"].mean().reset_index()

    high_error.sort_values("abs_error", ascending=False).head(2000).to_csv(
        REPORT_DIR / f"{prefix}_cases.csv", index=False
    )
    hourly.to_csv(REPORT_DIR / f"{prefix}_by_hour.csv", index=False)
    daily.to_csv(REPORT_DIR / f"{prefix}_by_dayofweek.csv", index=False)


def _pick_best_test_model(pred: pd.DataFrame) -> str:
    return (
        pred[pred["split"] == "test"]
        .groupby("model")["abs_error"]
        .mean()
        .sort_values()
        .index[0]
    )


def run_error_analysis(
    mode: str = "top_ratio",
    top_ratio: float = 0.1,
    abs_error_threshold: float = 0.8,
    silica_threshold: float = 3.0,
    segment_ratio: float = 0.05,
    segment_cols: tuple[str, ...] = ("shift",),
) -> None:
    pred_path = REPORT_DIR / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError("predictions.csv not found. Run training first.")

    pred = pd.read_csv(pred_path, parse_dates=["date"])
    test_best = _pick_best_test_model(pred)
    subset = pred[(pred["split"] == "test") & (pred["model"] == test_best)].copy()
    subset = _ensure_shift_column(subset)

    if mode == "top_ratio":
        cutoff = subset["abs_error"].quantile(1 - top_ratio)
        high_error = subset[subset["abs_error"] >= cutoff].copy()
        prefix = "high_error_top_ratio"
    elif mode == "absolute":
        high_error = subset[subset["abs_error"] >= abs_error_threshold].copy()
        prefix = "high_error_absolute"
    elif mode == "quality":
        quality_mask = subset["actual"] >= silica_threshold
        # Quality risk first, then sort by large error in that risky region.
        high_error = subset[quality_mask].copy()
        high_error = high_error.sort_values("abs_error", ascending=False)
        prefix = "high_error_quality"
    elif mode == "segmented_top":
        valid_cols = [c for c in segment_cols if c in subset.columns]
        if not valid_cols:
            raise ValueError(f"No valid segment columns found in prediction file: {segment_cols}")

        def pick_top(group: pd.DataFrame) -> pd.DataFrame:
            k = max(1, int(len(group) * segment_ratio))
            return group.nlargest(k, "abs_error")

        high_error = subset.groupby(valid_cols, dropna=False, group_keys=False).apply(pick_top)
        high_error = high_error.reset_index(drop=True)
        prefix = "high_error_segmented_top"
    else:
        raise ValueError("mode must be one of: top_ratio, absolute, quality, segmented_top")

    _summarize_and_save(high_error, prefix=prefix)

    print(f"Best test model: {test_best}")
    print(f"Mode: {mode}")
    print(f"High-error rows saved: {len(high_error)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible high-error analysis for regression tasks.")
    parser.add_argument(
        "--mode",
        type=str,
        default="top_ratio",
        choices=["top_ratio", "absolute", "quality", "segmented_top"],
        help="Selection strategy for high-error cases.",
    )
    parser.add_argument("--top-ratio", type=float, default=0.1, help="Ratio used in top_ratio mode.")
    parser.add_argument(
        "--abs-error-threshold",
        type=float,
        default=0.8,
        help="Absolute error threshold used in absolute mode.",
    )
    parser.add_argument(
        "--silica-threshold",
        type=float,
        default=3.0,
        help="Actual silica threshold used in quality mode.",
    )
    parser.add_argument(
        "--segment-ratio",
        type=float,
        default=0.05,
        help="Per-segment top ratio used in segmented_top mode.",
    )
    parser.add_argument(
        "--segment-cols",
        nargs="+",
        default=["shift"],
        help="Segment columns for segmented_top mode. Example: shift line_id",
    )
    args = parser.parse_args()
    run_error_analysis(
        mode=args.mode,
        top_ratio=args.top_ratio,
        abs_error_threshold=args.abs_error_threshold,
        silica_threshold=args.silica_threshold,
        segment_ratio=args.segment_ratio,
        segment_cols=tuple(args.segment_cols),
    )
