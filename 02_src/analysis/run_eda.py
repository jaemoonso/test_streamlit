from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "02_src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.mine_loader import build_training_table, load_raw_data, parse_mining_dataframe


TARGET_COL = "% Silica Concentrate"
REPORT_DIR = ROOT_DIR / "04_reports" / "eda_refresh"


def save_basic_profile(df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        ("rows", len(df)),
        ("columns", len(df.columns)),
        ("start_date", str(df["date"].min())),
        ("end_date", str(df["date"].max())),
        ("target_mean", float(df[TARGET_COL].mean())),
        ("target_std", float(df[TARGET_COL].std())),
        ("target_min", float(df[TARGET_COL].min())),
        ("target_max", float(df[TARGET_COL].max())),
    ]
    profile_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    profile_df.to_csv(REPORT_DIR / "basic_profile.csv", index=False)

    missing_df = (
        df.isna()
        .mean()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_ratio"})
        .sort_values("missing_ratio", ascending=False)
    )
    missing_df.to_csv(REPORT_DIR / "missing_ratio.csv", index=False)

    numeric_cols = [c for c in df.columns if c != "date"]
    corr = df[numeric_cols].corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    corr_df = corr.reset_index()
    corr_df.columns = ["feature", "corr_with_target"]
    corr_df["abs_corr"] = corr_df["corr_with_target"].abs()
    corr_df = corr_df.sort_values("abs_corr", ascending=False)
    corr_df.to_csv(REPORT_DIR / "target_correlation.csv", index=False)


def save_charts(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[TARGET_COL], kde=True, ax=ax, bins=60)
    ax.set_title("Target Distribution: % Silica Concentrate")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "target_distribution.png", dpi=140)
    plt.close(fig)

    sample = df.sample(min(50000, len(df)), random_state=42)
    numeric_cols = [c for c in sample.columns if c != "date"]
    corr = sample[numeric_cols].corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    top = corr.head(15).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax)
    ax.set_title("Top 15 |Correlation| with Target")
    ax.set_xlabel("|corr|")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "top15_abs_corr.png", dpi=140)
    plt.close(fig)

    ts = df[["date", TARGET_COL]].dropna().set_index("date").sort_index()
    ts_hourly = ts.resample("h").mean().rolling(24, min_periods=6).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts_hourly.index, ts_hourly[TARGET_COL], linewidth=1)
    ax.set_title("Smoothed Target Trend (Hourly + 24h rolling)")
    ax.set_ylabel(TARGET_COL)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "target_trend.png", dpi=140)
    plt.close(fig)

    hour_df = df.copy()
    hour_df["hour"] = hour_df["date"].dt.hour
    hourly = hour_df.groupby("hour")[TARGET_COL].agg(["mean", "std"]).reset_index()
    hourly.to_csv(REPORT_DIR / "target_by_hour.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=hourly, x="hour", y="mean", marker="o", ax=ax)
    ax.set_title("Average Target by Hour")
    ax.set_ylabel(TARGET_COL)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "target_by_hour.png", dpi=140)
    plt.close(fig)


def save_markdown_summary() -> None:
    profile = pd.read_csv(REPORT_DIR / "basic_profile.csv")
    missing = pd.read_csv(REPORT_DIR / "missing_ratio.csv")
    corr = pd.read_csv(REPORT_DIR / "target_correlation.csv")
    hour = pd.read_csv(REPORT_DIR / "target_by_hour.csv")

    top_missing = missing.head(10)
    top_corr = corr.head(10)
    peak_hour = hour.sort_values("mean", ascending=False).iloc[0]["hour"]
    low_hour = hour.sort_values("mean", ascending=True).iloc[0]["hour"]

    lines = [
        "# EDA Refresh Report",
        "",
        "## 1. Dataset Overview",
    ]
    for _, row in profile.iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")

    lines.extend(
        [
            "",
            "## 2. Missing Value Top 10",
            "",
            "| column | missing_ratio |",
            "|---|---:|",
        ]
    )
    for _, row in top_missing.iterrows():
        lines.append(f"| {row['column']} | {row['missing_ratio']:.4f} |")

    lines.extend(
        [
            "",
            "## 3. Top 10 Features by |corr| with Target",
            "",
            "| feature | corr_with_target |",
            "|---|---:|",
        ]
    )
    for _, row in top_corr.iterrows():
        lines.append(f"| {row['feature']} | {row['corr_with_target']:.4f} |")

    lines.extend(
        [
            "",
            "## 4. Time Pattern (Hour)",
            f"- Highest average target hour: {int(peak_hour)}",
            f"- Lowest average target hour: {int(low_hour)}",
            "",
            "## 5. Generated Figures",
            "- target_distribution.png",
            "- top15_abs_corr.png",
            "- target_trend.png",
            "- target_by_hour.png",
        ]
    )

    (REPORT_DIR / "EDA_REFRESH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw = load_raw_data()
    parsed = parse_mining_dataframe(raw)
    train_table = build_training_table(parsed, target_col=TARGET_COL)

    save_basic_profile(train_table)
    save_charts(train_table)
    save_markdown_summary()

    print(f"EDA refresh completed: {REPORT_DIR}")


if __name__ == "__main__":
    main()
