from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "00_data" / "quality_prediction_mining_process" / "MiningProcess_Flotation_Plant_Database.csv"
REPORT_DIR = ROOT_DIR / "04_reports"
EXPERIMENTS_DIR = REPORT_DIR / "experiments"
TARGET_COL = "% Silica Concentrate"
SRC_DIR = ROOT_DIR / "02_src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.mine_loader import build_training_table, load_raw_data, parse_mining_dataframe
from features.feature_builder import create_model_frame


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@st.cache_data
def load_all_experiment_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    sig_rows = []
    for summary_path in EXPERIMENTS_DIR.glob("**/experiment_summary.csv"):
        run_name = summary_path.parent.name if summary_path.parent != EXPERIMENTS_DIR else "root"
        sdf = pd.read_csv(summary_path)
        sdf["run_name"] = run_name
        summary_rows.append(sdf)

        sig_path = summary_path.parent / "significance_vs_baseline.csv"
        if sig_path.exists():
            gdf = pd.read_csv(sig_path)
            gdf["run_name"] = run_name
            sig_rows.append(gdf)

    all_summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    all_sig = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame()
    return all_summary, all_sig


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
            if value.strip():
                for kv in value.split(","):
                    if "=" not in kv:
                        continue
                    k, v = kv.split("=", 1)
                    params[k] = _auto_cast(v)
            out["params"] = params
        elif key == "impute":
            out["impute"] = value
        elif key == "outlier":
            out["outlier"] = value
        elif key == "sampling":
            out["sampling"] = value
        elif key == "scale":
            out["scale"] = value.lower() == "true"
        elif key == "target_transform":
            out["target_transform"] = value
    return out


def _auto_cast(v: str):
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d*", v):
        return float(v)
    return v


def build_model(parsed: dict) -> Pipeline:
    model_name = parsed["model"]
    params = parsed.get("params", {})
    if model_name == "ridge":
        model = Ridge(**params)
    elif model_name == "random_forest":
        model = RandomForestRegressor(random_state=42, n_jobs=1, **params)
    elif model_name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(random_state=42, n_jobs=1, objective="reg:squarederror", **params)
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(random_state=42, n_jobs=1, **params)
    else:
        model = Ridge(alpha=1.0)

    steps = [("imputer", SimpleImputer(strategy=parsed.get("impute", "median")))]
    if parsed.get("scale", False):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def _transform_target(y: pd.Series, mode: str) -> pd.Series:
    if mode == "none":
        return y
    if mode == "log1p":
        return np.log1p(np.clip(y, a_min=0, a_max=None))
    return y


def _inverse_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "log1p":
        return np.expm1(y)
    return y


@st.cache_data
def load_model_frame(sample_rows: int = 50000) -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    raw = load_raw_data(DATA_PATH)
    parsed = parse_mining_dataframe(raw)
    base = build_training_table(parsed, target_col=TARGET_COL)
    frame = create_model_frame(base, target_col=TARGET_COL)
    if sample_rows > 0 and len(frame) > sample_rows:
        frame = frame.iloc[-sample_rows:].reset_index(drop=True)
    return frame


@st.cache_data
def fit_best_model_and_extract_features(best_exp_id: str) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    frame = load_model_frame(sample_rows=60000)
    if frame.empty:
        raise FileNotFoundError(
            f"Raw dataset not found at `{DATA_PATH}`. "
            "Tabs 2/3 require source data to retrain local explainer models."
        )
    split = int(len(frame) * 0.8)
    train_df = frame.iloc[:split].reset_index(drop=True)
    test_df = frame.iloc[split:].reset_index(drop=True)

    feature_cols = [c for c in frame.columns if c not in {"date", TARGET_COL}]
    x_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    x_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    parsed = parse_exp_id(best_exp_id)
    model = build_model(parsed)
    y_train_model = _transform_target(y_train, parsed.get("target_transform", "none"))
    model.fit(x_train, y_train_model)
    return model, x_train, y_train, x_test, y_test


@st.cache_data
def compute_feature_importance(best_exp_id: str) -> pd.DataFrame:
    model, _, _, x_test, y_test = fit_best_model_and_extract_features(best_exp_id)
    n_sample = min(5000, len(x_test))
    x_eval = x_test.sample(n_sample, random_state=42) if len(x_test) > n_sample else x_test
    y_eval = y_test.loc[x_eval.index]
    p = permutation_importance(model, x_eval, y_eval, n_repeats=5, random_state=42, scoring="neg_root_mean_squared_error")
    out = pd.DataFrame(
        {
            "feature": x_eval.columns,
            "importance_mean": p.importances_mean,
            "importance_std": p.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return out.reset_index(drop=True)


def experiment_insights(summary: pd.DataFrame, sig: pd.DataFrame) -> tuple[str, str]:
    if summary.empty:
        return "실험 결과가 없습니다.", "실험 실행 후 다시 확인하세요."

    best = summary.sort_values("test_rmse").iloc[0]
    n_sig = int(sig["is_significant_0_05"].sum()) if not sig.empty and "is_significant_0_05" in sig.columns else 0
    text = (
        f"- 현재 test RMSE 기준 최고 조합은 `{best['exp_id']}` 입니다.\n"
        f"- CV 평균 RMSE `{best['rmse_mean']:.4f}`, test RMSE `{best['test_rmse']:.4f}`입니다.\n"
        f"- 통계 검정에서 유의 개선 조합 수는 `{n_sig}`개입니다.\n"
    )
    next_steps = (
        "- CV-우수 / Test-우수 조합 간 격차가 있는지 재검증(시간 분할 재설계).\n"
        "- 상위 3개 모델 앙상블(가중 평균) 실험.\n"
        "- outlier/결측 처리 조합을 고정하고 모델 하이퍼파라미터만 확장 탐색.\n"
        "- 목적함수를 RMSE/MAE 혼합 또는 품질 구간 가중으로 변경해 재평가.\n"
    )
    return text, next_steps


def compute_important_vs_unimportant_stats(frame: pd.DataFrame, fi: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    top_feats = fi["feature"].head(top_k).tolist()
    low_feats = fi["feature"].tail(top_k).tolist()
    target = frame[TARGET_COL]
    q = target.quantile(0.75)
    high_mask = target >= q
    low_mask = target < q

    rows = []
    for group_name, feats in [("important", top_feats), ("less_important", low_feats)]:
        for f in feats:
            a = frame.loc[high_mask, f].dropna()
            b = frame.loc[low_mask, f].dropna()
            if len(a) < 10 or len(b) < 10:
                continue
            stat = mannwhitneyu(a, b, alternative="two-sided")
            effect = float(a.median() - b.median())
            rows.append(
                {
                    "group": group_name,
                    "feature": f,
                    "pvalue": float(stat.pvalue),
                    "median_diff_high_vs_low_target": effect,
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "pvalue"])


def build_error_case_view(best_exp_id: str, threshold: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    parsed = parse_exp_id(best_exp_id)
    model, _, _, x_test, y_test = fit_best_model_and_extract_features(best_exp_id)
    pred_model = model.predict(x_test)
    pred = _inverse_target(pred_model, parsed.get("target_transform", "none"))
    err = pd.DataFrame({"actual": y_test.values, "pred": pred}, index=y_test.index)
    err["abs_error"] = (err["actual"] - err["pred"]).abs()
    err["margin_to_threshold_actual"] = (err["actual"] - threshold).abs()
    err["margin_to_threshold_pred"] = (err["pred"] - threshold).abs()
    err["is_threshold_miss"] = (err["actual"] >= threshold) != (err["pred"] >= threshold)

    merged = x_test.copy()
    merged["actual"] = err["actual"].values
    merged["pred"] = err["pred"].values
    merged["abs_error"] = err["abs_error"].values
    merged["is_threshold_miss"] = err["is_threshold_miss"].values
    merged["near_threshold_miss"] = err["is_threshold_miss"] & (err["margin_to_threshold_actual"] < 0.2)
    large_cut = err["abs_error"].quantile(0.95)
    merged["large_margin_error"] = err["abs_error"] >= large_cut
    # Label-quality suspicion heuristic: very high error + prediction near local median
    pred_med = float(np.median(pred))
    merged["possible_label_issue"] = (merged["abs_error"] >= err["abs_error"].quantile(0.99)) & (
        (merged["pred"] - pred_med).abs() < np.std(pred) * 0.3
    )
    return err.reset_index(drop=True), merged.reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Mining Quality Dashboard", layout="wide")
    st.title("Mining Quality Experiment Intelligence Dashboard")

    all_summary, all_sig = load_all_experiment_tables()
    if all_summary.empty:
        st.warning("실험 결과 파일을 찾지 못했습니다. experiment_runner를 먼저 실행하세요.")
        return

    best_row = all_summary.sort_values("test_rmse").iloc[0]
    best_exp_id = best_row["exp_id"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Experiment Rows", f"{len(all_summary):,}")
    c2.metric("Best Test RMSE", f"{best_row['test_rmse']:.4f}")
    c3.metric("Best Run", f"{best_row['run_name']}")
    st.caption(f"Best configuration: `{best_exp_id}`")

    tab1, tab2, tab3 = st.tabs(["탭1: 실험 결과", "탭2: 입력 변수 분석", "탭3: 오분류 데이터 심화 분석"])

    with tab1:
        st.subheader("전체 실험 테이블 + 통계 검증")
        merged = all_summary.merge(
            all_sig[["run_name", "exp_id", "ttest_pvalue", "wilcoxon_pvalue", "is_significant_0_05"]],
            on=["run_name", "exp_id"],
            how="left",
        )
        sort_col = st.selectbox("정렬 기준", ["test_rmse", "rmse_mean", "mae_mean"])
        st.dataframe(merged.sort_values(sort_col).reset_index(drop=True), use_container_width=True, height=360)

        fig = px.scatter(
            merged,
            x="rmse_mean",
            y="test_rmse",
            color="run_name",
            hover_data=["exp_id", "is_significant_0_05"],
            title="CV RMSE vs Test RMSE",
        )
        st.plotly_chart(fig, use_container_width=True)

        insight, future = experiment_insights(all_summary, all_sig)
        st.markdown("### 방법론 해석")
        st.markdown(insight)
        st.markdown("### 향후 시도 제안")
        st.markdown(future)

    with tab2:
        st.subheader("Feature Importance + 심화 EDA")
        try:
            fi = compute_feature_importance(best_exp_id)
        except FileNotFoundError as e:
            st.warning(
                "배포 환경에 원본 데이터(`00_data/...csv`)가 없어 탭2 분석을 실행할 수 없습니다. "
                "로컬에서는 정상 동작하며, 배포에서도 사용하려면 샘플 데이터 포함 또는 원격 다운로드 로직이 필요합니다."
            )
            st.code(str(e))
            fi = pd.DataFrame()

        if fi.empty:
            st.info("탭2는 데이터 소스가 준비되면 자동으로 활성화됩니다.")
        else:
            top_k = st.slider("중요 변수 개수", min_value=5, max_value=20, value=10, step=1)
            st.dataframe(fi.head(30), use_container_width=True)

            fig_fi = px.bar(
                fi.head(top_k).sort_values("importance_mean"),
                x="importance_mean",
                y="feature",
                orientation="h",
                title="Top Feature Importance (Permutation)",
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            frame = load_model_frame(sample_rows=50000)
            stat_df = compute_important_vs_unimportant_stats(frame, fi, top_k=top_k)
            st.markdown("#### 중요/비중요 변수의 통계적 차이 (High target vs Low target)")
            st.dataframe(stat_df.head(40), use_container_width=True, height=300)

            corr_target = (
                frame[[c for c in frame.columns if c != "date"]]
                .corr(numeric_only=True)[TARGET_COL]
                .drop(TARGET_COL, errors="ignore")
                .sort_values(key=lambda s: s.abs(), ascending=False)
            )
            selected = corr_target.head(top_k).index.tolist()
            heat_df = frame[selected + [TARGET_COL]].corr(numeric_only=True)
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=heat_df.values,
                    x=heat_df.columns,
                    y=heat_df.index,
                    colorscale="RdBu",
                    zmid=0,
                )
            )
            fig_heat.update_layout(title="Top Features Correlation Heatmap")
            st.plotly_chart(fig_heat, use_container_width=True)

            st.info("현재는 딥러닝 모델이 아니므로 GradCAM 대신 Feature Importance 기반 해석을 제공합니다.")

    with tab3:
        st.subheader("오분석 케이스 심화 분석")
        threshold = st.number_input("품질 임계치 (% Silica Concentrate)", value=3.0, step=0.1)
        try:
            err, merged_err = build_error_case_view(best_exp_id=best_exp_id, threshold=threshold)
            st.write(f"전체 테스트 샘플: `{len(err):,}`")
            st.write(f"임계치 기준 오판정 케이스: `{int(merged_err['is_threshold_miss'].sum()):,}`")

            fig_err = px.histogram(err, x="abs_error", nbins=60, title="Absolute Error Distribution")
            st.plotly_chart(fig_err, use_container_width=True)

            near_df = merged_err[merged_err["near_threshold_miss"]].copy().head(100)
            large_df = merged_err[merged_err["large_margin_error"]].copy().head(100)
            suspect_df = merged_err[merged_err["possible_label_issue"]].copy().head(50)

            st.markdown("#### 아쉽게 틀린 케이스 (임계치 근처 오판정)")
            st.dataframe(near_df[["actual", "pred", "abs_error"]], use_container_width=True, height=220)
            st.markdown("#### 크게 틀린 케이스 (상위 5% 오류)")
            st.dataframe(large_df[["actual", "pred", "abs_error"]], use_container_width=True, height=220)
            st.markdown("#### 라벨링 의심 케이스 (휴리스틱)")
            st.dataframe(suspect_df[["actual", "pred", "abs_error"]], use_container_width=True, height=220)

            if len(near_df) > 20 and len(large_df) > 20:
                compare_feats = [c for c in merged_err.columns if c not in {"actual", "pred", "abs_error", "is_threshold_miss", "near_threshold_miss", "large_margin_error", "possible_label_issue"}]
                compare_feats = compare_feats[:20]
                rows = []
                for f in compare_feats:
                    a = near_df[f].dropna()
                    b = large_df[f].dropna()
                    if len(a) < 10 or len(b) < 10:
                        continue
                    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
                    rows.append(
                        {
                            "feature": f,
                            "near_median": float(a.median()),
                            "large_median": float(b.median()),
                            "pvalue": float(p),
                        }
                    )
                cmp_df = pd.DataFrame(rows).sort_values("pvalue").head(15)
                st.markdown("#### Near-miss vs Large-error 차이 변수")
                st.dataframe(cmp_df, use_container_width=True)

            st.markdown("### 모델 고도화 아이디어 (추가 인사이트 기반)")
            st.markdown(
                "- 임계치 주변 샘플에 가중치를 주는 목적함수(quality-aware loss) 적용\n"
                "- Near-miss 전용 2단계 보정 모델(calibration regressor) 추가\n"
                "- 라벨링 의심 케이스를 전문가 검수 후 재학습 데이터 정제\n"
                "- 중요 변수의 drift 모니터링을 대시보드에 추가하여 운영 안정성 강화"
            )
        except FileNotFoundError as e:
            st.warning(
                "배포 환경에 원본 데이터(`00_data/...csv`)가 없어 탭3 분석을 실행할 수 없습니다. "
                "원본 데이터 접근이 가능해야 실패 케이스 재구성이 가능합니다."
            )
            st.code(str(e))
            st.info("탭1(실험 결과)은 기존 리포트 파일만으로 계속 확인 가능합니다.")


if __name__ == "__main__":
    main()
