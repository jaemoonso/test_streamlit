from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT_DIR / "04_reports"
EXPERIMENTS_DIR = REPORT_DIR / "experiments"
DEPLOY_DIR = REPORT_DIR / "deploy"
EDA_DIR = REPORT_DIR / "eda_refresh"


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


@st.cache_data
def load_deploy_artifacts() -> dict[str, pd.DataFrame | dict]:
    out: dict[str, pd.DataFrame | dict] = {}
    out["feature_importance"] = pd.read_csv(DEPLOY_DIR / "feature_importance.csv") if (DEPLOY_DIR / "feature_importance.csv").exists() else pd.DataFrame()
    out["important_stats"] = pd.read_csv(DEPLOY_DIR / "important_vs_unimportant_stats.csv") if (DEPLOY_DIR / "important_vs_unimportant_stats.csv").exists() else pd.DataFrame()
    out["error_cases"] = pd.read_csv(DEPLOY_DIR / "error_cases_enriched.csv") if (DEPLOY_DIR / "error_cases_enriched.csv").exists() else pd.DataFrame()
    out["corr"] = pd.read_csv(EDA_DIR / "target_correlation.csv") if (EDA_DIR / "target_correlation.csv").exists() else pd.DataFrame()
    context_path = DEPLOY_DIR / "deploy_model_context.json"
    if context_path.exists():
        out["context"] = json.loads(context_path.read_text(encoding="utf-8"))
    else:
        out["context"] = {}
    return out


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
    deploy = load_deploy_artifacts()
    context = deploy.get("context", {})
    if isinstance(context, dict) and context:
        st.caption(f"Deploy artifact model: `{context.get('best_exp_id', '-')}`")

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
        fi = deploy["feature_importance"]
        if fi.empty:
            st.warning("배포용 feature importance 리포트가 없습니다. `04_reports/deploy/feature_importance.csv`를 확인하세요.")
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

            stat_df = deploy["important_stats"]
            st.markdown("#### 중요/비중요 변수의 통계적 차이 (High target vs Low target)")
            st.dataframe(stat_df.head(40) if not stat_df.empty else stat_df, use_container_width=True, height=300)

            corr_df = deploy["corr"]
            if not corr_df.empty:
                corr_top = corr_df.head(top_k)
                fig_corr = px.bar(
                    corr_top.sort_values("abs_corr"),
                    x="abs_corr",
                    y="feature",
                    orientation="h",
                    title="Top Correlation with Target (EDA Refresh)",
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("상관 분석 리포트가 없어 heatmap 대신 중요도만 표시합니다.")

            st.info("현재는 딥러닝 모델이 아니므로 GradCAM 대신 Feature Importance 기반 해석을 제공합니다.")

    with tab3:
        st.subheader("오분석 케이스 심화 분석")
        threshold = st.number_input("품질 임계치 (% Silica Concentrate)", value=3.0, step=0.1)
        merged_err = deploy["error_cases"].copy()
        if merged_err.empty:
            st.warning("배포용 오류 케이스 리포트가 없습니다. `04_reports/deploy/error_cases_enriched.csv`를 확인하세요.")
        else:
            merged_err["is_threshold_miss"] = (merged_err["actual"] >= threshold) != (merged_err["pred"] >= threshold)
            merged_err["near_threshold_miss"] = merged_err["is_threshold_miss"] & ((merged_err["actual"] - threshold).abs() < 0.2)
            merged_err["large_margin_error"] = merged_err["abs_error"] >= merged_err["abs_error"].quantile(0.95)
            merged_err["possible_label_issue"] = merged_err["abs_error"] >= merged_err["abs_error"].quantile(0.99)

            st.write(f"전체 테스트 샘플: `{len(merged_err):,}`")
            st.write(f"임계치 기준 오판정 케이스: `{int(merged_err['is_threshold_miss'].sum()):,}`")

            fig_err = px.histogram(merged_err, x="abs_error", nbins=60, title="Absolute Error Distribution")
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

            compare_feats = [c for c in merged_err.columns if c not in {"actual", "pred", "abs_error", "is_threshold_miss", "near_threshold_miss", "large_margin_error", "possible_label_issue"}]
            compare_feats = compare_feats[:20]
            rows = []
            for f in compare_feats:
                a = near_df[f].dropna() if f in near_df.columns else pd.Series(dtype=float)
                b = large_df[f].dropna() if f in large_df.columns else pd.Series(dtype=float)
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
            cmp_df = pd.DataFrame(rows).sort_values("pvalue").head(15) if rows else pd.DataFrame()
            st.markdown("#### Near-miss vs Large-error 차이 변수")
            st.dataframe(cmp_df, use_container_width=True)

            st.markdown("### 모델 고도화 아이디어 (추가 인사이트 기반)")
            st.markdown(
                "- 임계치 주변 샘플에 가중치를 주는 목적함수(quality-aware loss) 적용\n"
                "- Near-miss 전용 2단계 보정 모델(calibration regressor) 추가\n"
                "- 라벨링 의심 케이스를 전문가 검수 후 재학습 데이터 정제\n"
                "- 중요 변수의 drift 모니터링을 대시보드에 추가하여 운영 안정성 강화"
            )


if __name__ == "__main__":
    main()
