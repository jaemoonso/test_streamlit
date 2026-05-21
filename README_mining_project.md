# Mining Quality Project Quickstart

## 1) 환경 설치
```powershell
pip install -r requirements.txt
```

## 2) 모델 학습/평가
```powershell
python 02_src/models/train_regression.py
```

학습 완료 후 아래 파일이 생성됩니다.
- `04_reports/metrics.csv`
- `04_reports/predictions.csv`
- `04_reports/summary.json`
- `03_models/ridge.joblib`
- `03_models/random_forest.joblib`

## 3) 고오차 케이스 분석
```powershell
python 02_src/analysis/error_analysis.py
```

결과 파일:
- `04_reports/high_error_top_ratio_cases.csv`
- `04_reports/high_error_top_ratio_by_hour.csv`
- `04_reports/high_error_top_ratio_by_dayofweek.csv`

### 현업형 모드 예시
- 절대 임계치(오차 0.8 이상):
```powershell
python 02_src/analysis/error_analysis.py --mode absolute --abs-error-threshold 0.8
```

- 품질 기준 기반(실제 Silica 3.0 이상 구간 우선):
```powershell
python 02_src/analysis/error_analysis.py --mode quality --silica-threshold 3.0
```

- 구간별 상위 5%(기본 shift 기준):
```powershell
python 02_src/analysis/error_analysis.py --mode segmented_top --segment-ratio 0.05 --segment-cols shift
```

- 라인 정보가 예측 파일에 있을 경우 shift+line 기준:
```powershell
python 02_src/analysis/error_analysis.py --mode segmented_top --segment-ratio 0.05 --segment-cols shift line
```

## 4) Streamlit 대시보드 실행
```powershell
streamlit run 05_streamlit_app/app.py
```

## 5) EDA 노트북
- `01_notebooks/EDA_quality_mining.ipynb`

## 6) 대규모 비교실험 + 통계검정
```powershell
python 02_src/models/experiment_runner.py --sample-rows 80000
```

생성 파일(`04_reports/experiments/`):
- `cv_fold_metrics.csv` : 5-fold fold별 성능
- `oof_predictions.csv` : OOF 예측값
- `test_predictions.csv` : fold-ensemble test 예측값
- `experiment_summary.csv` : 실험 요약(CV/Test)
- `significance_vs_baseline.csv` : baseline 대비 통계검정(p-value)
- `best_config.json` : 최적 실험 설정

실험 축:
- 모델: Ridge, RandomForest, XGBoost, LightGBM
- 결측: median, knn
- 이상치: none, iqr_clip, quantile_1_99
- 샘플링: none, target_bin_weighted, undersample_major_bins
- 스케일링: on/off

통계검정:
- 5-fold RMSE 기준 baseline 대비 paired t-test + Wilcoxon signed-rank
