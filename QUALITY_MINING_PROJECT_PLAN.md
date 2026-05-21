# Quality Prediction in Mining Process - Project Plan

## 1) 프로젝트 목표
- 데이터셋: `edumagalhaes/quality-prediction-in-a-mining-process`
- 핵심 문제: 플로테이션 공정 변수로 최종 품위(정광 품질) 예측
- 1차 타깃: `% Silica Concentrate` (낮을수록 품질 우수)
- 2차 타깃(확장): `% Iron Concentrate`

## 2) 데이터 위치 및 구조
- 경로: `00_data/quality_prediction_mining_process/MiningProcess_Flotation_Plant_Database.csv`
- 파일 크기: 약 184MB
- 레코드 수: 약 737K rows
- 주요 특징:
  - `date` 시계열 컬럼 포함
  - 숫자 값이 문자열 + 콤마 소수점(`"55,2"`) 형식이라 숫자 변환 전처리 필수
  - 공정 센서/유량/레벨/밀도/약품유량 등 다변량 tabular

## 3) 단계별 실행 계획

### Step A. EDA 및 전처리
- 데이터 로딩 시 `,` 소수점 문자열을 `.`으로 변환 후 `float` 캐스팅
- 결측/이상치 점검:
  - 컬럼별 null 비율
  - IQR 또는 robust z-score 기반 이상치 탐지
  - 시간순으로 급격한 점프 구간 확인
- 시계열 특성 생성:
  - hour/dayofweek
  - lag(예: 1, 3, 6 step), rolling mean/std
- 데이터 분할:
  - 랜덤 분할 대신 시계열 hold-out 권장
  - 예: 초기 70% train / 다음 15% valid / 마지막 15% test
- 스케일링:
  - 선형모델/신경망용 StandardScaler
  - 트리 모델은 무스케일 baseline 가능

### Step B. ML 모델링
- Baseline:
  - Linear Regression / Ridge / ElasticNet
- 트리 계열:
  - RandomForestRegressor
  - XGBoost 또는 LightGBM(가능 시)
- 후보 확장:
  - MLPRegressor (비선형 근사)
- 하이퍼파라미터 튜닝:
  - Optuna 또는 RandomizedSearchCV
  - TimeSeriesSplit 기준으로 검증

### Step C. 성능평가 대시보드(Streamlit)
- 필수 화면:
  1. 데이터 개요(KPI 카드: 행수, 기간, 결측률)
  2. 타깃/입력 분포 및 상관관계
  3. 모델별 성능 비교(MAE/RMSE/R2)
  4. 예측 vs 실제 산점도
  5. 시간축 잔차 트렌드
  6. 중요 특성(Feature Importance/SHAP)

### Step D. 오분류(고오차) 케이스 분석 및 고도화
- 회귀에서는 오분류 대신 고오차 케이스 분석으로 수행
- 상위 오차 구간(top 5~10%)을 추출해 공정 조건 비교
- 잔차의 체계적 편향 확인:
  - 특정 시간대/유량/레벨 구간에서 과소·과대예측 여부
- 고도화 방향:
  - 레짐 분리 모델(운전 상태별 모델)
  - 타깃 변환(log/Box-Cox)
  - lag/rolling feature 추가
  - 손실함수 가중(고품질 구간 오차에 가중치)

## 4) 평가 지표 및 합격 기준(초안)
- 기본 지표: MAE, RMSE, R2
- 운영 관점 추천:
  - `% Silica Concentrate` 임계치 초과 위험 탐지 정확도(보조 분류 지표)
- 목표(초안):
  - Baseline 대비 RMSE 10% 이상 개선
  - Test set에서 시간 구간별 성능 편차 축소

## 5) 폴더 구조 제안
- `00_data/quality_prediction_mining_process/` : raw data
- `01_notebooks/` : EDA, 실험 노트북
- `02_src/`
  - `data/` (로더/전처리)
  - `features/` (lag, rolling 생성)
  - `models/` (train/eval)
  - `analysis/` (고오차 분석)
- `03_models/` : 학습 모델 아티팩트
- `04_reports/` : 실험 리포트
- `05_streamlit_app/` : 대시보드 코드

## 6) 실행 순서(체크리스트)
- [ ] 1차 EDA 노트북 작성 및 데이터 타입 정합화
- [ ] 베이스라인 회귀 모델 학습
- [ ] 트리 기반 모델 튜닝 및 성능 비교
- [ ] 최적 모델 + 해석(importance/SHAP)
- [ ] Streamlit 대시보드 구현
- [ ] 고오차 케이스 분석 및 개선안 반영 재학습

## 7) 권장 패키지
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- `xgboost` 또는 `lightgbm`
- `streamlit`, `plotly`
- `shap`, `optuna`

## 8) 주의사항
- 데이터에 콤마 소수점 포맷이 많아 숫자 변환 실수 시 모델 성능이 비정상적으로 떨어질 수 있음
- 시계열 분할을 지키지 않으면 누수로 과대평가될 가능성 큼
- 공정 데이터 특성상 단일 지표보다 구간별 성능 진단이 중요
