# EDA Refresh Report

## 1. Dataset Overview
- rows: 737453
- columns: 25
- start_date: 2017-03-10 01:00:00
- end_date: 2017-09-09 23:00:00
- target_mean: 2.3267632513529675
- target_std: 1.1255535947546034
- target_min: 0.6
- target_max: 5.53

## 2. Missing Value Top 10

| column | missing_ratio |
|---|---:|
| date | 0.0000 |
| % Iron Feed | 0.0000 |
| % Silica Feed | 0.0000 |
| Starch Flow | 0.0000 |
| Amina Flow | 0.0000 |
| Ore Pulp Flow | 0.0000 |
| Ore Pulp pH | 0.0000 |
| Ore Pulp Density | 0.0000 |
| Flotation Column 01 Air Flow | 0.0000 |
| Flotation Column 02 Air Flow | 0.0000 |

## 3. Top 10 Features by |corr| with Target

| feature | corr_with_target |
|---|---:|
| Flotation Column 01 Air Flow | -0.2192 |
| Flotation Column 03 Air Flow | -0.2189 |
| Flotation Column 05 Level | -0.1692 |
| Flotation Column 02 Air Flow | -0.1674 |
| Amina Flow | 0.1567 |
| Flotation Column 04 Level | -0.1495 |
| Ore Pulp pH | -0.1477 |
| Flotation Column 07 Level | -0.1414 |
| Flotation Column 06 Level | -0.1024 |
| % Iron Feed | -0.0771 |

## 4. Time Pattern (Hour)
- Highest average target hour: 10
- Lowest average target hour: 23

## 5. Generated Figures
- target_distribution.png
- top15_abs_corr.png
- target_trend.png
- target_by_hour.png