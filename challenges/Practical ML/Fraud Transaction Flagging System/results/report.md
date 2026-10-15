# Fraud Transaction Flagging System: results

Train on hours 0-24, validate on 24-36, test on 36-48.

| Window | Transactions | Frauds | Fraud rate | Fraud amount |
|---|---|---|---|---|
| train | 144,236 | 272 | 0.189% | 32,469 |
| validation | 47,292 | 87 | 0.184% | 9,711 |
| test | 92,198 | 114 | 0.124% | 16,412 |

## Models (PR-AUC; 95% interval for the test value in brackets)

| Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| Isolation forest (no labels) | 0.192 | 0.028 (0.02-0.03) | 0.937 |
| Logistic regression | 0.839 | 0.770 (0.69-0.85) | 0.979 |
| Logistic regression, balanced weights | 0.840 | 0.771 (0.69-0.85) | 0.981 |
| LightGBM, no weighting | 0.772 | 0.768 (0.70-0.85) | 0.979 |
| LightGBM, sqrt weighting | 0.752 | 0.712 (0.63-0.79) | 0.954 |
| LightGBM, balanced weighting | 0.688 | 0.697 (0.61-0.78) | 0.931 |
| LightGBM, undersampled 20:1 | 0.806 | 0.781 (0.70-0.86) | 0.983 |
| Amount only | - | 0.002 | 0.423 |
| Random | - | 0.001 | 0.500 |

Main model, chosen on validation PR-AUC among the LightGBM variants: **LightGBM, undersampled 20:1**.

## Calibration

|  | Log-loss | Brier |
|---|---|---|
| Raw | 0.01046 | 0.00143 |
| Platt scaled | 0.00253 | 0.00038 |

## Operating points chosen on validation, judged on test

### Fixed precision

| Target | Threshold | Val precision / recall | Test precision (95%) | Test recall (95%) | Flagged | Fraud amount caught |
|---|---|---|---|---|---|---|
| 0.5 | 0.0663 | 50.0% / 81.6% | 43.1% (38.5% to 48.4%) | 77.2% (69.3% to 85.1%) | 204 | 12,157 |
| 0.8 | 0.3205 | 80.0% / 78.2% | 90.3% (84.2% to 95.6%) | 73.7% (65.8% to 81.6%) | 93 | 11,907 |
| 0.9 | 0.5889 | 90.8% / 67.8% | 98.8% (95.9% to 100.0%) | 69.3% (60.5% to 78.1%) | 80 | 10,579 |
| 0.95 | 0.8832 | 95.6% / 49.4% | 100.0% (100.0% to 100.0%) | 44.7% (36.0% to 53.5%) | 51 | 7,642 |

### Fixed recall

| Target | Threshold | Test precision | Test recall | Flagged | Alerts per 10k |
|---|---|---|---|---|---|
| 0.6 | 0.7873 | 98.6% | 61.4% | 71 | 7.7 |
| 0.7 | 0.5266 | 97.5% | 69.3% | 81 | 8.8 |
| 0.8 | 0.1866 | 84.3% | 75.4% | 102 | 11.1 |
| 0.9 | 0.0040 | 7.3% | 85.1% | 1331 | 144.4 |

### Cost (review cost c per flagged transaction; a missed fraud loses its amount)

| c | Flag nothing | Flag everything | Static threshold (flagged) | Amount-aware (flagged) | Oracle | Amount-aware 95% |
|---|---|---|---|---|---|---|
| 1 | 16,412 | 92,198 | 4,503 (1741) | 3,440 (602) | 99 | 1,292 to 6,562 |
| 5 | 16,412 | 460,990 | 4,970 (93) | 4,484 (179) | 357 | 2,012 to 7,649 |
| 25 | 16,412 | 2,304,950 | 8,089 (71) | 5,916 (61) | 1,356 | 3,196 to 9,561 |
| 100 | 16,412 | 9,219,800 | 16,752 (5) | 8,009 (27) | 4,259 | 5,214 to 11,580 |

### Review capacity (reviews per hour, highest expected loss first)

| Per hour | Flagged | Frauds caught | Precision | Recall | Fraud amount caught |
|---|---|---|---|---|---|
| 1 | 12 | 10 | 83.3% | 8.8% | 6,152 |
| 3 | 36 | 27 | 75.0% | 23.7% | 11,263 |
| 10 | 120 | 39 | 32.5% | 34.2% | 12,314 |

## How much a threshold moves with the validation draw

| Precision target | Threshold (5-95%) | Test precision (5-95%) | Test recall (5-95%) |
|---|---|---|---|
| 0.8 | 0.157 to 0.513 | 75.7% to 97.6% | 69.3% to 76.3% |
| 0.9 | 0.434 to 0.883 | 96.4% to 100.0% | 44.7% to 71.1% |
| 0.95 | 0.527 to 0.890 | 97.5% to 100.0% | 43.9% to 69.3% |

## Rolling origin (train on the past, score the next 6 hours)

| Hours | Frauds | LightGBM PR-AUC | Logistic PR-AUC |
|---|---|---|---|
| 24-30 | 60 | 0.860 | 0.893 |
| 30-36 | 27 | 0.725 | 0.762 |
| 36-42 | 62 | 0.821 | 0.823 |
| 42-48 | 52 | 0.762 | 0.744 |

## Feature groups (drop one, refit)

| Dropped | Test PR-AUC | Change |
|---|---|---|
| pca | 0.004 | -0.777 |
| amount | 0.784 | +0.003 |
| time | 0.773 | -0.008 |

## Errors at the precision-0.9 point

79 frauds caught, 35 missed. Median amount caught 4.69, missed 8.00; the median score of a missed fraud is 0.0049.

## Top features (mean |SHAP|)

| Feature | Group | Mean abs SHAP | Gain share |
|---|---|---|---|
| V14 | pca | 0.699 | 66.4% |
| V4 | pca | 0.395 | 5.1% |
| V8 | pca | 0.213 | 0.9% |
| V12 | pca | 0.209 | 2.8% |
| V10 | pca | 0.135 | 11.1% |
| V22 | pca | 0.090 | 0.3% |
| V19 | pca | 0.080 | 0.8% |
| V20 | pca | 0.080 | 0.9% |
| V3 | pca | 0.079 | 1.2% |
| V7 | pca | 0.078 | 0.7% |

## Drift (PSI, train to test)

log_amount 0.003, V4 0.217, V10 0.079, V12 0.067, V14 0.075, V17 0.054.
