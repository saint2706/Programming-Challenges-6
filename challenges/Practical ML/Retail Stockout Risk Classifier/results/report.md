# Retail Stockout Risk Classifier: results

Label: 4+ out-of-stock hours (of 16) the next day. Base rate 35.0% train, 34.2% validation, 32.0% test.
Label days: train 2024-04-05 to 2024-06-11, validation 2024-06-12 to 2024-06-25, test 2024-06-26 to 2024-07-02 (350,000 test rows).

## Models on the test window

| Model | PR-AUC | ROC-AUC | Log-loss | Brier |
|---|---|---|---|---|
| Yesterday's state | 0.400 | 0.622 | - | - |
| Same weekday last week | 0.342 | 0.544 | - | - |
| Heavy-day rate, 7 days | 0.409 | 0.627 | - | - |
| Mean out-of-stock hours, 14 days | 0.436 | 0.634 | - | - |
| Logistic regression | 0.531 | 0.727 | 0.559 | 0.190 |
| LightGBM (raw) | 0.546 | 0.737 | 0.548 | 0.186 |
| **LightGBM (isotonic)** | **0.544** | **0.737** | **0.546** | **0.185** |

Calibration (ECE): raw 0.0270, isotonic 0.0126.

## Worklist: check the top share of each day's store-products

| Top share | Precision | Lift | Recall | 14-day baseline precision | Baseline lift |
|---|---|---|---|---|---|
| 1.0% | 73.5% | 2.30 | 2.3% | 62.9% | 1.97 |
| 5.0% | 67.3% | 2.11 | 10.5% | 54.7% | 1.71 |
| 10.0% | 63.4% | 1.98 | 19.8% | 52.2% | 1.63 |
| 20.0% | 57.6% | 1.80 | 36.0% | 47.5% | 1.49 |

## Cost of a threshold

One check costs 1, a missed stockout costs `ratio`; a check is assumed to prevent the stockout it finds, which makes the saving an upper bound.

| Ratio | Val-picked threshold | 1/ratio | Checked | Cost | Best trivial | Saving |
|---|---|---|---|---|---|---|
| 2 | 0.499 | 0.500 | 20.8% | 0.606 | 0.639 | 5.1% |
| 5 | 0.201 | 0.200 | 67.5% | 0.869 | 1.000 | 13.1% |
| 10 | 0.108 | 0.100 | 86.1% | 0.940 | 1.000 | 6.0% |
| 20 | 0.046 | 0.050 | 94.1% | 0.969 | 1.000 | 3.1% |

## Newly at risk

Rows that were not a heavy stockout day today: 238,777, base rate 24.2%. LightGBM PR-AUC 0.427 / ROC-AUC 0.722; 14-day baseline 0.321 / 0.610.

## Categories

The model beats the 14-day baseline's PR-AUC in 23 of 24 first-level categories. Largest gain: category 3 (0.604 vs 0.375); smallest: category 30 (0.378 vs 0.388).

## Feature groups (drop one, refit)

| Dropped | PR-AUC | Change vs full |
|---|---|---|
| history | 0.544 | -0.0028 |
| close | 0.539 | -0.0069 |
| sales | 0.543 | -0.0029 |
| cross | 0.522 | -0.0246 |
| promo | 0.536 | -0.0107 |
| weather | 0.555 | +0.0090 |
| static | 0.542 | -0.0040 |

## Label definitions

| Stockout means | Base rate | Yesterday PR-AUC | 14-day PR-AUC | LightGBM PR-AUC | LightGBM ROC-AUC |
|---|---|---|---|---|---|
| 1+ hours | 41.0% | 0.506 | 0.528 | 0.639 | 0.741 |
| 4+ hours | 32.0% | 0.400 | 0.436 | 0.546 | 0.737 |
| 8+ hours | 15.9% | 0.218 | 0.270 | 0.401 | 0.764 |

## Censored sales

| Out-of-stock hours | Rows | Mean units sold | In-stock sales rate x 16 h |
|---|---|---|---|
| 0 | 1,890,937 | 0.96 | 0.91 |
| 1-3 | 313,701 | 1.40 | 1.52 |
| 4-7 | 593,834 | 1.22 | 1.73 |
| 8-11 | 325,300 | 0.92 | 2.05 |
| 12-15 | 134,928 | 0.60 | 2.43 |

## Top features (mean |SHAP|, log-odds)

| Feature | Group | Mean abs SHAP | Gain share |
|---|---|---|---|
| product_heavy_rate | cross | 0.4403 | 30.2% |
| oos_last4 | close | 0.2034 | 18.5% |
| early_share | close | 0.1245 | 2.5% |
| oos_close | close | 0.1202 | 7.0% |
| cat2 | static | 0.1110 | 10.1% |
| sale_mean7 | sales | 0.0855 | 1.5% |
| discount_next | promo | 0.0713 | 1.6% |
| oos_t | history | 0.0701 | 5.2% |
| dow_next | promo | 0.0670 | 2.7% |
| sale_t | sales | 0.0628 | 1.4% |
| holiday_next | promo | 0.0544 | 0.7% |
| precip_next | weather | 0.0482 | 1.1% |

## Drift

PSI of features, train to test: oos_mean14 0.017, product_heavy_rate 0.038, oos_last4 0.007, demand16 0.041, discount_next 0.026.
