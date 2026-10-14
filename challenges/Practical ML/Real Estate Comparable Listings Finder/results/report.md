# Comparable listings: results

21,613 King County sales, 2014-05-02 to 2015-05-27. Fit on 14,633, validation 2,228, test 4,752 (4,654 first sales of a property) from 2015-03-01. Settings chosen on validation: k=5, market-index extrapolation `flat`, comps combined by `mean`.

## Test window

| Method                               | MdAPE | 95% CI         | Within 10% | Within 20% | Median bias |
| ------------------------------------ | ----: | -------------- | ---------: | ---------: | ----------: |
| Scaled kNN                           | 11.2% | [10.7%, 11.6%] |      45.4% |      72.5% |       -3.3% |
| Ridge-weighted distance              | 10.9% | [10.5%, 11.3%] |      46.4% |      73.7% |       -3.3% |
| Hedonic-weighted distance (GBM SHAP) | 10.2% | [9.8%, 10.5%]  |      49.3% |      77.4% |       -3.8% |
| GBM leaf proximity                   |  9.6% | [9.2%, 9.9%]   |      51.7% |      79.8% |       -3.7% |
| GBM AVM (no comps)                   |  9.1% | [8.8%, 9.3%]   |      54.3% |      82.4% |       -4.2% |

Paired bootstrap, first minus second (negative MdAPE = first is better):

| Pair                               | MdAPE diff (pts) | 95% CI         | Within-10% diff (pts) | 95% CI          |
| ---------------------------------- | ---------------: | -------------- | --------------------: | --------------- |
| Hedonic-weighted vs Scaled kNN     |            -1.05 | [-1.46, -0.58] |                 +3.91 | [+2.30, +5.52]  |
| Leaf proximity vs Scaled kNN       |            -1.64 | [-2.11, -1.20] |                 +6.38 | [+4.79, +8.08]  |
| Leaf proximity vs Hedonic-weighted |            -0.59 | [-0.98, -0.25] |                 +2.47 | [+1.10, +3.78]  |
| Leaf proximity vs GBM AVM          |            +0.50 | [+0.16, +0.83] |                 -2.56 | [-3.91, -1.20]  |
| Hedonic-weighted vs GBM AVM        |            +1.10 | [+0.74, +1.46] |                 -5.03 | [-6.47, -3.57]  |
| Scaled kNN vs GBM AVM              |            +2.14 | [+1.70, +2.55] |                 -8.94 | [-10.40, -7.28] |

## Adjustments (MdAPE)

| Method           |  none | + market time | + attributes | GBM-adjusted |
| ---------------- | ----: | ------------: | -----------: | -----------: |
| Scaled kNN       | 12.1% |         11.5% |        11.2% |         9.3% |
| Ridge-weighted   | 11.5% |         11.0% |        10.9% |         9.1% |
| Hedonic-weighted | 10.9% |         10.2% |        10.2% |         9.2% |
| Leaf proximity   | 10.5% |          9.9% |         9.6% |         9.1% |

## Number of comps (MdAPE)

| Method           |   k=3 |   k=5 |   k=8 |  k=12 |
| ---------------- | ----: | ----: | ----: | ----: |
| Scaled kNN       | 11.4% | 11.2% | 11.1% | 11.1% |
| Ridge-weighted   | 11.3% | 10.9% | 10.9% | 10.8% |
| Hedonic-weighted | 10.3% | 10.2% | 10.0% | 10.0% |
| Leaf proximity   |  9.9% |  9.6% |  9.4% |  9.5% |

## Where the comps come from

| Method           | Median km | Same zip | Median days old | Median gross adjustment |
| ---------------- | --------: | -------: | --------------: | ----------------------: |
| Scaled kNN       |      3.87 |      35% |             191 |                    9.3% |
| Ridge-weighted   |      3.21 |      39% |             192 |                    7.3% |
| Hedonic-weighted |      1.70 |      56% |             191 |                    9.9% |
| Leaf proximity   |      1.88 |      55% |             191 |                   10.1% |

## Pool and extrapolation (MdAPE)

| Method           | pool = all earlier sales | pool = fit window only |
| ---------------- | -----------------------: | ---------------------: |
| Scaled kNN       |                    11.2% |                  11.4% |
| Hedonic-weighted |                    10.2% |                  10.4% |
| Leaf proximity   |                     9.6% |                   9.7% |

| Index beyond fit window | Scaled kNN | Hedonic-weighted | Leaf proximity | GBM AVM |
| ----------------------- | ---------: | ---------------: | -------------: | ------: |
| flat                    |      11.2% |            10.2% |           9.6% |    9.1% |
| trend                   |      11.1% |             9.9% |           9.4% |    8.8% |

## Bracketing

| Method           | Bracketed (top-k) | Bracketed (bracket-aware) | MdAPE top-k | MdAPE bracket-aware | MdAPE if bracketed | MdAPE if not |
| ---------------- | ----------------: | ------------------------: | ----------: | ------------------: | -----------------: | -----------: |
| Scaled kNN       |               78% |                       97% |       11.2% |               11.2% |              11.1% |        11.6% |
| Hedonic-weighted |               76% |                       95% |       10.2% |               10.2% |              10.1% |        10.6% |
| Leaf proximity   |               74% |                       95% |        9.6% |                9.5% |               9.4% |        10.0% |

## By distance to downtown (MdAPE)

| Segment |     n | Scaled kNN | Ridge-weighted | Hedonic-weighted | Leaf proximity | GBM AVM |
| ------- | ----: | ---------: | -------------: | ---------------: | -------------: | ------: |
| < 8 km  |   769 |      14.7% |          14.7% |            12.9% |          11.7% |   11.2% |
| 8-20 km | 2,085 |      13.4% |          12.6% |            11.4% |          10.8% |   10.1% |
| > 20 km | 1,800 |       7.9% |           8.2% |             7.8% |           7.4% |    7.3% |

## By price tier (by AVM estimate) (MdAPE)

| Segment      |     n | Scaled kNN | Ridge-weighted | Hedonic-weighted | Leaf proximity | GBM AVM |
| ------------ | ----: | ---------: | -------------: | ---------------: | -------------: | ------: |
| low third    | 1,551 |      10.9% |          11.0% |            10.6% |          10.3% |    9.4% |
| middle third | 1,551 |      10.8% |          10.4% |             9.8% |           9.1% |    8.8% |
| top third    | 1,552 |      12.1% |          11.3% |            10.2% |           9.5% |    9.0% |

## Resales in the test window (n=98)

| Estimator              | MdAPE |
| ---------------------- | ----: |
| GBM AVM                | 18.2% |
| knn without own prior  | 14.2% |
| knn with own prior     | 17.7% |
| leaf without own prior | 18.1% |
| leaf with own prior    | 18.5% |

## GBM AVM: zipcode

lat/long only 9.1%, plus zipcode as a categorical 8.9%.

## Speed

Milliseconds per subject (find comps and estimate, single process):

- Scaled kNN: 0.71
- Ridge-weighted: 0.82
- Hedonic-weighted: 0.81
- Leaf proximity: 4.43

## Market index (log, flat vs trend extrapolation)

| Month   | Fitted |   Flat |  Trend |
| ------- | ------ | -----: | -----: |
| 2014-05 | yes    | +0.000 | +0.000 |
| 2014-06 | yes    | +0.020 | +0.020 |
| 2014-07 | yes    | +0.014 | +0.014 |
| 2014-08 | yes    | +0.014 | +0.014 |
| 2014-09 | yes    | +0.008 | +0.008 |
| 2014-10 | yes    | +0.010 | +0.010 |
| 2014-11 | yes    | +0.010 | +0.010 |
| 2014-12 | yes    | +0.016 | +0.016 |
| 2015-01 | yes    | +0.017 | +0.017 |
| 2015-02 | yes    | +0.038 | +0.038 |
| 2015-03 | no     | +0.038 | +0.043 |
| 2015-04 | no     | +0.038 | +0.048 |
| 2015-05 | no     | +0.038 | +0.053 |

## GBM AVM signed error by sale month (test)

| Month   |    n | Median error |
| ------- | ---: | -----------: |
| 2015-03 | 1835 |        -2.7% |
| 2015-04 | 2189 |        -5.1% |
| 2015-05 |  630 |        -6.2% |
