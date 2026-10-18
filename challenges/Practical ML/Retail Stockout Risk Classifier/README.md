# Retail Stockout Risk Classifier

**Category:** Practical ML
**Difficulty:** I (brief: "Predict stockout risk from sales/inventory signals (classification, not forecasting).")

**Status:** Implemented (Python)

Source modules live in `src/stockout_risk/`; the tests are in `tests/`; the written report, metrics and charts are in `results/`.

At the end of each day, for every store-product: how likely is it to be out of stock for at least a quarter of tomorrow's selling hours? The answer is a calibrated probability, a ranked "check these first" worklist, and the three or four features that pushed one particular score up or down. The question the challenge answers is *how much does a model add over "it was out of stock yesterday"?* On the held-out final week: a lot in ranking, little in dollars-style cost, because stockouts here are too common for any threshold to filter much.

```bash
uv run stockout data                                     # download (sha256-checked) and summarise the data
uv run stockout evaluate                                 # train, calibrate, score the last week, write results/ (refits LightGBM 10 times)
uv run stockout evaluate --no-ablation --no-other-labels # the main model only
uv run stockout worklist --top 10                        # the store-products most likely to stock out tomorrow
uv run stockout score --store-id 714 --product-id 129    # one store-product, with the reasons
uv run streamlit run src/stockout_risk/app.py            # threshold / cost explorer, worklist, reliability
uv run pytest -q
```

`score` and `worklist` default to 2024-07-01, the last day whose next day is also in the data, so a score can be compared with what happened on 2024-07-02; `--date YYYY-MM-DD` picks another day and `--json` prints JSON. Scoring 2024-07-02 itself works but warns: tomorrow's promotion, discount and weather are not in the data, and the model never saw them missing. Data is read as parquet with polars (never pandas) and kept in `data/`; the model is LightGBM's text format plus a JSON file (nothing is pickled).

```text
$ uv run stockout score --store-id 714 --product-id 129
store 714, product 129: 94.4% chance of 4+ out-of-stock hours on 2024-07-02
  what happened: 16 out-of-stock hours
  +0.53  share of this product's stores heavily out of stock today = 0.54
  +0.32  out-of-stock hours in the last 4 selling hours today = 4
  +0.27  average out-of-stock hours, 14 days = 10.5
  ...
```

That is the top-ranked store-product for 2024-07-01 and it did stock out, but only 1 of the day's top 5 did (the 20 highest-risk rows were right 60% of the time), and store 183's product 129, scored 78.9%, had no stockout hours at all. The reasons are SHAP contributions in log-odds, not causes.

## Data

[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (CC BY 4.0; [arXiv 2505.16319](https://arxiv.org/abs/2505.16319)): 50,000 store-product series from 898 stores and 865 fresh-food products, one row per store-product-day with 24 hourly sales values and 24 hourly stock flags, plus discount, promotion, holiday and weather. `train` is 90 days (2024-03-28 to 2024-06-25, 4.5M rows) and `eval` the 7 days after it (2024-06-26 to 2024-07-02, 350k rows) for the same 50,000 series. No login is needed; both files are downloaded and checked against the sha256 that Hugging Face publishes for them.

What the dataset card does not say, found by looking at the rows:

- **`hours_stock_status` is 1 when the product is out of stock.** `stock_hour6_22_cnt` is the number of those hours among hours 6 to 21, a 16-hour selling window, and it equals the flags on all 4,850,000 rows. `hours_sale` sums to `sale_amount` on all of them.
- **There is no on-hand quantity.** Inventory enters only as the hourly in-stock flag, so "inventory signals" here means stockout history, not stock levels or replenishment schedules.
- **Stockouts are common and noisy.** 44% of store-product-days have at least one out-of-stock hour in the window. Stocked out today, a product is heavily out of stock tomorrow 49% of the time; stocked today, 24%. Yesterday is a weak predictor on its own.
- **Sales are censored.** A product that is out of stock sells nothing, so sales on stockout days understate demand. Mean units sold fall from 1.40 (1 to 3 hours out) to 0.60 (12 to 15 hours), while the sales *rate of the hours that were in stock*, scaled to 16 hours, rises from 1.52 to 2.43 (see "Censored sales" below).
- **The stockout rate is not stationary.** The share of heavy-stockout days (below) was 49% in the week of 2024-04-01, about 29% in late May, 33 to 35% in the validation weeks and 30% then 36.5% in the two test weeks.

## Method

**Label.** A heavy stockout: 4 or more of the 16 selling hours out of stock the next day (a quarter of the window). The base rate is 35.0% in train, 34.2% in validation and 32.0% in test. 1 or more hours (41%) and 8 or more hours (16%) are measured as alternatives.

**Features, as of the end of day t** (`features.py`, seven groups): out-of-stock hours today and 1 to 6 days ago, 3/7/14-day averages, days since the last heavy or fully stocked day (`history`); how the day ended: stocked out at open or close, in the last four hours, when it began (`close`); sales and a censoring-corrected demand estimate (`sales`); today's heavy-stockout rate across the store's products and across the product's stores (`cross`); tomorrow's promotion, discount, holiday and weekday (`promo`); tomorrow's weather (`weather`); city, group and category ids (`static`). Tomorrow's promotion, discount and holiday are things a planner would know a day ahead. Tomorrow's weather in this data is the *observed* weather, not a forecast, which flatters it; it is kept as its own group for that reason.

**Split by time, on the label day.** A row for day t predicts day t+1, and the windows are cut on t+1: train 2024-04-05 to 2024-06-11 (3.4M rows), validation 2024-06-12 to 2024-06-25 (0.7M), test 2024-06-26 to 2024-07-02 (0.35M; the `eval` file, which the model has never seen). The first 7 days are dropped while rolling windows fill. A test checks that rewriting every day after day t changes none of day t's features (apart from the tomorrow covariates, which describe day t+1 on purpose).

**Models.** Four scores that need no model (yesterday's state; the same weekday last week; the heavy-day rate over 7 days; the mean out-of-stock hours over 14 days), a standardised logistic regression (1M training rows, numeric features, weekday one-hot, no ids), and LightGBM (63 leaves, learning rate 0.08, early stopping on validation log-loss, which picked 100 trees). The LightGBM probabilities are then calibrated with binned isotonic regression (see Calibration) fitted on the validation window. Nothing is refit on validation, so the deployed model is the one scored below.

## Results (final week, 350,000 store-product-days)

| Model                            | PR-AUC    | ROC-AUC   | Log-loss  | Brier     |
| -------------------------------- | --------- | --------- | --------- | --------- |
| Base rate                        | 0.320     | 0.500     |           |           |
| Same weekday last week           | 0.342     | 0.544     |           |           |
| Yesterday's state                | 0.400     | 0.622     |           |           |
| Heavy-day rate, 7 days           | 0.409     | 0.627     |           |           |
| Mean out-of-stock hours, 14 days | 0.436     | 0.634     |           |           |
| Logistic regression              | 0.531     | 0.727     | 0.559     | 0.190     |
| LightGBM, raw                    | 0.546     | 0.737     | 0.548     | 0.186     |
| **LightGBM, isotonic**           | **0.544** | **0.737** | **0.546** | **0.185** |

Log-loss and Brier are shown only for the models that output probabilities. PR-AUC is 0.54 against a 0.32 base rate and 0.44 for the best simple rule. Logistic regression gets most of it (0.531), so much of the signal is additive and monotone; the trees add about 0.015.

**Worklist** (the operational use: each day, check the top share of that day's store-products), isotonic LightGBM against the 14-day baseline:

| Top share | Precision | Lift | Recall | Baseline precision | Baseline lift |
| --------- | --------- | ---- | ------ | ------------------ | ------------- |
| 1%        | 73.5%     | 2.30 | 2.3%   | 62.9%              | 1.97          |
| 5%        | 67.3%     | 2.11 | 10.5%  | 54.7%              | 1.71          |
| 10%       | 63.4%     | 1.98 | 19.8%  | 52.2%              | 1.63          |
| 20%       | 57.6%     | 1.80 | 36.0%  | 47.5%              | 1.49          |

**Newly at risk.** The 238,777 rows (68%) that were *not* heavily out of stock today, where "yesterday" has nothing to say: base rate 24.2%, LightGBM PR-AUC 0.427 and ROC-AUC 0.722 against 0.321 and 0.610 for the 14-day baseline.

**Calibration.** The raw LightGBM probabilities run low in the middle (predicted 0.35, observed 0.40). Isotonic regression fitted on the validation window cuts the expected calibration error from 0.027 to 0.013, at a cost of 0.002 PR-AUC (it flattens some ties). Plain isotonic regression on every row has a tail problem, so the calibrator first cuts the validation rows into equal-count bins of at least 1,000 and fits on the bin means: the unbinned version gave "100%" to the 64 highest-scored validation rows, which all happened to be positives (one test row reached that score). The binned version tops out at 94.4%. It does not fix the top of the range completely: the 325 test rows scored 0.9 or more were out of stock 82% of the time against 92.5% predicted. That gap is larger than sampling noise (about 2 points) and looks like a shift between the validation and test weeks, so read the highest scores as "very likely", not as 90%+.

### What a threshold is worth

A check costs 1 and a missed stockout costs `ratio` checks; a check is assumed to find *and prevent* the stockout it finds, so the savings below are an upper bound. The threshold is picked on validation and judged on the test week:

| Ratio | Threshold | 1/ratio | Checked | Cost  | Best of never/always | Saving |
| ----- | --------- | ------- | ------- | ----- | -------------------- | ------ |
| 2     | 0.499     | 0.500   | 20.8%   | 0.606 | 0.639                | 5.1%   |
| 5     | 0.201     | 0.200   | 67.5%   | 0.869 | 1.000                | 13.1%  |
| 10    | 0.108     | 0.100   | 86.1%   | 0.940 | 1.000                | 6.0%   |
| 20    | 0.046     | 0.050   | 94.1%   | 0.969 | 1.000                | 3.1%   |

The validation-picked thresholds land close to `1/ratio`, which is what a calibrated probability should do. The point of the table is the saving column: with a third of all store-product-days stocking out, "check almost everything" is nearly optimal as soon as a miss costs more than a handful of checks, so a threshold filters little. The 14-day baseline, thresholded the same way, saves 0.0 to 3.6%. The model is a ranking tool for a fixed review capacity (the worklist table), not a filter.

### What the model uses

| Feature              | Group  | Mean abs SHAP | Share of gain |
| -------------------- | ------ | ------------- | ------------- |
| `product_heavy_rate` | cross  | 0.440         | 30.2%         |
| `oos_last4`          | close  | 0.203         | 18.5%         |
| `early_share`        | close  | 0.125         | 2.5%          |
| `oos_close`          | close  | 0.120         | 7.0%          |
| `cat2`               | static | 0.111         | 10.1%         |

The strongest single signal is not the product's own history but how many of its stores are heavily out of stock *today* (a shared supply problem). Dropping one feature group and refitting (single run, one seed; differences of about 0.003 are probably noise, which I did not measure):

| Dropped | PR-AUC | Change |
| ------- | ------ | ------ |
| none    | 0.546  |        |
| cross   | 0.522  | -0.025 |
| promo   | 0.536  | -0.011 |
| close   | 0.539  | -0.007 |
| static  | 0.542  | -0.004 |
| sales   | 0.543  | -0.003 |
| history | 0.544  | -0.003 |
| weather | 0.555  | +0.009 |

Groups overlap (the rolling history and the close-of-day state describe the same days), so dropping one understates it. **Dropping the weather group improved the test PR-AUC.** I left it in rather than choose features on the test week, but the likely reading is that day-level weather lets the trees memorise the 90 training days. Confirming it would take another time window.

### Other label definitions

| Stockout means | Base rate | Yesterday PR-AUC | 14-day PR-AUC | LightGBM PR-AUC | LightGBM ROC-AUC |
| -------------- | --------- | ---------------- | ------------- | --------------- | ---------------- |
| 1+ hours       | 41.0%     | 0.506            | 0.528         | 0.639           | 0.741            |
| **4+ hours**   | 32.0%     | 0.400            | 0.436         | 0.546           | 0.737            |
| 8+ hours       | 15.9%     | 0.218            | 0.270         | 0.401           | 0.764            |

The longer the stockout, the harder it is to hit in PR-AUC terms but the more the model separates it (ROC-AUC rises); the gain over the baselines holds at all three. 4+ hours was chosen before looking at these, from the baseline numbers alone.

### Censored sales

Training rows with no in-stock hour excluded; "in-stock sales rate x 16 h" is units sold per in-stocked hour times 16.

| Out-of-stock hours | Rows      | Mean units sold | In-stock sales rate x 16 h |
| ------------------ | --------- | --------------- | -------------------------- |
| 0                  | 1,890,937 | 0.96            | 0.91                       |
| 1-3                | 313,701   | 1.40            | 1.52                       |
| 4-7                | 593,834   | 1.22            | 1.73                       |
| 8-11               | 325,300   | 0.92            | 2.05                       |
| 12-15              | 134,928   | 0.60            | 2.43                       |

Sales alone make a deeply stocked-out product look slow. That is part of why the demand estimate is a feature, though the `sales` group moved PR-AUC by only 0.003 when dropped. Part of the rising rate is that products sell out *because* demand was high, and which hours were in stock is not random (evenings go first), so the estimate is a hint, not a measurement.

## Caveats

- **One retailer, 97 days, perishable goods.** The last 7 days are the only test, so there are no confidence intervals and rows within a day and a series are not independent. The weekly base rate moved from 49% to 29% inside the training period, so the next quarter may look different.
- **Stockout status, not stock.** There is no quantity and no replenishment schedule, and I take the retailer's hourly flag as given.
- **A prevented stockout is assumed.** The cost table is an upper bound; a real check fixes only some stockouts.
- **Observed, not forecast, tomorrow covariates.** Promotion and discount are plausibly planned ahead; weather is not forecast data.
- **Single seed, no hyper-parameter search.** Beyond early stopping, nothing was tuned, and results come from one training run each.
- **Validation numbers are optimistic.** The model is early-stopped and calibrated on that window (validation PR-AUC 0.598 against 0.544 on test).

## Where this is actually used

**Censored demand is an old inventory problem.** When a product sells out, the sales figure is only
a lower bound on demand. Nahmias (1994) treated demand estimation for lost-sales inventory systems
on that basis. The censoring-corrected demand feature here belongs to that line of work.

**The dataset targets this question.** FreshRetailNet-50K (arXiv 2505.16319) is a retail dataset
with sales and stockout records, built around recovering latent demand during stockouts. This
challenge uses the stockout flags as a classification target, which is the classification side of
the split the brief asks for.

**Where the gap is.** A deployed worklist would use the planner's own promotion and weather inputs,
and it would be refitted as the base rate drifts. The README records the base rate moving within its
own training period.

## Layout

`data.py` (download, parsing, hourly lists to daily features), `features.py` (the feature groups and their leakage rules), `models.py` (split, models, calibrator, saving), `evaluate.py` (ranking, calibration, worklist and cost metrics), `pipeline.py` (the experiment), `report.py` (report and charts), `predict.py` (scoring with reasons), `cli.py`, `app.py`. Tests run offline on a synthetic table in the real schema (`tests/helpers.py`), never the 100 MB download.
