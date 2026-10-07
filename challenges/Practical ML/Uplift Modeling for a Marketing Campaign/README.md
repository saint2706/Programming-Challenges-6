# Uplift Modeling for a Marketing Campaign

**Category:** Practical ML
**Difficulty:** I (brief: "Two-model approach to estimate incremental campaign effect.")

**Status:** Implemented (Python)

Who should a marketing campaign contact? Not the customers most likely to convert, but the ones whose
chance of converting goes **up because they were contacted**. This project estimates that
per-customer incremental effect (uplift) on the real Criteo ad campaign, with the brief's
two-model approach as the headline learner, benchmarked against the other standard meta-learners, and
turns the ranking into a budget decision. Everything is judged on a held-out test split with bootstrap
intervals, and the learners are also checked on semi-synthetic outcomes where the true effect is known,
the only setting where "did it recover the effect" has an answer.

The most useful finding is about the data, not the learners: the campaign is **almost, but not exactly,
randomized**, and that small imbalance inflates the headline effect by about 40% if you ignore it.

## Data

[Criteo Uplift Prediction Dataset v2.1](https://huggingface.co/datasets/criteo/criteo-uplift) (Diemert et al.,
AdKDD 2018): 13,979,592 users from incrementality tests, 12 anonymized dense features, a `treatment` flag
(85% treated), and two outcomes, `visit` (4.7%) and `conversion` (0.29%). License CC BY-NC-SA 4.0
(non-commercial), so the data is downloaded, never committed.

- **Mirror, not the documented URL.** `scikit-uplift` documents `go.criteo.net/...`; it returns 404 now (checked
  2026-10-07). `cli.py fetch` downloads Criteo's own HuggingFace mirror once (297 MB) into a gitignored
  `data/`, takes a seeded random sample of **2,000,000 rows** and caches it as parquet.
- **Intent-to-treat.** `treatment` is the intervention. The dataset's `exposure` column (was the ad actually
  shown) is a post-treatment variable and is **refused as a feature** (`data.xy` raises), because it would leak
  the outcome.
- **Splits.** Random 60/20/20 train/validation/test (1.2M / 400k / 400k rows), stratified on treatment x visit.
  Hyperparameters are tuned on validation only, the test split is scored once, and final fits use the train
  split only. Because the split is stratified on the outcome, every split has the same *unadjusted* visit
  rates per arm, so the unadjusted ATE on test equals the full-sample number by construction.
- **Primary outcome: `visit`.** `conversion` is the business outcome but it is very sparse (591
  control-arm conversions in 2M rows), so it is reported as secondary with its wider intervals.

### Is it a randomized trial? Nearly, and the difference matters

`cli.py check` runs two tests. A per-feature gate (standardized mean difference between arms, failing only
beyond both 0.1 and 4 standard errors, so it neither false-alarms on small clean samples nor needs tuning on
large ones) passes: the worst feature is 0.051. That looks balanced. It is not quite:

- A model predicting `treatment` from the 12 features together reaches a held-out **AUC of 0.510
  (95% CI 0.508 to 0.513)**; pure randomization is 0.5.
- Sorted by predicted propensity, the treated share runs from 0.846 in the lowest quintile to 0.859 in the
  highest, and the highest quintile has by far the highest visit rate (about 14% against 0.4% to 3.8% in the
  others), so heavy visitors are over-represented among the treated.
- That is enough to matter: the plain treated-minus-control difference says **10.52** extra visits per 1,000
  customers; weighting each customer by 1 / P(their arm | features) says **7.37**.

So the estimates are **inverse-propensity weighted**: a propensity model e(x) (LightGBM, fit on the train
split, clipped to [0.05, 0.95]) gives each test customer a weight, and the X, transformed-outcome and DR
learners use e(x) instead of a constant (T and S never use it). The plain numbers are kept next to the adjusted
ones as a sensitivity. The per-feature gate alone would have let this through, which is why the check also
reports the joint predictability.

## Learners

All on LightGBM. Propensity is e(x) as above.

| Learner                     | tau(x) estimate                                                                                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **T** (the brief, headline) | separate treated and control outcome models; `mu1(x) - mu0(x)`                                                                                                                                          |
| S                           | one model with the treatment flag as a feature; `f(x,1) - f(x,0)`                                                                                                                                       |
| X                           | Kunzel et al. 2019: impute each arm's missing outcome with the other arm's model, fit two effect models, blend them with weight `e(x)`                                                                  |
| TO (transformed outcome)    | regress `Y (t - e) / (e (1 - e))` on x. The textbook class-variable transformation `z = tY + (1-t)(1-Y)` is only valid at e = 0.5. It is also the learner most exposed to a wrong e: see the note below |
| DR                          | doubly robust (Kennedy 2020): cross-fitted outcome models, regress the pseudo-outcome `mu1 - mu0 + t(Y-mu1)/e - (1-t)(Y-mu0)/(1-e)`                                                                     |
| response (baseline)         | `P(visit \| x, treated)`: who converts when contacted, which is not who is persuadable                                                                                                                  |
| random (baseline)           | the chance reference                                                                                                                                                                                    |

Why e(x) matters most for TO: with a constant e under even mild confounding, the transformed outcome's mean is
off by `delta * (mu1/e + mu0/(1-e))`. In a test dataset where heavy users are both treated more and convert more,
the true effect is 0.02 and the constant-e target averages **0.110**; with the true e(x) it averages 0.020.

## Evaluation

For a ranking, take the top share of customers and compare the outcome rate of treated and control
customers inside it, with each customer weighted by their inverse propensity.

- **Uplift curve** `u(x)`: incremental outcomes per customer of the whole population if the top `x` share is
  contacted; `u(1)` is the ATE. This drives the policy numbers.
- **Qini curve and coefficient** (Radcliffe): the same idea with the control group rescaled to the treated
  group's size. Both coefficients are areas *above the random-targeting line*, so a useless ranking scores 0.
  Under inverse-propensity weights the two arms become equal-sized pseudo-populations, so the Qini rescaling is
  about 1 and **the Qini coefficient and AUUC coincide** in the results below.
- **Bootstrap** (200 draws, paired across learners): every learner sees the same resampled test rows, so
  "X beats T" is a paired interval. It resamples the test set only: the fitted learners and the propensity
  model are held fixed, so the intervals leave out training variance (rerunning with another seed could move
  point estimates by more than the interval width suggests).
- **Decile calibration**: customers sorted into ten predicted-uplift groups and the observed (weighted)
  difference in means per group. Individual effects are unobservable, so this is the real check.
- **Semi-synthetic ground truth**: the real features with simulated outcomes whose true effect is known.

## Results (real Criteo data, test split, n = 400,002)

Overall effect of the campaign, propensity-adjusted: **+7.37 extra visits per 1,000 customers**
(95% CI 5.49 to 9.05; unadjusted 10.52); for conversion **+0.94 per 1,000** (0.50 to 1.35; unadjusted 1.10).

### Visits

| Learner  | Qini = AUUC (95% CI)     | Incremental visits / 1,000 if top 20% contacted | Qini minus T               |
| -------- | ------------------------ | ----------------------------------------------- | -------------------------- |
| T        | 0.0027 [0.0020, 0.0032]  | 6.56 [5.30, 7.77]                               | -                          |
| S        | 0.0027 [0.0020, 0.0035]  | 6.43 [5.02, 7.92]                               | 0.0001 [-0.0005, 0.0008]   |
| X        | 0.0028 [0.0022, 0.0035]  | 6.47 [5.07, 7.68]                               | 0.0001 [-0.0004, 0.0007]   |
| TO       | 0.0028 [0.0022, 0.0035]  | 6.22 [4.93, 7.52]                               | 0.0002 [-0.0006, 0.0009]   |
| DR       | 0.0030 [0.0023, 0.0036]  | 6.50 [5.12, 7.78]                               | 0.0004 [-0.0002, 0.0010]   |
| response | 0.0030 [0.0021, 0.0036]  | 6.71 [5.11, 8.21]                               | 0.0003 [-0.0004, 0.0011]   |
| random   | 0.0002 [-0.0003, 0.0006] | 1.86 [1.13, 2.61]                               | -0.0025 [-0.0032, -0.0017] |

### Conversions (secondary, underpowered)

| Learner  | Qini = AUUC (95% CI)     | Incremental conversions / 1,000 if top 20% contacted | Qini minus T              |
| -------- | ------------------------ | ---------------------------------------------------- | ------------------------- |
| T        | 0.0002 [0.0000, 0.0004]  | 0.68 [0.35, 1.06]                                    | -                         |
| S        | 0.0004 [0.0002, 0.0005]  | 0.85 [0.42, 1.21]                                    | 0.0001 [-0.0000, 0.0003]  |
| X        | 0.0003 [0.0001, 0.0005]  | 0.78 [0.37, 1.17]                                    | 0.0001 [-0.0001, 0.0002]  |
| TO       | 0.0004 [0.0002, 0.0006]  | 0.86 [0.50, 1.16]                                    | 0.0001 [-0.0001, 0.0004]  |
| DR       | 0.0003 [0.0001, 0.0005]  | 0.76 [0.40, 1.09]                                    | 0.0001 [-0.0001, 0.0002]  |
| response | 0.0004 [0.0002, 0.0006]  | 0.87 [0.43, 1.30]                                    | 0.0002 [0.0000, 0.0004]   |
| random   | 0.0001 [-0.0000, 0.0002] | 0.30 [0.15, 0.45]                                    | -0.0002 [-0.0004, 0.0001] |

### What this says (read before quoting any number)

- **Targeting works.** Every uplift learner beats random targeting on visits, with Qini intervals well above
  zero. Contacting the top 20% by the T-learner yields about **6.6 incremental visits per 1,000 customers,
  roughly 89% of the 7.4 you would get by contacting everyone**, against 1.5 for contacting a random 20%
  (0.2 x the ATE), about 4.5x.
- **The correction changes the size of the effect, not the ranking of the learners.** Unadjusted, the headline is
  10.5 and the top-20% figure 7.9; adjusted, 7.4 and 6.6. The learner comparison is nearly identical both ways
  (visit Qini for T: 0.0028 unadjusted, 0.0027 adjusted; response: 0.0031 and 0.0030). Anyone quoting the
  campaign's total effect from a plain difference in means on this data is overstating it by about 40%.
- **The learners cannot be told apart.** On visits, the "minus T" interval of every other uplift learner and of
  the response model contains zero. With 400k test rows and a 4.7% outcome the data does not separate S, T, X,
  TO and DR. The headline T-learner is as good as any, and no ranking of the others is supported.
- **The "likely responders" baseline is not worse here.** On visits the response model matches the uplift
  learners. On conversions its paired interval against T just excludes zero (0.0000 to 0.0004), but that is one of
  twelve comparisons on the outcome this README calls underpowered, with training variance excluded, and would
  not survive a multiple-comparison correction. The honest reading is "not worse", not "better". On Criteo the
  customers who respond most to an ad are largely the ones it moves. The test suite includes a dataset where
  that fails (likely converters put off by contact) and the response model's Qini interval is entirely below zero
  there, so the evaluation can see the failure; Criteo just does not have it.
- **Conversion is underpowered.** Treat its results as consistent with the visit story, not as independent
  confirmation.
- **Calibration** (weighted). For the visit T-learner, the top predicted-uplift decile has an observed uplift of
  **52 extra visits per 1,000 (5.2 points, +/- 12)**, seven times the 7.4 average; deciles 3 to 9 are
  indistinguishable from zero. The *lowest* decile is not negative: its observed uplift is +5.5 (+/- 8.2). The
  model finds the persuadable customers but does not find any who are put off.
- Tuning picked the smaller model for the headline T-learner on visits (15 leaves, 200 minimum samples per leaf).

## Does a learner recover a known effect? (semi-synthetic)

Real features, simulated outcomes with a known individual effect tau(x) (winners, "sleeping dogs" whose
chance falls, and an interaction), 200k rows, 5 seeds, mean (95% CI over seeds). Treatment is randomized by
construction here, so propensity is a constant:

| Scenario                    | Learner         | Spearman with true tau   | RMSE vs tau    | Qini (oracle 0.0075) |
| --------------------------- | --------------- | ------------------------ | -------------- | -------------------- |
| heterogeneous               | T               | 0.833 [0.809, 0.858]     | 0.0272         | 0.0063               |
|                             | S               | **0.940** [0.920, 0.960] | **0.0118**     | **0.0070**           |
|                             | X               | 0.851 [0.828, 0.874]     | 0.0220         | 0.0066               |
|                             | TO              | 0.830 [0.804, 0.856]     | 0.0299         | 0.0062               |
|                             | DR              | 0.845 [0.818, 0.873]     | 0.0277         | 0.0064               |
| none (tau = 0 for everyone) | T / X / TO / DR | -                        | 0.021 to 0.030 | at most 0.0003       |
|                             | S               | -                        | 0.003          | about 0              |

All five learners recover the ranking of a real heterogeneous effect (Spearman 0.83 to 0.94). The S-learner is
best here, but this is partly the generator: the effect is smooth and additive on the logit scale, which is
the easy case for one joint model. Do not read it as "S beats T" in general (on the real data they tie).
When there is no effect at all, the differencing learners (T, X, TO, DR) still predict a noisy non-zero effect
(RMSE about 0.02 to 0.03) while S shrinks to zero; no learner's Qini exceeds 0.0003 (the oracle ordering is
exactly 0). With a constant-sign effect that varies only in size, even the oracle ordering scores just 0.0022
and the learners 0.0007 to 0.0015.

## Targeting policy

`cli.py target` turns a ranking into a decision: expected incremental outcomes from contacting the top
`--budget` share (propensity-weighted, with a bootstrap CI), against random targeting of the same share and
treating everyone, plus the break-even cost per contact and the profit-maximizing share for a **value per
incremental outcome** and **cost per contact that you supply**. Criteo has no revenue or cost data, so neither is
invented. The profit-maximizing share is picked on the same test data it is evaluated on, so its profit is
slightly optimistic.

## Cross-checks

- The uplift and Qini curves match `scikit-uplift` to floating-point precision (largest difference 3e-17 on a
  20k-row probe; the tests assert 1e-9) on continuous and tied scores, and the Qini coefficient matches the area
  under its curve.
- The learners are compared with `causalml` (constant propensity): T and X correlate 1.0000 with its
  `BaseTClassifier` and `BaseXClassifier` on the same data, S 0.995 (column order differs), DR 0.93 (causalml's
  DR uses regressors for the outcome nuisances and its own folds, so only the ranking is expected to agree).
- The metrics are validated by hand-computed cases (a 6-customer example worked out on paper: Qini 1/8, AUUC
  11/54), and the inverse-propensity weights by a dataset with a known effect of 0.02 where the plain difference
  in means reads above 0.05.

## Usage

```bash
uv run python cli.py fetch                       # download the mirror once, cache a 2M-row sample
uv run python cli.py check                       # per-feature gate + how predictable treatment is
uv run python cli.py benchmark                   # propensity, tune, score, bootstrap; writes results/report.json
uv run python cli.py benchmark --stage visit     # one stage at a time; finished stages are reused if the seed,
                                                 # bootstrap size, grid and data are unchanged (--fresh recomputes)
uv run python cli.py target --budget 0.2 --value 5 --cost 0.05
uv run python cli.py score customers.parquet --output scored.parquet   # adds uplift and uplift_rank
uv run streamlit run app.py                      # the targeting dashboard
uv run pytest -q                                 # 143 tests, no network
```

The full benchmark takes about 15 minutes on a CPU. `results/report.json` is committed; the scored test
split and fitted models it produces are git-ignored.

## Limitations

- One campaign and one anonymized feature set: the learners cannot be interpreted, and nothing here says how they
  would behave on a campaign with a different effect structure.
- The propensity correction assumes the 12 features capture everything that made treatment depend on the
  customer. A weak dependence on something unobserved would remain, and the data cannot show it.
- The 4.7% visit rate and 0.29% conversion rate mean intervals are wide; a 2M sample is a fraction of the 14M
  rows. A larger sample would tighten them, and might separate the learners, which this one cannot.
- Intervals do not include training variance (fitted models and the propensity model are held fixed).
- Final models are fit on the train split only (validation is used for tuning, not refit).
- Visits are a weaker business outcome than conversions, but they are the only one with enough events to
  support conclusions.
