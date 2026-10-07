# Uplift Modeling for a Marketing Campaign

**Category:** Practical ML
**Difficulty:** I (brief: "Two-model approach to estimate incremental campaign effect.")

**Status:** Implemented (Python)

Who should a marketing campaign contact? Not the customers most likely to convert, but the ones whose
chance of converting goes **up because they were contacted**. This project estimates that
per-customer incremental effect (uplift) on the real, randomized Criteo ad campaign, with the brief's
two-model approach as the headline learner, benchmarked against the other standard meta-learners, and
turns the ranking into a budget decision. Everything is judged on a held-out test split with paired
bootstrap intervals, and the learners are also checked on semi-synthetic outcomes where the true effect
is known, the only setting where "did it recover the effect" has an answer.

## Data

[Criteo Uplift Prediction Dataset v2.1](https://huggingface.co/datasets/criteo/criteo-uplift) (Diemert et al.,
AdKDD 2018): 13,979,592 users from randomized incrementality tests, 12 anonymized dense features, a
`treatment` flag (85% treated), and two outcomes, `visit` (4.7%) and `conversion` (0.29%). License
CC BY-NC-SA 4.0 (non-commercial), so the data is downloaded, never committed.

- **Mirror, not the documented URL.** `scikit-uplift` documents `go.criteo.net/...`; it returns 404 now (checked
  2026-10-07). `cli.py fetch` downloads Criteo's own HuggingFace mirror once (297 MB) into a gitignored
  `data/`, takes a seeded random sample of **2,000,000 rows** and caches it as parquet.
- **Intent-to-treat.** `treatment` is the intervention. The dataset's `exposure` column (was the ad actually
  shown) is a post-treatment variable and is **refused as a feature** (`data.xy` raises), because it would leak
  the outcome.
- **Randomization check first.** The standardized mean difference of every feature between treated and control
  is computed before any modeling and the run aborts if any exceeds 0.1. On the real sample the worst is
  0.051 (`f3`), so the arms are balanced.
- **Splits.** Random 60/20/20 train/validation/test (1.2M / 400k / 400k rows), stratified on treatment x visit.
  Hyperparameters are tuned on validation only, the test split is scored once, and final fits use the train
  split only.
- **Primary outcome: `visit`.** `conversion` is the business outcome but it is very sparse (591
  control-arm conversions in 2M rows), so it is reported as secondary with its wider intervals.

## Learners

All on LightGBM, with a constant known propensity e = 0.85 (an RCT, so no propensity model).

| Learner                     | tau(x) estimate                                                                                                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **T** (the brief, headline) | separate treated and control outcome models; `mu1(x) - mu0(x)`                                                                                                                              |
| S                           | one model with the treatment flag as a feature; `f(x,1) - f(x,0)`                                                                                                                           |
| X                           | Kunzel et al. 2019: impute each arm's missing outcome with the other arm's model, fit two effect models, blend them with weight `e`                                                         |
| TO (transformed outcome)    | regress `Y (t - e) / (e (1 - e))` on x. The textbook class-variable transformation `z = tY + (1-t)(1-Y)` is only valid at e = 0.5; with 85% treated it needs this propensity-corrected form |
| DR                          | doubly robust (Kennedy 2020): cross-fitted outcome models, regress the pseudo-outcome `mu1 - mu0 + t(Y-mu1)/e - (1-t)(Y-mu0)/(1-e)`                                                         |
| response (baseline)         | `P(visit \| x, treated)`: who converts when contacted, which is not who is persuadable                                                                                                      |
| random (baseline)           | the chance reference                                                                                                                                                                        |

## Evaluation

For a ranking, take the top share of customers and compare the outcome rate of treated and control
customers inside it.

- **Uplift curve** `u(x)`: incremental outcomes per customer of the whole population if the top `x` share is
  contacted; `u(1)` is the ATE. This drives the policy numbers.
- **Qini curve and coefficient** (Radcliffe): the same idea with the control group rescaled to the treated
  group's size. The Qini coefficient and AUUC reported here are areas *above the random-targeting line*, so
  a useless ranking scores 0.
- **Paired bootstrap** (200 draws): every learner sees the same resampled test rows, so "X beats T" is a
  paired interval. A ranking is sorted once and the bootstrap only varies integer row weights, so 200 draws
  cost seconds.
- **Decile calibration**: customers sorted into ten predicted-uplift groups and the *observed* difference in
  means per group. Individual effects are unobservable, so this is the real check.
- **Semi-synthetic ground truth**: the real features with simulated outcomes whose true effect is known.

## Results (real Criteo data, test split, n = 400,002)

Overall effect of the campaign: **+10.52 extra visits per 1,000 customers** (95% CI 8.78 to 12.13); for
conversion, **+1.10 per 1,000** (0.70 to 1.50).

### Visits

| Learner  | Qini (95% CI)            | AUUC                     | Incremental visits / 1,000 if top 20% contacted | Qini minus T               |
| -------- | ------------------------ | ------------------------ | ----------------------------------------------- | -------------------------- |
| T        | 0.0028 [0.0023, 0.0033]  | 0.0033 [0.0026, 0.0038]  | 7.94 [6.70, 9.13]                               | -                          |
| S        | 0.0029 [0.0023, 0.0035]  | 0.0033 [0.0027, 0.0041]  | 7.64 [6.23, 9.01]                               | 0.0001 [-0.0004, 0.0007]   |
| X        | 0.0029 [0.0024, 0.0035]  | 0.0033 [0.0028, 0.0040]  | 7.82 [6.56, 8.97]                               | 0.0001 [-0.0004, 0.0006]   |
| TO       | 0.0027 [0.0022, 0.0033]  | 0.0032 [0.0025, 0.0039]  | 7.24 [6.05, 8.44]                               | -0.0001 [-0.0007, 0.0006]  |
| DR       | 0.0030 [0.0024, 0.0035]  | 0.0035 [0.0029, 0.0041]  | 7.71 [6.39, 8.92]                               | 0.0002 [-0.0003, 0.0007]   |
| response | 0.0031 [0.0024, 0.0036]  | 0.0036 [0.0028, 0.0042]  | 7.90 [6.45, 9.49]                               | 0.0003 [-0.0003, 0.0009]   |
| random   | 0.0001 [-0.0003, 0.0005] | 0.0002 [-0.0003, 0.0006] | 2.43 [1.73, 3.12]                               | -0.0027 [-0.0033, -0.0020] |

### Conversions (secondary, underpowered)

| Learner  | Qini (95% CI)            | Incremental conversions / 1,000 if top 20% contacted | Qini minus T              |
| -------- | ------------------------ | ---------------------------------------------------- | ------------------------- |
| T        | 0.0002 [0.0001, 0.0004]  | 0.74 [0.41, 1.11]                                    | -                         |
| S        | 0.0003 [0.0002, 0.0005]  | 0.90 [0.50, 1.24]                                    | 0.0001 [-0.0000, 0.0002]  |
| X        | 0.0003 [0.0001, 0.0004]  | 0.85 [0.45, 1.24]                                    | 0.0001 [-0.0001, 0.0002]  |
| TO       | 0.0002 [-0.0000, 0.0003] | 0.73 [0.35, 1.06]                                    | -0.0000 [-0.0002, 0.0001] |
| DR       | 0.0001 [-0.0000, 0.0003] | 0.66 [0.29, 1.01]                                    | -0.0001 [-0.0002, 0.0001] |
| response | 0.0004 [0.0002, 0.0005]  | 0.93 [0.55, 1.34]                                    | 0.0002 [0.0000, 0.0003]   |
| random   | 0.0001 [-0.0000, 0.0001] | 0.32 [0.17, 0.47]                                    | -0.0002 [-0.0003, 0.0000] |

### What this says (read before quoting any number)

- **Targeting works.** Every uplift learner beats random targeting on visits, with Qini intervals well above
  zero. Contacting the top 20% by the T-learner yields about **7.9 incremental visits per 1,000 customers,
  roughly 75% of the 10.5 you would get by contacting everyone**, and 3.8x what contacting a random 20% would give (2.1).
- **The learners cannot be told apart.** On visits, the "minus T" interval of every other uplift learner and of the response model contains zero. With
  400k test rows and a 4.7% outcome the data does not separate S, T, X, TO and DR. The headline T-learner is as good as
  any, and no ranking of the others is supported.
- **The "likely responders" baseline is not worse here, and on conversions it is better.** On visits the
  response model matches the uplift learners. On conversions it scores *higher* than T, with a paired interval that
  just excludes zero (0.0000 to 0.0003). On Criteo the customers who respond most to an ad are largely the same ones
  who would have responded anyway *and* the ones the ad moves, so the textbook claim that response models
  mis-target does not hold on this dataset. The test suite includes a dataset where it does hold (likely converters
  put off by contact) and the response model's Qini interval is entirely below zero there, so the evaluation
  can see the failure; Criteo just does not have it.
- **Conversion is underpowered.** Its intervals for TO and DR include zero. Treat conversion results as
  consistent with the visit story, not as an independent confirmation.
- **Calibration.** For the visit T-learner, the top predicted-uplift decile has an observed uplift of
  **56 extra visits per 1,000 (5.6 points, +/- 12)**, five times the 10.5 average; deciles 3 to 9 are
  indistinguishable from zero. The *lowest* decile is not negative: its observed uplift is +8 (+/- 8). The model finds the
  persuadable customers but does not find any who are put off.
- Tuning picked the smaller model for the headline (15 leaves, 200 minimum samples per leaf) for visits.

## Does a learner recover a known effect? (semi-synthetic)

Real features, simulated outcomes with a known individual effect tau(x) (winners, "sleeping dogs" whose
chance falls, and an interaction), 200k rows, 5 seeds, mean (95% CI over seeds):

| Scenario                    | Learner         | Spearman with true tau   | RMSE vs tau    | Qini (oracle 0.0075) |
| --------------------------- | --------------- | ------------------------ | -------------- | -------------------- |
| heterogeneous               | T               | 0.833 [0.809, 0.858]     | 0.0272         | 0.0063               |
|                             | S               | **0.940** [0.920, 0.960] | **0.0118**     | **0.0070**           |
|                             | X               | 0.851 [0.828, 0.874]     | 0.0220         | 0.0066               |
|                             | TO              | 0.830 [0.804, 0.856]     | 0.0299         | 0.0062               |
|                             | DR              | 0.845 [0.818, 0.873]     | 0.0277         | 0.0064               |
| none (tau = 0 for everyone) | T / X / TO / DR | -                        | 0.021 to 0.030 | about 0              |
|                             | S               | -                        | 0.003          | about 0              |

All five learners recover the ranking of a real heterogeneous effect (Spearman 0.83 to 0.94). The S-learner is
best here, but this is partly the generator: the effect is smooth and additive on the logit scale, which is
the easy case for one joint model. Do not read it as "S beats T" in general (on the real data they tie).
When there is no effect at all, the differencing learners (T, X, TO, DR) still predict a noisy non-zero effect
(RMSE about 0.02 to 0.03) while S shrinks to zero; no learner's Qini exceeds 0.0003 (the oracle ordering is exactly 0). With a constant-sign effect that varies only in size, even the oracle ordering scores just 0.0022 and the learners 0.0007 to 0.0015.

## Targeting policy

`cli.py target` turns a ranking into a decision: expected incremental outcomes from contacting the top
`--budget` share with a bootstrap CI, against random targeting of the same share and treating everyone, plus the
break-even cost per contact and the profit-maximizing share for a **value per incremental outcome** and **cost
per contact that you supply**. Criteo has no revenue or cost data, so neither is invented.

## Cross-checks

- The uplift and Qini curves match `scikit-uplift` to 1e-17 on continuous and tied scores, and the Qini coefficient
  matches the area under its curve.
- The learners are compared with `causalml`: T and X correlate 1.0000 with its `BaseTClassifier` and
  `BaseXClassifier` on the same data, S 0.995 (column order differs), DR 0.93 (causalml's DR uses regressors for the
  outcome nuisances and its own folds, so only the ranking is expected to agree).
- The metrics are validated by hand-computed cases (a 6-customer example worked out on paper: Qini 1/8, AUUC 11/54).

## Usage

```bash
uv run python cli.py fetch                       # download the mirror once, cache a 2M-row sample
uv run python cli.py check                       # randomization check
uv run python cli.py benchmark                   # tune, score, bootstrap; writes results/report.json
uv run python cli.py benchmark --stage visit     # one stage at a time (stages are cached; --fresh recomputes)
uv run python cli.py target --budget 0.2 --value 5 --cost 0.05
uv run python cli.py score customers.parquet --output scored.parquet   # adds uplift and uplift_rank
uv run streamlit run app.py                      # the targeting dashboard
uv run pytest -q                                 # 116 tests, no network
```

The full benchmark takes about 12 minutes on a CPU (visit 5 minutes, conversion and the semi-synthetic runs about 7). `results/report.json` is committed; the scored test
split and fitted models it produces are git-ignored.

## Limitations

- One campaign and one anonymized feature set: the learners cannot be interpreted, and nothing here says how they
  would behave on a campaign with a different effect structure.
- The 4.7% visit rate and 0.29% conversion rate mean intervals are wide; a 2M sample is a fraction of the 14M
  rows. A larger sample would tighten them, and might separate the learners, which this one cannot.
- Final models are fit on the train split only (validation is used for tuning, not refit).
- Visits are a weaker business outcome than conversions, but they are the only one with enough events to
  support conclusions.
