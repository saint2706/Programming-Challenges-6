# Interactive Hypothesis-Testing Notebook

**Category:** Data Analytics
**Difficulty:** A
**Status:** Implemented (Python)

A marimo notebook and a small typer CLI that take a table and a question, choose the test from
the design of the data (never from a p-value or an assumption check), run it with an effect size,
a confidence interval, a power-based minimum detectable difference, and a plain-language verdict.
Post-hoc comparisons for three or more groups are adjusted. A counter tracks every test run in a
session and shows the adjusted threshold alpha / (tests tried).

## Data

| Dataset                                       | Rows                            | Licence                                   | Used for                                             |
| --------------------------------------------- | ------------------------------- | ----------------------------------------- | ---------------------------------------------------- |
| Palmer penguins (`penguins.csv`)              | 344 birds, 3 species            | CC0 1.0                                   | two-group and grouped comparisons                    |
| Fisher's iris (`iris.csv`)                    | 150 flowers, 3 species          | CC BY 4.0 (UCI)                           | one-way ANOVA, Welch ANOVA, Kruskal-Wallis, post-hoc |
| Red wine quality (`wine_red.csv`)             | 1,599 wines                     | CC BY 4.0 (UCI)                           | ordinal outcome, Spearman correlation                |
| Sleep (`sleep.csv`)                           | 10 patients, paired             | published historical table (R `datasets`) | paired and Welch tests                               |
| ToothGrowth (`toothgrowth.csv`)               | 60 guinea pigs                  | published historical table (R `datasets`) | correlations                                         |
| UC Berkeley admissions (`ucb_admissions.csv`) | 4,526 applicants, 6 departments | published historical table (R `datasets`) | stratified 2x2 (Mantel-Haenszel)                     |

The three downloaded datasets (Palmer penguins, iris, wine quality) are small and their licences
permit redistribution, so they are committed under `sample_data/`. The R tables are copies of
historical data bundled with R, exported by `scripts/export_r_datasets.R`. `sample_data/SOURCES.json`
records the SHA-256 hash of every committed CSV and the URLs of the three downloads.
`scripts/fetch_data.py` re-downloads the downloads and rebuilds all six files.

## Method

The decision tree (`src/hypothesis_wizard/design.py`) takes a design: question (compare,
association, correlation), number of groups, paired or not, stratified or not, level of the
outcome (binary, nominal, ordinal, continuous), expected counts, heavy tails, and whether IDs repeat.

- Compare, two groups, continuous, independent: Welch t. Alternatives: Brunner-Munzel, permutation test of means.
- Compare, two groups, ordinal: Brunner-Munzel, with Mann-Whitney and a permutation test of means as alternatives.
- Compare, paired: paired t for continuous outcomes, Wilcoxon signed-rank for ordinal ones.
- Compare, three or more groups: Welch ANOVA with Games-Howell post-hoc for continuous outcomes; Kruskal-Wallis with Dunn (Holm) for ordinal ones.
- Association, 2x2: Fisher exact when an expected count is below 5, otherwise chi-square. Stratified 2x2: Cochran-Mantel-Haenszel, with a crude and a pooled odds ratio reported together.
- Larger tables: chi-square with warnings about expected counts.
- Correlation: Pearson for continuous data, Spearman for ordinal data. Heavy tails add a note; they never change the primary test.

Two rules hold throughout. First, the primary test never depends on a p-value or on an assumption
diagnostic: diagnostics (Levene-type spread ratios, heavy-tail measures) change the wording and the
order of the alternatives only. Second, every alternative run is counted. The session threshold is
alpha / max(1, tests tried), so the verdict for each test reflects how many tests the session has run.

Effect sizes: Hedges g with a large-sample normal interval (Hedges-Olkin standard error), dz for paired
data with a normal-approximation interval, Cliff's delta and rank-biserial correlation for ordinal
comparisons, Cramer's V and epsilon-squared for the other designs. The minimum detectable difference
at 80% power comes from statsmodels' t-test power solver for the sample sizes given.

## Results

### Cross-checks

Every statistic the wizard reports is compared with an independent implementation in
`tests/test_methods.py`. Relative tolerances are 1e-6 unless noted.

- R 4.6.1 (base R, `scripts/make_reference.R` writes `tests/reference/r_tests.json`): Welch t on sleep (t = -1.8608, df 17.78, p = 0.07939), Student t, paired t (t = -4.062, df 9, p = 0.002833), Welch ANOVA on iris petal length (F = 1826.58, df 2 and 78.05, p = 2.9e-66), ANOVA, Kruskal-Wallis, chi-square, Mantel-Haenszel on UCB (statistic 1.5246, p = 0.2169, pooled OR 1.105), Pearson, Spearman (rho = 0.8283 on ToothGrowth), Kendall, Holm and Benjamini-Hochberg adjustments.
- pingouin (dev dependency, computed live): Welch ANOVA, Games-Howell post-hoc and Hedges g.
- Two documented gaps, both recorded in the tests rather than hidden:
  - Mann-Whitney on sleep (tied data): R's default `wilcox.test` gives p = 0.06582; the normal approximation with continuity correction that scipy uses (matched to R with `exact = FALSE`) gives p = 0.06933. The wizard reports the normal approximation.
  - Fisher exact on [[8, 1], [2, 5]]: this code solves the conditional maximum-likelihood estimate and the exact interval to 1e-14. R's root finder stops early, so R's estimate (15.4697) differs from the exact root (15.4658) by 2.5e-4 relative, and R's upper bound (1049.8) gives a tail probability of 0.02523 instead of 0.02500. The test checks that this code's interval satisfies its defining equations.
- Dunn (Holm) and Brunner-Munzel have no independent R or pingouin oracle in this environment. Dunn is tested with properties only (adjusted p-values never shrink, separated groups are significant, identical groups are not). Brunner-Munzel uses scipy only.

### Simulation study

`scripts/run_simulation.py` runs 4,000 replicates per cell at alpha = 0.05 with seed 20261027, and
1,000 relabellings for the permutation test (run only on the four null scenarios where it is
included). Each cell is the rejection rate with its Monte Carlo standard error, sqrt(p (1 - p) / R).
The full tables are in `results/simulation.md`; `results/simulation.json` has the same numbers.

Null scenarios (type I error, nominal 0.05):

| Pipeline                        | Normal, equal SD, n 20/20 | SD 3 vs 1, n 10/30 | SD 3 vs 1, n 30/10 | Lognormal, n 8/24 | Student t (3 df), n 15/15 |
| ------------------------------- | ------------------------- | ------------------ | ------------------ | ----------------- | ------------------------- |
| wizard (= Welch here)           | 0.046 (0.003)             | 0.050 (0.003)      | 0.054 (0.004)      | 0.069 (0.004)     | 0.040 (0.003)             |
| Student t always                | 0.046 (0.003)             | 0.208 (0.006)      | 0.004 (0.001)      | 0.046 (0.003)     | 0.040 (0.003)             |
| Mann-Whitney always             | 0.047 (0.003)             | 0.120 (0.005)      | 0.018 (0.002)      | 0.046 (0.003)     | 0.046 (0.003)             |
| Levene, then Student or Welch   | 0.046 (0.003)             | 0.061 (0.004)      | 0.050 (0.003)      | 0.028 (0.003)     | 0.040 (0.003)             |
| Shapiro, then t or Mann-Whitney | 0.048 (0.003)             | 0.206 (0.006)      | 0.009 (0.001)      | 0.048 (0.003)     | 0.048 (0.003)             |

Power scenarios (normal shift of 0.5 SD, n 20/20 unless stated):

| Pipeline                        | Normal shift, n 20/20 | Shift, SD 3 vs 1, n 10/30 | Lognormal, shift on group 2, n 20/20 |
| ------------------------------- | --------------------- | ------------------------- | ------------------------------------ |
| wizard (= Welch here)           | 0.318 (0.007)         | 0.076 (0.004)             | 0.157 (0.006)                        |
| Student t always                | 0.320 (0.007)         | 0.276 (0.007)             | 0.161 (0.006)                        |
| Mann-Whitney always             | 0.297 (0.007)         | 0.167 (0.006)             | 0.397 (0.008)                        |
| Shapiro, then t or Mann-Whitney | 0.318 (0.007)         | 0.266 (0.007)             | 0.397 (0.008)                        |

Reading the tables: the wizard keeps the type I error near 0.05 in these settings because it runs
Welch on these designs. The rule it follows is not the same as the pre-test pipelines: a test chosen
after looking at the variances or normality of the same sample inherits that pre-test's error
rates. Student's t fails in both directions when the groups differ in size and spread. The power
comparison in the lognormal column is not like for like: Mann-Whitney and Welch test different
quantities, so higher power for Mann-Whitney there does not mean it is the better test for the mean.

Flip-through (null, normal, n 20/20, four tests in order: Welch, Student, Mann-Whitney,
Brunner-Munzel): rejecting when any test has p < 0.05 gives a family-wise rate of 0.068 (0.004).
Using the counter rule (reject only if the minimum p is below 0.05 / 4) gives 0.022 (0.002). The
Bonferroni-style rule is conservative for correlated tests, so it is the safe default, not the most
powerful one.

### Example output

On sleep, the paired t test gives t = -4.062, df = 9, p = 0.002833, dz = -1.285 (large), with a 95%
interval for the mean paired difference of [-2.46, -0.70]. The minimum detectable difference at 80%
power is 1.00 SD. Adding the Wilcoxon signed-rank as an alternative gives W+ = 0, p = 0.003906, and
the threshold becomes 0.025 (two tests tried).

## Limits

- Literature claims about the pre-test problem and the default choices in R and scipy were not
  verified by fetching sources in this environment; the numbers above come from the simulation.
- The wizard is not a separate estimator: on the designs in the simulation it runs Welch, so its
  error rates are Welch's.
- Only Python 3.14 was run here; Python 3.12 (the declared minimum) was not tested.
- The sample datasets are small and historical. The notebook is for analysis guidance, not a
  substitute for a design review.

## Where this is actually used

The default choice of test differs between the two main statistics libraries. R's `t.test` uses
Welch's approximation by default: its documentation sets `var.equal = FALSE` and says the variance is
then estimated separately for each group. SciPy's `ttest_ind` defaults to `equal_var=True`, the
Student test; passing `equal_var=False` performs Welch's test.

Multiple-comparison adjustment is an explicit choice in both ecosystems. R's `p.adjust` offers Holm,
Hochberg, Hommel, Bonferroni, Benjamini-Hochberg (`BH`), Benjamini-Yekutieli (`BY`) and `none`.
statsmodels' `multipletests` offers Holm, Bonferroni, Benjamini-Hochberg and others, and its default
is `hs`, not plain Holm.

Online experimentation platforms handle repeated looks at results in different ways. GrowthBook
describes its sequential testing as the frequentist answer to the peeking problem. It uses asymptotic
confidence sequences (Waudby-Smith et al., 2023), so results can be checked repeatedly without
inflating the false positive rate beyond the chosen alpha. Eppo offers classical frequentist tests such
as t-tests, a sequential analysis it describes as always valid, and Bayesian statistics.

## Run it

Run from this folder. The Python environment is managed by `uv`.

```bash
uv run --locked pytest -q              # fast tests (slow ones are marked and skipped)
uv run --locked pytest -q -m slow      # slow tests: notebook export, full simulation checks
uv run marimo edit notebook.py         # interactive notebook
uv run marimo export html notebook.py -o notebook.html   # render without a browser
uv run python scripts/run_simulation.py  # re-run the Monte Carlo study (several minutes)
```

To regenerate the R reference values, run `Rscript scripts/make_reference.R` (needs R with
`jsonlite`). To re-fetch the datasets, run `uv run python scripts/fetch_data.py`.

## Usage

```bash
uv run hypo datasets
uv run hypo ask sleep --outcome extra --group group --pair ID --paired \
    --alternative wilcoxon_signed_rank
```

`hypo ask` takes a dataset key or a CSV path. Use `--ordinal COLUMN` (repeatable) for ordinal
columns, `--stratum COLUMN` for 2x2 strata, `--kind compare|association|correlation` to override
inference, and `--alpha` for the significance level. Each `--alternative` runs and counts toward the
tests-tried threshold; unlisted tests are refused.

## Layout

```text
notebook.py                    marimo notebook (dataset, question, alternatives, session log)
pyproject.toml, uv.lock        uv project; console script `hypo`
pytest.ini                     slow marker; slow tests deselected by default
sample_data/                   committed CSVs and SOURCES.json
scripts/                       fetch_data.py, export_r_datasets.R, make_reference.R, run_simulation.py
src/hypothesis_wizard/         decision tree, methods, effects, post-hoc, diagnostics, wizard, CLI, simulation
tests/                         unit, cross-check, decision-tree property, wizard, CLI, simulation, notebook tests
tests/reference/r_tests.json   R 4.6.1 reference values
results/                       simulation.json and simulation.md
```

## Files

- `src/hypothesis_wizard/design.py`: decision tree from design to primary test, alternatives, post-hoc, steps and warnings.
- `src/hypothesis_wizard/methods.py`: the statistical tests and their result records.
- `src/hypothesis_wizard/effects.py`: effect sizes, intervals and minimum detectable differences.
- `src/hypothesis_wizard/posthoc.py`: Holm, Benjamini-Hochberg, Games-Howell and Dunn (Holm).
- `src/hypothesis_wizard/diagnostics.py`: group summaries, heavy-tail and spread diagnostics.
- `src/hypothesis_wizard/wizard.py`: session counter, specification, analysis and verdict.
- `src/hypothesis_wizard/simulate.py`: scenarios, pipelines, rejection rates and the flip-through study.
- `src/hypothesis_wizard/cli.py`: the `hypo` command.
- `src/hypothesis_wizard/data.py`: dataset registry with licences and loaders.

## Sources

- Palmer penguins: Horst, Hill and Gorman (2020), `palmerpenguins` package, CC0 1.0; https://github.com/allisonhorst/palmerpenguins
- Iris: Fisher (1936); UCI Machine Learning Repository, CC BY 4.0; https://archive.ics.uci.edu/dataset/53/iris
- Wine quality: Cortez et al. (2009); UCI Machine Learning Repository, CC BY 4.0; https://archive.ics.uci.edu/dataset/186/wine+quality
- Sleep, ToothGrowth, UCBAdmissions: R `datasets` package (base R 4.6.1)
- R: R Core Team, base statistics functions (`t.test`, `wilcox.test`, `oneway.test`, `fisher.test`, `mantelhaen.test`, `cor.test`, `p.adjust`)
- SciPy and statsmodels documentation for the Python implementations
- pingouin documentation for the cross-check implementations
