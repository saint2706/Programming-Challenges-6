# Simulation results

Replicates per cell: 4000; permutation test: 1000 relabellings on its scenarios only. Alpha = 0.05. Seed 20261027.
Each cell is the rejection rate with its Monte Carlo SE in brackets: sqrt(p (1 - p) / R). On null scenarios a rate is the type I error; on power scenarios it is power.

## Null scenarios (type I error)

| Pipeline                     | null: normal, equal SD, n 20/20 | null: normal, SD 3 vs 1, n 10/30 | null: normal, SD 3 vs 1, n 30/10 | null: lognormal(0,1) both groups, n 20/20 | null: lognormal(0,1) both groups, n 8/24 | null: Student t (3 df), n 15/15 | null: centred exponential vs normal, same mean and SD, n 15/15 |
| ---------------------------- | ------------------------------- | -------------------------------- | -------------------------------- | ----------------------------------------- | ---------------------------------------- | ------------------------------- | -------------------------------------------------------------- |
| wizard                       | 0.046 (±0.003)                  | 0.050 (±0.003)                   | 0.054 (±0.004)                   | 0.037 (±0.003)                            | 0.069 (±0.004)                           | 0.040 (±0.003)                  | 0.048 (±0.003)                                                 |
| welch_always                 | 0.046 (±0.003)                  | 0.050 (±0.003)                   | 0.054 (±0.004)                   | 0.037 (±0.003)                            | 0.069 (±0.004)                           | 0.040 (±0.003)                  | 0.048 (±0.003)                                                 |
| student_always               | 0.046 (±0.003)                  | 0.208 (±0.006)                   | 0.004 (±0.001)                   | 0.042 (±0.003)                            | 0.046 (±0.003)                           | 0.040 (±0.003)                  | 0.050 (±0.003)                                                 |
| mann_whitney_always          | 0.047 (±0.003)                  | 0.120 (±0.005)                   | 0.018 (±0.002)                   | 0.049 (±0.003)                            | 0.046 (±0.003)                           | 0.046 (±0.003)                  | 0.059 (±0.004)                                                 |
| brunner_munzel_always        | 0.051 (±0.003)                  | 0.056 (±0.004)                   | 0.053 (±0.004)                   | 0.054 (±0.004)                            | 0.064 (±0.004)                           | 0.056 (±0.004)                  | 0.063 (±0.004)                                                 |
| levene_then_student_or_welch | 0.046 (±0.003)                  | 0.061 (±0.004)                   | 0.050 (±0.003)                   | 0.039 (±0.003)                            | 0.028 (±0.003)                           | 0.040 (±0.003)                  | 0.049 (±0.003)                                                 |
| shapiro_then_t_or_mwu        | 0.048 (±0.003)                  | 0.206 (±0.006)                   | 0.009 (±0.001)                   | 0.049 (±0.003)                            | 0.048 (±0.003)                           | 0.048 (±0.003)                  | 0.061 (±0.004)                                                 |
| permutation_mean             | 0.040 (±0.006)                  | 0.152 (±0.011)                   | not run                          | not run                                   | 0.066 (±0.008)                           | not run                         | 0.046 (±0.007)                                                 |

## Power scenarios

| Pipeline                     | power: normal shift 0.5 SD, n 20/20 | power: normal shift 0.5, SD 3 vs 1, n 10/30 | power: lognormal, shift +0.5 on the second group, n 20/20 |
| ---------------------------- | ----------------------------------- | ------------------------------------------- | --------------------------------------------------------- |
| wizard                       | 0.318 (±0.007)                      | 0.076 (±0.004)                              | 0.157 (±0.006)                                            |
| welch_always                 | 0.318 (±0.007)                      | 0.076 (±0.004)                              | 0.157 (±0.006)                                            |
| student_always               | 0.320 (±0.007)                      | 0.276 (±0.007)                              | 0.161 (±0.006)                                            |
| mann_whitney_always          | 0.297 (±0.007)                      | 0.167 (±0.006)                              | 0.397 (±0.008)                                            |
| brunner_munzel_always        | 0.310 (±0.007)                      | 0.085 (±0.004)                              | 0.392 (±0.008)                                            |
| levene_then_student_or_welch | 0.320 (±0.007)                      | 0.090 (±0.005)                              | 0.160 (±0.006)                                            |
| shapiro_then_t_or_mwu        | 0.318 (±0.007)                      | 0.266 (±0.007)                              | 0.397 (±0.008)                                            |
| permutation_mean             | not run                             | not run                                     | not run                                                   |

## Flip-through (null, normal, n 20/20)

Tests tried in order: welch_t, student_t, mann_whitney, brunner_munzel.

| Rule                                      | Rejection rate (±MC SE) |
| ----------------------------------------- | ----------------------- |
| Reject if any test has p < 0.05           | 0.068 (±0.004)          |
| Reject if min p < 0.05 / 4 (counter rule) | 0.022 (±0.002)          |
