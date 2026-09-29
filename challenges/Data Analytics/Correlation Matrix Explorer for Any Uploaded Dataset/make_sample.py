"""Regenerate sample_data/health_survey.csv deterministically (seed 42).

Run with:  uv run python make_sample.py
"""

import csv
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)
n = 300
age = rng.normal(40, 10, n).round(1)
education = np.clip(12 + 0.05 * (age - 40) + rng.normal(0, 2.5, n), 8, 22).round(1)
income = (20000 + 900 * age + 2500 * (education - 14) + rng.normal(0, 8000, n)).round(0)
exercise = np.clip(rng.normal(4, 2, n), 0, 14).round(1)
resting_hr = (75 - 1.5 * exercise + rng.normal(0, 4, n)).round(0)
# Monotonic but strongly nonlinear (exponential): Spearman >> Pearson.
dose = rng.uniform(0, 5, n).round(2)
response = (np.exp(1.6 * dose) + rng.normal(0, 3, n)).round(2)
noise = rng.normal(0, 1, n).round(3)
region = rng.choice(["north", "south", "east", "west"], n)

header = [
    "age",
    "income",
    "education_years",
    "weekly_exercise_hrs",
    "resting_hr",
    "dose_mg",
    "response",
    "noise",
    "site_code",
    "region",
]
out = Path(__file__).parent / "sample_data" / "health_survey.csv"
out.parent.mkdir(exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for i in range(n):
        row = [
            age[i],
            income[i],
            education[i],
            exercise[i],
            resting_hr[i],
            dose[i],
            response[i],
            noise[i],
            1,  # constant column: should be skipped
            region[i],
        ]
        if i % 23 == 0:
            row[3] = ""  # missing exercise
        if i % 41 == 0:
            row[1] = ""  # missing income
        w.writerow(row)
