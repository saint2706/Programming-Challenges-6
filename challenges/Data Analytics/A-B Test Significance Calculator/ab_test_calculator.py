"""A/B Test Significance Calculator: chi-square and Welch's t-test with plain-English output.

Run with:  uv run --with polars --with plotly --with scipy --with numpy python ab_test_calculator.py data.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from scipy import stats

# Common header aliases, tried case-insensitively, same spirit as the categorizer's column mapping.
GROUP_ALIASES = ["group", "variant", "arm", "cohort", "bucket"]
OUTCOME_ALIASES = ["outcome", "converted", "conversion", "value", "metric", "result"]


@dataclass
class ProportionResult:
    kind: str
    control_label: str
    variant_label: str
    control_conversions: int
    control_total: int
    variant_conversions: int
    variant_total: int
    control_rate: float
    variant_rate: float
    diff: float
    diff_ci: tuple[float, float]
    relative_lift: float
    chi2: float
    p_value: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


@dataclass
class MeansResult:
    kind: str
    control_label: str
    variant_label: str
    control_mean: float
    control_std: float
    control_n: int
    variant_mean: float
    variant_std: float
    variant_n: int
    diff: float
    diff_ci: tuple[float, float]
    cohens_d: float
    t_stat: float
    p_value: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def resolve_column(
    columns: list[str], aliases: list[str], override: str | None, role: str
) -> str:
    if override:
        if override not in columns:
            raise ValueError(
                f"column {override!r} not found; available columns: {columns}"
            )
        return override
    lowered = [(c, c.lower()) for c in columns]
    for alias in aliases:
        for original, low in lowered:
            if low == alias:
                return original
    for alias in aliases:
        for original, low in lowered:
            if alias in low:
                return original
    raise ValueError(
        f"couldn't auto-detect the {role} column; pass --{role}-col explicitly. Available columns: {columns}"
    )


def is_binary_outcome(series: pl.Series) -> bool:
    values = set(
        series.drop_nulls().cast(pl.Utf8).str.to_lowercase().unique().to_list()
    )
    return values <= {
        "0",
        "1",
        "true",
        "false",
        "yes",
        "no",
        "converted",
        "not converted",
    }


def to_binary(series: pl.Series) -> pl.Series:
    truthy = {"1", "true", "yes", "converted"}
    return series.cast(pl.Utf8).str.to_lowercase().is_in(truthy)


def two_proportion_test(
    control_label: str,
    variant_label: str,
    control_conversions: int,
    control_total: int,
    variant_conversions: int,
    variant_total: int,
    alpha: float,
) -> ProportionResult:
    table = np.array(
        [
            [control_conversions, control_total - control_conversions],
            [variant_conversions, variant_total - variant_conversions],
        ]
    )
    chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)

    p_c = control_conversions / control_total
    p_v = variant_conversions / variant_total
    diff = p_v - p_c

    # Wald CI on the difference in proportions (unpooled variance) -- gives direction
    # and a plausible-range effect size that the chi-square test alone doesn't.
    se_diff = (
        (p_c * (1 - p_c) / control_total) + (p_v * (1 - p_v) / variant_total)
    ) ** 0.5
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    relative_lift = (diff / p_c) if p_c > 0 else float("nan")

    return ProportionResult(
        kind="proportions",
        control_label=control_label,
        variant_label=variant_label,
        control_conversions=control_conversions,
        control_total=control_total,
        variant_conversions=variant_conversions,
        variant_total=variant_total,
        control_rate=p_c,
        variant_rate=p_v,
        diff=diff,
        diff_ci=ci,
        relative_lift=relative_lift,
        chi2=chi2,
        p_value=p_value,
        alpha=alpha,
    )


def welch_t_test_from_stats(
    control_label: str,
    variant_label: str,
    control_mean: float,
    control_std: float,
    control_n: int,
    variant_mean: float,
    variant_std: float,
    variant_n: int,
    alpha: float,
) -> MeansResult:
    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1=variant_mean,
        std1=variant_std,
        nobs1=variant_n,
        mean2=control_mean,
        std2=control_std,
        nobs2=control_n,
        equal_var=False,
    )
    diff = variant_mean - control_mean
    se = ((control_std**2 / control_n) + (variant_std**2 / variant_n)) ** 0.5
    df = se**4 / (
        (control_std**2 / control_n) ** 2 / (control_n - 1)
        + (variant_std**2 / variant_n) ** 2 / (variant_n - 1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = (diff - t_crit * se, diff + t_crit * se)

    pooled_std = (
        ((control_n - 1) * control_std**2 + (variant_n - 1) * variant_std**2)
        / (control_n + variant_n - 2)
    ) ** 0.5
    cohens_d = diff / pooled_std if pooled_std > 0 else float("nan")

    return MeansResult(
        kind="means",
        control_label=control_label,
        variant_label=variant_label,
        control_mean=control_mean,
        control_std=control_std,
        control_n=control_n,
        variant_mean=variant_mean,
        variant_std=variant_std,
        variant_n=variant_n,
        diff=diff,
        diff_ci=ci,
        cohens_d=cohens_d,
        t_stat=t_stat,
        p_value=p_value,
        alpha=alpha,
    )


def welch_t_test_from_raw(
    control_label: str,
    variant_label: str,
    control: np.ndarray,
    variant: np.ndarray,
    alpha: float,
) -> MeansResult:
    return welch_t_test_from_stats(
        control_label,
        variant_label,
        control_mean=float(control.mean()),
        control_std=float(control.std(ddof=1)),
        control_n=len(control),
        variant_mean=float(variant.mean()),
        variant_std=float(variant.std(ddof=1)),
        variant_n=len(variant),
        alpha=alpha,
    )


def analyze_csv(
    path: Path, group_col: str | None, outcome_col: str | None, alpha: float
) -> ProportionResult | MeansResult:
    df = pl.read_csv(path, infer_schema_length=10_000)
    group_col = resolve_column(df.columns, GROUP_ALIASES, group_col, "group")
    outcome_col = resolve_column(df.columns, OUTCOME_ALIASES, outcome_col, "outcome")

    groups = df[group_col].unique().to_list()
    if len(groups) != 2:
        raise ValueError(
            f"expected exactly 2 groups in {group_col!r}, found {len(groups)}: {groups}"
        )
    groups.sort()
    control_label, variant_label = groups

    outcome = df[outcome_col]
    if is_binary_outcome(outcome):
        binary = to_binary(outcome)
        df = df.with_columns(binary.alias("_binary_outcome"))
        control_rows = df.filter(pl.col(group_col) == control_label)
        variant_rows = df.filter(pl.col(group_col) == variant_label)
        return two_proportion_test(
            control_label,
            variant_label,
            control_conversions=int(control_rows["_binary_outcome"].sum()),
            control_total=len(control_rows),
            variant_conversions=int(variant_rows["_binary_outcome"].sum()),
            variant_total=len(variant_rows),
            alpha=alpha,
        )

    control_values = (
        df.filter(pl.col(group_col) == control_label)[outcome_col]
        .drop_nulls()
        .to_numpy()
    )
    variant_values = (
        df.filter(pl.col(group_col) == variant_label)[outcome_col]
        .drop_nulls()
        .to_numpy()
    )
    return welch_t_test_from_raw(
        control_label, variant_label, control_values, variant_values, alpha
    )


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def plain_english(result: ProportionResult | MeansResult) -> str:
    verdict = (
        "statistically significant"
        if result.significant
        else "not statistically significant"
    )
    conclusion = (
        "it's unlikely this difference is due to random chance alone."
        if result.significant
        else "there isn't enough evidence to conclude a real difference exists -- consider a larger sample size."
    )

    if isinstance(result, ProportionResult):
        return (
            f"{result.variant_label}'s conversion rate ({format_pct(result.variant_rate)}, "
            f"{result.variant_conversions}/{result.variant_total}) vs {result.control_label}'s "
            f"({format_pct(result.control_rate)}, {result.control_conversions}/{result.control_total}): "
            f"an absolute difference of {result.diff * 100:+.2f}pp "
            f"({result.relative_lift * 100:+.1f}% relative), 95% CI [{result.diff_ci[0] * 100:.2f}pp, "
            f"{result.diff_ci[1] * 100:.2f}pp]. This is {verdict} at alpha={result.alpha} "
            f"(chi-square p={result.p_value:.4f}): {conclusion}"
        )

    return (
        f"{result.variant_label}'s mean ({result.variant_mean:.3f}, n={result.variant_n}) vs "
        f"{result.control_label}'s ({result.control_mean:.3f}, n={result.control_n}): "
        f"a difference of {result.diff:+.3f}, 95% CI [{result.diff_ci[0]:.3f}, {result.diff_ci[1]:.3f}], "
        f"Cohen's d={result.cohens_d:.2f}. This is {verdict} at alpha={result.alpha} "
        f"(Welch's t-test p={result.p_value:.4f}): {conclusion}"
    )


def _fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False}
    )


def build_chart(result: ProportionResult | MeansResult) -> str:
    if isinstance(result, ProportionResult):
        y = [result.control_rate * 100, result.variant_rate * 100]
        labels = [result.control_label, result.variant_label]
        errors = [0, abs(result.diff_ci[1] - result.diff_ci[0]) / 2 * 100]
        title, ylabel = "Conversion rate", "Rate (%)"
    else:
        y = [result.control_mean, result.variant_mean]
        labels = [result.control_label, result.variant_label]
        errors = [0, abs(result.diff_ci[1] - result.diff_ci[0]) / 2]
        title, ylabel = "Mean value", "Mean"

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=y,
            marker_color=["#4C78A8", "#F58518"],
            error_y=dict(type="data", array=errors, visible=True),
        )
    )
    fig.update_layout(
        title=title, yaxis_title=ylabel, margin=dict(l=50, r=20, t=40, b=30), height=320
    )
    return _fig_to_div(fig)


CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
       background: #0f1115; color: #e8e8e8; }
h1 { font-size: 1.5rem; }
.verdict { background: #1a1d24; border-radius: 8px; padding: 1rem 1.25rem; font-size: 1.05rem; max-width: 720px; }
.verdict.significant { border-left: 4px solid #54A24B; }
.verdict.not-significant { border-left: 4px solid #9aa0aa; }
"""


def render_report(result: ProportionResult | MeansResult) -> str:
    plotly_js = pio.to_html(go.Figure(), include_plotlyjs="inline", full_html=False)
    script_start = plotly_js.index("<script")
    script_end = plotly_js.index("</script>") + len("</script>")
    plotly_bundle = plotly_js[script_start:script_end]

    verdict_class = "significant" if result.significant else "not-significant"
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>A/B Test Result</title><style>{CSS}</style></head>
<body>
{plotly_bundle}
<h1>A/B Test Significance Report ({result.kind})</h1>
<div class="verdict {verdict_class}">{plain_english(result)}</div>
<div class="chart">{build_chart(result)}</div>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        help="CSV with a group column and an outcome column",
    )
    parser.add_argument("--group-col", type=str, default=None)
    parser.add_argument("--outcome-col", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Optional HTML report path"
    )

    group = parser.add_argument_group("summary stats: proportions")
    group.add_argument("--control-conversions", type=int)
    group.add_argument("--control-total", type=int)
    group.add_argument("--variant-conversions", type=int)
    group.add_argument("--variant-total", type=int)

    means = parser.add_argument_group("summary stats: means")
    means.add_argument("--control-mean", type=float)
    means.add_argument("--control-std", type=float)
    means.add_argument("--control-n", type=int)
    means.add_argument("--variant-mean", type=float)
    means.add_argument("--variant-std", type=float)
    means.add_argument("--variant-n", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    proportions_flags = [
        args.control_conversions,
        args.control_total,
        args.variant_conversions,
        args.variant_total,
    ]
    means_flags = [
        args.control_mean,
        args.control_std,
        args.control_n,
        args.variant_mean,
        args.variant_std,
        args.variant_n,
    ]
    has_proportions_flags = any(f is not None for f in proportions_flags)
    has_means_flags = any(f is not None for f in means_flags)

    if args.csv_path and (has_proportions_flags or has_means_flags):
        print(
            "error: pass either a CSV path or summary-stat flags, not both",
            file=sys.stderr,
        )
        return 1
    if has_proportions_flags and has_means_flags:
        print(
            "error: pass either proportions flags or means flags, not both",
            file=sys.stderr,
        )
        return 1

    try:
        if has_proportions_flags:
            if not all(f is not None for f in proportions_flags):
                raise ValueError(
                    "all of --control-conversions/--control-total/--variant-conversions/--variant-total are required together"
                )
            result = two_proportion_test(
                "control",
                "variant",
                args.control_conversions,
                args.control_total,
                args.variant_conversions,
                args.variant_total,
                args.alpha,
            )
        elif has_means_flags:
            if not all(f is not None for f in means_flags):
                raise ValueError(
                    "all of --control-mean/--control-std/--control-n/--variant-mean/--variant-std/--variant-n are required together"
                )
            result = welch_t_test_from_stats(
                "control",
                "variant",
                args.control_mean,
                args.control_std,
                args.control_n,
                args.variant_mean,
                args.variant_std,
                args.variant_n,
                args.alpha,
            )
        elif args.csv_path:
            if not args.csv_path.exists():
                print(f"error: {args.csv_path} does not exist", file=sys.stderr)
                return 1
            result = analyze_csv(
                args.csv_path, args.group_col, args.outcome_col, args.alpha
            )
        else:
            print(
                "error: provide a CSV path or a complete set of summary-stat flags",
                file=sys.stderr,
            )
            return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(plain_english(result))
    if args.output:
        args.output.write_text(render_report(result), encoding="utf-8")
        print(f"HTML report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
