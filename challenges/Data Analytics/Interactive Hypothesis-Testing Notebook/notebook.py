import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from hypothesis_wizard import data, wizard

    # Each entry records one question the reader chose to keep: its tests and p-values.
    get_log, set_log = mo.state([])
    return data, get_log, mo, pl, set_log, wizard


@app.cell
def _(mo):
    mo.md(
        """
        # Hypothesis-testing wizard

        Pick a table and a question. The wizard chooses the test from the design (never from a
        p-value pre-test), runs it with an effect size and a confidence interval, and says what
        the result does and does not show. Alternatives run only when you select them, and each
        one counts toward the significance threshold, which becomes alpha / (tests tried).
        """
    )


@app.cell
def _(data, mo):
    dataset = mo.ui.dropdown(
        options=list(data.DATASETS), value="sleep", label="Dataset"
    )
    mo.vstack([dataset])
    return (dataset,)


@app.cell
def _(data, dataset, mo):
    raw = data.load(dataset.value)
    meta = data.DATASETS[dataset.value]
    frame = data.expand_counts(raw, "Freq") if "Freq" in raw.columns else raw
    cols = list(raw.columns)
    defaults = {
        "sleep": ("extra", "group"),
        "iris": ("petal_length", "species"),
        "penguins": ("bill_length_mm", "sex"),
        "wine_red": ("quality", "alcohol"),
        "toothgrowth": ("len", "supp"),
        "ucb_admissions": ("Admit", "Gender"),
    }
    out_default, grp_default = defaults[dataset.value]
    outcome = mo.ui.dropdown(options=cols, value=out_default, label="Outcome column")
    group = mo.ui.dropdown(
        options=cols, value=grp_default, label="Group or second column"
    )
    pair = mo.ui.dropdown(
        options=["none", *cols],
        value="ID" if dataset.value == "sleep" else "none",
        label="Subject ID (paired or repeated measures)",
    )
    paired = mo.ui.checkbox(value=dataset.value == "sleep", label="Paired measurements")
    stratum = mo.ui.dropdown(
        options=["none", *cols],
        value="Dept" if dataset.value == "ucb_admissions" else "none",
        label="Stratum (2x2 tables)",
    )
    ordinal = mo.ui.multiselect(
        options=cols,
        value=["quality"] if dataset.value == "wine_red" else [],
        label="Treat these columns as ordinal",
    )
    alpha = mo.ui.slider(
        start=0.01, stop=0.10, step=0.01, value=0.05, label="alpha", show_value=True
    )
    shape = (
        f"{frame.height} rows after expanding counts"
        if frame.height != raw.height
        else f"{raw.height} rows"
    )
    mo.vstack(
        [
            mo.md(
                f"**{meta.title}**  \nSource: {meta.source}  \nLicence: {meta.license}. "
                f"{meta.note}  \n{shape}, columns: {', '.join(cols)}"
            ),
            mo.hstack([outcome, group]),
            mo.hstack([pair, paired, stratum]),
            mo.hstack([ordinal, alpha]),
        ]
    )
    return alpha, frame, group, ordinal, outcome, pair, paired, stratum


@app.cell
def _(alpha, group, ordinal, outcome, pair, paired, stratum, wizard):
    spec = wizard.Spec(
        outcome=outcome.value,
        group=group.value,
        pair=None if pair.value == "none" else pair.value,
        paired=paired.value,
        stratum=None if stratum.value == "none" else stratum.value,
        ordinal=tuple(ordinal.value),
        alpha=alpha.value,
    )
    return (spec,)


@app.cell
def _(frame, mo, pl, spec, wizard):
    # Only the plan is needed here, to list the alternatives; no test is counted in this probe.
    try:
        probe = wizard.analyse(frame, spec, wizard.Session(alpha=spec.alpha))
        options = list(probe.plan.alternatives)
        probe_error = None
    except (ValueError, KeyError, pl.exceptions.PolarsError) as exc:
        options, probe_error = [], str(exc)
    alts = mo.ui.multiselect(
        options=options,
        label="Alternatives to run (each one counts toward the threshold)",
    )
    alts if probe_error is None else mo.callout(
        mo.md(f"Could not analyse these inputs: {probe_error}"), kind="danger"
    )
    return (alts,)


@app.cell
def _(alts, frame, mo, pl, spec, wizard):
    def _describe(result):
        text = f"`{result.test}`: {result.statistic_name} = {result.statistic:.4g}"
        if result.df is not None:
            text += f", df = {result.df:.4g}"
        text += f", p = {result.p_value:.4g}"
        if result.effect is not None:
            text += f"; {result.effect_name} = {result.effect:.3f}"
        if result.ci is not None:
            text += f"; {result.ci_name} [{result.ci[0]:.3f}, {result.ci[1]:.3f}]"
        return text

    # A fresh session for every change, so re-running this cell never double-counts a test.
    session = wizard.Session(alpha=spec.alpha)
    report = None
    try:
        report = wizard.analyse(frame, spec, session)
        for name in report.plan.alternatives:
            if name in alts.value:
                wizard.run_alternative(report, name, session, spec)
        error = None
    except (ValueError, KeyError, pl.exceptions.PolarsError) as exc:
        report, error = None, str(exc)

    if error is not None:
        view = mo.callout(mo.md(f"Could not run these tests: {error}"), kind="danger")
    else:
        final = session.threshold
        p = report.primary.p_value
        lines = [
            f"**Question:** {report.kind}. Groups: {', '.join(report.names)}.",
            "",
            "**Why this test**",
        ]
        lines += [f"- {step.branch}: {step.reason}" for step in report.plan.steps]
        lines += [f"- Warning: {w}" for w in report.plan.warnings]
        lines += ["", f"**Primary result:** {_describe(report.primary)}"]
        if report.posthoc:
            lines += [
                "",
                "**Post-hoc comparisons (adjusted p-values)**",
                "",
                "| Pair | Adjusted p |",
                "|---|---|",
            ]
            lines += [
                f"| {row['a']} vs {row['b']} | {row['p']:.4g} |"
                for row in report.posthoc
            ]
        if report.mde is not None:
            lines += [
                "",
                f"Minimum detectable difference at 80% power: {report.mde:.2f} SD.",
            ]
        if report.alternatives:
            lines += ["", "**Alternatives run**"]
            lines += [f"- {_describe(r)}" for r in report.alternatives.values()]
        lines += [
            "",
            (
                f"**Tests tried in this question:** {len(session.tried)} "
                f"({', '.join(session.tried)}). Threshold for every test here: "
                f"alpha / {len(session.tried)} = {final:.4g}. The primary p = {p:.4g} is "
                f"{'below' if p < final else 'not below'} that threshold."
            ),
            "",
            f"**Verdict for the primary test at alpha = {spec.alpha}:** {report.verdict}",
        ]
        view = mo.md("\n".join(lines))
    mo.vstack([view])
    return report, session


@app.cell
def _(dataset, get_log, mo, report, session, set_log, spec):
    def _record(_):
        if report is None:
            return
        pvalues = [
            report.primary.p_value,
            *(r.p_value for r in report.alternatives.values()),
        ]
        entry = {
            "question": f"{dataset.value}: {spec.outcome} by {spec.group}",
            "tests": list(session.tried),
            "pvalues": pvalues,
        }
        set_log(get_log() + [entry])

    keep = mo.ui.button(
        on_click=_record, label="Keep this question's tests in the session log"
    )
    rows = get_log()
    total = sum(len(r["tests"]) for r in rows)
    if rows:
        listing = "\n".join(
            f"- {r['question']}: {len(r['tests'])} test(s), p = "
            + ", ".join(f"{p:.4g}" for p in r["pvalues"])
            for r in rows
        )
        summary = (
            f"**Across questions:** {total} test(s) kept. A family-wide threshold is "
            f"alpha / {total} = {spec.alpha / max(1, total):.4g}.\n\n{listing}"
        )
    else:
        summary = (
            "Nothing kept yet. Keeping a question adds its tests to a family-wide count, "
            "which is what matters when you look across many questions."
        )
    mo.vstack([keep, mo.md(summary)])


if __name__ == "__main__":
    app.run()
