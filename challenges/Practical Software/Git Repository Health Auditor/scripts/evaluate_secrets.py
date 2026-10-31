"""Measure the secret scanner's precision and recall on a labelled corpus.

Positives are real-shaped provider tokens, assembled from fragments at runtime so this file never
contains a complete credential. Negatives are values that look secret-like but are not (hashes,
UUIDs, placeholders, example keys, code). Run from the challenge directory:

    uv run python scripts/evaluate_secrets.py          # writes results/secret_eval.{json,md}
    GITAUDIT_SLOW=1 uv run pytest -q -m slow           # the test that re-runs this script
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from gitaudit.secrets import Allowlist, scan_text
from helpers import fake_aws_key_id, secret_fixtures

RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Sample:
    label: str  # "positive" or "negative"
    name: str
    text: str
    expect: str | None = None  # rule expected for a positive


def corpus() -> list[Sample]:
    fx = secret_fixtures()
    out: list[Sample] = []
    # One positive per provider fixture, in a realistic assignment context.
    rule_for = {
        "aws_key": "SEC001",
        "aws_secret": "SEC002",
        "github_classic": "SEC003",
        "github_fine": "SEC004",
        "gitlab": "SEC005",
        "slack": "SEC006",
        "slack_hook": "SEC007",
        "stripe": "SEC008",
        "google": "SEC009",
        "private_key_header": "SEC010",
        "jwt": "SEC011",
        "url_password": "SEC012",
        "generic": "SEC013",
        "npm_token": "SEC014",
    }
    aws_contexts = [
        "aws_secret_access_key = '{v}'",
        "AWS_SECRET_ACCESS_KEY={v}",
        "secret_key: {v}",
        'aws_secret = "{v}"',
    ]
    contexts = [
        "KEY = '{v}'",
        'export TOKEN="{v}"',
        "  api_key: {v}",
        "headers = {{'Authorization': '{v}'}}",
    ]
    for name, rule in rule_for.items():
        value = fx[name]
        # The AWS secret has no prefix, so it is only findable next to its key name; the
        # generic fixture is already a whole `password = ...` statement.
        ctxs = (
            aws_contexts
            if name == "aws_secret"
            else (["{v}"] if name == "generic" else contexts)
        )
        for n, ctx in enumerate(ctxs):
            out.append(
                Sample("positive", f"{name}/{n}", ctx.format(v=value) + "\n", rule)
            )
    out.append(Sample("positive", "aws_key/bare", fake_aws_key_id() + "\n", "SEC001"))
    out.append(
        Sample(
            "positive", "stripe/json", '{"stripe": "' + fx["stripe"] + '"}\n', "SEC008"
        )
    )
    # Negatives: values that must not be reported as credentials.
    negatives = {
        "sha256": "digest = '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'",
        "sha1_git": "commit = 'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3'",
        "uuid": "id = '123e4567-e89b-12d3-a456-426614174000'",
        "placeholder": "password = 'changeme-placeholder-value'",
        # Joined with + (ruff keeps it) so the issue-tracking bot does not read this sample as a
        # real comment marker.
        "todo_text": "# TO" + "DO: rotate the key before release",
        "plain_word": "name = 'configuration_manager'",
        "lorem": "text = 'lorem ipsum dolor sit amet consectetur'",
        "sequential": "pad = 'abcdefghijklmnopqrstuvwxyz0123456789'",
        "short_hex": "color = '#1a2b3c'",
        "version": "requires = 'requests==2.32.3'",
        "url_no_password": "url = 'https://example.com/path/to/resource'",
        "python_code": "def total(items):\n    return sum(i.price for i in items)",
        "integrity": "integrity = 'sha512-' + '" + "A" * 40 + "'",
        "import_path": "from django.contrib.auth.models import User",
        "base64_image": "img = 'data:image/png;base64," + "iVBORw0KGgo" * 20 + "'",
        "aws_doc_example": "aws_secret_access_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'",
        "markdown_link": "[docs](https://docs.example.org/guide.html)",
    }
    for name, text in negatives.items():
        out.append(Sample("negative", name, text + "\n"))
    return out


def evaluate(samples: list[Sample]) -> dict[str, object]:
    allow = Allowlist()
    tp = fp = fn = tn = 0
    per_rule: dict[str, dict[str, int]] = {}
    misses: list[str] = []
    false_alarms: list[str] = []
    for s in samples:
        hits = {h.rule_id for h in scan_text(s.text.encode("utf-8"), allow)}
        if s.label == "positive":
            ok = s.expect in hits
            stats = per_rule.setdefault(s.expect or "?", {"tp": 0, "fn": 0, "fp": 0})
            if ok:
                tp += 1
                stats["tp"] += 1
            else:
                fn += 1
                stats["fn"] += 1
                misses.append(s.name)
        else:
            if hits:
                fp += 1
                false_alarms.append(f"{s.name}: {sorted(hits)}")
                for rule in hits:
                    per_rule.setdefault(rule, {"tp": 0, "fn": 0, "fp": 0})["fp"] += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "samples": len(samples),
        "positives": tp + fn,
        "negatives": tn + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "per_rule": dict(sorted(per_rule.items())),
        "misses": misses,
        "false_alarms": false_alarms,
    }


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def render_markdown(report: dict[str, object]) -> str:
    rows = [
        [rule, str(v["tp"]), str(v["fn"]), str(v["fp"])]
        for rule, v in report["per_rule"].items()  # type: ignore[union-attr]
    ]
    parts = [
        "# Secret scanner evaluation",
        "",
        (
            f"Samples: {report['samples']} ({report['positives']} positive, "
            f"{report['negatives']} negative). Generated by `scripts/evaluate_secrets.py`."
        ),
        "",
        f"- Precision: {report['precision']:.4f}",
        f"- Recall: {report['recall']:.4f}",
        "",
        _table(["Rule", "TP", "FN", "FP"], rows),
        "",
    ]
    if report["misses"]:
        parts += ["Missed positives: " + ", ".join(report["misses"]), ""]  # type: ignore[arg-type]
    if report["false_alarms"]:
        parts += ["False alarms:", ""] + [f"- {a}" for a in report["false_alarms"]]  # type: ignore[union-attr]
        parts.append("")
    return "\n".join(parts)


def main() -> int:
    samples = corpus()
    report = evaluate(samples)
    RESULTS.mkdir(exist_ok=True)
    # newline="\n" keeps the outputs LF on Windows, where text mode would write CRLF.
    (RESULTS / "secret_eval.json").write_text(
        json.dumps(report, indent=4) + "\n", encoding="utf-8", newline="\n"
    )
    (RESULTS / "secret_eval.md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    print(
        f"precision={report['precision']:.4f} recall={report['recall']:.4f} "
        f"samples={report['samples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
