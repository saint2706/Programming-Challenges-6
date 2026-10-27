"""Score the extractor against the labelled corpus.

Gold items are located by quote: the quote must occur on exactly one line, and that line number
is the gold location. A predicted item matches a gold action when it is on the same line and at
least half of the gold task's words appear in the predicted task. Owners match leniently (one
name's tokens are a subset of the other's). Due dates must be equal, with None meaning none.
Every predicted item that does not match a gold action is a false positive, including items
that match a gold non-action (a decision, question or status line).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from actionitems import llm
from actionitems.errors import InputError
from actionitems.extract import Extraction, Options, extract_document
from actionitems.ingest import Document, decode_bytes
from actionitems.models import ActionItem
from actionitems.paths import corpus_dir, results_dir
from actionitems.text import one_line, safe_text

SPLITS = ("dev", "test")
MATCH_CONTAINMENT = 0.5
SCHEMA_VERSION = 1
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(value: str | None) -> set[str]:
    return set(_TOKEN.findall((value or "").lower()))


def locate_quote(text: str, quote: str) -> int:
    """1-based number of the single line holding ``quote``; anything else is a corpus error."""
    if not quote or "\n" in quote:
        raise InputError(f"gold quote must be one non-empty line: {quote!r}")
    hits = [n for n, line in enumerate(text.split("\n"), start=1) if quote in line]
    if len(hits) != 1 or text.count(quote) != 1:
        raise InputError(
            f"gold quote must occur exactly once on one line (found {len(hits)} lines): {quote!r}"
        )
    return hits[0]


def task_matches(gold_task: str, predicted: str) -> bool:
    wanted = _tokens(gold_task)
    return (
        bool(wanted)
        and len(wanted & _tokens(predicted)) / len(wanted) >= MATCH_CONTAINMENT
    )


def owner_matches(gold: str | None, predicted: str | None) -> bool:
    if gold is None or predicted is None:
        return gold is None and predicted is None
    g, p = _tokens(gold), _tokens(predicted)
    return bool(g and p) and (g <= p or p <= g)


def _predicted_line(item: ActionItem) -> int:
    return min((s.line for s in item.sources), default=0)


@dataclass
class DocScore:
    doc_id: str
    fmt: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    pairs: list[tuple[bool, bool]] = field(default_factory=list)  # (owner ok, due ok)
    missed: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)


def score_document(entry: dict, text: str, extraction: Extraction) -> DocScore:
    score = DocScore(entry["id"], entry["format"])
    predicted = list(extraction.items)
    used = [False] * len(predicted)
    for gold in entry["items"]:
        if not gold["action"]:
            continue
        line = locate_quote(text, gold["quote"])
        hit = next(
            (
                i
                for i, item in enumerate(predicted)
                if not used[i]
                and _predicted_line(item) == line
                and task_matches(gold["task"], item.task)
            ),
            None,
        )
        if hit is None:
            score.fn += 1
            score.missed.append(gold["task"])
            continue
        used[hit] = True
        item = predicted[hit]
        score.tp += 1
        score.pairs.append(
            (owner_matches(gold.get("owner"), item.owner), item.due == gold.get("due"))
        )
    for i, item in enumerate(predicted):
        if not used[i]:
            score.fp += 1
            score.spurious.append(item.task)
    return score


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def aggregate(scores: list[DocScore]) -> dict:
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    pairs = [p for s in scores for p in s.pairs]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "documents": len(scores),
        "gold_actions": tp + fn,
        "predicted_items": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "matched": len(pairs),
        "owner_accuracy": _ratio(sum(o for o, _ in pairs), len(pairs)),
        "due_accuracy": _ratio(sum(d for _, d in pairs), len(pairs)),
    }


def load_gold(path: Path | None = None) -> dict:
    return json.loads((path or corpus_dir() / "gold.json").read_text(encoding="utf-8"))


def gold_digest(path: Path | None = None) -> str:
    return hashlib.sha256((path or corpus_dir() / "gold.json").read_bytes()).hexdigest()


def _corpus_file(name: str) -> Path:
    """Gold ``file`` entries must stay inside the corpus folder (no absolute paths, no ``..``)."""
    base = corpus_dir().resolve()
    if not isinstance(name, str):
        raise InputError("gold entry file must be a string")
    target = (base / name).resolve()
    if not target.is_relative_to(base):
        raise InputError(f"gold entry points outside the corpus: {safe_text(name)!r}")
    return target


def _plain(value: str) -> str:
    return safe_text(one_line(value))


def run_split(
    gold: dict,
    split: str,
    base: Options,
    *,
    llm_url: str | None = None,
    allow_remote: bool = False,
) -> tuple[dict, list[DocScore]]:
    scores: list[DocScore] = []
    backends: set[str] = set()
    for entry in gold["documents"]:
        if entry["split"] != split:
            continue
        text = decode_bytes(_corpus_file(entry["file"]).read_bytes())
        options = replace(
            base,
            attendees=tuple(entry["attendees"]),
            meeting_date=entry["meeting_date"],
            timezone=entry["timezone"],
        )
        extraction = extract_document(Document(entry["id"], text, None), options)
        if llm_url:
            extraction = llm.refine(
                extraction,
                text,
                set(entry["attendees"]),
                llm_url,
                allow_remote=allow_remote,
                min_confidence=base.min_confidence,
            )
        backends.add(extraction.backend)
        scores.append(score_document(entry, text, extraction))
    formats = sorted({s.fmt for s in scores})
    result = {
        "documents": len(scores),
        "backends": sorted(backends),
        "overall": aggregate(scores),
        "by_format": {f: aggregate([s for s in scores if s.fmt == f]) for f in formats},
    }
    return result, scores


def evaluate_splits(
    splits: tuple[str, ...] = SPLITS,
    base: Options | None = None,
    *,
    llm_url: str | None = None,
    allow_remote: bool = False,
    gold: dict | None = None,
) -> tuple[dict, dict[str, list[DocScore]]]:
    gold = gold or load_gold()
    base = base or Options()
    metrics: dict = {
        "schema": SCHEMA_VERSION,
        "gold_sha256": gold_digest(),
        "llm_pass": bool(llm_url),
        "options": {"use_nlp": base.use_nlp, "min_confidence": base.min_confidence},
        "splits": {},
    }
    all_scores: dict[str, list[DocScore]] = {}
    for split in splits:
        result, scores = run_split(
            gold, split, base, llm_url=llm_url, allow_remote=allow_remote
        )
        metrics["splits"][split] = result
        all_scores[split] = scores
    return metrics, all_scores


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return (
            "| "
            + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))
            + " |"
        )

    out = [line(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def _cell(value: float | None) -> str:
    return (
        "n/a"
        if value is None
        else f"{value:.3f}"
        if isinstance(value, float)
        else str(value)
    )


_METRIC_HEADERS = [
    "docs",
    "gold",
    "pred",
    "tp",
    "fp",
    "fn",
    "P",
    "R",
    "F1",
    "owner acc",
    "due acc",
]


def _metric_row(label: str, m: dict) -> list[str]:
    return [
        label,
        str(m["documents"]),
        str(m["gold_actions"]),
        str(m["predicted_items"]),
        str(m["tp"]),
        str(m["fp"]),
        str(m["fn"]),
        _cell(m["precision"]),
        _cell(m["recall"]),
        _cell(m["f1"]),
        _cell(m["owner_accuracy"]),
        _cell(m["due_accuracy"]),
    ]


def summary(metrics: dict) -> str:
    lines = []
    for split, result in metrics["splits"].items():
        rows = [_metric_row("overall", result["overall"])]
        rows += [_metric_row(f, m) for f, m in result["by_format"].items()]
        backends = _plain(", ".join(result["backends"]))
        lines.append(f"split: {split}  backends: {backends}")
        lines.append(_table(["scope", *_METRIC_HEADERS], rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_report(metrics: dict, scores: dict[str, list[DocScore]]) -> str:
    out = ["# Evaluation report", ""]
    out.append(
        f"Gold file SHA-256: `{metrics['gold_sha256']}`. "
        f"LLM pass: {'yes' if metrics['llm_pass'] else 'no'}. "
        f"spaCy NER: {'on' if metrics['options']['use_nlp'] else 'off'}."
    )
    out.append("")
    for split, result in metrics["splits"].items():
        out.append(f"## {split.upper()} split")
        out.append("")
        out.append(f"Backend: {_plain(', '.join(result['backends']))}")
        out.append("")
        rows = [_metric_row("overall", result["overall"])]
        rows += [_metric_row(f, m) for f, m in result["by_format"].items()]
        out.append(_table(["scope", *_METRIC_HEADERS], rows))
        out.append("")
        out.append("### Errors by document")
        out.append("")
        for score in scores[split]:
            if not score.missed and not score.spurious:
                continue
            out.append(f"- `{score.doc_id}`")
            for task in score.missed:
                out.append(f"  - missed: {task}")
            for task in score.spurious:
                out.append(f"  - spurious: {task}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_results(
    metrics: dict, scores: dict[str, list[DocScore]], out_dir: Path | None = None
) -> tuple[Path, Path]:
    target = out_dir or results_dir()
    target.mkdir(parents=True, exist_ok=True)
    metrics_path = target / "metrics.json"
    report_path = target / "report.md"
    metrics_path.write_text(
        json.dumps(metrics, indent=4) + "\n", encoding="utf-8", newline="\n"
    )
    report_path.write_text(
        build_report(metrics, scores), encoding="utf-8", newline="\n"
    )
    return metrics_path, report_path
