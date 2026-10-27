"""Four outputs for one `AuditResult`: a terminal table, JSON, SARIF 2.1.0 and a self-contained HTML page.

Every string that came from the audited repository is escaped for its target: Rich `Text` cells
(no markup parsing), `html.escape` for HTML, `json` for JSON and SARIF. The HTML page has no
script and a CSP that blocks all network loads.
"""

from __future__ import annotations

import html
import json
import urllib.parse
from datetime import UTC, datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text

from gitaudit import __version__
from gitaudit.audit import AuditResult
from gitaudit.findings import CATALOG, Finding, Severity
from gitaudit.safety import sanitize

SCHEMA_VERSION = 1
_SEVERITY_STYLE = {
    Severity.CRITICAL: "bold white on red",
    Severity.HIGH: "bold red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}
_SEVERITY_COLOUR = {
    Severity.CRITICAL: "#8b0000",
    Severity.HIGH: "#b00020",
    Severity.MEDIUM: "#8a5a00",
    Severity.LOW: "#1d5f8a",
    Severity.INFO: "#555555",
}


def render_table(
    result: AuditResult, console: Console, limit: int | None = None
) -> None:
    active = result.active()
    table = Table(title=f"gitaudit: {len(active)} findings", show_lines=False)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Rule", no_wrap=True)
    table.add_column("Where", overflow="fold")
    table.add_column("Finding", overflow="fold")
    shown = active if limit is None else active[:limit]
    for f in shown:
        where = f.path or f.ref or (f.commit or "")[:12] or "-"
        table.add_row(
            Text(f.severity.label.upper(), style=_SEVERITY_STYLE[f.severity]),
            Text(f.rule_id),
            Text(sanitize(where, 120)),
            Text(sanitize(f.title, 200)),
        )
    console.print(table)
    if limit is not None and len(active) > limit:
        console.print(
            Text(f"... {len(active) - limit} more; use --format json for all")
        )
    suppressed = len(result.findings) - len(active)
    if suppressed:
        console.print(Text(f"{suppressed} findings accepted by baseline (not counted)"))
    console.print(
        Text(
            f"commit {result.commit[:12]} on {sanitize(result.default_branch or 'unknown branch', 60)}; "
            f"{result.stats.get('commits', 0)} commits, "
            f"{result.stats.get('distinct_blobs', 0)} blobs, {result.stats.get('seconds', 0)} s"
        )
    )


def _finding_dict(f: Finding) -> dict[str, object]:
    return f.to_dict()


def render_json(result: AuditResult) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "tool": {"name": "gitaudit", "version": __version__},
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "repo": result.repo,
        "ref": result.ref,
        "commit": result.commit,
        "default_branch": result.default_branch,
        "config": result.config_source,
        "stats": result.stats,
        "findings": [_finding_dict(f) for f in result.findings],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _sarif_uri(path: str) -> str:
    return urllib.parse.quote(path.replace("\\", "/"), safe="/")


def render_sarif(result: AuditResult) -> str:
    used = sorted({f.rule_id for f in result.findings})
    rules = [
        {
            "id": rid,
            "name": CATALOG[rid].name,
            "shortDescription": {"text": CATALOG[rid].description},
            "defaultConfiguration": {"level": CATALOG[rid].severity.sarif_level},
            "properties": {"category": CATALOG[rid].category},
        }
        for rid in used
    ]
    results = []
    for f in result.findings:
        item: dict[str, object] = {
            "ruleId": f.rule_id,
            "level": f.severity.sarif_level,
            "message": {
                "text": sanitize(f"{f.title}. {f.evidence}. {f.remediation}", 1000)
            },
            "partialFingerprints": {"gitauditFingerprint/v1": f.fingerprint},
            "properties": {"confidence": f.confidence, "severity": f.severity.label},
        }
        if f.path:
            region = {"startLine": f.line} if f.line else None
            loc: dict[str, object] = {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": _sarif_uri(f.path),
                        "uriBaseId": "%SRCROOT%",
                    },
                }
            }
            if region:
                loc["physicalLocation"]["region"] = region  # type: ignore[index]
            item["locations"] = [loc]
        if f.suppressed:
            item["suppressions"] = [{"kind": "external", "justification": "baseline"}]
        results.append(item)
    doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "gitaudit",
                        "version": __version__,
                        "informationUri": "https://github.com/",
                        "rules": rules,
                    }
                },
                "results": results,
                # A partial audit (budget or cap reached) did not complete its checks.
                "invocations": [{"executionSuccessful": not result.partial}],
            }
        ],
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


_HTML_STYLE = """
:root { color-scheme: light dark; --bg:#fbfbfd; --fg:#1b1b1f; --muted:#5b5b66; --line:#dcdce4; }
@media (prefers-color-scheme: dark) { :root { --bg:#16161a; --fg:#ececf1; --muted:#a0a0ad; --line:#33333d; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 16px; }
h1 { font-size: 1.4rem; margin: 0 0 8px; }
.meta { color: var(--muted); font-size: 0.9rem; margin-bottom: 16px; word-break: break-all; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th, td { text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid var(--line); }
td.path, td.ev { word-break: break-all; font-family: ui-monospace, monospace; }
.sev { font-weight: 600; text-transform: uppercase; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; color: #fff; }
.count { font-weight: 600; }
"""


def render_html(result: AuditResult) -> str:
    rows = []
    for f in result.findings:
        colour = _SEVERITY_COLOUR[f.severity]
        where = f.path or f.ref or (f.commit or "")[:12] or "-"
        rows.append(
            "<tr>"
            f'<td><span class="sev" style="background:{colour}">{html.escape(f.severity.label)}</span></td>'
            f"<td>{html.escape(f.rule_id)}</td>"
            f'<td class="path">{html.escape(sanitize(where, 200))}</td>'
            f"<td>{html.escape(sanitize(f.title, 300))}</td>"
            f'<td class="ev">{html.escape(sanitize(f.evidence, 300))}</td>'
            f"<td>{html.escape(sanitize(f.remediation, 400))}</td>"
            f"<td>{'accepted' if f.suppressed else 'open'}</td>"
            "</tr>"
        )
    body_rows = "\n".join(rows) or '<tr><td colspan="7">No findings.</td></tr>'
    active = len(result.active())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<meta name="referrer" content="no-referrer">
<title>gitaudit report</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
<h1>gitaudit: <span class="count">{active}</span> open findings</h1>
<div class="meta">repo {html.escape(result.repo)} &middot; ref {html.escape(result.ref)} &middot;
commit {html.escape(result.commit[:12])} &middot; default {html.escape(result.default_branch or "unknown")}
&middot; {html.escape(str(result.stats.get("seconds", 0)))} s</div>
<table>
<thead><tr><th>Severity</th><th>Rule</th><th>Where</th><th>Finding</th><th>Evidence</th><th>Remediation</th><th>State</th></tr></thead>
<tbody>
{body_rows}
</tbody>
</table>
</body>
</html>
"""
