"""Check 3: leaked secrets in file contents and file paths.

Content rules run over every distinct blob once (added lines of every commit are blobs). Each
line is scanned in overlapping 4 KiB windows, so one huge minified line costs linear time and a
token that straddles a window edge is still seen. Regexes use bounded quantifiers only.

Scoring is static: severity comes from the rule, confidence from pattern specificity and entropy.
Nothing is verified against a live provider. Secrets are never stored in full: see `redact`.
"""

from __future__ import annotations

import fnmatch
import itertools
import math
import posixpath
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from gitaudit.findings import CATALOG, Finding, Severity, make_fingerprint
from gitaudit.safety import redact, sanitize

WINDOW = 4096
OVERLAP = 256
MAX_ALLOW_VALUE = 512

MAX_HITS_PER_BLOB = 200

# Matched as substrings of the secret value. Kept specific: a broad word such as "password" or
# "todo" would hide real values that happen to contain it.
DEFAULT_STOPWORDS = (
    "example", "placeholder", "changeme", "change_me", "dummy", "sample", "redacted",
    "your_", "your-", "yourtoken", "xxxx", "****",
    "<", "${", "{{", "%s", "process.env", "os.environ", "getenv",
)  # fmt: skip

DATA_URI = re.compile(r"data:[A-Za-z0-9.+/-]{1,60};base64,[A-Za-z0-9+/=]{0,200000}")
HEX_HASH = re.compile(r"[0-9a-fA-F]{32,128}")
UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
INTEGRITY = re.compile(r"(?:sha|md)\d{0,3}-.*")


@dataclass(frozen=True)
class SecretRule:
    rule_id: str
    pattern: re.Pattern[str]
    group: int
    confidence: str
    anchors: tuple[str, ...]
    entropy_min: float | None = None
    keyword: bool = False
    whole_match: bool = False


RULES: tuple[SecretRule, ...] = (
    SecretRule("SEC001", re.compile(r"(?<![A-Z0-9])((?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16})(?![A-Z0-9])"),
               1, "high", ("akia", "asia", "agpa", "aida", "aroa", "aipa", "anpa", "anva", "asca")),
    SecretRule("SEC002", re.compile(r"(?i)aws_?secret(?:_?access)?_?key[\"']?\s{0,3}[:=]\s{0,3}[\"']?([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"),
               1, "high", ("secret",), keyword=True),
    SecretRule("SEC003", re.compile(r"(?<![A-Za-z0-9_])((?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36})(?![A-Za-z0-9])"),
               1, "high", ("ghp_", "gho_", "ghu_", "ghs_", "ghr_")),
    SecretRule("SEC004", re.compile(r"(?<![A-Za-z0-9_])(github_pat_[A-Za-z0-9_]{82})(?![A-Za-z0-9_])"),
               1, "high", ("github_pat_",)),
    SecretRule("SEC005", re.compile(r"(?<![A-Za-z0-9_-])(glpat-[A-Za-z0-9_-]{20})(?![A-Za-z0-9_-])"),
               1, "high", ("glpat-",)),
    SecretRule("SEC006", re.compile(r"(?<![A-Za-z0-9])(xox[abposr]-[A-Za-z0-9-]{10,72})(?![A-Za-z0-9-])"),
               1, "high", ("xox",)),
    SecretRule("SEC007", re.compile(r"(https://hooks\.slack\.com/services/T[A-Za-z0-9]{8,12}/B[A-Za-z0-9]{8,12}/[A-Za-z0-9]{24})"),
               1, "high", ("hooks.slack.com",)),
    SecretRule("SEC008", re.compile(r"(?<![A-Za-z0-9])((?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,99})(?![A-Za-z0-9])"),
               1, "high", ("_live_", "_test_")),
    SecretRule("SEC009", re.compile(r"(?<![A-Za-z0-9_-])(AIza[0-9A-Za-z_-]{35})(?![A-Za-z0-9_-])"),
               1, "high", ("aiza",)),
    SecretRule("SEC010", re.compile(r"-----BEGIN ((?:[A-Z0-9]+ )?PRIVATE KEY(?: BLOCK)?)-----"),
               0, "high", ("private key",), whole_match=True),
    SecretRule("SEC011", re.compile(r"(?<![A-Za-z0-9_-])(eyJ[A-Za-z0-9_-]{10,500}\.eyJ[A-Za-z0-9_-]{10,2000}\.[A-Za-z0-9_-]{10,500})(?![A-Za-z0-9_-])"),
               1, "medium", ("eyj",)),
    SecretRule("SEC012", re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]{1,20}://[A-Za-z0-9._~%+-]{1,100}:([^\s/@'\"<>{}$]{3,200})@[A-Za-z0-9._~:\[\]-]{1,255}"),
               1, "high", ("://",)),
    SecretRule("SEC013", re.compile(r"(?i)(?<![A-Za-z])(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|auth[_-]?token|client[_-]?secret|token)(?![A-Za-z])[\"']?\s{0,3}[:=]\s{0,3}[\"']?([^\s\"'`,;<>{}()\[\]$\\]{8,128})"),
               1, "medium", ("password", "passwd", "pwd", "secret", "key", "token"), keyword=True),
    SecretRule("SEC014", re.compile(r"(?i)_auth(?:token)?\s{0,3}=\s{0,3}([A-Za-z0-9+/._~=-]{16,256})"),
               1, "high", ("_auth",)),
)  # fmt: skip

ENTROPY_RULE = SecretRule(
    "SEC015",
    re.compile(r"[\"'`]([A-Za-z0-9+/_=-]{20,200})[\"'`]"),
    1,
    "low",
    ('"', "'", "`"),
    entropy_min=4.0,
)

# Path rules: (rule id, basename globs, excluded basename globs, path-suffix globs).
PATH_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    ("PTH001", (".env", ".env.*"), (".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults"), ()),
    ("PTH002", ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "*.ppk"), (), ()),
    ("PTH003", ("*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore"), (), ()),
    ("PTH004", ("*.tfstate", "*.tfstate.backup"), (), ()),
    ("PTH005", ("kubeconfig", "*.kubeconfig"), (), (".kube/config",)),
    ("PTH006", (".netrc", "_netrc", ".git-credentials", ".htpasswd", ".pgpass", ".pypirc"), (), (".aws/credentials", ".docker/config.json")),
)  # fmt: skip

PATH_ONLY_CONFIDENCE = {
    "PTH003": "low"
}  # keys and PEMs are often public material; SEC010 is authoritative


def shannon(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in Counter(text).values())


def _has_sequential_run(text: str, length: int = 6) -> bool:
    run = 1
    for prev, cur in itertools.pairwise(text):
        run = run + 1 if ord(cur) - ord(prev) == 1 else 1
        if run >= length:
            return True
    return False


class Allowlist:
    """Path globs, value regexes and stopwords. Value checks run on the matched secret only."""

    def __init__(
        self,
        paths: Iterable[str] = (),
        regexes: Iterable[str] = (),
        stopwords: Iterable[str] = (),
        *,
        honor_inline: bool = False,
    ) -> None:
        self.paths = tuple(paths)
        self.regexes = tuple(re.compile(r) for r in regexes)
        self.stopwords = tuple(
            sorted({w.lower() for w in (*DEFAULT_STOPWORDS, *stopwords)})
        )
        # `gitaudit:allow` in a line is honoured only when the operator opts in; a committed
        # comment must not be able to hide a secret from the audit by default.
        self.honor_inline = honor_inline

    def path_allowed(self, path: str) -> bool:
        return any(fnmatch.fnmatchcase(path, pattern) for pattern in self.paths)

    def value_allowed(self, value: str) -> bool:
        if len(value) > MAX_ALLOW_VALUE:
            return False
        lowered = value.lower()
        if any(word in lowered for word in self.stopwords):
            return True
        return any(rx.search(value) for rx in self.regexes)


@dataclass(frozen=True)
class Hit:
    rule_id: str
    line: int
    shape: str
    confidence: str
    severity_override: Severity | None = None
    occurrence: int = 0  # index among hits of the same rule on the same line


def _accept_entropy(value: str) -> bool:
    if len(value) < 20 or HEX_HASH.fullmatch(value) or UUID.fullmatch(value):
        return False
    if INTEGRITY.fullmatch(value) or _has_sequential_run(value):
        return False
    # Random tokens mix upper case, lower case and digits. Package names and slugs
    # (`t9-predictive-text`) are lower case and fail here, which removes most noise.
    if not (
        any(c.isdigit() for c in value)
        and any(c.islower() for c in value)
        and any(c.isupper() for c in value)
    ):
        return False
    return shannon(value) >= (ENTROPY_RULE.entropy_min or 0.0)


def _accept_keyword(value: str) -> tuple[bool, float]:
    entropy = shannon(value)
    if len(set(value)) <= 2 or entropy < 2.5:
        return False, entropy
    return any(not c.isalnum() for c in value) or any(
        c.isdigit() for c in value
    ), entropy


def scan_text(data: bytes, allow: Allowlist, *, entropy: bool = True) -> list[Hit]:
    """Findings for one blob. Caller guarantees the blob is text (no NUL in the sniff window)."""
    """Hits for one blob, capped at MAX_HITS_PER_BLOB (the caller reports a capped blob)."""
    hits: list[Hit] = []
    seen: set[tuple[str, int, str]] = set()
    occurrences: Counter[tuple[str, int]] = Counter()
    for lineno, raw in enumerate(data.split(b"\n"), start=1):
        if allow.honor_inline and b"gitaudit:allow" in raw:
            continue  # inline marker: the author has reviewed this line
        for start in range(0, max(len(raw), 1), WINDOW - OVERLAP):
            window = raw[start : start + WINDOW].decode("utf-8", "replace")
            if "base64," in window:
                window = DATA_URI.sub(" ", window)
            _scan_window(window, lineno, allow, entropy, hits, seen, occurrences)
            if len(hits) >= MAX_HITS_PER_BLOB:
                return hits
            if start + WINDOW >= len(raw):
                break
    return hits


def _scan_window(
    text: str,
    lineno: int,
    allow: Allowlist,
    entropy: bool,
    hits: list[Hit],
    seen: set[tuple[str, int, str]],
    occurrences: Counter[tuple[str, int]],
) -> None:
    lowered = text.lower()
    rules: Sequence[SecretRule] = RULES + ((ENTROPY_RULE,) if entropy else ())
    for rule in rules:
        if not any(anchor in lowered for anchor in rule.anchors):
            continue
        for match in rule.pattern.finditer(text):
            value = match.group(rule.group)
            key = (rule.rule_id, lineno, value)
            if key in seen:
                continue
            if rule.whole_match:
                seen.add(key)
                occ = occurrences[(rule.rule_id, lineno)]
                occurrences[(rule.rule_id, lineno)] += 1
                hits.append(
                    Hit(rule.rule_id, lineno, value, rule.confidence, None, occ)
                )
                continue
            if allow.value_allowed(value):
                continue
            confidence = rule.confidence
            if rule.rule_id == ENTROPY_RULE.rule_id:
                if not _accept_entropy(value):
                    continue
                confidence = "medium" if shannon(value) >= 4.5 else "low"
            elif rule.keyword:
                ok, ent = _accept_keyword(value)
                if not ok:
                    continue
                confidence = "medium" if ent >= 3.5 else "low"
            seen.add(key)
            occ = occurrences[(rule.rule_id, lineno)]
            occurrences[(rule.rule_id, lineno)] += 1
            hits.append(Hit(rule.rule_id, lineno, redact(value), confidence, None, occ))


def path_rule_ids(path: str) -> list[str]:
    """Rule ids whose path globs match. Matching is on the basename, plus suffix globs on the path."""
    base = posixpath.basename(path).lower()
    lowered = path.lower()
    matched: list[str] = []
    for rule_id, globs, excluded, suffixes in PATH_RULES:
        if any(fnmatch.fnmatchcase(base, g) for g in excluded):
            continue
        if any(fnmatch.fnmatchcase(base, g) for g in globs) or any(
            lowered.endswith(s) for s in suffixes
        ):
            matched.append(rule_id)
    return matched


def path_findings(
    path: str, blob: str, commit: str | None, allow: Allowlist
) -> list[Finding]:
    if allow.path_allowed(path):
        return []
    out = []
    for rule_id in path_rule_ids(path):
        info = CATALOG[rule_id]
        confidence = PATH_ONLY_CONFIDENCE.get(rule_id, "high")
        severity = Severity.MEDIUM if rule_id in PATH_ONLY_CONFIDENCE else info.severity
        out.append(
            Finding(
                rule_id=rule_id,
                severity=severity,
                confidence=confidence,
                title=f"{info.name}: {sanitize(path, 160)}",
                category="secrets",
                evidence="file name matches a credential-file pattern; contents not inspected for this rule",
                remediation=(
                    "Rotate anything the file held, untrack it (git rm --cached), add it to .gitignore, "
                    "then purge history: git filter-repo --invert-paths --path <path>"
                ),
                fingerprint=make_fingerprint(rule_id, path, "path", blob),
                path=path,
                commit=commit,
                blob=blob,
            )
        )
    return out


def content_finding(
    rule_id: str,
    path: str,
    commit: str | None,
    blob: str,
    line: int,
    shape: str,
    confidence: str,
    occurrence: int = 0,
) -> Finding:
    info = CATALOG[rule_id]
    return Finding(
        rule_id=rule_id,
        severity=info.severity,
        confidence=confidence,
        title=f"{info.name} in {sanitize(path, 160)}",
        category="secrets",
        evidence=f"line {line}: {shape}",
        remediation=(
            "Treat the credential as compromised: revoke and rotate it. Then rewrite history with "
            "git filter-repo --replace-text (a file of 'literal==>***REMOVED***' lines), and move the value "
            "to an environment variable or secret manager."
        ),
        # Position, not value: two secrets on one line get distinct fingerprints, and the
        # redacted shape is not part of the key.
        fingerprint=make_fingerprint(rule_id, path, blob, str(line), str(occurrence)),
        path=path,
        commit=commit,
        blob=blob,
        line=line,
    )
