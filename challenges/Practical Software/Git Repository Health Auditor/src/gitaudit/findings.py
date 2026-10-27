"""Finding model, severity ordering and the rule catalogue shared by every check and reporter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from gitaudit.safety import fingerprint


class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: str) -> Severity:
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            choices = ", ".join(s.name.lower() for s in cls)
            raise ValueError(f"severity must be one of: {choices}") from exc

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def sarif_level(self) -> str:
        if self >= Severity.HIGH:
            return "error"
        if self == Severity.MEDIUM:
            return "warning"
        return "note"


@dataclass(frozen=True)
class RuleInfo:
    rule_id: str
    name: str
    category: str
    severity: Severity
    description: str


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: Severity
    confidence: str
    title: str
    category: str
    evidence: str
    remediation: str
    fingerprint: str
    path: str | None = None
    commit: str | None = None
    blob: str | None = None
    ref: str | None = None
    line: int | None = None
    suppressed: bool = False
    extra: dict[str, str | int | None] = field(default_factory=dict)

    def with_suppressed(self, value: bool) -> Finding:
        return Finding(**{**self.__dict__, "suppressed": value})

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule_id,
            "severity": self.severity.label,
            "confidence": self.confidence,
            "category": self.category,
            "title": self.title,
            "evidence": self.evidence,
            "remediation": self.remediation,
            "fingerprint": self.fingerprint,
            "path": self.path,
            "commit": self.commit,
            "blob": self.blob,
            "ref": self.ref,
            "line": self.line,
            "suppressed": self.suppressed,
            "extra": dict(self.extra),
        }


def make_fingerprint(rule_id: str, *key: str) -> str:
    return fingerprint(rule_id, *key)


CATALOG: dict[str, RuleInfo] = {
    info.rule_id: info
    for info in [
        RuleInfo(
            "BR001",
            "merged-branch",
            "branches",
            Severity.LOW,
            "Branch is fully merged into the default branch and can be deleted.",
        ),
        RuleInfo(
            "BR002",
            "stale-unmerged-branch",
            "branches",
            Severity.MEDIUM,
            "Branch has unmerged commits and has not been updated for longer than the stale threshold.",
        ),
        RuleInfo(
            "BR003",
            "gone-upstream",
            "branches",
            Severity.MEDIUM,
            "Local branch tracks an upstream that no longer exists on the remote.",
        ),
        RuleInfo(
            "BR004",
            "long-lived-branch",
            "branches",
            Severity.LOW,
            "Unmerged branch that diverged from the default branch long ago.",
        ),
        RuleInfo(
            "BR005",
            "tag-non-commit",
            "branches",
            Severity.LOW,
            "Tag points at a blob or tree instead of a commit or tag object.",
        ),
        RuleInfo(
            "BR006",
            "default-branch-unknown",
            "branches",
            Severity.INFO,
            "The default branch could not be determined; merged/unmerged checks were skipped.",
        ),
        RuleInfo(
            "BL001",
            "oversized-blob-in-head",
            "blobs",
            Severity.MEDIUM,
            "A blob in the audited tree is larger than the configured maximum.",
        ),
        RuleInfo(
            "BL002",
            "oversized-blob-in-history",
            "blobs",
            Severity.MEDIUM,
            "A blob larger than the maximum is no longer in the audited tree but still bloats history.",
        ),
        RuleInfo(
            "BL003",
            "lfs-candidate",
            "blobs",
            Severity.LOW,
            "Binary or archive file committed to the tree at or above the Git LFS threshold.",
        ),
        RuleInfo(
            "BL004",
            "gc-advised",
            "blobs",
            Severity.LOW,
            "Object store has loose or unreachable objects that `git gc` would clean up.",
        ),
        RuleInfo(
            "SEC001",
            "aws-access-key-id",
            "secrets",
            Severity.HIGH,
            "AWS access key ID in a committed file.",
        ),
        RuleInfo(
            "SEC002",
            "aws-secret-access-key",
            "secrets",
            Severity.HIGH,
            "AWS secret access key assigned to a variable.",
        ),
        RuleInfo(
            "SEC003",
            "github-token",
            "secrets",
            Severity.HIGH,
            "GitHub classic or app token (ghp_, gho_, ghu_, ghs_, ghr_).",
        ),
        RuleInfo(
            "SEC004",
            "github-fine-grained-pat",
            "secrets",
            Severity.HIGH,
            "GitHub fine-grained personal access token.",
        ),
        RuleInfo(
            "SEC005",
            "gitlab-pat",
            "secrets",
            Severity.HIGH,
            "GitLab personal access token.",
        ),
        RuleInfo(
            "SEC006", "slack-token", "secrets", Severity.HIGH, "Slack API token (xox*)."
        ),
        RuleInfo(
            "SEC007",
            "slack-webhook",
            "secrets",
            Severity.HIGH,
            "Slack incoming webhook URL.",
        ),
        RuleInfo(
            "SEC008",
            "stripe-key",
            "secrets",
            Severity.HIGH,
            "Stripe live or test secret/restricted key.",
        ),
        RuleInfo(
            "SEC009",
            "google-api-key",
            "secrets",
            Severity.HIGH,
            "Google API key (AIza...).",
        ),
        RuleInfo(
            "SEC010",
            "private-key-block",
            "secrets",
            Severity.CRITICAL,
            "PEM or OpenSSH private key block.",
        ),
        RuleInfo(
            "SEC011",
            "jwt",
            "secrets",
            Severity.MEDIUM,
            "JSON Web Token; often a sample, but a live session token is still a leak.",
        ),
        RuleInfo(
            "SEC012",
            "connection-string-password",
            "secrets",
            Severity.HIGH,
            "URL with an embedded username and password.",
        ),
        RuleInfo(
            "SEC013",
            "generic-credential-assignment",
            "secrets",
            Severity.MEDIUM,
            "password, secret, api_key or token assigned a literal value.",
        ),
        RuleInfo(
            "SEC014",
            "registry-auth-token",
            "secrets",
            Severity.HIGH,
            "npm or registry _authToken/_auth value.",
        ),
        RuleInfo(
            "SEC015",
            "high-entropy-literal",
            "secrets",
            Severity.LOW,
            "Quoted high-entropy token-like literal that matched no provider rule.",
        ),
        RuleInfo(
            "PTH001",
            "env-file",
            "secrets",
            Severity.HIGH,
            "Dotenv file committed (.env, .env.local ...).",
        ),
        RuleInfo(
            "PTH002",
            "ssh-private-key-file",
            "secrets",
            Severity.HIGH,
            "SSH private key file committed.",
        ),
        RuleInfo(
            "PTH003",
            "key-or-certificate-file",
            "secrets",
            Severity.HIGH,
            "Private key or keystore file (.pem, .key, .p12, .pfx, .jks) committed.",
        ),
        RuleInfo(
            "PTH004",
            "terraform-state",
            "secrets",
            Severity.HIGH,
            "Terraform state file committed; state often contains plaintext secrets.",
        ),
        RuleInfo(
            "PTH005",
            "kubeconfig",
            "secrets",
            Severity.HIGH,
            "Kubernetes kubeconfig committed.",
        ),
        RuleInfo(
            "PTH006",
            "credential-store-file",
            "secrets",
            Severity.HIGH,
            "Credential store committed (.netrc, .git-credentials, .htpasswd, cloud credentials).",
        ),
        RuleInfo(
            "HY001",
            "tracked-ignorable-file",
            "hygiene",
            Severity.LOW,
            "Tracked file matches a pattern that is normally ignored (caches, OS junk, logs).",
        ),
        RuleInfo(
            "HY002",
            "missing-gitignore",
            "hygiene",
            Severity.MEDIUM,
            "No .gitignore at the repository root.",
        ),
        RuleInfo(
            "HY003",
            "empty-gitignore",
            "hygiene",
            Severity.LOW,
            ".gitignore exists but is empty.",
        ),
        RuleInfo(
            "HY004",
            "committed-dependency-dir",
            "hygiene",
            Severity.MEDIUM,
            "Dependency or virtualenv directory is committed (node_modules, .venv, ...).",
        ),
        RuleInfo(
            "HY005",
            "huge-commit",
            "hygiene",
            Severity.LOW,
            "A single commit changes an unusually large number of files or adds many bytes.",
        ),
        RuleInfo(
            "HY006",
            "mixed-line-endings",
            "hygiene",
            Severity.LOW,
            "Text file mixes CRLF and LF line endings.",
        ),
        RuleInfo(
            "HY007",
            "case-colliding-paths",
            "hygiene",
            Severity.MEDIUM,
            "Two paths differ only by case; they collide on Windows and macOS.",
        ),
        RuleInfo(
            "HY008",
            "submodule-url-oddity",
            "hygiene",
            Severity.MEDIUM,
            "Submodule URL uses an unusual or risky transport, or embeds credentials.",
        ),
        RuleInfo(
            "HY009",
            "missing-license",
            "hygiene",
            Severity.LOW,
            "No LICENSE file at the repository root.",
        ),
        RuleInfo(
            "HY010",
            "missing-readme",
            "hygiene",
            Severity.LOW,
            "No README file at the repository root.",
        ),
        RuleInfo(
            "HY011",
            "unverified-commit",
            "hygiene",
            Severity.LOW,
            "Recent commit on the default branch has no good signature (opt-in check).",
        ),
        RuleInfo(
            "HY012",
            "suspicious-author-identity",
            "hygiene",
            Severity.LOW,
            "Commit author has an empty, placeholder or bot-looking email (opt-in check).",
        ),
        RuleInfo(
            "HY013",
            "audit-budget-exhausted",
            "hygiene",
            Severity.HIGH,
            "A time or scan budget ran out, so the results are partial.",
        ),
        RuleInfo(
            "HY014",
            "blobs-not-content-scanned",
            "hygiene",
            Severity.INFO,
            "Blobs skipped by the content scan: over the size limit or binary.",
        ),
    ]
}


def info(rule_id: str) -> RuleInfo:
    return CATALOG[rule_id]
