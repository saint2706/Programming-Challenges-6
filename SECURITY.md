# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, not in a public issue or pull request.

Use GitHub's private reporting: open the repository's **Security** tab, choose **Report a
vulnerability**, and describe the problem (what, where, and how to reproduce it). Or go straight to
<https://github.com/saint2706/Programming-Challenges-6/security/advisories/new>.

This is a personal project, so there is no response-time guarantee, but reports are read and
acknowledged on a best-effort basis.

## Scope

This repository is a collection of standalone programming challenges, not a deployed service. Only
the `main` branch is maintained. Reports about code under `challenges/` (for example XSS in a web
demo, path traversal on a file input, or command injection in a CLI or workflow) are in scope.
Vulnerabilities in third-party dependencies should go to the upstream project; Dependabot tracks
them here.
