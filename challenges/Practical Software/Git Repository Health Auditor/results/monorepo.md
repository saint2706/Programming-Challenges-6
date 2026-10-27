# Monorepo run

Repository: this challenge monorepo (`Programming-Challenges-6`), full history, default options.
Command: `uv run gitaudit <repo> --fail-on none --format json -o data/monorepo.json` (output git-ignored).

- Commits: 458, distinct blobs: 3793, content-scanned: 3615, binary: 173, over the size limit: 5
- Audit time: 17.5 s (earlier run before the security review: 16.5 s)
- Peak memory: not measured (no portable counter in the standard library on Windows)

| Rule   | Findings |
| ------ | -------- |
| BL003  | 4        |
| BL004  | 1        |
| HY004  | 1        |
| HY014  | 1        |
| SEC008 | 1        |
| SEC013 | 61       |
| SEC015 | 50       |

Triage summary: SEC015 (entropy) fell from 834 to 50 after requiring upper, lower and digit characters, which removed package names and slugs. The remaining SEC015 hits are mostly CamelCase identifiers in JSON metrics files. SEC013 (keyword rule) rose from 41 to 61 in the security review. The blanket `password` stopword was removed, so `password = "..."` test values and fixtures now count. Most of the rise is in two files: 16 hits in `Local Network File Server with Access Logs/server.py` and 16 in `Voice Memo Transcriber & Organizer/tests/test_web.py`. The rest are spread across about 20 more files. Honouring inline `gitaudit:allow` comments (now opt-in) changes nothing on this repository. SEC008 and SEC013 on tests/test_no_leak.py are deliberate leak-test fixtures. The 4 BL003 hits are sample data and textures (gz, glb, hdr) that belong in Git LFS if the repo grows. HY004 is a vendored stylesheet. HY014 is one INFO finding: 173 binary blobs and 5 over the size limit were not content-scanned.
