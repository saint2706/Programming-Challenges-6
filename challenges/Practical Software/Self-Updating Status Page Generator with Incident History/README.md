# Self-Updating Status Page Generator with Incident History

**Category:** Practical Software
**Difficulty:** X

Poll services, publish a static status page, log incidents.

**Status:** Not started

## Where this is actually used

**Upptime runs a status page on GitHub alone.** Scheduled GitHub Actions workflows check endpoints,
with a shortest interval of 5 minutes, and response times are committed to the git history. Incidents
are GitHub Issues that open on failure and close on recovery, and GitHub Pages serves the generated
site. It is the closest working version of this brief: poll, publish static files, log incidents.

**Incidents as an append-only record.** Keeping incidents in issues or commits means each entry is
dated and reviewable after the fact, which is what an incident history is for.

**Where the gap is.** Scheduled runs are not a continuous probe, so a 5-minute check can miss a short
outage that a self-hosted poller would catch. Nothing here is built yet, so nothing is measured.
