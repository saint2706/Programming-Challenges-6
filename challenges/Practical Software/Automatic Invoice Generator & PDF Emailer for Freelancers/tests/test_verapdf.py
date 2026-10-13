"""Independent PDF/A-3b validation with veraPDF, the industry reference validator.

veraPDF is a Java tool, so this runs only where it is installed: put ``verapdf`` on PATH or
point ``INVOICER_VERAPDF`` at it (it also needs ``java`` on PATH). Everywhere else it is skipped
rather than failed; the local run is what backs the README's PDF/A claim.
"""

import os
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest
from invoicer import render, service


def _verapdf() -> str | None:
    candidate = os.environ.get("INVOICER_VERAPDF") or shutil.which("verapdf")
    if candidate is None:
        default = Path.home() / "tools" / "verapdf" / "verapdf.bat"
        candidate = str(default) if default.exists() else None
    return candidate if candidate and shutil.which("java") else None


pytestmark = pytest.mark.skipif(
    _verapdf() is None,
    reason="veraPDF (and Java) not installed; see this module's docstring",
)


def validate(pdf: Path) -> str:
    done = subprocess.run(
        [_verapdf(), "--flavour", "3b", str(pdf)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return done.stdout


def failed_rules(report: str) -> str:
    return re.search(r'failedRules="(\d+)"', report).group(1)


def test_issued_invoice_is_valid_pdfa3b(session, settings, make_invoice, tmp_path):
    inv = make_invoice(
        lines=("Design & branding | 10 | 85.00 | 19", "Hosting | 1 | 20 | 7"),
        discount="10%",
    )
    service.record_payment(session, inv, "100", paid_on=date(2026, 10, 5))
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(
        render.render_pdf(service.load_view(session, inv, date(2026, 10, 13)), settings)
    )
    report = validate(pdf)
    assert 'isCompliant="true"' in report, report[:1500]
    assert failed_rules(report) == "0"


def test_the_validator_does_reject_a_plain_pdf(
    session, settings, make_invoice, tmp_path
):
    """Control: without PDF/A mode WeasyPrint's output is not PDF/A-3b, so a pass above means
    something."""
    view = service.load_view(session, make_invoice(), date(2026, 10, 13))
    plain = tmp_path / "plain.pdf"
    plain.write_bytes(render.render_pdf(view, settings, facturx=False))
    assert 'isCompliant="false"' in validate(plain)
