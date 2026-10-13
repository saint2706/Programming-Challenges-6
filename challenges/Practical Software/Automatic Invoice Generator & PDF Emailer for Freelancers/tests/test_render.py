import io
import json
from datetime import date

import facturx
import pypdfium2 as pdfium
import pytest
from invoicer import render, service
from invoicer.errors import InvoicerError
from invoicer.facturx_xml import build_data, build_xml
from pypdf import PdfReader


def view_of(session, inv, today=date(2026, 10, 5)):
    return service.load_view(session, inv, today)


def page_text(pdf: bytes) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)


def ink(pdf: bytes) -> int:
    doc = pdfium.PdfDocument(pdf)
    try:
        image = doc[0].render(scale=0.5).to_pil().convert("RGB")
    finally:
        doc.close()
    return sum(1 for px in image.get_flattened_data() if px != (255, 255, 255))


# ------------------------------------------------------------------------------- HTML


def test_html_shows_the_numbers(session, settings, make_invoice):
    html = render.render_html(view_of(session, make_invoice(discount="10%")), settings)
    for expected in (
        "INV-2026-0001",
        "1 Oct 2026",
        "15 Oct 2026",
        "€1,000.00",  # line net
        "Discount (10%)",
        "€900.00",  # net after discount
        "€171.00",  # 19 % of 900
        "€1,071.00",  # total
        "GB29NWBK60161331926819",
        "Acme GmbH",
    ):
        assert expected in html, expected


def test_untrusted_text_is_escaped(session, settings, acme):
    service.edit_client(
        session,
        acme,
        name="<script>alert(1)</script>",
        address='"><img src=x onerror=alert(1)>',
    )
    lines = [service.parse_line("<b>bold</b> & {{ 7*7 }} | 1 | 10", "EUR")]
    inv = service.create_invoice(session, acme, lines, notes="<i>note</i>")
    html = render.render_html(view_of(session, inv), settings)
    for raw in (
        "<script>alert(1)</script>",
        "<img src=x",
        "<b>bold</b>",
        "<i>note</i>",
    ):
        assert raw not in html
    assert "&lt;script&gt;" in html and "&amp;" in html
    assert "{{ 7*7 }}" in html and "49" not in html.split("{{ 7*7 }}")[0][-5:]


def test_user_templates_override_and_run_in_the_sandbox(
    session, settings, make_invoice, tmp_path
):
    folder = tmp_path / "tpl"
    folder.mkdir()
    (folder / "invoice.html").write_text(
        "<p>{{ number }} / {{ seller.name }}</p>", encoding="utf-8"
    )
    custom = type(settings)(**{**settings.__dict__, "template_dir": folder})
    html = render.render_html(view_of(session, make_invoice()), custom)
    assert html == "<p>INV-2026-0001 / Ada Lovelace Consulting</p>"
    (folder / "invoice.html").write_text("{{ ''.__class__.__mro__ }}", encoding="utf-8")
    with pytest.raises(Exception, match="unsafe|SecurityError|access"):
        render.render_html(view_of(session, make_invoice()), custom)


def test_missing_variables_in_a_template_are_errors_not_blanks(
    session, settings, make_invoice, tmp_path
):
    folder = tmp_path / "tpl"
    folder.mkdir()
    (folder / "invoice.html").write_text("{{ nonexistent_field }}", encoding="utf-8")
    custom = type(settings)(**{**settings.__dict__, "template_dir": folder})
    with pytest.raises(Exception, match="nonexistent_field"):
        render.render_html(view_of(session, make_invoice()), custom)


def test_stamps(session, settings, make_invoice):
    inv = make_invoice()
    assert 'class="stamp issued"' not in render.render_html(
        view_of(session, inv), settings
    )
    service.record_payment(session, inv, "1190")
    assert ">PAID<" in render.render_html(view_of(session, inv), settings)
    other = make_invoice()
    service.void_invoice(session, other)
    assert ">VOID<" in render.render_html(view_of(session, other), settings)


def test_draft_renders_without_a_number(session, settings, acme):
    inv = service.create_invoice(
        session, acme, [service.parse_line("x | 1 | 10", "EUR")]
    )
    html = render.render_html(view_of(session, inv), settings)
    assert ">DRAFT<" in html and "Acme GmbH" in html


def test_issued_invoice_uses_the_frozen_parties_not_the_live_ones(
    session, settings, acme, make_invoice
):
    inv = make_invoice()
    service.edit_client(session, acme, name="Renamed Corp")
    html = render.render_html(view_of(session, inv), settings)
    assert "Acme GmbH" in html and "Renamed Corp" not in html


def test_logo_is_embedded_and_validated(session, settings, make_invoice, tmp_path):
    from PIL import Image

    png = tmp_path / "logo.png"
    Image.new("RGB", (40, 20), (200, 0, 0)).save(png)
    business = type(settings.business)(**{**settings.business.__dict__, "logo": png})
    with_logo = type(settings)(**{**settings.__dict__, "business": business})
    assert 'src="data:image/png;base64,' in render.render_html(
        view_of(session, make_invoice()), with_logo
    )
    text = tmp_path / "evil.txt"
    text.write_text("not an image")
    business = type(settings.business)(**{**settings.business.__dict__, "logo": text})
    bad = type(settings)(**{**settings.__dict__, "business": business})
    with pytest.raises(InvoicerError, match="not an image"):
        render.render_html(view_of(session, make_invoice()), bad)


def test_pdf_filename_is_filesystem_safe():
    assert render.pdf_filename("INV-2026-0001", 1) == "INV-2026-0001.pdf"
    assert render.pdf_filename("../../etc/passwd", 1) == ".._.._etc_passwd.pdf"
    assert render.pdf_filename(None, 7) == "draft-7.pdf"
    assert "/" not in render.pdf_filename("a/b\\c:d", 1)


# -------------------------------------------------------------------------------- PDF


def test_pdf_has_the_invoice_text(session, settings, make_invoice):
    pdf = render.render_pdf(view_of(session, make_invoice()), settings, facturx=False)
    assert pdf.startswith(b"%PDF")
    text = page_text(pdf)
    assert "INV-2026-0001" in text and "€1,190.00" in text and "Acme GmbH" in text
    assert ink(pdf) > 1000  # something visible was drawn


def test_external_resources_are_never_fetched(
    session, settings, make_invoice, tmp_path
):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    folder = tmp_path / "tpl"
    folder.mkdir()
    (folder / "invoice.html").write_text(
        f'<img src="{secret.as_uri()}"><link rel="stylesheet" href="http://127.0.0.1:9/x.css">'
        f"<p>ok {{{{ number }}}}</p>",
        encoding="utf-8",
    )
    custom = type(settings)(**{**settings.__dict__, "template_dir": folder})
    # WeasyPrint logs and skips blocked resources; the render still succeeds without them.
    pdf = render.render_pdf(view_of(session, make_invoice()), custom, facturx=False)
    assert "ok INV-2026-0001" in page_text(pdf)
    assert b"TOP SECRET" not in pdf


def test_the_fetcher_allows_only_data_urls():
    fetcher = render._safe_fetcher(render.load_weasyprint())
    assert fetcher.fetch("data:text/plain;base64,aGk=").read() == b"hi"
    for url in (
        "file:///etc/hosts",
        "http://127.0.0.1:9/x.png",
        "https://example.com/x.css",
        "ftp://h/x",
    ):
        with pytest.raises(ValueError, match="disallowed protocol"):
            fetcher.fetch(url)


# ----------------------------------------------------------------------- PDF/A + Factur-X


def _root(pdf: bytes):
    return PdfReader(io.BytesIO(pdf)).trailer["/Root"]


def test_issued_pdf_embeds_a_valid_facturx_xml(session, settings, make_invoice):
    inv = make_invoice(discount="10%")
    pdf = render.render_pdf(view_of(session, inv), settings)
    name, xml = facturx.get_xml_from_pdf(pdf)[:2]
    assert name == "factur-x.xml"
    facturx.xml_check_xsd(
        xml, flavor="factur-x", level="basicwl"
    )  # raises when invalid
    assert b"urn:factur-x.eu:1p0:basicwl" in xml and b"INV-2026-0001" in xml
    root = _root(pdf)
    assert "/AF" in root  # associated file: how PDF/A-3 links the XML to the document
    assert "factur-x.xml" in str(root["/Names"]["/EmbeddedFiles"])
    assert "INV-2026-0001" in page_text(pdf)  # still a readable invoice


def test_pdf_declares_pdfa3_and_the_facturx_profile(session, settings, make_invoice):
    pdf = render.render_pdf(view_of(session, make_invoice()), settings)
    xmp = _root(pdf)["/Metadata"].get_object().get_data().decode("utf-8")
    assert "<pdfaid:part>3</pdfaid:part>" in xmp
    assert "<fx:DocumentFileName>factur-x.xml</fx:DocumentFileName>" in xmp
    assert "<fx:ConformanceLevel>BASIC WL</fx:ConformanceLevel>" in xmp


def test_drafts_and_opt_out_get_plain_pdfs(session, settings, acme, make_invoice):
    draft = service.create_invoice(
        session, acme, [service.parse_line("x | 1 | 10", "EUR")]
    )
    pdf = render.render_pdf(
        view_of(session, draft), settings
    )  # facturx is on, but a draft has no number
    assert "/AF" not in _root(pdf)
    assert "/AF" not in _root(
        render.render_pdf(view_of(session, make_invoice()), settings, facturx=False)
    )
    assert "/AF" in _root(
        render.render_pdf(view_of(session, make_invoice()), settings)
    )  # control


def test_xml_amounts_match_the_invoice(session, settings, make_invoice):
    inv = make_invoice(
        lines=(
            "Design | 10 | 85.00 | 19",
            "Hosting | 1 | 20 | 7",
            "Travel | 1 | 12.50 | 0",
        ),
        discount="10%",
    )
    service.record_payment(session, inv, "100")
    view = view_of(session, inv)
    data = build_data(view)
    assert data["BT-1"] == "INV-2026-0001"
    assert (data["BT-106"], data["BT-107"], data["BT-109"]) == (
        "882.50",
        "88.25",
        "794.25",
    )
    assert (data["BT-110"], data["BT-112"]) == ("146.61", "940.86")
    assert (data["BT-113"], data["BT-115"]) == ("100.00", "840.86")
    assert [(g["BT-118"], g["BT-119"], g["BT-117"]) for g in data["BG-23"]] == [
        ("Z", "0.00", "0.00"),
        ("S", "7.00", "1.26"),
        ("S", "19.00", "145.35"),
    ]
    # One allowance per tax rate, adding up to the discount.
    assert sum(float(a["amount"]) for a in data["BG-20"]) == pytest.approx(88.25)
    xml = build_xml(view)
    facturx.xml_check_xsd(xml, flavor="factur-x", level="basicwl")
    for fragment in (b"INV-2026-0001", b">840.86<", b">940.86<", b"Discount"):
        assert fragment in xml


def test_xml_money_is_exact_for_awkward_amounts(session, settings, acme):
    lines = [service.parse_line("Part | 3 | 33.33 | 21", "EUR")] * 3
    inv = service.issue_invoice(
        session,
        service.create_invoice(session, acme, lines),
        settings,
        date(2026, 10, 1),
    )
    data = build_data(view_of(session, inv))
    # 9 x 33.33 = 299.97 net; 21 % -> 62.9937 -> 62.99
    assert (data["BT-109"], data["BT-110"], data["BT-112"]) == (
        "299.97",
        "62.99",
        "362.96",
    )


def test_xml_for_a_zero_decimal_currency(session, settings):
    client = service.add_client(session, "Tokyo KK", country="JP", currency="JPY")
    inv = service.issue_invoice(
        session,
        service.create_invoice(
            session, client, [service.parse_line("Work | 3 | 10000 | 10", "JPY")]
        ),
        settings,
        date(2026, 4, 1),
    )
    data = build_data(view_of(session, inv))
    assert (data["BT-5"], data["BT-112"]) == ("JPY", "33000")
    facturx.xml_check_xsd(
        build_xml(view_of(session, inv)), flavor="factur-x", level="basicwl"
    )


def test_xml_requires_an_issued_invoice(session, settings, acme):
    draft = service.create_invoice(
        session, acme, [service.parse_line("x | 1 | 10", "EUR")]
    )
    with pytest.raises(InvoicerError, match="issued"):
        build_data(view_of(session, draft))


def test_xml_survives_unicode_and_reference(session, settings, acme):
    service.edit_client(
        session,
        acme,
        name="Müller & Söhne GmbH",
        address="Königsallee 1\n40212 Düsseldorf",
    )
    inv = service.create_invoice(
        session,
        acme,
        [service.parse_line("Beratung | 1 | 100 | 19", "EUR")],
        reference="PO-ä-1",
    )
    service.issue_invoice(session, inv, settings, date(2026, 10, 1))
    xml = build_xml(view_of(session, inv))
    assert (
        "Müller &amp; Söhne GmbH".encode() in xml
        or "Müller & Söhne GmbH".encode() in xml
    )
    assert b"PO-" in xml
    facturx.xml_check_xsd(xml, flavor="factur-x", level="basicwl")


def test_snapshot_drives_the_xml(session, settings, acme, make_invoice):
    inv = make_invoice()
    service.edit_client(session, acme, name="Changed Name")
    snap = json.loads(inv.snapshot)
    assert (
        build_data(view_of(session, inv))["BG-7"]["name"]
        == snap["buyer"]["name"]
        == "Acme GmbH"
    )
