import json
from datetime import date

import facturx
import pytest
from invoicer.cli import main
from pypdf import PdfReader
from smtpserver import smtp_server


def run(*argv):
    return main([str(a) for a in argv])


def out_json(capsys, *argv):
    capsys.readouterr()
    assert run(*argv, "--json") == 0
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def configured(_isolated_home):
    """A real invoicer.toml in the isolated home, as `invoicer init` would write it."""
    (_isolated_home / "invoicer.toml").write_text(
        '[business]\nname = "Ada Ltd"\naddress = "1 Road\\nTown"\ncountry = "GB"\n'
        'email = "ada@example.com"\ncurrency = "EUR"\ntax_id = "GB1"\n'
        'iban = "GB29NWBK60161331926819"\n'
        '[email]\nhost = "127.0.0.1"\nport = 1\nsecurity = "none"\n'
        'from_address = "ada@example.com"\n[pdf]\nlocale = "en_GB"\n',
        encoding="utf-8",
    )
    return _isolated_home


def seed(capsys):
    assert (
        run(
            "client",
            "add",
            "Acme GmbH",
            "--email",
            "ap@acme.example",
            "--country",
            "DE",
        )
        == 0
    )
    assert (
        run(
            "invoice",
            "new",
            "--client",
            "acme",
            "--item",
            "Design | 10 | 85.00 | 19",
            "--item",
            "Hosting | 1 | 20 | 7",
            "--discount",
            "10%",
            "--issue",
            "--date",
            "2026-10-01",
        )
        == 0
    )
    capsys.readouterr()


def test_init_writes_config_and_database_once(_isolated_home, capsys):
    assert run("init") == 0
    assert (_isolated_home / "invoicer.toml").exists() and (
        _isolated_home / "invoicer.db"
    ).exists()
    first = (_isolated_home / "invoicer.toml").read_text()
    capsys.readouterr()
    assert run("init") == 0
    assert "already exists" in capsys.readouterr().out
    assert (_isolated_home / "invoicer.toml").read_text() == first


def test_client_commands(configured, capsys):
    assert (
        run(
            "client",
            "add",
            "Acme GmbH",
            "--email",
            "ap@acme.example",
            "--terms-days",
            30,
        )
        == 0
    )
    assert run("client", "add", "acme gmbh") == 1  # duplicate, case-insensitive
    assert "already exists" in capsys.readouterr().err
    rows = out_json(capsys, "client", "list")
    assert rows[0]["name"] == "Acme GmbH" and rows[0]["terms_days"] == 30
    assert (
        rows[0]["country"] == "GB" and rows[0]["currency"] == "EUR"
    )  # defaults from business
    assert (
        run(
            "client",
            "edit",
            "acme",
            "--email",
            "billing@acme.example",
            "--address",
            "A\\nB",
        )
        == 0
    )
    assert run("client", "add", "Bad", "--email", "nope") == 1
    assert run("client", "rm", "acme") == 0
    assert out_json(capsys, "client", "list") == []


def test_invoice_lifecycle(configured, capsys):
    seed(capsys)
    data = out_json(capsys, "invoice", "show", "INV-2026-0001", "--as-of", "2026-10-05")
    assert (data["status"], data["total"], data["balance"]) == (
        "issued",
        "929.61",
        "929.61",
    )
    assert data["due_date"] == "2026-10-15"
    assert [t["tax"] for t in data["tax"]] == ["1.26", "145.35"]

    assert run("invoice", "pay", "INV-2026-0001", "100", "--method", "bank") == 0
    listing = out_json(capsys, "invoice", "list", "--as-of", "2026-10-20")
    assert listing["outstanding"] == {"EUR": "829.61"}
    assert listing["invoices"][0]["status"] == "overdue"
    assert run("invoice", "pay", "INV-2026-0001", "5000") == 1
    assert "exceeds" in capsys.readouterr().err
    assert run("invoice", "pay", "INV-2026-0001", "829.61") == 0
    assert (
        out_json(capsys, "invoice", "list", "--status", "paid")["invoices"][0][
            "balance"
        ]
        == "0.00"
    )


def test_drafts_then_issue_then_void_and_delete(configured, capsys):
    run("client", "add", "Acme GmbH", "--email", "ap@acme.example")
    run("invoice", "new", "--client", "acme", "--item", "x | 1 | 10")
    run("invoice", "new", "--client", "acme", "--item", "y | 1 | 20")
    capsys.readouterr()
    rows = out_json(capsys, "invoice", "list")["invoices"]
    assert [r["number"] for r in rows] == [None, None] and {
        r["status"] for r in rows
    } == {"draft"}
    assert run("invoice", "issue", "1", "--date", "2026-01-05") == 0
    assert "INV-2026-0001" in capsys.readouterr().out
    assert run("invoice", "void", "INV-2026-0001") == 0
    assert run("invoice", "delete", "1") == 1  # issued/void invoices cannot be deleted
    assert run("invoice", "delete", "2") == 0
    assert out_json(capsys, "invoice", "list")["invoices"][0]["status"] == "void"


def test_invoice_from_a_toml_file(configured, tmp_path, capsys):
    run("client", "add", "Acme GmbH")
    spec = tmp_path / "job.toml"
    spec.write_text(
        'client = "acme"\ndiscount = "5%"\nnotes = "Thanks"\nreference = "PO-1"\ntax = 20\n'
        '[[items]]\ndescription = "Workshop"\nqty = 2\nprice = "500.00"\n'
        '[[items]]\ndescription = "Travel"\nqty = 1\nprice = "40"\ntax = 0\n',
        encoding="utf-8",
    )
    assert run("invoice", "new", "--file", spec, "--issue", "--date", "2026-02-01") == 0
    data = out_json(capsys, "invoice", "show", "INV-2026-0001")
    assert (data["reference"], data["notes"], data["subtotal"]) == (
        "PO-1",
        "Thanks",
        "1040.00",
    )
    assert [line["tax_percent"] for line in data["lines"]] == ["20", "0"]
    assert data["discount"] == "52.00"


def test_json_job_files_and_bad_job_files(configured, tmp_path, capsys):
    run("client", "add", "Acme GmbH")
    good = tmp_path / "job.json"
    good.write_text(
        json.dumps(
            {"client": "acme", "items": [{"description": "x", "qty": 1, "price": "10"}]}
        )
    )
    assert run("invoice", "new", "--file", good) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert run("invoice", "new", "--file", bad) == 1
    huge = tmp_path / "huge.json"
    huge.write_text(" " * 1_000_001)
    assert run("invoice", "new", "--file", huge) == 1
    assert "over 1 MB" in capsys.readouterr().err
    not_object = tmp_path / "list.json"
    not_object.write_text("[1, 2]")
    assert run("invoice", "new", "--file", not_object) == 1


def test_error_messages_not_tracebacks(configured, capsys):
    for argv in (
        ["invoice", "show", "INV-0000"],
        ["invoice", "new", "--item", "x | 1 | 5"],
        ["invoice", "new", "--client", "ghost", "--item", "x | 1 | 5"],
        ["invoice", "pdf", "99"],
    ):
        assert run(*argv) == 1
        err = capsys.readouterr().err
        assert err.startswith("error:") and "Traceback" not in err


def test_pdf_command_writes_a_facturx_pdf(configured, capsys, tmp_path):
    seed(capsys)
    target = tmp_path / "out" / "inv.pdf"
    assert run("invoice", "pdf", "INV-2026-0001", "-o", target) == 0
    assert target.read_bytes().startswith(b"%PDF")
    assert facturx.get_xml_from_pdf(target.read_bytes())[0] == "factur-x.xml"
    assert run("invoice", "pdf", "INV-2026-0001", "--no-facturx", "-o", target) == 0
    assert "/AF" not in PdfReader(target).trailer["/Root"]  # plain PDF: no attached XML
    assert run("invoice", "pdf", "1") == 0  # default location under the home folder
    assert (configured / "pdf" / "INV-2026-0001.pdf").exists()


def test_send_is_a_dry_run_unless_asked(configured, capsys):
    seed(capsys)
    with smtp_server() as server:
        # port 1 in the config: a real send would fail to connect, so success proves no attempt.
        assert run("invoice", "send", "INV-2026-0001") == 0
        out = capsys.readouterr().out
        assert (
            "dry run" in out and "ap@acme.example" in out and "nothing was sent" in out
        )
        assert not server.messages
    assert out_json(capsys, "invoice", "show", "INV-2026-0001")["emails"] == []


def test_send_for_real_logs_and_prevents_double_sends(configured, capsys):
    seed(capsys)
    with smtp_server() as server:
        text = (
            (configured / "invoicer.toml")
            .read_text()
            .replace("port = 1", f"port = {server.port}")
        )
        (configured / "invoicer.toml").write_text(text)
        assert run("invoice", "send", "INV-2026-0001", "--send") == 0
        assert "sent INV-2026-0001 to ap@acme.example" in capsys.readouterr().out
        (message,) = server.messages
        pdf = next(message.iter_attachments()).get_payload(decode=True)
        assert PdfReader(
            __import__("io").BytesIO(pdf)
        ).pages  # a real, parseable PDF arrived
        assert run("invoice", "send", "INV-2026-0001", "--send") == 1
        assert "already sent" in capsys.readouterr().err
        assert (
            run(
                "invoice",
                "send",
                "INV-2026-0001",
                "--send",
                "--resend",
                "--to",
                "other@x.example",
            )
            == 0
        )
        assert (
            len(server.messages) == 2 and server.messages[1]["To"] == "other@x.example"
        )
    emails = out_json(capsys, "invoice", "show", "INV-2026-0001")["emails"]
    assert [e["status"] for e in emails] == ["sent", "sent"]


def test_failed_send_exits_nonzero_and_is_recorded(configured, capsys):
    seed(capsys)
    with smtp_server(replies=["550 rejected"]) as server:
        text = (
            (configured / "invoicer.toml")
            .read_text()
            .replace("port = 1", f"port = {server.port}")
        )
        (configured / "invoicer.toml").write_text(text)
        assert run("invoice", "send", "INV-2026-0001", "--send") == 1
    assert "FAILED" in capsys.readouterr().err
    emails = out_json(capsys, "invoice", "show", "INV-2026-0001")["emails"]
    assert emails[0]["status"] == "failed" and "550" in emails[0]["error"]
    # A failure does not count as sent: it can be sent again without --resend.
    assert run("invoice", "send", "INV-2026-0001") == 0


def test_drafts_cannot_be_sent(configured, capsys):
    run("client", "add", "Acme GmbH", "--email", "ap@acme.example")
    run("invoice", "new", "--client", "acme", "--item", "x | 1 | 10")
    capsys.readouterr()
    assert run("invoice", "send", "1", "--send") == 1
    assert "issue it first" in capsys.readouterr().err


def test_reminders_over_time(configured, capsys):
    seed(capsys)  # due 2026-10-15
    assert run("remind", "--as-of", "2026-10-20") == 0
    assert "no reminders due" in capsys.readouterr().out
    assert run("remind", "--as-of", "2026-10-23") == 0  # 8 days late: stage 7
    assert "would remind: INV-2026-0001" in capsys.readouterr().out
    with smtp_server() as server:
        text = (
            (configured / "invoicer.toml")
            .read_text()
            .replace("port = 1", f"port = {server.port}")
        )
        (configured / "invoicer.toml").write_text(text)
        assert run("remind", "--as-of", "2026-10-23", "--send") == 0
        assert run("remind", "--as-of", "2026-10-23", "--send") == 0  # idempotent
        assert len(server.messages) == 1
        assert "Reminder" in server.messages[0]["Subject"]
        assert run("remind", "--as-of", "2026-11-20", "--send") == 0  # stage 30
        assert len(server.messages) == 2
    emails = out_json(capsys, "invoice", "show", "INV-2026-0001")["emails"]
    assert [(e["kind"], e["stage"]) for e in emails] == [
        ("reminder", 7),
        ("reminder", 30),
    ]


def test_recurring_commands(configured, capsys):
    run("client", "add", "Acme GmbH", "--email", "ap@acme.example")
    assert (
        run(
            "recurring",
            "add",
            "Retainer",
            "--client",
            "acme",
            "--item",
            "Retainer | 1 | 500 | 19",
            "--frequency",
            "monthly",
            "--start",
            "2026-01-31",
        )
        == 0
    )
    capsys.readouterr()
    assert run("recurring", "run", "--as-of", "2026-03-31") == 0
    out = capsys.readouterr().out
    assert out.count("created INV-2026-") == 3 and "(2026-02-28)" in out
    assert run("recurring", "run", "--as-of", "2026-03-31") == 0
    assert "nothing due" in capsys.readouterr().out
    assert (
        run("recurring", "list") == 0 and "next 2026-04-30" in capsys.readouterr().out
    )
    assert run("recurring", "stop", "Retainer") == 0
    assert run("recurring", "run", "--as-of", "2026-12-31") == 0
    assert "nothing due" in capsys.readouterr().out
    assert run("recurring", "stop", "Nope") == 1


def test_recurring_auto_send(configured, capsys):
    run("client", "add", "Acme GmbH", "--email", "ap@acme.example")
    run(
        "recurring",
        "add",
        "R",
        "--client",
        "acme",
        "--item",
        "x | 1 | 10",
        "--start",
        "2026-01-01",
        "--auto-send",
    )
    with smtp_server() as server:
        text = (
            (configured / "invoicer.toml")
            .read_text()
            .replace("port = 1", f"port = {server.port}")
        )
        (configured / "invoicer.toml").write_text(text)
        assert (
            run("recurring", "run", "--as-of", "2026-01-01") == 0
        )  # created, not sent: no --send
        assert not server.messages
        run(
            "recurring",
            "add",
            "S",
            "--client",
            "acme",
            "--item",
            "y | 1 | 10",
            "--start",
            "2026-02-01",
            "--auto-send",
        )
        # Both definitions are due on 2026-02-01 (R's February invoice and S's first), and both
        # auto-send, so two messages go out.
        assert run("recurring", "run", "--as-of", "2026-02-01", "--send") == 0
        assert len(server.messages) == 2


def test_summary(configured, capsys):
    assert run("summary") == 0 and "nothing outstanding" in capsys.readouterr().out
    seed(capsys)
    assert run("summary", "--as-of", "2026-10-20") == 0
    assert "EUR: 929.61 outstanding, 929.61 overdue" in capsys.readouterr().out


def test_config_check(configured, capsys):
    assert run("config-check") == 0
    assert capsys.readouterr().out.rstrip().endswith("ok")
    (configured / "invoicer.toml").write_text(
        (configured / "invoicer.toml")
        .read_text()
        .replace('security = "none"', 'security = "none"\nusername = "u"')
    )
    assert run("config-check") == 1
    assert "password_env" in capsys.readouterr().out


def test_help_lists_every_command(capsys):
    assert run("--help") == 0
    out = capsys.readouterr().out
    for command in (
        "client",
        "invoice",
        "recurring",
        "remind",
        "serve",
        "summary",
        "init",
    ):
        assert command in out


def test_a_dated_invoice_defaults_to_today(configured, capsys):
    run("client", "add", "Acme GmbH")
    run("invoice", "new", "--client", "acme", "--item", "x | 1 | 10", "--issue")
    data = out_json(capsys, "invoice", "show", f"INV-{date.today().year}-0001")
    assert data["issue_date"] == date.today().isoformat()
