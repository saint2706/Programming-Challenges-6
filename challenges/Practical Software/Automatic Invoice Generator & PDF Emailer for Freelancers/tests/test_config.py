import pytest
from invoicer.config import (
    SAMPLE_CONFIG,
    load_settings,
    number_scope,
    validate_number_format,
)
from invoicer.errors import InvoicerError
from invoicer.paths import project_root


def write(home, text):
    path = home / "invoicer.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_a_file(_isolated_home):
    settings = load_settings()
    assert settings.business.currency == "USD"
    assert settings.number_format == "INV-{year}-{seq:04d}"
    assert settings.reminder_days == (7, 14, 30)
    assert settings.root == _isolated_home
    assert settings.db_path == _isolated_home / "invoicer.db"


def test_the_sample_config_is_valid_and_complete(_isolated_home):
    write(_isolated_home, SAMPLE_CONFIG)
    s = load_settings()
    assert s.business.name == "Ada Lovelace Consulting"
    assert (s.business.country, s.business.currency) == ("GB", "GBP")
    assert s.email.security == "starttls" and s.email.port == 587
    assert s.email.from_address == "ada@example.com"
    assert s.reminder_days == (7, 14, 30)
    assert s.locale == "en_GB" and s.facturx is True


def test_relative_paths_resolve_against_the_config_folder(_isolated_home):
    write(
        _isolated_home,
        '[business]\nlogo = "logo.png"\n[pdf]\ntemplate_dir = "tpl"\n[email]\nca_file = "ca.pem"\n',
    )
    s = load_settings()
    assert s.business.logo == _isolated_home / "logo.png"
    assert s.template_dir == _isolated_home / "tpl"
    assert s.email.ca_file == _isolated_home / "ca.pem"


def test_smtp_password_comes_from_the_environment(_isolated_home, monkeypatch):
    write(_isolated_home, '[email]\npassword_env = "MY_SMTP_PW"\n')
    s = load_settings()
    assert s.email.password is None
    monkeypatch.setenv("MY_SMTP_PW", "hunter2")
    assert s.email.password == "hunter2"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[business]\ncountry = 'DEU'", "two-letter"),
        ("[business]\ncurrency = 'ZZ'", "currency"),
        ("[business]\nname = 5", "must be a string"),
        ("[email]\nsecurity = 'tls13'", "security"),
        ("[email]\nport = 0", "port"),
        ("[email]\nport = '587'", "port"),
        ("[reminders]\ndays_after_due = [0, 7]", "positive"),
        ("[reminders]\ndays_after_due = 'soon'", "positive"),
        ("[numbering]\nformat = 'INV-{year}'", "exactly one"),
        ("business = 3", "must be a table"),
        ("not toml [", "cannot read"),
    ],
)
def test_bad_config_is_a_clear_error(_isolated_home, text, message):
    write(_isolated_home, text)
    with pytest.raises(InvoicerError, match=message):
        load_settings()


def test_reminder_days_are_sorted_and_deduplicated(_isolated_home):
    write(_isolated_home, "[reminders]\ndays_after_due = [30, 7, 7, 14]\n")
    assert load_settings().reminder_days == (7, 14, 30)


@pytest.mark.parametrize(
    "fmt",
    ["INV-{year}-{seq:04d}", "{seq}", "{year}{month:02d}/{seq:03d}", "A{seq:d}-{year}"],
)
def test_valid_number_formats(fmt):
    assert validate_number_format(fmt) == fmt


@pytest.mark.parametrize(
    "fmt",
    [
        "{seq.__class__}",  # attribute access reaches into objects
        "{seq[0]}",
        "{0}",
        "{}",
        "{year!r}-{seq}",
        "{seq:{year}}",  # nested format spec
        "{year}",  # no sequence
        "{seq}{seq}",  # two sequences
        "{other}-{seq}",
        "{seq",
        "{seq:zzz}",
    ],
)
def test_dangerous_or_broken_number_formats_are_refused(fmt):
    with pytest.raises(InvoicerError):
        validate_number_format(fmt)


def test_number_scope_restarts_the_sequence_only_when_the_number_changes():
    fmt = "INV-{year}-{seq:04d}"
    assert number_scope(fmt, 2026, 1) == number_scope(fmt, 2026, 12) == "INV-2026-#"
    assert number_scope(fmt, 2027, 1) == "INV-2027-#"
    assert number_scope("{seq}", 2026, 1) == number_scope("{seq}", 2030, 5) == "#"
    assert number_scope("{year}{month:02d}-{seq}", 2026, 3) == "202603-#"


def test_project_root_respects_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv("INVOICER_HOME", str(tmp_path / "elsewhere"))
    assert project_root() == tmp_path / "elsewhere"
    monkeypatch.delenv("INVOICER_HOME")
    assert (project_root() / "pyproject.toml").exists()
