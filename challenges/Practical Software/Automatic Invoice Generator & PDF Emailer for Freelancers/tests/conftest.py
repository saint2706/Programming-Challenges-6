import pytest
from invoicer import service
from invoicer.config import Business, EmailSettings, Settings
from invoicer.db import init_db, make_engine, open_session


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own INVOICER_HOME so nothing touches a real database or config."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("INVOICER_HOME", str(home))
    monkeypatch.delenv("INVOICER_SMTP_PASSWORD", raising=False)
    return home


@pytest.fixture
def settings(_isolated_home):
    return Settings(
        business=Business(
            name="Ada Lovelace Consulting",
            address="12 Analytical Way\nLondon N1 9GU",
            country="GB",
            email="ada@example.com",
            tax_id="GB123456789",
            currency="EUR",
            iban="GB29NWBK60161331926819",
            bic="NWBKGB2L",
            payment_note="Please pay by bank transfer.",
        ),
        email=EmailSettings(
            host="127.0.0.1",
            port=0,
            security="none",
            from_address="ada@example.com",
            from_name="Ada Lovelace Consulting",
        ),
        locale="en_GB",
        root=_isolated_home,
    )


@pytest.fixture
def engine(settings):
    engine = make_engine(settings.db_path)
    init_db(engine)
    return engine


@pytest.fixture
def session(engine):
    with open_session(engine) as session:
        yield session


@pytest.fixture
def acme(session):
    return service.add_client(
        session,
        "Acme GmbH",
        email="ap@acme.example",
        address="Hauptstr. 1\n10115 Berlin",
        country="DE",
        tax_id="DE123456789",
        currency="EUR",
    )


@pytest.fixture
def make_invoice(session, acme, settings):
    """Factory: an issued invoice for Acme with one 19 % line (1000.00 net unless overridden)."""
    from datetime import date

    def make(
        *,
        lines=("Consulting | 10 | 100.00 | 19",),
        discount=None,
        issue_on=date(2026, 10, 1),
    ):
        specs = [service.parse_line(spec, "EUR") for spec in lines]
        inv = service.create_invoice(
            session, acme, specs, discount=service.parse_discount(discount, "EUR")
        )
        return service.issue_invoice(session, inv, settings, issue_on)

    return make
