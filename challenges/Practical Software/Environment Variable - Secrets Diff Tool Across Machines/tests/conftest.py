import pytest


@pytest.fixture(autouse=True)
def _no_ambient_passphrase(monkeypatch):
    monkeypatch.delenv("ENVDIFF_PASSPHRASE", raising=False)
