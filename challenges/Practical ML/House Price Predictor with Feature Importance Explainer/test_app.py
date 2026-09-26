from __future__ import annotations

import pytest
from app import STATE, app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_models_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": True}


def test_dashboard_renders_a_field_for_every_editable_feature(client):
    r = client.get("/")
    assert r.status_code == 200
    for col in STATE["editable_features"]:
        assert f'name="{col}"' in r.text


def _valid_form() -> dict[str, str]:
    default_row = STATE["default_row"]
    return {col: str(default_row.iloc[0][col]) for col in STATE["editable_features"]}


def test_predict_happy_path_shows_both_model_predictions(client):
    r = client.post("/predict", data=_valid_form())
    assert r.status_code == 200
    assert "Ridge:" in r.text
    assert "LightGBM:" in r.text
    assert "$" in r.text


def test_predict_with_missing_fields_falls_back_to_defaults(client):
    # Submitting nothing at all should still produce a valid prediction --
    # every field defaults to the training median/mode.
    r = client.post("/predict", data={})
    assert r.status_code == 200
    assert "Ridge:" in r.text


def test_predict_rejects_garbage_numeric_value_with_422_not_500(client):
    numeric_cols = [
        c for c in STATE["editable_features"] if c not in STATE["categorical_cols"]
    ]
    assert numeric_cols, "expected at least one numeric editable feature"
    form = _valid_form()
    form[numeric_cols[0]] = "not-a-number"
    r = client.post("/predict", data=form)
    assert r.status_code == 422


def test_predict_rejects_unknown_categorical_value_with_422_not_500(client):
    categorical_cols = [
        c for c in STATE["editable_features"] if c in STATE["categorical_cols"]
    ]
    assert categorical_cols, "expected at least one categorical editable feature"
    form = _valid_form()
    form[categorical_cols[0]] = "DefinitelyNotARealCategory"
    r = client.post("/predict", data=form)
    assert r.status_code == 422


def test_predict_rejects_adversarial_script_payload_in_categorical_field(client):
    categorical_cols = [
        c for c in STATE["editable_features"] if c in STATE["categorical_cols"]
    ]
    assert categorical_cols
    form = _valid_form()
    form[categorical_cols[0]] = "<script>alert(1)</script>"
    r = client.post("/predict", data=form)
    # Not a real training category, so it's rejected before it ever
    # reaches HTML rendering. FastAPI's default error body is JSON, where
    # a raw "<script>" string is inert -- the real invariant is that the
    # value never reaches an *HTML* response.
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/json")
