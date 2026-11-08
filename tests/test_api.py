# API tests: do the doors work, and are they locked?
import json
import os

os.environ["API_KEY"] = "ci-test-key"      # test-only key, set before the app loads

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
GOOD = {"X-API-Key": "ci-test-key"}
FLOW = json.load(open("tests/sample_flow.json", encoding="utf-8-sig"))

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_predict_works_with_key():
    r = client.post("/predict", json=FLOW, headers=GOOD)
    assert r.status_code == 200
    assert r.json()["prediction"] in ("normal", "attack")

def test_no_key_is_rejected():
    assert client.post("/predict", json=FLOW).status_code == 401

def test_wrong_key_is_rejected():
    assert client.post("/predict", json=FLOW, headers={"X-API-Key": "nope"}).status_code == 401

def test_bad_protocol_is_rejected():
    assert client.post("/predict", json={**FLOW, "protocol_type": "icmpx"}, headers=GOOD).status_code == 422

def test_negative_value_is_rejected():
    assert client.post("/predict", json={**FLOW, "src_bytes": -5}, headers=GOOD).status_code == 422

def test_extra_field_is_rejected():
    assert client.post("/predict", json={**FLOW, "hacker": 1}, headers=GOOD).status_code == 422
