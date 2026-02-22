"""Tests for Milestone 4 — FastAPI endpoints."""

import os
import pytest
from fastapi.testclient import TestClient

# Force lightweight model for tests
os.environ["MODEL_VARIANT"] = "logreg"

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Health ───────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert set(data["categories"]) == {"Billing", "Technical", "HR", "General"}
    assert data["uptime_seconds"] >= 0


# ── Route endpoint ───────────────────────────────────────────────────────

def test_route_success(client):
    resp = client.post("/route", json={
        "subject": "Invoice issue",
        "body": "Charged twice for last month subscription",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] in {"Billing", "Technical", "HR", "General"}
    assert data["urgency"] in {"1(HIGH)", "0(NORMAL)"}
    assert "model_used" in data


def test_route_urgent(client):
    resp = client.post("/route", json={
        "subject": "URGENT",
        "body": "Server is down critical outage ASAP",
    })
    assert resp.status_code == 200
    assert resp.json()["urgency"] == "1(HIGH)"


def test_route_missing_fields(client):
    resp = client.post("/route", json={"subject": "hello"})
    assert resp.status_code == 422


def test_route_empty_subject(client):
    resp = client.post("/route", json={"subject": "", "body": "some body"})
    assert resp.status_code == 422
