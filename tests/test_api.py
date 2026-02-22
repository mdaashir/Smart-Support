"""Tests for FastAPI endpoints — Milestones 1-3."""

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
    assert set(data["categories"]) == {"Billing", "Technical", "Legal"}
    assert data["uptime_seconds"] >= 0


# ── Route endpoint ───────────────────────────────────────────────────────

def test_route_success(client):
    resp = client.post("/route", json={
        "subject": "Invoice issue",
        "body": "Charged twice for last month subscription",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] in {"Billing", "Technical", "Legal"}
    assert data["urgency"] in {"1(HIGH)", "0(NORMAL)"}
    assert "model_used" in data
    assert "urgency_score" in data


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


# ── Async route (202) ───────────────────────────────────────────────────

def test_route_async_accepted(client):
    """Milestone 2: POST /route/async returns 202 with job_id."""
    resp = client.post("/route/async", json={
        "subject": "test ticket",
        "body": "testing async broker pattern",
    })
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "accepted"


# ── Batch route ──────────────────────────────────────────────────────────

def test_route_batch(client):
    resp = client.post("/route/batch", json={
        "tickets": [
            {"subject": "billing question", "body": "need a refund"},
            {"subject": "tech issue", "body": "app crashes on startup"},
        ]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    for r in data["results"]:
        assert r["category"] in {"Billing", "Technical", "Legal"}


# ── Stats ────────────────────────────────────────────────────────────────

def test_stats(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "tickets_routed" in data
    assert "webhook_fires" in data
    assert "master_incidents" in data


# ── Agents status ────────────────────────────────────────────────────────

def test_agents_endpoint(client):
    resp = client.get("/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data
