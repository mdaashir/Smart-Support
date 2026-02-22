"""Tests for Milestone 2 — TF-IDF(char) + LinearSVC (synthetic fallback)."""

import pytest
from sklearn.model_selection import train_test_split

from src.data.synthetic_generator import generate_dataset
from src.models.tfidf_svc import TfidfSVCClassifier
from src.routing.router import route_ticket


@pytest.fixture(scope="module")
def svc_artifacts():
    df = generate_dataset(n_per_class=500, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"],
        test_size=0.2, stratify=df["category"], random_state=42,
    )
    clf = TfidfSVCClassifier()
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


def test_svc_accuracy(svc_artifacts):
    clf, X_test, y_test = svc_artifacts
    metrics = clf.evaluate(X_test, y_test)
    assert metrics["accuracy"] >= 0.85


def test_svc_predict_one(svc_artifacts):
    clf, *_ = svc_artifacts
    assert clf.predict_one("refund request") in {"Billing", "Technical", "HR", "General"}


def test_route_ticket_schema(svc_artifacts):
    clf, *_ = svc_artifacts
    result = route_ticket("Invoice issue", "Charged twice for last month", classifier=clf)
    assert "category" in result
    assert "urgency" in result
    assert result["category"] in {"Billing", "Technical", "HR", "General"}
    assert result["urgency"] in {"1(HIGH)", "0(NORMAL)"}


def test_route_ticket_urgency_detection(svc_artifacts):
    clf, *_ = svc_artifacts
    result = route_ticket("URGENT", "server down critical ASAP", classifier=clf)
    assert result["urgency"] == "1(HIGH)"
