"""Tests for Milestone 2 — TF-IDF(char) + LinearSVC + urgency regressor."""

import pytest
from sklearn.model_selection import train_test_split

from src.config import CATEGORIES
from src.models.tfidf_svc import TfidfSVCClassifier
from src.routing.router import route_ticket
from src.routing.urgency_regressor import UrgencyRegressor


@pytest.fixture(scope="module")
def svc_artifacts():
    from src.data.dataset_loader import load_dataset
    df = load_dataset()
    # 5 % stratified sample for speed
    _, sample = train_test_split(
        df, test_size=0.05, stratify=df["label"], random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        sample["text"], sample["label"],
        test_size=0.2, stratify=sample["label"], random_state=42,
    )
    clf = TfidfSVCClassifier()
    clf.fit(X_train, y_train)
    return clf, X_test, y_test, sample


def test_svc_accuracy(svc_artifacts):
    clf, X_test, y_test, _ = svc_artifacts
    metrics = clf.evaluate(X_test, y_test)
    assert metrics["accuracy"] >= 0.45  # real data


def test_svc_predict_one(svc_artifacts):
    clf, *_ = svc_artifacts
    assert clf.predict_one("refund request") in set(CATEGORIES)


def test_route_ticket_schema(svc_artifacts):
    clf, *_ = svc_artifacts
    result = route_ticket("Invoice issue", "Charged twice for last month", classifier=clf)
    assert "category" in result
    assert "urgency" in result
    assert result["category"] in set(CATEGORIES)
    assert result["urgency"] in {"1(HIGH)", "0(NORMAL)"}


def test_route_ticket_urgency_detection(svc_artifacts):
    clf, *_ = svc_artifacts
    result = route_ticket("URGENT", "server down critical ASAP", classifier=clf)
    assert result["urgency"] == "1(HIGH)"


# ── Urgency regressor ───────────────────────────────────────────────────

def test_urgency_regressor_score_range(svc_artifacts):
    _, _, _, sample = svc_artifacts
    urg = UrgencyRegressor()
    urg.fit(
        sample["text"].tolist()[:500],
        sample["priority_num"].tolist()[:500],
    )
    score = urg.predict_score("URGENT critical server outage ASAP")
    assert 0.0 <= score <= 1.0


def test_urgency_regressor_keyword_fallback():
    """Without a trained model, falls back to keyword heuristic."""
    urg = UrgencyRegressor()
    assert not urg.is_trained
    score = urg.predict_score("urgent critical asap")
    assert score > 0.0
