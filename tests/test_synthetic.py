"""Tests for Milestone 1 — synthetic data, TF-IDF+LogReg, urgency, queue."""

import pytest
from sklearn.model_selection import train_test_split

from src.data.synthetic_generator import generate_dataset
from src.models.tfidf_logreg import TfidfLogRegClassifier
from src.routing.urgency import detect_urgency, PRIORITY_HIGH, PRIORITY_NORMAL
from src.routing.queue import TicketQueue


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_df():
    return generate_dataset(n_per_class=500, seed=42)


@pytest.fixture(scope="module")
def trained_logreg(synthetic_df):
    clf = TfidfLogRegClassifier()
    X_train, _, y_train, _ = train_test_split(
        synthetic_df["text"], synthetic_df["category"],
        test_size=0.2, stratify=synthetic_df["category"], random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


# ── Data tests ───────────────────────────────────────────────────────────

def test_dataset_shape(synthetic_df):
    assert synthetic_df.shape == (2000, 2)  # 4 categories × 500


def test_class_balance(synthetic_df):
    counts = synthetic_df["category"].value_counts()
    assert set(counts.index) == {"Billing", "Technical", "HR", "General"}
    for c in counts:
        assert c == 500


# ── Model tests ──────────────────────────────────────────────────────────

def test_logreg_accuracy(synthetic_df, trained_logreg):
    _, X_test, _, y_test = train_test_split(
        synthetic_df["text"], synthetic_df["category"],
        test_size=0.2, stratify=synthetic_df["category"], random_state=42,
    )
    metrics = trained_logreg.evaluate(X_test, y_test)
    assert metrics["accuracy"] >= 0.90


def test_logreg_predict_returns_valid_classes(trained_logreg):
    preds = trained_logreg.predict(["refund request", "server down", "PTO request", "general inquiry"])
    for p in preds:
        assert p in {"Billing", "Technical", "HR", "General"}


# ── Urgency tests ────────────────────────────────────────────────────────

def test_urgency_high():
    assert detect_urgency("This is urgent, fix ASAP") == PRIORITY_HIGH


def test_urgency_normal():
    assert detect_urgency("Please update my billing address") == PRIORITY_NORMAL


# ── Queue tests ──────────────────────────────────────────────────────────

def test_queue_ordering(trained_logreg):
    q = TicketQueue()
    q.ingest("server down urgent", classifier=trained_logreg, urgency_fn=detect_urgency)
    q.ingest("please update billing address", classifier=trained_logreg, urgency_fn=detect_urgency)

    assert q.size == 2
    first = q.process_next()
    second = q.process_next()
    assert first["priority"] <= second["priority"]
    assert q.size == 0
