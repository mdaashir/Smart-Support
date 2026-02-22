"""Tests for Milestone 1 — real data, TF-IDF+LogReg, urgency, queue, cross-validation."""

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from src.config import CATEGORIES
from src.models.tfidf_logreg import TfidfLogRegClassifier
from src.routing.urgency import detect_urgency, PRIORITY_HIGH, PRIORITY_NORMAL
from src.routing.queue import TicketQueue


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_data():
    """Load a small slice of the real dataset for fast tests."""
    from src.data.dataset_loader import load_dataset
    df = load_dataset()
    # Take a stratified 5 % sample for speed
    _, sample = train_test_split(
        df, test_size=0.05, stratify=df["label"], random_state=42,
    )
    return sample


@pytest.fixture(scope="module")
def trained_logreg(real_data):
    clf = TfidfLogRegClassifier()
    X_train, _, y_train, _ = train_test_split(
        real_data["text"], real_data["label"],
        test_size=0.2, stratify=real_data["label"], random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


# ── Data tests ───────────────────────────────────────────────────────────

def test_dataset_has_required_columns(real_data):
    for col in ("text", "label", "priority_num"):
        assert col in real_data.columns


def test_categories_match_pdf(real_data):
    """PDF specifies exactly 3 categories: Billing, Technical, Legal."""
    assert set(real_data["label"].unique()) == set(CATEGORIES)


# ── Model tests ──────────────────────────────────────────────────────────

def test_logreg_accuracy(real_data, trained_logreg):
    _, X_test, _, y_test = train_test_split(
        real_data["text"], real_data["label"],
        test_size=0.2, stratify=real_data["label"], random_state=42,
    )
    metrics = trained_logreg.evaluate(X_test, y_test)
    assert metrics["accuracy"] >= 0.50  # real data — more realistic bar


def test_logreg_predict_returns_valid_classes(trained_logreg):
    preds = trained_logreg.predict([
        "refund request", "server down", "legal compliance notice",
    ])
    for p in preds:
        assert p in set(CATEGORIES)


def test_cross_validation_no_overfit(real_data):
    """CV variance should be low → model generalises, not memorises."""
    clf = TfidfLogRegClassifier()
    X, y = real_data["text"], real_data["label"]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf.pipeline, X, y, cv=skf, scoring="f1_macro")
    assert scores.std() < 0.15, f"High CV variance: {scores}"


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
