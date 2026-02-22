"""Tests for Milestone 3 — DistilBERT embedder + classifier (smoke tests).

These tests only verify shapes and interface, not full training accuracy,
to keep the test suite fast without GPU.
"""

import pytest
import numpy as np

from backend.config import CATEGORIES
from backend.models.distilbert_classifier import DistilBertEmbedder, DistilBertTicketClassifier


# ── Embedder tests ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    return DistilBertEmbedder(batch_size=2)


def test_embed_shape(embedder):
    texts = ["server down", "refund request", "legal compliance notice"]
    embs = embedder.embed(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (3, 768)


def test_embed_deterministic(embedder):
    texts = ["hello world"]
    a = embedder.embed(texts)
    b = embedder.embed(texts)
    np.testing.assert_allclose(a, b, atol=1e-5)


# ── Classifier smoke test ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def mini_classifier():
    clf = DistilBertTicketClassifier()
    texts = [
        "charged twice", "refund request", "invoice incorrect",
        "server down", "app not loading", "500 internal error",
        "data breach lawsuit", "contract violation", "GDPR complaint",
    ]
    labels = ["Billing"] * 3 + ["Technical"] * 3 + ["Legal"] * 3
    clf.fit(texts, labels)
    return clf


def test_classifier_predict_returns_valid_labels(mini_classifier):
    preds = mini_classifier.predict(["refund needed", "server outage"])
    for p in preds:
        assert p in set(CATEGORIES)


def test_classifier_predict_one(mini_classifier):
    pred = mini_classifier.predict_one("contract dispute about last invoice")
    assert pred in set(CATEGORIES)
