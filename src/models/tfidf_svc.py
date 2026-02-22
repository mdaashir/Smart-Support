"""Milestone 2 — Multilingual TF-IDF (char n-grams) + LinearSVC classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

from src.config import TFIDF_SVC

logger = logging.getLogger(__name__)


class TfidfSVCClassifier:
    """Char-level TF-IDF + LinearSVC — handles multilingual tickets."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer=TFIDF_SVC["analyzer"],
            ngram_range=TFIDF_SVC["ngram_range"],
            min_df=TFIDF_SVC["min_df"],
            max_features=TFIDF_SVC["max_features"],
        )
        self.clf = LinearSVC(
            C=TFIDF_SVC["C"],
            class_weight="balanced",
            max_iter=5_000,
        )

    # ── Training ─────────────────────────────────────────────────────
    def fit(self, X, y) -> "TfidfSVCClassifier":
        logger.info("Training TF-IDF(char) + LinearSVC on %d samples …", len(X))
        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)
        return self

    # ── Inference ────────────────────────────────────────────────────
    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.clf.predict(X_vec)

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]

    @property
    def classes_(self):
        return self.clf.classes_

    # ── Evaluation ───────────────────────────────────────────────────
    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "report": report,
        }
        logger.info(
            "Evaluation — acc=%.4f  macro-F1=%.4f  weighted-F1=%.4f",
            metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"],
        )
        return metrics

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "clf": self.clf}, path)
        logger.info("Model saved → %s", path)

    def load(self, path: str | Path) -> "TfidfSVCClassifier":
        bundle = joblib.load(path)
        self.vectorizer = bundle["vectorizer"]
        self.clf = bundle["clf"]
        logger.info("Model loaded ← %s", path)
        return self
