"""Milestone 1 — TF-IDF + Logistic Regression classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from backend.config import TFIDF_LOGREG

logger = logging.getLogger(__name__)


class TfidfLogRegClassifier:
    """Scikit-learn pipeline: TF-IDF (word n-grams) → Logistic Regression."""

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=TFIDF_LOGREG["ngram_range"],
                max_features=TFIDF_LOGREG["max_features"],
                min_df=TFIDF_LOGREG["min_df"],
                sublinear_tf=TFIDF_LOGREG["sublinear_tf"],
                stop_words=TFIDF_LOGREG["stop_words"],
            )),
            ("clf", LogisticRegression(
                max_iter=TFIDF_LOGREG["max_iter"],
                C=TFIDF_LOGREG["C"],
                solver=TFIDF_LOGREG["solver"],
                class_weight="balanced",
            )),
        ])

    # ── Training ─────────────────────────────────────────────────────
    def fit(self, X, y) -> "TfidfLogRegClassifier":
        logger.info("Training TF-IDF + LogReg on %d samples …", len(X))
        self.pipeline.fit(X, y)
        return self

    # ── Inference ────────────────────────────────────────────────────
    def predict(self, X):
        return self.pipeline.predict(X)

    @property
    def classes_(self):
        return self.pipeline.classes_

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
        joblib.dump(self.pipeline, path)
        logger.info("Model saved → %s", path)

    def load(self, path: str | Path) -> "TfidfLogRegClassifier":
        self.pipeline = joblib.load(path)
        logger.info("Model loaded ← %s", path)
        return self
