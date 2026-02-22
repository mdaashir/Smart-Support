"""Milestone 2 -- Multilingual TF-IDF (word + char n-grams) + LinearSVC classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from backend.config import TFIDF_SVC

logger = logging.getLogger(__name__)


class TfidfSVCClassifier:
    """FeatureUnion of word + char TF-IDF vectors fed into LinearSVC.

    Combining word n-grams (semantic) with char n-grams (morphological,
    multilingual) consistently outperforms either individually.
    """

    def __init__(self) -> None:
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=TFIDF_SVC["word_ngram_range"],
            max_features=TFIDF_SVC["word_max_features"],
            min_df=TFIDF_SVC["word_min_df"],
            sublinear_tf=TFIDF_SVC["word_sublinear_tf"],
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_SVC["char_ngram_range"],
            max_features=TFIDF_SVC["char_max_features"],
            min_df=TFIDF_SVC["char_min_df"],
        )
        self.pipeline = Pipeline([
            ("features", FeatureUnion([
                ("word", word_vec),
                ("char", char_vec),
            ])),
            ("clf", LinearSVC(
                C=TFIDF_SVC["C"],
                class_weight="balanced",
                max_iter=5_000,
                dual=True,
            )),
        ])

    # Training
    def fit(self, X, y) -> "TfidfSVCClassifier":
        logger.info("Training word+char TF-IDF + LinearSVC on %d samples ...", len(X))
        self.pipeline.fit(X, y)
        return self

    # Inference
    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]

    @property
    def classes_(self):
        return self.pipeline.named_steps["clf"].classes_

    # Evaluation
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
            "Evaluation -- acc=%.4f  macro-F1=%.4f  weighted-F1=%.4f",
            metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"],
        )
        return metrics

    # Persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Model saved -> %s", path)

    def load(self, path: str | Path) -> "TfidfSVCClassifier":
        self.pipeline = joblib.load(path)
        logger.info("Model loaded <- %s", path)
        return self
