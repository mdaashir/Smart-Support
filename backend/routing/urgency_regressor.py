"""Milestone 2 — Urgency regression: continuous score S ∈ [0, 1].

Combines keyword heuristics with a lightweight ML model trained on
priority labels from the real dataset.  Falls back to the regex
heuristic when no trained model is available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from backend.config import URGENCY_WEBHOOK_THRESHOLD
from backend.routing.urgency import _URGENT_RE  # shared compiled regex

logger = logging.getLogger(__name__)


def _keyword_score(text: str) -> float:
    """Simple keyword-density score in [0, 1]."""
    matches = _URGENT_RE.findall(text.lower())
    if not matches:
        return 0.0
    # More matches → higher score, but clamp at 1.0
    return min(len(matches) * 0.25, 1.0)


class UrgencyRegressor:
    """Predict urgency score S ∈ [0, 1] from ticket text.

    Training target: priority_num mapped to [0, 1] via
        ``S = 1 - (priority_num - 1) / 4``
    so critical=1 → S=1.0, very_low=5 → S=0.0.
    """

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    # ── Training ─────────────────────────────────────────────────────
    def fit(self, texts: list[str], priority_nums: list[int]) -> "UrgencyRegressor":
        """Train on ticket texts and numeric priorities (1–5)."""
        # Map priority_num to S ∈ [0, 1]
        y = np.array([1.0 - (p - 1) / 4.0 for p in priority_nums])
        y = np.clip(y, 0.0, 1.0)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), max_features=10_000, stop_words="english",
            )),
            ("ridge", Ridge(alpha=1.0)),
        ])
        self.pipeline.fit(texts, y)
        logger.info("Urgency regressor trained on %d samples", len(texts))
        return self

    # ── Inference ────────────────────────────────────────────────────
    def predict_score(self, text: str) -> float:
        """Return urgency score S ∈ [0, 1].  Falls back to keyword heuristic."""
        if self.pipeline is not None:
            raw = float(self.pipeline.predict([text])[0])
            # Blend: 70 % model + 30 % keyword heuristic for robustness
            kw = _keyword_score(text)
            score = 0.7 * raw + 0.3 * kw
        else:
            score = _keyword_score(text)
        return float(np.clip(score, 0.0, 1.0))

    def is_urgent(self, text: str) -> bool:
        """True when score exceeds webhook threshold."""
        return self.predict_score(text) > URGENCY_WEBHOOK_THRESHOLD

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Urgency regressor saved → %s", path)

    def load(self, path: str | Path) -> "UrgencyRegressor":
        self.pipeline = joblib.load(path)
        logger.info("Urgency regressor loaded ← %s", path)
        return self
