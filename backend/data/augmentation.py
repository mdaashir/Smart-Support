"""Data augmentation via SMOTE on TF-IDF features (replaces naive synthetic).

Only used as a fallback when the real dataset is unavailable or to balance
underrepresented classes.  Generates *augmented* samples in feature space
rather than naive template strings — avoids overfitting to templates.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def augment_with_smote(
    X_texts: pd.Series | list[str],
    y_labels: pd.Series | list[str],
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Apply SMOTE to TF-IDF vectors to balance classes.

    Returns (X_resampled, y_resampled, fitted_vectorizer).
    Both are numpy arrays; X is sparse→dense after SMOTE.
    """
    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=15_000, stop_words="english",
    )
    X_vec = vec.fit_transform(X_texts)

    class_counts = pd.Series(y_labels).value_counts()
    logger.info("Pre-SMOTE class distribution:\n%s", class_counts.to_string())

    min_count = class_counts.min()
    max_count = class_counts.max()

    if min_count < max_count * 0.5:
        logger.info("Applying SMOTE to balance classes …")
        sm = SMOTE(random_state=random_state, k_neighbors=min(5, min_count - 1))
        X_res, y_res = sm.fit_resample(X_vec, y_labels)
        logger.info("Post-SMOTE: %d samples", X_res.shape[0])
    else:
        logger.info("Classes are roughly balanced — skipping SMOTE.")
        X_res = X_vec.toarray() if hasattr(X_vec, "toarray") else X_vec
        y_res = np.array(y_labels)

    return X_res, y_res, vec
