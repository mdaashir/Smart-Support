"""Load and prepare the real multilingual ticket dataset (Milestone 2+)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import LABEL_MAP, DEFAULT_LABEL, RANDOM_STATE

logger = logging.getLogger(__name__)


def _map_queue_to_category(queue: str) -> str:
    q = str(queue).lower()
    for keyword, cat in LABEL_MAP.items():
        if keyword in q:
            return cat
    return DEFAULT_LABEL


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV and return a DataFrame with ``text`` and ``label`` columns.

    Expected CSV columns: ``subject``, ``body``, ``queue``.
    """
    df = pd.read_csv(path)
    df["text"] = (
        df["subject"].fillna("").astype(str) + " " +
        df["body"].fillna("").astype(str)
    ).str.lower().str.strip()
    df["label"] = df["queue"].apply(_map_queue_to_category)

    logger.info(
        "Loaded %d tickets — class distribution:\n%s",
        len(df),
        df["label"].value_counts().to_string(),
    )
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    text_col: str = "text",
    label_col: str = "label",
):
    """Stratified train/test split. Returns (X_train, X_test, y_train, y_test)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df[text_col],
        df[label_col],
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )
