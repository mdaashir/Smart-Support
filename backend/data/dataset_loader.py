"""Load real open-source multilingual ticket data (Milestone 2 +).

Uses **Tobi-Bueck/customer-support-tickets** from HuggingFace Hub
(61 765 real bilingual EN/DE support tickets with queue, priority,
language, and tags).  Downloads once and caches as local CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backend.config import (
    CACHED_CSV,
    DEFAULT_LABEL,
    HF_DATASET_ID,
    HF_DATASET_SPLIT,
    PRIORITY_MAP,
    QUEUE_TO_CATEGORY,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


# ── Queue → category mapping ────────────────────────────────────────────

def _map_queue_to_category(queue: str) -> str:
    q = str(queue).strip().lower()
    if q in QUEUE_TO_CATEGORY:
        return QUEUE_TO_CATEGORY[q]
    # Partial-match fallback for the long-tail niche queues
    for keyword, cat in QUEUE_TO_CATEGORY.items():
        if keyword in q:
            return cat
    return DEFAULT_LABEL


def _map_priority(pri: str) -> int:
    return PRIORITY_MAP.get(str(pri).strip().lower(), 3)


# ── Download / cache ────────────────────────────────────────────────────

def _download_dataset() -> pd.DataFrame:
    """Pull from HuggingFace Hub → CSV cache."""
    from datasets import load_dataset as hf_load

    logger.info("Downloading %s from HuggingFace Hub …", HF_DATASET_ID)
    ds = hf_load(HF_DATASET_ID, split=HF_DATASET_SPLIT)
    df = ds.to_pandas()

    CACHED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHED_CSV, index=False)
    logger.info("Cached → %s (%d rows)", CACHED_CSV, len(df))
    return df


# ── Public API ───────────────────────────────────────────────────────────

def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the real ticket dataset.

    1. If *path* is given, reads that CSV directly.
    2. Otherwise uses the local cache.
    3. If no cache exists, downloads from HuggingFace.

    Returns a DataFrame with columns:
        ``text``, ``label``, ``priority``, ``language``,
        ``type``, ``queue_raw``, ``tags``
    """
    if path is not None:
        df = pd.read_csv(path)
    elif CACHED_CSV.exists():
        df = pd.read_csv(CACHED_CSV)
        logger.info("Loaded cached data (%d rows) from %s", len(df), CACHED_CSV)
    else:
        df = _download_dataset()

    # ── Feature engineering ──────────────────────────────────────────
    df["text"] = (
        df["subject"].fillna("").astype(str) + " " +
        df["body"].fillna("").astype(str)
    ).str.strip()

    df["label"] = df["queue"].apply(_map_queue_to_category)
    df["priority_num"] = df["priority"].apply(_map_priority)
    df["language"] = df["language"].fillna("en")

    # Preserve raw queue for analysis
    df["queue_raw"] = df["queue"]

    # Collapse tag columns into a single comma-separated string
    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    df["tags"] = df[tag_cols].apply(
        lambda row: ", ".join(t for t in row if pd.notna(t) and t), axis=1
    )

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
    """Stratified train/test split → (X_train, X_test, y_train, y_test)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df[text_col],
        df[label_col],
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )
