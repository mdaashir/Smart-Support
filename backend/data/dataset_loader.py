"""Load real open-source multilingual ticket data (Milestone 2 +).

Primary source  -- Tobi-Bueck/customer-support-tickets: 61 765 real
bilingual EN/DE support tickets (queue, priority, language, tags).

Secondary source -- bitext/Bitext-customer-support-llm-chatbot-training-dataset:
~26 000 English customer-service samples labelled with fine-grained intents,
mapped to our 3 categories via BITEXT_INTENT_MAP.

Both sources are merged once and cached as merged_tickets.csv; subsequent
runs read from cache so there is no repeated network traffic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backend.config import (
    BITEXT_DATASET_ID,
    BITEXT_DATASET_SPLIT,
    BITEXT_INTENT_MAP,
    BITEXT_MERGED_CSV,
    CACHED_CSV,
    DEFAULT_LABEL,
    HF_DATASET_ID,
    HF_DATASET_SPLIT,
    PRIORITY_MAP,
    QUEUE_TO_CATEGORY,
    RANDOM_STATE,
)
from backend.preprocessing.text_cleaner import clean

logger = logging.getLogger(__name__)


def _map_queue_to_category(queue: str) -> str:
    q = str(queue).strip().lower()
    if q in QUEUE_TO_CATEGORY:
        return QUEUE_TO_CATEGORY[q]
    for keyword, cat in QUEUE_TO_CATEGORY.items():
        if keyword in q:
            return cat
    return DEFAULT_LABEL


def _map_priority(pri: str) -> int:
    return PRIORITY_MAP.get(str(pri).strip().lower(), 3)


# ── Primary dataset (Tobi-Bueck) ──────────────────────────────────────────

def _download_primary() -> pd.DataFrame:
    """Pull Tobi-Bueck/customer-support-tickets from HuggingFace Hub."""
    from datasets import load_dataset as hf_load

    logger.info("Downloading %s from HuggingFace Hub ...", HF_DATASET_ID)
    ds = hf_load(HF_DATASET_ID, split=HF_DATASET_SPLIT)
    df = ds.to_pandas()
    CACHED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHED_CSV, index=False)
    logger.info("Primary dataset cached -> %s (%d rows)", CACHED_CSV, len(df))
    return df


def _build_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise primary dataset into the canonical schema."""
    df = df.copy()
    df["text"] = (
        df["subject"].fillna("").astype(str) + " " +
        df["body"].fillna("").astype(str)
    ).apply(clean)

    df["label"]        = df["queue"].apply(_map_queue_to_category)
    df["priority_num"] = df["priority"].apply(_map_priority)
    df["language"]     = df["language"].fillna("en")
    df["queue_raw"]    = df["queue"]

    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    df["tags"] = df[tag_cols].apply(
        lambda row: ", ".join(t for t in row if pd.notna(t) and t), axis=1
    )
    return df[["text", "label", "priority_num", "language", "queue_raw", "tags"]]


# ── Secondary dataset (Bitext) ────────────────────────────────────────────

def _download_bitext() -> pd.DataFrame:
    """Pull bitext customer-support dataset from HuggingFace."""
    from datasets import load_dataset as hf_load

    logger.info("Downloading %s from HuggingFace Hub ...", BITEXT_DATASET_ID)
    ds = hf_load(BITEXT_DATASET_ID, split=BITEXT_DATASET_SPLIT)
    df = ds.to_pandas()
    logger.info("Bitext: %d rows, columns: %s", len(df), list(df.columns))
    return df


def _build_bitext(df: pd.DataFrame) -> pd.DataFrame:
    """Map bitext intent labels -> our 3 categories, clean text."""
    df = df.copy()
    text_col   = "instruction" if "instruction" in df.columns else df.columns[0]
    intent_col = "intent" if "intent" in df.columns else "category"

    df["text"]  = df[text_col].fillna("").astype(str).apply(clean)
    df["label"] = df[intent_col].str.strip().str.lower().map(BITEXT_INTENT_MAP)
    df = df.dropna(subset=["label"])

    df["priority_num"] = 3
    df["language"]     = "en"
    df["queue_raw"]    = df[intent_col]
    df["tags"]         = ""
    return df[["text", "label", "priority_num", "language", "queue_raw", "tags"]]


# ── Merged cache builder ──────────────────────────────────────────────────

def _load_or_build_merged() -> pd.DataFrame:
    """Return merged df; builds and caches it if not already present."""
    if BITEXT_MERGED_CSV.exists():
        df = pd.read_csv(BITEXT_MERGED_CSV)
        logger.info("Loaded merged cache (%d rows) from %s", len(df), BITEXT_MERGED_CSV)
        return df

    raw_primary = pd.read_csv(CACHED_CSV) if CACHED_CSV.exists() else _download_primary()
    primary = _build_primary(raw_primary)
    logger.info("Primary: %d rows", len(primary))

    try:
        raw_bitext = _download_bitext()
        bitext     = _build_bitext(raw_bitext)
        logger.info("Bitext: %d rows after intent mapping", len(bitext))
        merged = pd.concat([primary, bitext], ignore_index=True)
    except Exception as exc:
        logger.warning("Could not load bitext dataset (%s) -- using primary only", exc)
        merged = primary

    BITEXT_MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(BITEXT_MERGED_CSV, index=False)
    logger.info(
        "Merged dataset saved -> %s (%d rows)\n%s",
        BITEXT_MERGED_CSV, len(merged),
        merged["label"].value_counts().to_string(),
    )
    return merged


# ── Public API ────────────────────────────────────────────────────────────

def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the full merged ticket dataset.

    Priority order:
    1. *path* -- explicit CSV path (for testing / custom data).
    2. Merged cache (primary + bitext).
    3. Build merge from individual sources, downloading if needed.

    Returns a DataFrame with columns:
        ``text``, ``label``, ``priority_num``, ``language``,
        ``queue_raw``, ``tags``
    """
    if path is not None:
        df = pd.read_csv(path)
        if "subject" in df.columns:
            df["text"] = (
                df["subject"].fillna("").astype(str) + " " +
                df["body"].fillna("").astype(str)
            ).apply(clean)
        elif "text" in df.columns:
            df["text"] = df["text"].apply(clean)
        if "label" not in df.columns:
            df["label"] = df["queue"].apply(_map_queue_to_category)
        if "priority_num" not in df.columns:
            df["priority_num"] = df.get(
                "priority", pd.Series(["medium"] * len(df))
            ).apply(_map_priority)
        df["language"]  = df.get("language", pd.Series(["en"] * len(df))).fillna("en")
        df["queue_raw"] = df.get("queue_raw", df.get("queue", ""))
        df["tags"]      = df.get("tags", "")
    else:
        df = _load_or_build_merged()

    # Sanitise: ensure no NaN text values make it downstream
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    logger.info(
        "Dataset ready: %d tickets -- class distribution:\n%s",
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
