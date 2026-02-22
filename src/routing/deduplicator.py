"""Milestone 3 — Semantic deduplication via sentence embeddings.

Detects duplicate / near-duplicate tickets using cosine similarity
of sentence embeddings.  When ≥ ``DEDUP_COUNT_THRESHOLD`` similar
tickets arrive within ``DEDUP_WINDOW_SECONDS``, suppresses further
duplicates and creates a "Master Incident".
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from src.config import (
    DEDUP_COUNT_THRESHOLD,
    DEDUP_SIMILARITY_THRESHOLD,
    DEDUP_WINDOW_SECONDS,
    SENTENCE_MODEL,
)

logger = logging.getLogger(__name__)


@dataclass
class MasterIncident:
    """A cluster of near-duplicate tickets."""
    incident_id: str
    representative_text: str
    ticket_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class SemanticDeduplicator:
    """Tracks recent embeddings and detects duplicate clusters.

    Uses ``sentence-transformers`` for embeddings and cosine similarity
    for matching.  Lazy-loads the model to avoid slow import at startup.
    """

    def __init__(self) -> None:
        self._model = None
        self._recent: list[dict] = []         # {id, text, embedding, ts}
        self._incidents: dict[str, MasterIncident] = {}

    # ── Lazy model loading ───────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(SENTENCE_MODEL)
            logger.info("Loaded sentence model: %s", SENTENCE_MODEL)
        return self._model

    # ── Public API ───────────────────────────────────────────────────

    def check(self, ticket_id: str, text: str) -> dict:
        """Check a new ticket for duplication.

        Returns
        -------
        dict with keys:
            is_duplicate : bool
            master_incident_id : str | None  (set when suppressed)
            similarity : float               (max similarity to cluster)
        """
        now = time.time()
        self._prune_old(now)

        model = self._get_model()
        embedding = model.encode([text], normalize_embeddings=True)[0]

        # Find similar recent tickets
        similar_ids: list[str] = []
        max_sim = 0.0
        for entry in self._recent:
            sim = float(np.dot(embedding, entry["embedding"]))
            if sim > max_sim:
                max_sim = sim
            if sim >= DEDUP_SIMILARITY_THRESHOLD:
                similar_ids.append(entry["id"])

        # Store this ticket
        self._recent.append({
            "id": ticket_id,
            "text": text,
            "embedding": embedding,
            "ts": now,
        })

        # Check if threshold count reached
        if len(similar_ids) >= DEDUP_COUNT_THRESHOLD - 1:
            # Create or update master incident
            incident = self._find_or_create_incident(
                ticket_id, text, similar_ids,
            )
            return {
                "is_duplicate": True,
                "master_incident_id": incident.incident_id,
                "similarity": round(max_sim, 4),
            }

        return {
            "is_duplicate": False,
            "master_incident_id": None,
            "similarity": round(max_sim, 4),
        }

    def get_incidents(self) -> list[MasterIncident]:
        return list(self._incidents.values())

    # ── Internal ─────────────────────────────────────────────────────

    def _prune_old(self, now: float) -> None:
        cutoff = now - DEDUP_WINDOW_SECONDS
        self._recent = [e for e in self._recent if e["ts"] >= cutoff]

    def _find_or_create_incident(
        self,
        trigger_id: str,
        trigger_text: str,
        similar_ids: list[str],
    ) -> MasterIncident:
        # Check if any similar ticket already belongs to an incident
        for iid, inc in self._incidents.items():
            if any(sid in inc.ticket_ids for sid in similar_ids):
                inc.ticket_ids.append(trigger_id)
                return inc

        # New incident
        inc = MasterIncident(
            incident_id=f"MI-{uuid.uuid4().hex[:8].upper()}",
            representative_text=trigger_text[:300],
            ticket_ids=similar_ids + [trigger_id],
        )
        self._incidents[inc.incident_id] = inc
        logger.warning(
            "🚨 Master Incident %s created — %d clustered tickets",
            inc.incident_id, len(inc.ticket_ids),
        )
        return inc
