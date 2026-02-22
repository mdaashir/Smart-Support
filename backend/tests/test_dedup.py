"""Tests for Milestone 3 — Semantic deduplication.

Validates the full storm → Master Incident lifecycle including:
- Cosine similarity detection of near-duplicate tickets
- Master Incident creation when ≥ DEDUP_COUNT_THRESHOLD similar tickets arrive
- Window pruning (old tickets are dropped after DEDUP_WINDOW_SECONDS)
- Non-duplicate tickets remain unaffected
- get_incidents() API returns correct state
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.routing.deduplicator import MasterIncident, SemanticDeduplicator


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_embedding(seed: int = 0, dim: int = 384) -> np.ndarray:
    """Create a deterministic, unit-normalised embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _near_duplicate(base: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
    """Return a vector very close to *base* (cosine sim > 0.99)."""
    rng = np.random.RandomState(42)
    noisy = base + rng.randn(*base.shape).astype(np.float32) * noise_scale
    return noisy / np.linalg.norm(noisy)


def _make_mock_model(embeddings: list[np.ndarray]):
    """Return a mock SentenceTransformer whose .encode() yields *embeddings* in order."""
    model = MagicMock()
    call_idx = {"i": 0}

    def _encode(texts, normalize_embeddings=True):
        idx = call_idx["i"]
        call_idx["i"] += 1
        return [embeddings[idx]]

    model.encode = _encode
    return model


# ── Tests ────────────────────────────────────────────────────────────────

class TestSemanticDeduplicator:
    """Unit tests for SemanticDeduplicator."""

    def _make_dedup(self, embeddings: list[np.ndarray]) -> SemanticDeduplicator:
        """Create a deduplicator with a mocked sentence model."""
        dedup = SemanticDeduplicator()
        dedup._model = _make_mock_model(embeddings)
        return dedup

    # ── Basic duplicate detection ────────────────────────────────────

    def test_no_duplicate_on_first_ticket(self):
        emb = _make_embedding(0)
        dedup = self._make_dedup([emb])
        result = dedup.check("T-001", "Server is down")
        assert result["is_duplicate"] is False
        assert result["master_incident_id"] is None
        assert result["similarity"] == 0.0  # nothing to compare to

    def test_similar_tickets_detected(self):
        base = _make_embedding(0)
        dup = _near_duplicate(base)
        dedup = self._make_dedup([base, dup])

        dedup.check("T-001", "Server is down")
        result = dedup.check("T-002", "Server is down again")
        # Should be similar but not yet enough for Master Incident (need 10)
        assert result["is_duplicate"] is False
        assert result["similarity"] > 0.9

    def test_dissimilar_tickets_not_flagged(self):
        emb1 = _make_embedding(0)
        emb2 = _make_embedding(999)  # very different seed → different vector
        dedup = self._make_dedup([emb1, emb2])

        dedup.check("T-001", "Server is down")
        result = dedup.check("T-002", "Billing issue")
        assert result["similarity"] < 0.5  # different topics

    # ── Master Incident creation (storm detection) ───────────────────

    def test_master_incident_created_at_threshold(self):
        """When ≥ 10 similar tickets arrive, a Master Incident is created."""
        base = _make_embedding(0)
        # 10 near-duplicate embeddings (first + 9 more → 10th triggers)
        embeddings = [base] + [_near_duplicate(base, noise_scale=0.005 + i * 0.001) for i in range(10)]
        dedup = self._make_dedup(embeddings)

        # First 9 tickets should NOT trigger
        for i in range(1, 10):
            result = dedup.check(f"T-{i:03d}", f"Server crash #{i}")
            assert result["is_duplicate"] is False, f"Ticket {i} falsely flagged as dup"

        # 10th ticket should trigger (9 similar + itself = meets threshold of 10-1=9 similar)
        result = dedup.check("T-010", "Server crash #10")
        assert result["is_duplicate"] is True
        assert result["master_incident_id"] is not None
        assert result["master_incident_id"].startswith("MI-")
        assert result["similarity"] > 0.9

    def test_master_incident_in_incidents_list(self):
        """get_incidents() returns the created Master Incident."""
        base = _make_embedding(0)
        embeddings = [base] + [_near_duplicate(base, noise_scale=0.005 + i * 0.001) for i in range(10)]
        dedup = self._make_dedup(embeddings)

        for i in range(1, 11):
            dedup.check(f"T-{i:03d}", f"Server crash #{i}")

        incidents = dedup.get_incidents()
        assert len(incidents) >= 1
        inc = incidents[0]
        assert isinstance(inc, MasterIncident)
        assert len(inc.ticket_ids) >= 10
        assert "T-010" in inc.ticket_ids

    def test_subsequent_dupes_join_existing_incident(self):
        """Tickets after the 10th in a storm join the same Master Incident."""
        base = _make_embedding(0)
        embeddings = [base] + [_near_duplicate(base, noise_scale=0.005 + i * 0.001) for i in range(11)]
        dedup = self._make_dedup(embeddings)

        # First 10 tickets
        for i in range(1, 11):
            dedup.check(f"T-{i:03d}", f"Server crash #{i}")

        # 11th ticket
        result = dedup.check("T-011", "Server crash #11")
        assert result["is_duplicate"] is True

        incidents = dedup.get_incidents()
        assert len(incidents) == 1  # still just one incident
        assert "T-011" in incidents[0].ticket_ids

    # ── Window pruning ───────────────────────────────────────────────

    def test_old_tickets_pruned(self):
        """Tickets older than DEDUP_WINDOW_SECONDS are pruned."""
        base = _make_embedding(0)
        dup = _near_duplicate(base)
        dedup = self._make_dedup([base, dup])

        # Insert first ticket
        dedup.check("T-001", "Server is down")

        # Artificially age the entry beyond the window
        dedup._recent[0]["ts"] = time.time() - 400  # > 300s window

        # Second ticket — the old one should be pruned, so no similarity
        result = dedup.check("T-002", "Server is down again")
        assert result["similarity"] == 0.0  # old ticket was pruned

    def test_storm_not_triggered_across_windows(self):
        """If 5 tickets arrive in window 1 and 5 in window 2, no Master Incident."""
        base = _make_embedding(0)
        embeddings = [_near_duplicate(base, noise_scale=0.005 + i * 0.001) for i in range(11)]
        dedup = self._make_dedup(embeddings)

        # First 5 tickets
        for i in range(1, 6):
            dedup.check(f"T-{i:03d}", f"Server crash #{i}")

        # Age first 5 beyond window
        now = time.time()
        for entry in dedup._recent:
            entry["ts"] = now - 400

        # Next 5 tickets — should not trigger (only 4 similar in current window)
        for i in range(6, 11):
            result = dedup.check(f"T-{i:03d}", f"Server crash #{i}")
            assert result["is_duplicate"] is False

    # ── Edge cases ───────────────────────────────────────────────────

    def test_empty_incidents_initially(self):
        dedup = SemanticDeduplicator()
        assert dedup.get_incidents() == []

    def test_master_incident_dataclass_fields(self):
        inc = MasterIncident(
            incident_id="MI-TEST1234",
            representative_text="Server is down",
            ticket_ids=["T-001", "T-002"],
        )
        assert inc.incident_id == "MI-TEST1234"
        assert inc.representative_text == "Server is down"
        assert len(inc.ticket_ids) == 2
        assert inc.created_at > 0
