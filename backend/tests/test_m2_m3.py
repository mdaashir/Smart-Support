"""Tests for Milestone 2 & 3 components — broker, webhook, dedup, circuit breaker, skill router."""

import asyncio
import pytest

from backend.config import CATEGORIES
from backend.routing.skill_router import SkillRouter
from backend.routing.circuit_breaker import CircuitBreaker, BreakerState
from backend.routing.webhook import fire_webhook, get_webhook_log, clear_webhook_log


# ── Skill-based routing ─────────────────────────────────────────────────

class TestSkillRouter:
    def test_assign_returns_best_agent(self):
        router = SkillRouter()
        result = router.assign("Technical")
        assert result["agent"] is not None
        assert result["affinity"] > 0

    def test_assign_respects_capacity(self):
        registry = {
            "Agent_X": {"skills": {"Billing": 1.0}, "capacity": 1},
        }
        router = SkillRouter(registry)
        r1 = router.assign("Billing")
        assert r1["agent"] == "Agent_X"
        r2 = router.assign("Billing")
        assert r2["agent"] is None  # at capacity

    def test_release_frees_slot(self):
        registry = {
            "Agent_X": {"skills": {"Billing": 1.0}, "capacity": 1},
        }
        router = SkillRouter(registry)
        router.assign("Billing")
        router.release("Agent_X")
        r = router.assign("Billing")
        assert r["agent"] == "Agent_X"

    def test_status(self):
        router = SkillRouter()
        status = router.status()
        assert len(status) > 0
        for name, info in status.items():
            assert "load" in info
            assert "capacity" in info


# ── Circuit breaker ──────────────────────────────────────────────────────

class _FastModel:
    def predict(self, texts):
        return ["Billing"] * len(texts)

class _SlowModel:
    def predict(self, texts):
        import time; time.sleep(0.6)  # > 500ms threshold
        return ["Technical"] * len(texts)


class TestCircuitBreaker:
    def test_closed_uses_primary(self):
        cb = CircuitBreaker(primary=_FastModel(), fallback=_FastModel())
        assert cb.state == BreakerState.CLOSED
        result = cb.predict(["test"])
        assert result == ["Billing"]

    def test_trips_on_slow_primary(self):
        cb = CircuitBreaker(primary=_SlowModel(), fallback=_FastModel())
        _ = cb.predict(["test"])  # This will be slow → trip
        assert cb.state == BreakerState.OPEN

    def test_fallback_after_trip(self):
        cb = CircuitBreaker(primary=_SlowModel(), fallback=_FastModel())
        _ = cb.predict(["test"])  # trip
        result = cb.predict(["test"])  # uses fallback
        assert result == ["Billing"]


# ── Webhook ──────────────────────────────────────────────────────────────

class TestWebhook:
    def test_fire_webhook_logs(self):
        clear_webhook_log()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(fire_webhook(
            ticket_id="T-001",
            category="Technical",
            urgency_score=0.95,
            text_preview="Server is down",
        ))
        loop.close()
        log = get_webhook_log()
        assert len(log) == 1
        assert "T-001" in log[0]["text"]

    def test_webhook_log_clears(self):
        clear_webhook_log()
        assert len(get_webhook_log()) == 0
