"""Milestone 3 — Circuit breaker: fail-over to M1 when transformer is slow.

Monitors the latency of the primary (transformer) model.  If any single
inference exceeds ``CIRCUIT_BREAKER_LATENCY_MS`` (default 500 ms), the
breaker *opens* and routes subsequent requests through the fast M1 model
for a cool-down period.
"""

from __future__ import annotations

import logging
import time
from enum import Enum

from src.config import CIRCUIT_BREAKER_LATENCY_MS

logger = logging.getLogger(__name__)

# Cool-down before retrying the primary model (seconds)
_COOLDOWN_SECONDS = 30.0


class BreakerState(Enum):
    CLOSED = "closed"          # Normal — using primary
    OPEN = "open"              # Tripped — using fallback
    HALF_OPEN = "half-open"    # Testing primary again


class CircuitBreaker:
    """Wraps two classifiers: *primary* (transformer) and *fallback* (M1).

    Usage::

        breaker = CircuitBreaker(primary=distilbert, fallback=logreg)
        category = breaker.predict(text)
    """

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback
        self.state = BreakerState.CLOSED
        self._tripped_at: float = 0.0
        self._latency_ms: float = 0.0

    # ── Public ───────────────────────────────────────────────────────

    def predict(self, texts: list[str]):
        """Route through primary or fallback depending on breaker state."""
        if self.state == BreakerState.OPEN:
            if self._cooldown_elapsed():
                self.state = BreakerState.HALF_OPEN
                logger.info("Circuit breaker → HALF-OPEN (testing primary)")
            else:
                return self._use_fallback(texts)

        if self.state in (BreakerState.CLOSED, BreakerState.HALF_OPEN):
            return self._try_primary(texts)

        return self._use_fallback(texts)

    @property
    def current_state(self) -> str:
        return self.state.value

    @property
    def last_latency_ms(self) -> float:
        return self._latency_ms

    # ── Internal ─────────────────────────────────────────────────────

    def _try_primary(self, texts: list[str]):
        t0 = time.perf_counter()
        result = self.primary.predict(texts)
        self._latency_ms = (time.perf_counter() - t0) * 1000

        if self._latency_ms > CIRCUIT_BREAKER_LATENCY_MS:
            logger.warning(
                "Primary model latency %.0f ms > %d ms → OPEN circuit",
                self._latency_ms, CIRCUIT_BREAKER_LATENCY_MS,
            )
            self.state = BreakerState.OPEN
            self._tripped_at = time.time()
            # Still return the (slow) result this time
            return result

        # Success on HALF_OPEN → close
        if self.state == BreakerState.HALF_OPEN:
            logger.info("Primary recovered (%.0f ms) → CLOSED", self._latency_ms)
            self.state = BreakerState.CLOSED

        return result

    def _use_fallback(self, texts: list[str]):
        logger.debug("Using fallback model (circuit OPEN)")
        t0 = time.perf_counter()
        result = self.fallback.predict(texts)
        self._latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def _cooldown_elapsed(self) -> bool:
        return (time.time() - self._tripped_at) >= _COOLDOWN_SECONDS
