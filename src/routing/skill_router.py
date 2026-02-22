"""Milestone 3 — Skill-based routing with constraint optimisation.

Each agent has a *skill vector* (proficiency per category) and a
finite *capacity*.  Given a classified ticket, the router picks the
best available agent using a simple greedy / Hungarian-style approach.
"""

from __future__ import annotations

import logging
from copy import deepcopy

from src.config import AGENT_REGISTRY

logger = logging.getLogger(__name__)


class SkillRouter:
    """Route tickets to the best-matching agent by skill affinity.

    Parameters
    ----------
    registry : dict
        ``{agent_name: {"skills": {cat: float}, "capacity": int}}``
    """

    def __init__(self, registry: dict | None = None) -> None:
        self._registry = deepcopy(registry or AGENT_REGISTRY)
        # Runtime load counter per agent
        self._load: dict[str, int] = {a: 0 for a in self._registry}

    # ── Public ───────────────────────────────────────────────────────

    def assign(self, category: str, urgency_score: float = 0.0) -> dict:
        """Pick the best agent for *category*.

        Returns ``{"agent": "Agent_X", "affinity": 0.9, "load": 2}``
        or ``{"agent": None, ...}`` if all agents are at capacity.
        """
        candidates: list[tuple[float, str]] = []
        for name, info in self._registry.items():
            if self._load[name] >= info["capacity"]:
                continue
            affinity = info["skills"].get(category, 0.0)
            # Boost score slightly if very urgent (tie-break)
            score = affinity + urgency_score * 0.05
            candidates.append((score, name))

        if not candidates:
            logger.warning("All agents at capacity — ticket unassigned")
            return {"agent": None, "affinity": 0.0, "load": 0}

        candidates.sort(reverse=True)
        best_score, best_name = candidates[0]
        self._load[best_name] += 1

        logger.info(
            "Assigned to %s (affinity=%.2f, load=%d/%d)",
            best_name, best_score, self._load[best_name],
            self._registry[best_name]["capacity"],
        )
        return {
            "agent": best_name,
            "affinity": round(best_score, 3),
            "load": self._load[best_name],
        }

    def release(self, agent_name: str) -> None:
        """Free one slot when an agent finishes a ticket."""
        if agent_name in self._load and self._load[agent_name] > 0:
            self._load[agent_name] -= 1

    # ── Introspection ────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            name: {
                "load": self._load[name],
                "capacity": info["capacity"],
                "utilisation": round(self._load[name] / info["capacity"], 2),
            }
            for name, info in self._registry.items()
        }
