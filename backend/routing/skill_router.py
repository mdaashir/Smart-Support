"""Milestone 3 — Skill-based routing with constraint optimisation.

Each agent has a *skill vector* (proficiency per category) and a
finite *capacity*.  Given a classified ticket, the router picks the
best available agent.

For **single tickets** the ``assign()`` method uses greedy best-fit.

For **batches** the ``batch_assign()`` method solves a formal
constraint-optimisation problem via the Hungarian algorithm
(``scipy.optimize.linear_sum_assignment``) — globally optimal
assignment that respects capacity limits.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.config import AGENT_REGISTRY

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

    # ── Public (single ticket) ───────────────────────────────────────

    def assign(self, category: str, urgency_score: float = 0.0) -> dict:
        """Pick the best agent for *category* (greedy single-ticket).

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

    # ── Public (batch — constraint optimisation) ─────────────────────

    def batch_assign(
        self,
        tickets: list[dict],
    ) -> list[dict]:
        """Optimally assign a batch of tickets using the Hungarian algorithm.

        Each ticket is ``{"id": str, "category": str, "urgency": float}``.

        Returns a list of ``{"ticket_id", "agent", "affinity", "load"}``
        dicts in the same order as *tickets*.

        The method expands agents into *slots* (one per remaining unit of
        capacity), then builds a **cost matrix** (negative affinity) and
        solves with ``scipy.optimize.linear_sum_assignment`` for the
        globally optimal assignment that maximises total affinity while
        respecting every capacity constraint.
        """
        if not tickets:
            return []

        # Build expanded slot list: one entry per free slot
        slots: list[tuple[str, dict]] = []  # (agent_name, info)
        for name, info in self._registry.items():
            free = info["capacity"] - self._load[name]
            for _ in range(free):
                slots.append((name, info))

        n_tickets = len(tickets)
        n_slots = len(slots)

        if n_slots == 0:
            logger.warning("All agents at capacity — batch unassigned")
            return [
                {"ticket_id": t["id"], "agent": None, "affinity": 0.0, "load": 0}
                for t in tickets
            ]

        # Cost matrix: rows = tickets, cols = agent-slots
        # We minimise → use negative affinity as cost
        big = 1e6  # penalty for infeasible (more tickets than slots)
        size = max(n_tickets, n_slots)
        cost = np.full((size, size), big, dtype=np.float64)

        for i, t in enumerate(tickets):
            cat = t.get("category", "")
            urg = t.get("urgency", 0.0)
            for j, (agent_name, info) in enumerate(slots):
                affinity = info["skills"].get(cat, 0.0)
                score = affinity + urg * 0.05
                cost[i, j] = -score  # minimise negative = maximise

        row_idx, col_idx = linear_sum_assignment(cost)

        # Build result
        results: list[dict] = [
            {"ticket_id": t["id"], "agent": None, "affinity": 0.0, "load": 0}
            for t in tickets
        ]
        for r, c in zip(row_idx, col_idx):
            if r >= n_tickets or c >= n_slots:
                continue  # padding row/col
            if cost[r, c] >= big:
                continue  # no feasible slot
            agent_name = slots[c][0]
            affinity = -cost[r, c]
            self._load[agent_name] += 1
            results[r] = {
                "ticket_id": tickets[r]["id"],
                "agent": agent_name,
                "affinity": round(float(affinity), 3),
                "load": self._load[agent_name],
            }

        assigned = sum(1 for r in results if r["agent"] is not None)
        logger.info(
            "Batch assigned %d/%d tickets via Hungarian algorithm",
            assigned, n_tickets,
        )
        return results

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
