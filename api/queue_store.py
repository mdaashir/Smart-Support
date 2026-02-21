"""
In-memory priority queue for Milestone 1.

The heap stores (priority, counter, TicketOut).
  priority 1 = HIGH  → processed first (lowest value wins in heapq)
  priority 5 = NORMAL

A per-increment counter breaks ties so TicketOut dicts are never compared.
"""

from __future__ import annotations

import heapq
import threading
from typing import Optional

from api.schemas import TicketOut

# ──────────────────────────────────────────────
# Global state  (single-threaded OK for M1)
# ──────────────────────────────────────────────
_heap: list[tuple[int, int, dict]] = []
_counter: int = 0
_lock = threading.Lock()   # safe to include even in single-threaded mode


def push(ticket: TicketOut) -> None:
    """Add a ticket to the priority queue."""
    global _counter
    with _lock:
        heapq.heappush(_heap, (ticket.urgency_level, _counter, ticket.model_dump()))
        _counter += 1


def pop() -> Optional[dict]:
    """Remove and return the highest-priority ticket, or None if empty."""
    with _lock:
        if not _heap:
            return None
        _, _, ticket_dict = heapq.heappop(_heap)
        return ticket_dict


def peek() -> Optional[dict]:
    """Return the highest-priority ticket without removing it."""
    with _lock:
        if not _heap:
            return None
        return _heap[0][2]


def size() -> int:
    """Current number of tickets in the queue."""
    with _lock:
        return len(_heap)


def drain() -> list[dict]:
    """Return all tickets ordered by priority (clears the queue)."""
    with _lock:
        result = []
        while _heap:
            _, _, ticket_dict = heapq.heappop(_heap)
            result.append(ticket_dict)
        return result
