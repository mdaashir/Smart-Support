"""Priority queue for ticket processing (Milestone 1)."""

from __future__ import annotations

import heapq
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class _PrioritisedTicket:
    priority: int
    counter: int = field(compare=True)
    ticket: dict = field(compare=False)


class TicketQueue:
    """Min-heap priority queue for support tickets."""

    def __init__(self) -> None:
        self._heap: list[_PrioritisedTicket] = []
        self._counter: int = 0

    # ── Public API ───────────────────────────────────────────────────
    def ingest(self, text: str, *, classifier, urgency_fn) -> dict:
        """Classify *text*, detect urgency, push to queue, return ticket dict."""
        from backend.routing.urgency import detect_urgency as _default_urgency

        urg_fn = urgency_fn or _default_urgency
        category = classifier.predict([text])[0]
        priority = urg_fn(text)

        ticket = {
            "ticket_id": str(uuid.uuid4()),
            "text": text,
            "category": category,
            "priority": priority,
        }
        heapq.heappush(
            self._heap,
            _PrioritisedTicket(priority, self._counter, ticket),
        )
        self._counter += 1
        return ticket

    def process_next(self) -> dict | None:
        """Pop the highest-priority ticket, or ``None`` if empty."""
        if self._heap:
            return heapq.heappop(self._heap).ticket
        return None

    @property
    def size(self) -> int:
        return len(self._heap)

    def __len__(self) -> int:
        return self.size
