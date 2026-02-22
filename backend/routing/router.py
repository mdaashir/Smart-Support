"""Ticket router — dependency-injectable, works with any milestone's model."""

from __future__ import annotations

from backend.preprocessing.text_cleaner import combine_fields
from backend.routing.urgency import urgency_label


def route_ticket(
    subject: str,
    body: str,
    *,
    classifier,
) -> dict:
    """Classify a ticket and detect urgency.

    Parameters
    ----------
    subject, body:
        Raw ticket fields.
    classifier:
        Any object with a ``.predict([text])`` method returning a category
        array (TfidfLogReg, TfidfSVC, DistilBert — all share this interface).

    Returns
    -------
    dict with keys ``category`` and ``urgency``.
    """
    text = combine_fields(subject, body)
    category = classifier.predict([text])[0]
    urgency = urgency_label(text)
    return {"category": category, "urgency": urgency}
