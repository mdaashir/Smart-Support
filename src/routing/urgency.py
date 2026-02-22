"""Urgency detection — shared by all milestones."""

from __future__ import annotations

import re

from src.config import URGENT_KEYWORDS

_PATTERN = r"\b(" + "|".join(re.escape(k) for k in URGENT_KEYWORDS) + r")\b"
_URGENT_RE = re.compile(_PATTERN, re.IGNORECASE)


# Numeric priorities — low number = high urgency (heap-friendly)
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 5


def detect_urgency(text: str) -> int:
    """Return ``1`` (HIGH) if urgent keywords found, else ``5`` (NORMAL)."""
    return PRIORITY_HIGH if _URGENT_RE.search(text) else PRIORITY_NORMAL


def urgency_label(text: str) -> str:
    """Return a human-readable label like ``'1(HIGH)'`` or ``'0(NORMAL)'``."""
    if _URGENT_RE.search(text):
        return "1(HIGH)"
    return "0(NORMAL)"
