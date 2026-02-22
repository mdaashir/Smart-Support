"""Shared text preprocessing utilities."""

from __future__ import annotations

import re

# ── Compiled patterns (compiled once at import time) ─────────────────────────
_URL     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL   = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)
_HTML    = re.compile(r"<[^>]+>")
_TICKET  = re.compile(r"\b(ticket|case|ref|order)[#:\s]*\w+", re.IGNORECASE)
_REPEAT  = re.compile(r"([!?.]){2,}")          # !!!  → !
_NUMBER  = re.compile(r"\b\d{4,}\b")            # long numeric codes add noise
_MULTI   = re.compile(r"\s+")


def clean(text: str) -> str:
    """Full normalisation pipeline: lower-case, strip noise, collapse spaces.

    Steps applied in order:
    1. HTML tag removal
    2. URL / email replacement with type tokens
    3. Ticket/case/order-number scrubbing
    4. Long numeric code removal
    5. Repeated punctuation collapse
    6. Lower-case
    7. Whitespace normalisation
    """
    if not text:
        return ""
    t = str(text)
    t = _HTML.sub(" ", t)
    t = _URL.sub(" __url__ ", t)
    t = _EMAIL.sub(" __email__ ", t)
    t = _TICKET.sub(" ", t)
    t = _NUMBER.sub(" ", t)
    t = _REPEAT.sub(r"\1", t)
    t = t.lower().strip()
    return _MULTI.sub(" ", t)


def combine_fields(subject: str | None, body: str | None) -> str:
    """Join subject + body with a space, handling None/NaN."""
    s = str(subject) if subject else ""
    b = str(body) if body else ""
    return clean(f"{s} {b}")
