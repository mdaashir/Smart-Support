"""Shared text preprocessing utilities."""

from __future__ import annotations

import re


_MULTI_SPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    """Lower-case, strip, and collapse whitespace."""
    if not text:
        return ""
    return _MULTI_SPACE.sub(" ", str(text).lower().strip())


def combine_fields(subject: str | None, body: str | None) -> str:
    """Join subject + body with a space, handling None/NaN."""
    s = str(subject) if subject else ""
    b = str(body) if body else ""
    return clean(f"{s} {b}")
