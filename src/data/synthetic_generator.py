"""Synthetic ticket data generator for Milestone 1 proof-of-concept."""

from __future__ import annotations

import random

import pandas as pd

from src.config import SYNTHETIC_SAMPLES_PER_CLASS, RANDOM_STATE

# ── Corpus ───────────────────────────────────────────────────────────────
BILLING_ISSUES = [
    "charged twice", "refund request", "invoice incorrect",
    "payment failed", "subscription renewal issue",
    "billing address update", "credit card declined",
    "unexpected charge", "pricing discrepancy",
]

TECHNICAL_ISSUES = [
    "app not loading", "server down", "login failed",
    "500 internal error", "feature not working",
    "API timeout", "database crash",
    "slow performance", "bug in dashboard",
]

HR_ISSUES = [
    "employee onboarding delay", "payroll discrepancy",
    "PTO request not approved", "benefits enrollment issue",
    "workplace harassment report", "interview scheduling",
    "training programme access", "contract renewal query",
]

GENERAL_ISSUES = [
    "general inquiry about services", "office address question",
    "hours of operation", "feedback on experience",
    "how to contact support", "account overview request",
    "status update on request", "question about policies",
]

URGENCY_WORDS = [
    "ASAP", "urgent", "immediately", "critical",
    "right now", "as soon as possible",
]

NOISE_WORDS = [
    "since yesterday", "after update", "in production",
    "for multiple users", "this morning", "in staging",
]

TYPOS: dict[str, str] = {
    "payment": "paymnt",
    "server": "servr",
    "agreement": "agreemnt",
    "subscription": "subscrption",
}

CATEGORY_ISSUES = {
    "Billing": BILLING_ISSUES,
    "Technical": TECHNICAL_ISSUES,
    "HR": HR_ISSUES,
    "General": GENERAL_ISSUES,
}

ALL_ISSUES = BILLING_ISSUES + TECHNICAL_ISSUES + HR_ISSUES + GENERAL_ISSUES


# ── Helpers ──────────────────────────────────────────────────────────────

def _introduce_typo(text: str) -> str:
    for correct, wrong in TYPOS.items():
        if random.random() < 0.1:
            text = text.replace(correct, wrong)
    return text


def _generate_ticket(issue_list: list[str], label: str) -> tuple[str, str]:
    issue = random.choice(issue_list)
    noise = random.choice(NOISE_WORDS)
    text = f"We are facing {issue} {noise}"

    # 25 % mixed-category noise
    if random.random() < 0.25:
        mixed = random.choice(ALL_ISSUES)
        text += f" and also {mixed}"

    # 30 % urgency injection
    if random.random() < 0.30:
        text += f". This is {random.choice(URGENCY_WORDS)}"

    text = _introduce_typo(text)
    return text, label


# ── Public API ───────────────────────────────────────────────────────────

def generate_dataset(
    n_per_class: int = SYNTHETIC_SAMPLES_PER_CLASS,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a DataFrame with columns ``text`` and ``category``.

    Parameters
    ----------
    n_per_class:
        Number of synthetic tickets **per category**.
    seed:
        Random seed for reproducibility.
    """
    random.seed(seed)
    rows: list[tuple[str, str]] = []
    for _ in range(n_per_class):
        for label, issues in CATEGORY_ISSUES.items():
            rows.append(_generate_ticket(issues, label))
    return pd.DataFrame(rows, columns=["text", "category"])
