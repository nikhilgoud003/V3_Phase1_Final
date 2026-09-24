#!/usr/bin/env python3
"""Unit tests: docket_ner must not emit mentions from procedural title phrases."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.docket_ner import extract_judges_from_docket_text


PROCEDURAL_ZERO = [
    "Notice of Judge Assignment To",
    "Magistrate Judge Consent Form",
    "Pursuant to",
    "Signed by Judge X",
    "Referred to Judge Y",
    "Judge Assignment To District Court",
    "Magistrate Judge Consent Form filed",
    "ORDER regarding Magistrate Judge Consent Form",
    "Notice of Judge Assignment To the Honorable Calendar",
]


POSITIVE = [
    (
        "Signed by Judge John Smith on 1/1/2017",
        {"john smith", "john s. smith"},  # accept full or with middle
    ),
    (
        "Referred to Magistrate Judge Jane Doe for settlement",
        {"jane doe"},
    ),
    (
        "ORDER Signed by Judge Ronnie Abrams on 12/1/2017",
        {"ronnie abrams"},
    ),
    (
        "Signed by Judge X. Y. Zzz on 1/1/2017",  # unlikely; at least no crash
        set(),  # optional
    ),
]


def test_procedural_phrases_yield_zero():
    for phrase in PROCEDURAL_ZERO:
        spans = extract_judges_from_docket_text(phrase)
        assert spans == [], f"Expected 0 mentions from {phrase!r}, got {spans}"


def test_signed_by_and_referred_extract_real_names():
    spans = extract_judges_from_docket_text("Signed by Judge John Smith on 1/1/2017")
    norms = {re_norm(s["raw"]) for s in spans}
    assert "john smith" in norms, spans

    spans = extract_judges_from_docket_text(
        "Referred to Magistrate Judge Jane Doe for settlement"
    )
    norms = {re_norm(s["raw"]) for s in spans}
    assert "jane doe" in norms, spans

    spans = extract_judges_from_docket_text(
        "ORDER Signed by Judge Ronnie Abrams on 12/1/2017"
    )
    norms = {re_norm(s["raw"]) for s in spans}
    assert "ronnie abrams" in norms, spans


def test_no_middle_initial_truncation():
    """Regression: 'Judge John Smith' must not become 'John S'."""
    spans = extract_judges_from_docket_text("Signed by Judge John Smith on 1/1/2017")
    norms = {re_norm(s["raw"]) for s in spans}
    assert "john s" not in norms, spans
    assert any(n.startswith("john smith") or n == "john smith" for n in norms), spans


def re_norm(s: str) -> str:
    import re

    return re.sub(r"\s+", " ", s).strip().lower()


if __name__ == "__main__":
    test_procedural_phrases_yield_zero()
    test_signed_by_and_referred_extract_real_names()
    test_no_middle_initial_truncation()
    # Also assert Signed by Judge X / Referred to Judge Y produce nothing useful
    # (single letter) — gate rejects
    assert extract_judges_from_docket_text("Signed by Judge X") == []
    assert extract_judges_from_docket_text("Referred to Judge Y") == []
    print("PASS test_docket_ner_procedural")
