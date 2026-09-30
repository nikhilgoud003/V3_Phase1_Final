"""Tests for Tier3 LLM output normalization (signals coercion)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.tiers import normalize_tier3_signals


def test_normalize_empty_and_strings():
    assert normalize_tier3_signals(None) == []
    assert normalize_tier3_signals([]) == []
    assert normalize_tier3_signals(["same_court", "name_mismatch"]) == [
        "same_court",
        "name_mismatch",
    ]


def test_normalize_key_value_objects():
    raw = [
        {"key": "same_ucid", "value": False},
        {"key": "same_contact_info", "value": False},
    ]
    assert normalize_tier3_signals(raw) == ["same_ucid", "same_contact_info"]


def test_normalize_evidence_mirror_object():
    raw = [{"same_court": True, "same_ucid": False, "embedding_similarity": 0.85}]
    assert normalize_tier3_signals(raw) == [
        "same_court",
        "same_ucid",
        "embedding_similarity",
    ]


def test_normalize_mixed():
    raw = ["same_court", {"key": "embedding_similarity", "value": 0.88}]
    assert normalize_tier3_signals(raw) == ["same_court", "embedding_similarity"]
