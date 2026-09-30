#!/usr/bin/env python3
"""Trailing stripper rescues Appearances / recused / Dispositive / Deft remanded."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.name_validity import (
    classify_name_validity,
    strip_trailing_lexemes,
    strip_trailing_procedural,
    DEFAULT_PROCEDURAL_STOPWORDS,
)

CASES = [
    ("stephen smith appearances", "stephen smith"),
    ("Stephen Smith Appearances", "stephen smith"),
    ("rolando olvera recused", "rolando olvera"),
    ("nelva gonzales ramos appearances", "nelva gonzales ramos"),
    ("christopher dos santos dispositive", "christopher dos santos"),
    ("rolando olvera deft remanded", "rolando olvera"),
]


def test_strip_trailing_lexemes_rescue():
    for raw, want in CASES:
        got = strip_trailing_lexemes(raw.lower())
        assert got == want, (raw, got, want)


def test_strip_trailing_procedural_rescue():
    for raw, want in CASES:
        got = strip_trailing_procedural(raw.lower(), DEFAULT_PROCEDURAL_STOPWORDS)
        assert got == want, (raw, got, want)


def test_classify_validity_after_strip():
    for raw, want in CASES:
        ok, reasons = classify_name_validity(raw, english_words=frozenset())
        assert ok, (raw, reasons)
        cleaned = strip_trailing_procedural(raw.lower(), DEFAULT_PROCEDURAL_STOPWORDS)
        assert cleaned == want, (raw, cleaned)
