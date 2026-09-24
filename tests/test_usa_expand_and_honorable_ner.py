#!/usr/bin/env python3
"""Party expand_abbreviations + Honorable docket NER (no hardcoded merge logic in engine)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config
from engine.docket_ner import extract_judges_from_docket_text
from engine.normalize import apply_replace_regex, normalize_name


def _party_norm(raw: str) -> str:
    cfg = load_config(ROOT / "configs/parties.yaml")
    norm = cfg["normalization"]
    n = normalize_name(
        raw,
        honorifics=norm.get("strip_honorifics") or [],
        strip_chars=norm.get("strip_chars") or "",
        lowercase=True,
        collapse_whitespace=True,
        corp_suffixes=norm.get("strip_corp_suffixes") or [],
        replace_tokens=norm.get("replace_tokens") or {},
        replace_regex=norm.get("replace_regex") or [],
    )
    return apply_replace_regex(n, norm.get("expand_abbreviations") or [])


def test_usa_variants_collapse_via_config():
    canon = "united states of america"
    for raw in ("USA", "U.S.A.", "U.S.", "United States", "UNITED STATES OF AMERICA", "U S A"):
        assert _party_norm(raw) == canon, (raw, _party_norm(raw))


def test_honorable_without_judge_token():
    spans = extract_judges_from_docket_text(
        "MINUTE entry before Honorable Ronald A. Guzman: status hearing held."
    )
    norms = {s["raw"].lower() for s in spans}
    assert any("guzman" in n for n in norms), spans

    spans = extract_judges_from_docket_text(
        "before the Honorable Mary M. Rowland for settlement."
    )
    norms = {s["raw"].lower() for s in spans}
    assert any("rowland" in n for n in norms), spans
