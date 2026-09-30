#!/usr/bin/env python3
"""Judges header/party/NER extract goes through the generic path walker."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config
from engine.extract import extract_from_case

CFG = load_config(ROOT / "configs/judges.yaml")


def test_header_and_party_and_ner_via_walker():
    case = {
        "ucid": "ilnd;;1:16-cv-00001",
        "court": "ilnd",
        "case_id": "1:16-cv-00001",
        "case_type": "cv",
        "case_name": "Test v. Test",
        "filing_date": "2016-01-02",
        "judge": "Honorable James B. Zagel",
        "referred_judges": ["Magistrate Judge Sidney A. Fitzwater"],
        "parties": [
            {
                "name": "USA",
                "role": "Plaintiff",
                "pacer_id": "p0",
                "judge": None,
                "referred_judges": [],
            }
        ],
        "docket": [
            {"docket_text": "Signed by Judge James B. Zagel on 1/3/2016."},
        ],
    }
    ments, _ = extract_from_case(case, CFG, source_file="t.json")
    methods = {m["extraction_method"] for m in ments}
    assert "header_assigned_judge" in methods
    assert "header_referred_judges" in methods
    assert "SPACY_EN_CORE_WEB_SM" in methods
    assigned = [m for m in ments if m["extraction_method"] == "header_assigned_judge"]
    assert assigned and "zagel" in assigned[0]["normalized_name"]
    ner = [m for m in ments if m.get("docket_source") == "line_entry"]
    assert ner and ner[0].get("docket_index") == 0
