#!/usr/bin/env python3
"""research_dev-aligned judge noise rejection (procedural docket junk)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.name_validity import (
    classify_name_validity,
    load_fjc_name_sets,
    preclean_header_junk,
)


def _fjc_sets():
    honorifics = ["honorable", "hon", "judge", "magistrate", "chief", "senior", "district"]
    return load_fjc_name_sets(ROOT / "data/judges_fjc.csv", honorifics, ".,;:")


PROCEDURAL_REJECT = [
    "telephonic scheduling",
    "telephonic",
    "update in case",
    "unassigned disposition",
    "unassigned - cri",
    "unassigned",
    "who participates",
    "will notify parties",
    "calendar call",
    "sentencing memos",
    "jeg",
    "notify",
    "participates",
    "accept",
    "issues",
    "call",
    "whose name is indicated",
    "status hearing set",
    "granted",
    "none unassigned vjdistrict",
    "Notice of Hearing",
    "Motion to Dismiss",
    "johnston's web page",
    "ellis' website",
    "hurd advises",
    "schedules page",
    "parker's individual rules",
    "4/27/2017 blm1",
    "wood's decision",
]

REAL_JUDGES_KEEP = [
    "james b zagel",
    "thomas j rueter",
    "david r strawbridge",
    "robert w gettleman",
    "john z lee",
    "j thomas marten",
    "michael north",
    "deborah m fine",
    "fernando l aenlle-rocha",
    "sarah a l merriam",
    "beverly reid o'connell",
    "t s ellis",
    "sr robert c brack",
    "benjamin h settle",
]


def test_preclean_truncates_header_glue():
    assert preclean_header_junk("John Smith cause: criminal") == "John Smith"
    assert preclean_header_junk("designated John Smith") == "John Smith"
    assert preclean_header_junk("Jane Doe: Sentencing") == "Jane Doe"


def test_procedural_junk_quarantined():
    for name in PROCEDURAL_REJECT:
        ok, reasons = classify_name_validity(name)
        assert not ok, (name, reasons)


def test_baxter_accepts_rejected():
    ok, reasons = classify_name_validity("baxter accepts")
    assert not ok, reasons


def test_real_judges_kept():
    fjc_surnames, fjc_full = _fjc_sets()
    for name in REAL_JUDGES_KEEP:
        ok, reasons = classify_name_validity(
            name,
            fjc_surnames=fjc_surnames,
            fjc_full_names=fjc_full,
        )
        assert ok, (name, reasons)


def test_benjamin_h_settle_fjc_rescue():
    fjc_surnames, fjc_full = _fjc_sets()
    ok, reasons = classify_name_validity(
        "benjamin h settle",
        fjc_surnames=fjc_surnames,
        fjc_full_names=fjc_full,
    )
    assert ok, reasons


def test_gold_parent_entity_keep_rate():
    """≥99% of research_dev gold Parent_Entity names must pass the gate."""
    import json

    fjc_surnames, fjc_full = _fjc_sets()
    ok_n = fail_n = 0
    for line in (ROOT / "data/gold/SEL_pilot1000.jsonl").open():
        d = json.loads(line)
        name = d.get("Parent_Entity") or d.get("Extracted_Entity")
        # gold noise row
        if "unassigned" in str(name).lower():
            continue
        ok, _ = classify_name_validity(
            str(name).lower(),
            fjc_surnames=fjc_surnames,
            fjc_full_names=fjc_full,
        )
        if ok:
            ok_n += 1
        else:
            fail_n += 1
    rate = ok_n / max(1, ok_n + fail_n)
    assert rate >= 0.99, (rate, ok_n, fail_n)