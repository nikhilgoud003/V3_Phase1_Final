#!/usr/bin/env python3
"""Generic extract-gap rules: see-above fill, location strip, apostrophe/US, pro se skip."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config
from engine.extract import extract_from_case
from engine.normalize import normalize_name, split_trailing_location
from engine.path_extract import apply_record_copy_fill


def test_split_trailing_location_generic():
    seps = [" - ", "\n"]
    pats = [
        r"(?i),\s*[A-Za-z]{2}(\s+\d{5})?\s*$",
        r"(?i)\b[A-Za-z0-9.]+\s+(ave|avenue|street|blvd|ste|suite)\.?\b",
        r"(?i)^[A-Za-z][A-Za-z .'-]{0,40},\s*[A-Za-z]{2}\s*$",
        r"(?i)^(suite|ste\.?|room|rm\.?)\s+\S+",
        r"(?i).*\b(suite|ste\.?|room|rm\.?|street|avenue|courthouse)\b.*",
    ]
    core, loc = split_trailing_location(
        "Acme Partners LLP - Maple Ave., Dallas, TX",
        separators=seps,
        remainder_patterns=pats,
    )
    assert core == "Acme Partners LLP"
    assert "Dallas" in (loc or "")
    core2, loc2 = split_trailing_location(
        "North & South LLC - Atlanta, GA",
        separators=seps,
        remainder_patterns=pats,
    )
    assert core2 == "North & South LLC"
    assert loc2 == "Atlanta, GA"
    core3, loc3 = split_trailing_location(
        "Smith - Jones LLP",
        separators=seps,
        remainder_patterns=pats,
    )
    core4, loc4 = split_trailing_location(
        "UNITED STATES ATTORNEYS OFFICE - St. Louis",
        separators=seps,
        remainder_patterns=pats,
    )
    assert loc4 is None, loc4
    assert "St. Louis" in core4


def test_strip_trailing_address_contamination_suite_street_courthouse():
    from engine.normalize import strip_trailing_address_contamination

    junk = [
        r"(?i)[,;\s]+(suite|ste\.?|room|rm\.?)\s+[\w-]+\s*$",
        r"(?i)\s+\d{1,5}\s+[A-Za-z0-9 .'-]{1,60}\b(street|st\.|avenue|ave\.?|blvd)\b.*$",
        r"(?i)\s+[A-Za-z0-9 .'-]{2,50}\bcourthouse\b.*$",
        r"(?i)\s*\([^)]*\d[^)]*\)\s*$",
    ]
    # Newline + building + suite via separator pass is tested in extract; peel suite alone:
    c, loc = strip_trailing_address_contamination(
        "Darger & Errante, LLP 116 East 27th Street at Park Avenue",
        patterns=junk,
        junk_chars="*",
    )
    assert "Darger" in c and "Street" not in c
    assert loc and "Street" in loc

    c2, loc2 = strip_trailing_address_contamination(
        "Us Attorney's Office Toree Chardon Ste 1201",
        patterns=junk,
        junk_chars="*",
    )
    assert c2.endswith("Chardon") or "Toree" in c2
    assert loc2 and "1201" in loc2

    c3, loc3 = strip_trailing_address_contamination(
        "United States Marshal Hugo Black Courthouse, Room 240",
        patterns=junk,
        junk_chars="*",
    )
    assert "Marshal" in c3
    assert loc3 and ("Courthouse" in loc3 or "240" in loc3)

    c4, loc4 = strip_trailing_address_contamination(
        "US Attorney's Office - FLM*",
        patterns=junk,
        junk_chars="*",
    )
    assert "*" not in c4
    assert loc4 == "*" or (loc4 and "*" in loc4)

    # Chapter 13 is identity — digit alone must not strip
    c5, loc5 = strip_trailing_address_contamination(
        "Office Of The Chapter 13 Trustee",
        patterns=junk,
        junk_chars="*",
    )
    assert c5 == "Office Of The Chapter 13 Trustee"
    assert loc5 is None

    c6, loc6 = strip_trailing_address_contamination(
        "Proskauer Rose LLP (70W)",
        patterns=junk,
        junk_chars="*",
    )
    assert c6 == "Proskauer Rose LLP"
    assert loc6 and "70W" in loc6


def test_watergate_newline_suite_via_separator():
    seps = [" - ", "\n"]
    pats = [
        r"(?i)^(suite|ste\.?|room|rm\.?)\s+\S+",
        r"(?i).*\b(suite|ste\.?|room|rm\.?|courthouse)\b.*",
    ]
    core, loc = split_trailing_location(
        "Gray, Plant, Mooty, Mooty & Bennett, P.A.\nThe Watergate - Suite 700",
        separators=seps,
        remainder_patterns=pats,
    )
    assert "Gray" in core
    assert "Suite" not in core
    assert loc and "Suite" in loc


def test_apostrophe_delete_and_us_expand():
    n = normalize_name(
        "U.S. Attorney's Office",
        honorifics=[],
        strip_chars='.,;:"()[]{}',
        delete_chars="'",
        replace_regex=[
            {
                "pattern": r"(?i)(?<![A-Za-z])u\.?\s*s\.?(?![A-Za-z])",
                "replacement": "united states",
            }
        ],
    )
    assert n == "united states attorneys office"


def test_see_above_fill_same_name_other_party():
    case = {
        "parties": [
            {
                "name": "A",
                "counsel": [
                    {
                        "name": "Pat Lee",
                        "has_see_above": False,
                        "is_pro_se": False,
                        "entity_info": {
                            "office_name": "Lee Law",
                            "address": "1 Main",
                            "phone": "555",
                            "email": "p@lee.com",
                            "raw_info": "full",
                        },
                    }
                ],
            },
            {
                "name": "B",
                "counsel": [
                    {
                        "name": "Pat Lee",
                        "has_see_above": True,
                        "is_pro_se": False,
                        "entity_info": {
                            "office_name": None,
                            "address": None,
                            "phone": None,
                            "email": None,
                            "raw_info": "see above",
                        },
                    }
                ],
            },
        ]
    }
    spec = {
        "enabled": True,
        "list_path": "parties[].counsel[]",
        "match_field": "name",
        "flag_field": "has_see_above",
        "payload_field": "entity_info",
        "donor_requires_key": "office_name",
        "ignore_payload_keys": ["raw_info", "terminating_date"],
    }
    out, stats = apply_record_copy_fill(copy.deepcopy(case), spec)
    assert stats["filled"] == 1
    ei = out["parties"][1]["counsel"][0]["entity_info"]
    assert ei["office_name"] == "Lee Law"
    assert ei["raw_info"] == "see above"


def test_pro_se_boolean_skip_not_string():
    cfg = load_config(ROOT / "configs/firms.yaml")
    case = {
        "ucid": "x;;1",
        "court": "x",
        "case_type": "cv",
        "filing_date": "2016-01-01",
        "parties": [
            {
                "name": "Self",
                "role": "Plaintiff",
                "counsel": [
                    {
                        "name": "Self Person",
                        "is_pro_se": True,
                        "has_see_above": False,
                        "entity_info": {
                            "office_name": "Self Person",
                            "address": "9 Elm St",
                            "phone": None,
                            "email": None,
                        },
                    }
                ],
            }
        ],
    }
    ments, _ = extract_from_case(case, cfg, source_file="t.json")
    assert ments == []


def test_parties_apply_party_names_false_extracts_party_span():
    """Party names must not be dropped by the party-name negative list."""
    cfg = load_config(ROOT / "configs/parties.yaml")
    case = {
        "ucid": "nysd;;1:17-cv-00001",
        "court": "nysd",
        "case_id": "1:17-cv-00001",
        "case_type": "cv",
        "filing_date": "2017-01-01",
        "parties": [
            {
                "name": "Alice Smith",
                "role": "Plaintiff",
                "party_type": "plaintiff",
                "entity_info": {},
            }
        ],
    }
    ments, _ = extract_from_case(case, cfg, source_file="t.json")
    assert len(ments) == 1
    assert ments[0]["raw_name"] == "Alice Smith"
    assert ments[0]["entity_type"] == "party"
    assert ments[0]["docket_source"] == "case_parties"
