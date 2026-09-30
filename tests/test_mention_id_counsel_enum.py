"""mention_id: counsel_enum disambiguates firms; judges keep the 7-tuple hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from engine.config_loader import load_config
from engine.extract import _mention_id, extract_from_case

ROOT = Path(__file__).resolve().parents[1]


def test_seven_tuple_hash_is_stable():
    parts = [
        "nysd;;1:16-cv-01139",
        "assigned_judge",
        "Edgardo Ramos",
        None,
        0,
        None,
        "judge",
    ]
    expected = "mnt_" + hashlib.sha1(
        "|".join("" if p is None else str(p) for p in parts).encode()
    ).hexdigest()[:16]
    assert _mention_id(parts) == expected


def test_two_counsel_same_office_get_distinct_ids():
    cfg = load_config(ROOT / "configs/firms.yaml")
    case = {
        "ucid": "ilnd;;1:16-cv-00001",
        "court": "ilnd",
        "case_id": "1:16-cv-00001",
        "case_type": "cv",
        "case_name": "Test v. Test",
        "filing_date": "2016-01-01",
        "parties": [
            {
                "name": "Acme Corp",
                "role": "Defendant",
                "counsel": [
                    {
                        "name": "Ann Alpha",
                        "has_see_above": False,
                        "is_pro_se": False,
                        "entity_info": {
                            "office_name": "Locke Lord",
                            "email": "a@lockelord.com",
                            "phone": "312-000-0001",
                        },
                    },
                    {
                        "name": "Bob Beta",
                        "has_see_above": False,
                        "is_pro_se": False,
                        "entity_info": {
                            "office_name": "Locke Lord",
                            "email": "b@lockelord.com",
                            "phone": "312-000-0002",
                        },
                    },
                ],
            }
        ],
    }
    mentions, _ = extract_from_case(case, cfg, source_file="synthetic.json")
    offices = [m for m in mentions if (m.get("normalized_name") or "") == "locke lord"]
    assert len(offices) == 2, [m.get("normalized_name") for m in mentions]
    assert offices[0]["mention_id"] != offices[1]["mention_id"]
    assert {m.get("counsel_enum") for m in offices} == {0, 1}


def test_gold_judge_header_ids_unchanged_on_real_case():
    """Party/header judge IDs on a gold case still match recall_fix (no counsel_enum in hash)."""
    gold_path = ROOT / "data/runs/judges_pilot_recall_fix/mentions/judges_mentions.jsonl"
    json_path = ROOT / "data/json/pilot_1000/akd-3-16-cr-00074.json"
    if not gold_path.exists() or not json_path.exists():
        return
    gold_ids = set()
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            if m.get("ucid") != "akd;;3:16-cr-00074":
                continue
            if m.get("docket_source") == "line_entry":
                continue
            gold_ids.add(m["mention_id"])
            assert m.get("counsel_enum") is None
    assert gold_ids
    cfg = load_config(ROOT / "configs/judges.yaml")
    case = json.loads(json_path.read_text(encoding="utf-8"))
    mentions, _ = extract_from_case(case, cfg, source_file="akd-3-16-cr-00074.json")
    new_ids = {
        m["mention_id"]
        for m in mentions
        if m.get("docket_source") != "line_entry"
    }
    assert new_ids == gold_ids, (sorted(gold_ids - new_ids)[:5], sorted(new_ids - gold_ids)[:5])
