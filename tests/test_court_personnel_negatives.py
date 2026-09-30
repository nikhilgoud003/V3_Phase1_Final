"""Court-personnel negative evidence (USPO / reporter / clerk signature)."""

from __future__ import annotations

import json
from pathlib import Path

from engine.config_loader import load_config
from engine.extract import extract_from_case, mine_court_personnel_negatives

ROOT = Path(__file__).resolve().parents[1]
ADENUGA = ROOT / "data/json/nyed_smoke_14/1-16-cr-00002.json"

ADENUGA_MINUTE = (
    "Minute Entry for proceedings held before Judge Ann M Donnelly: Status Conference "
    "as to Faud Adenuga held on 4/25/2016. USPO Yara Suarez present. "
    "(Court Reporter Stacy Mace.) (Greene, Donna) (Entered: 04/25/2016)"
)

CLERK_ONLY = "ORDER as to Faud Adenuga. (Galeano, Sonia) (Entered: 03/30/2016)"


def _personnel_cfg() -> dict:
    return {
        "extraction": {
            "negative_evidence": {
                "court_personnel": {
                    "enabled": True,
                    "path": "docket[].docket_text",
                    "patterns": [
                        {
                            "name": "uspo",
                            "regex": r"\b(?i:USPO|U\.?S\.?\s*P(?:robation)?\s*O(?:fficer)?)\s+([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){0,3})",
                        },
                        {
                            "name": "court_reporter",
                            "regex": r"\b(?i:Court\s+Reporters?)\s*:?\s*([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){0,3})",
                        },
                        {
                            "name": "clerk_signature",
                            "regex": r"\(([A-Z][A-Za-z'\-]+),\s*([A-Z][A-Za-z'\-]+)\)(?=[^\n]{0,80}\(Entered:)",
                            "last_first": True,
                        },
                    ],
                }
            }
        }
    }


def test_mine_uspo_reporter_clerk_names():
    case = {"docket": [{"docket_text": ADENUGA_MINUTE}, {"docket_text": CLERK_ONLY}]}
    names = mine_court_personnel_negatives(case, _personnel_cfg(), honorifics=[], strip_chars=".,;:'\"")
    assert "yara suarez" in names
    assert "stacy mace" in names
    assert "donna greene" in names
    assert "sonia galeano" in names


def test_disabled_when_unset():
    case = {"docket": [{"docket_text": ADENUGA_MINUTE}]}
    names = mine_court_personnel_negatives(case, {"extraction": {"negative_evidence": {}}}, honorifics=[], strip_chars="")
    assert names == set()


def test_adenuga_extract_drops_personnel_keeps_judges():
    cfg = load_config(ROOT / "configs/judges.yaml")
    case = json.loads(ADENUGA.read_text(encoding="utf-8"))
    mentions, _ = extract_from_case(case, cfg, source_file="1-16-cr-00002.json")
    norms = {m["normalized_name"] for m in mentions}
    assert "yara suarez" not in norms, mentions
    assert "stacy mace" not in norms
    assert "sonia galeano" not in norms
    assert "donna greene" not in norms
    assert "linda danelczyk" not in norms
def test_header_assigned_not_dropped_by_clerk_signature():
    """(Last, First) near (Entered:) must not drop a header/party assigned judge."""
    cfg = load_config(ROOT / "configs/judges.yaml")
    case = {
        "ucid": "nysd;;1:16-cv-01139",
        "court": "nysd",
        "case_type": "cv",
        "case_id": "1:16-cv-01139",
        "filing_date": "02/16/2016",
        "judge": "Judge Edgardo Ramos",
        "referred_judges": [],
        "parties": [{"name": "P", "role": "Plaintiff", "judge": None, "counsel": []}],
        "docket": [
            {
                "docket_text": (
                    "ORDER granting motion. (HEREBY ORDERED by Judge Edgardo Ramos) "
                    "(Ramos, Edgardo) (Entered: 03/14/2016)"
                )
            }
        ],
    }
    mentions, _ = extract_from_case(case, cfg, source_file="nysd-1-16-cv-01139.json")
    assigned = [
        m
        for m in mentions
        if m.get("extraction_method") in {"header_assigned_judge", "party_assigned_judge"}
        and m.get("normalized_name") == "edgardo ramos"
    ]
    assert assigned, [(m.get("normalized_name"), m.get("extraction_method"), m.get("docket_source")) for m in mentions]
    # Docket-line clerk/judge signature spans may still be dropped as personnel.
    docket_ramos = [
        m
        for m in mentions
        if m.get("docket_source") == "line_entry" and m.get("normalized_name") == "edgardo ramos"
    ]
    assert not docket_ramos
