#!/usr/bin/env python3
"""FJC alias match: george carol hanks → george c hanks, tagged fjc_alias_match."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.fjc import fjc_alias_forms, load_fjc_index, link_mentions_to_fjc


HONORIFICS = ["honorable", "hon", "judge", "magistrate", "chief", "senior", "district"]
STRIP = '.,;:"()[]{}'


def _write_mini_fjc(tmp: Path) -> tuple[Path, Path]:
    csv_path = tmp / "fjc.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "nid",
                "First Name",
                "Middle Name",
                "Last Name",
                "Suffix",
                "Court Name (1)",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "nid": "1394776",
                "First Name": "George",
                "Middle Name": "Carol",
                "Last Name": "Hanks",
                "Suffix": "Jr.",
                "Court Name (1)": "U.S. District Court for the Southern District of Texas",
            }
        )
        w.writerow(
            {
                "nid": "6840201",
                "First Name": "Jeffrey",
                "Middle Name": "Vincent",
                "Last Name": "Brown",
                "Suffix": "",
                "Court Name (1)": "U.S. District Court for the Southern District of Texas",
            }
        )
    xwalk = tmp / "xwalk.json"
    xwalk.write_text(
        json.dumps(
            {"U.S. District Court for the Southern District of Texas": "txsd"}
        ),
        encoding="utf-8",
    )
    return csv_path, xwalk


def test_alias_forms_include_initial_middle():
    aliases = fjc_alias_forms("George", "Carol", "Hanks", "Jr")
    joined = " | ".join(a.lower() for a in aliases)
    assert "george c hanks" in joined
    assert "g c hanks" in joined
    assert "george hanks" in joined


def test_george_c_hanks_links_as_alias(tmp_path):
    csv_path, xwalk = _write_mini_fjc(tmp_path)
    idx = load_fjc_index(csv_path, xwalk, HONORIFICS, STRIP)
    ments = [
        {"mention_id": "e1", "normalized_name": "george carol hanks", "court": "txsd"},
        {"mention_id": "a1", "normalized_name": "george c hanks", "court": "txsd"},
        {"mention_id": "a2", "normalized_name": "george hanks", "court": "txsd"},
        {"mention_id": "a3", "normalized_name": "george c hanks jr", "court": "txsd"},
        {"mention_id": "b1", "normalized_name": "jeffrey vincent brown", "court": "txsd"},
        {"mention_id": "b2", "normalized_name": "jeffrey v brown", "court": "txsd"},
    ]
    stats = link_mentions_to_fjc(ments, idx)
    by = {m["mention_id"]: m for m in ments}

    assert by["e1"]["fjc_nid"] == "1394776"
    assert by["e1"]["fjc_match_method"] == "fjc_exact"
    assert by["a1"]["fjc_nid"] == "1394776"
    assert by["a1"]["fjc_match_method"] == "fjc_alias_match"
    assert by["a2"]["fjc_nid"] == "1394776"
    assert by["a2"]["fjc_match_method"] == "fjc_alias_match"
    assert by["a3"]["fjc_nid"] == "1394776"
    assert by["a3"]["fjc_match_method"] == "fjc_alias_match"
    assert by["b1"]["fjc_nid"] == "6840201"
    assert by["b1"]["fjc_match_method"] == "fjc_exact"
    assert by["b2"]["fjc_nid"] == "6840201"
    assert by["b2"]["fjc_match_method"] == "fjc_alias_match"
    assert stats["linked_court_unique"] >= 2
    assert stats["linked_court_unique_alias"] >= 3


def test_s_smith_does_not_global_alias_to_sidney(tmp_path):
    """Regression: 2-token initial+surname must not attach to an out-of-court judge."""
    csv_path = tmp_path / "fjc.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["nid", "First Name", "Middle Name", "Last Name", "Suffix", "Court Name (1)"],
        )
        w.writeheader()
        w.writerow(
            {
                "nid": "1388011",
                "First Name": "Sidney",
                "Middle Name": "Oslin",
                "Last Name": "Smith",
                "Suffix": "Jr.",
                "Court Name (1)": "U.S. District Court for the Northern District of Georgia",
            }
        )
        w.writerow(
            {
                "nid": "1394776",
                "First Name": "George",
                "Middle Name": "Carol",
                "Last Name": "Hanks",
                "Suffix": "Jr.",
                "Court Name (1)": "U.S. District Court for the Southern District of Texas",
            }
        )
    xwalk = tmp_path / "xwalk.json"
    xwalk.write_text(
        json.dumps(
            {
                "U.S. District Court for the Northern District of Georgia": "gand",
                "U.S. District Court for the Southern District of Texas": "txsd",
            }
        ),
        encoding="utf-8",
    )
    idx = load_fjc_index(csv_path, xwalk, HONORIFICS, STRIP)
    ments = [
        {"mention_id": "s1", "normalized_name": "s smith", "court": "txsd"},
        {"mention_id": "g1", "normalized_name": "g c hanks", "court": "txsd"},
        {"mention_id": "g2", "normalized_name": "george hanks", "court": "txsd"},
    ]
    link_mentions_to_fjc(ments, idx)
    by = {m["mention_id"]: m for m in ments}
    assert by["s1"].get("fjc_nid") is None
    assert by["s1"].get("fjc_match_method") is None
    assert by["g1"]["fjc_nid"] == "1394776"
    assert by["g1"]["fjc_match_method"] == "fjc_alias_match"
    assert by["g2"]["fjc_nid"] == "1394776"
    assert by["g2"]["fjc_match_method"] == "fjc_alias_match"


def test_colliding_alias_is_not_linked(tmp_path):
    csv_path = tmp_path / "fjc.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["nid", "First Name", "Middle Name", "Last Name", "Suffix", "Court Name (1)"],
        )
        w.writeheader()
        w.writerow(
            {
                "nid": "1",
                "First Name": "John",
                "Middle Name": "Adam",
                "Last Name": "Smith",
                "Suffix": "",
                "Court Name (1)": "U.S. District Court for the Southern District of Texas",
            }
        )
        w.writerow(
            {
                "nid": "2",
                "First Name": "John",
                "Middle Name": "Brian",
                "Last Name": "Smith",
                "Suffix": "",
                "Court Name (1)": "U.S. District Court for the Southern District of Texas",
            }
        )
    xwalk = tmp_path / "xwalk.json"
    xwalk.write_text(
        json.dumps(
            {"U.S. District Court for the Southern District of Texas": "txsd"}
        ),
        encoding="utf-8",
    )
    idx = load_fjc_index(csv_path, xwalk, HONORIFICS, STRIP)
    ments = [{"mention_id": "x", "normalized_name": "john smith", "court": "txsd"}]
    link_mentions_to_fjc(ments, idx)
    assert ments[0].get("fjc_nid") is None
    assert ments[0].get("fjc_match_method") is None
