"""Tests for Tier0 USA alias groups and Tier3 cross-UCID skip."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.provenance import DecisionJournal
from engine.tiers import (
    UnionFind,
    apply_tier0_alias_groups,
    pair_in_skipped_alias_group_cross_ucid,
    tier0_alias_group_id,
)


CFG = {
    "entity_type": "party",
    "version": "test",
    "tier0": {
        "alias_groups": [
            {
                "id": "united_states_government",
                "names": ["usa", "u s", "united states", "united states of america"],
                "merge_scope": "same_ucid",
                "confidence": 100,
                "method": "tier0.gov_usa_alias_ucid",
            }
        ]
    },
    "tier3": {
        "routing": {"skip_alias_groups_cross_ucid": ["united_states_government"]}
    },
}


def test_alias_group_id():
    assert tier0_alias_group_id("usa", CFG) == "united_states_government"
    assert tier0_alias_group_id("united states of america", CFG) == "united_states_government"
    assert tier0_alias_group_id("bank of america", CFG) is None


def test_alias_merge_same_ucid_only(tmp_path):
    mentions = [
        {"mention_id": "a", "normalized_name": "usa", "ucid": "x;;1", "court": "paed"},
        {
            "mention_id": "b",
            "normalized_name": "united states of america",
            "ucid": "x;;1",
            "court": "paed",
        },
        {"mention_id": "c", "normalized_name": "usa", "ucid": "y;;2", "court": "paed"},
    ]
    uf = UnionFind()
    for m in mentions:
        uf.add(m["mention_id"])
    journal = DecisionJournal(tmp_path / "dec.jsonl")
    stats = apply_tier0_alias_groups(mentions, CFG, journal, uf)
    assert stats["merges"] == 1
    assert uf.find("a") == uf.find("b")
    assert uf.find("a") != uf.find("c")


def test_cross_ucid_skip():
    ma = {"normalized_name": "usa", "ucid": "paed;;1"}
    mb = {"normalized_name": "usa", "ucid": "paed;;2"}
    assert pair_in_skipped_alias_group_cross_ucid(ma, mb, CFG) == "united_states_government"
    mb2 = {"normalized_name": "usa", "ucid": "paed;;1"}
    assert pair_in_skipped_alias_group_cross_ucid(ma, mb2, CFG) is None
    mb3 = {"normalized_name": "wells fargo", "ucid": "paed;;2"}
    assert pair_in_skipped_alias_group_cross_ucid(ma, mb3, CFG) is None
