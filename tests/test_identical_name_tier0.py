"""Tests for identical-name Tier0 rule and US-prefix normalize."""

from __future__ import annotations

from engine.normalize import normalize_name, strip_name_prefixes
from engine.provenance import DecisionJournal
from engine.tiers import UnionFind, tier0_merge_groups


def test_strip_us_prefix():
    assert strip_name_prefixes("US CLAY D LAND", ["us", "u.s.", "u.s"]) == "CLAY D LAND"
    assert (
        normalize_name(
            "US CLAY D LAND",
            honorifics=[],
            name_prefixes=["us", "u.s.", "u.s"],
        )
        == "clay d land"
    )


def test_identical_name_same_court_merges_across_years(tmp_path):
    cfg = {
        "entity_type": "judge",
        "version": "test",
        "tier0": {
            "enabled": True,
            "rules": [
                {
                    "id": "exact_normalized_name_same_court",
                    "type": "conjunction",
                    "fields": ["normalized_name", "court"],
                    "min_name_tokens": 2,
                    "confidence": 99,
                    "method": "tier0.exact_name_court",
                }
            ],
        },
    }
    ments = [
        {
            "mention_id": "a",
            "normalized_name": "james b zagel",
            "court": "ilnd",
            "year": 2009,
            "ucid": "ilnd;;1",
        },
        {
            "mention_id": "b",
            "normalized_name": "james b zagel",
            "court": "ilnd",
            "year": 2014,
            "ucid": "ilnd;;2",
        },
        {
            "mention_id": "c",
            "normalized_name": "james b zagel",
            "court": "paed",  # different court — must NOT merge
            "year": 2014,
            "ucid": "paed;;1",
        },
    ]
    journal = DecisionJournal(tmp_path / "d.jsonl")
    uf = UnionFind()
    stats = tier0_merge_groups(ments, cfg, journal, uf)
    assert uf.find("a") == uf.find("b")
    assert uf.find("a") != uf.find("c")
    assert stats["rules_fired"].get("exact_normalized_name_same_court", 0) >= 1


def test_identical_name_blocked_by_distinct_fjc(tmp_path):
    cfg = {
        "entity_type": "judge",
        "version": "test",
        "tier0": {
            "enabled": True,
            "rules": [
                {
                    "id": "exact_normalized_name_same_court",
                    "type": "conjunction",
                    "fields": ["normalized_name", "court"],
                    "min_name_tokens": 2,
                    "confidence": 99,
                    "method": "tier0.exact_name_court",
                }
            ],
        },
    }
    ments = [
        {
            "mention_id": "a",
            "normalized_name": "john smith",
            "court": "ilnd",
            "fjc_nid": "1",
        },
        {
            "mention_id": "b",
            "normalized_name": "john smith",
            "court": "ilnd",
            "fjc_nid": "2",
        },
    ]
    journal = DecisionJournal(tmp_path / "d.jsonl")
    uf = UnionFind()
    tier0_merge_groups(ments, cfg, journal, uf)
    assert uf.find("a") != uf.find("b")


def test_exact_name_office_court_rejects_single_token(tmp_path):
    rule = {
        "id": "exact_name_office_court",
        "type": "conjunction",
        "fields": ["normalized_name", "office_name", "court"],
        "require_non_null": ["office_name"],
        "min_name_tokens": 2,
        "confidence": 99,
        "method": "tier0.exact_name_office_court",
    }
    cfg = {"entity_type": "judge", "version": "test", "tier0": {"enabled": True, "rules": [rule]}}
    ments = [
        {
            "mention_id": "g1",
            "normalized_name": "guillermo",
            "office_name": "guillermo r garcia",
            "court": "txsd",
            "ucid": "txsd;;1",
        },
        {
            "mention_id": "g2",
            "normalized_name": "guillermo",
            "office_name": "guillermo r garcia",
            "court": "txsd",
            "ucid": "txsd;;2",
        },
        {
            "mention_id": "f1",
            "normalized_name": "guillermo r garcia",
            "office_name": "garcia law",
            "court": "txsd",
            "ucid": "txsd;;3",
        },
        {
            "mention_id": "f2",
            "normalized_name": "guillermo r garcia",
            "office_name": "garcia law",
            "court": "txsd",
            "ucid": "txsd;;4",
        },
    ]
    journal = DecisionJournal(tmp_path / "d.jsonl")
    uf = UnionFind()
    stats = tier0_merge_groups(ments, cfg, journal, uf)
    assert uf.find("g1") != uf.find("g2"), "single-token office_court must not cross-UCID merge"
    assert uf.find("f1") == uf.find("f2"), "≥2 token office_court should still merge"
    assert stats["rules_fired"].get("exact_name_office_court", 0) >= 1


def test_exact_name_office_court_without_min_tokens_would_glue(tmp_path):
    """Control: same rule minus min_name_tokens reproduces the mega-entity glue."""
    rule = {
        "id": "exact_name_office_court",
        "type": "conjunction",
        "fields": ["normalized_name", "office_name", "court"],
        "require_non_null": ["office_name"],
        "confidence": 99,
        "method": "tier0.exact_name_office_court",
    }
    cfg = {"entity_type": "judge", "version": "test", "tier0": {"enabled": True, "rules": [rule]}}
    ments = [
        {
            "mention_id": "g1",
            "normalized_name": "guillermo",
            "office_name": "guillermo r garcia",
            "court": "txsd",
            "ucid": "txsd;;1",
        },
        {
            "mention_id": "g2",
            "normalized_name": "guillermo",
            "office_name": "guillermo r garcia",
            "court": "txsd",
            "ucid": "txsd;;2",
        },
    ]
    journal = DecisionJournal(tmp_path / "d.jsonl")
    uf = UnionFind()
    tier0_merge_groups(ments, cfg, journal, uf)
    assert uf.find("g1") == uf.find("g2")


def test_suffix_equiv_generational_same_court_merges(tmp_path):
    cfg = {
        "entity_type": "judge",
        "version": "test",
        "tier0": {
            "enabled": True,
            "rules": [
                {
                    "id": "exact_normalized_name_same_court",
                    "type": "conjunction",
                    "fields": ["normalized_name", "court"],
                    "min_name_tokens": 2,
                    "suffix_equiv_generational": True,
                    "confidence": 99,
                    "method": "tier0.exact_name_court",
                }
            ],
        },
    }
    ments = [
        {
            "mention_id": "a",
            "normalized_name": "joseph c wilkinson jr",
            "court": "laed",
            "ucid": "laed;;2:16-cv-08827",
        },
        {
            "mention_id": "b",
            "normalized_name": "joseph c wilkinson",
            "court": "laed",
            "ucid": "laed;;2:17-cv-08848",
        },
        {
            "mention_id": "c",
            "normalized_name": "william h yohn",
            "court": "paed",
            "ucid": "paed;;2:08-cv-05845",
        },
        {
            "mention_id": "d",
            "normalized_name": "william h yohn jr",
            "court": "paed",
            "ucid": "paed;;2:02-cv-03462",
        },
    ]
    journal = DecisionJournal(tmp_path / "d2.jsonl")
    uf = UnionFind()
    tier0_merge_groups(ments, cfg, journal, uf)
    assert uf.find("a") == uf.find("b")
    assert uf.find("c") == uf.find("d")
