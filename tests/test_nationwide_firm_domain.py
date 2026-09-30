"""Nationwide firm-specific domain T0: prefix + exact, with institutional exclusions."""

from __future__ import annotations

from engine.provenance import DecisionJournal
from engine.tiers import UnionFind, tier0_merge_groups

NATIONWIDE_MATCH = {
    "forbid_office_class": ["doj", "fpd", "ag", "city_county_state"],
    "forbid_domains": ["gmail.com", "usdoj.gov", "fd.org", "comcast.net"],
    "forbid_domain_suffixes": [".gov", ".rr.com"],
    "forbid_normalized_names": ["attorney at law", "law office"],
}

RULES = [
    {
        "id": "office_domain_nationwide",
        "type": "conjunction",
        "fields": ["first_two_tokens", "domain"],
        "require_non_null": ["domain"],
        "min_name_tokens": 2,
        "match": NATIONWIDE_MATCH,
        "confidence": 97,
        "method": "tier0.office_domain_nationwide",
    },
    {
        "id": "office_domain_nationwide_exact",
        "type": "conjunction",
        "fields": ["normalized_name", "domain"],
        "require_non_null": ["domain"],
        "min_name_tokens": 1,
        "match": NATIONWIDE_MATCH,
        "confidence": 96,
        "method": "tier0.office_domain_nationwide_exact",
    },
]


def _cfg():
    return {
        "entity_type": "firm",
        "version": "test",
        "identity_exclusions": {
            "non_identifying_domains": NATIONWIDE_MATCH["forbid_domains"],
            "non_identifying_domain_suffixes": NATIONWIDE_MATCH["forbid_domain_suffixes"],
            "institutional_office_classes": NATIONWIDE_MATCH["forbid_office_class"],
            "generic_office_names": NATIONWIDE_MATCH["forbid_normalized_names"],
        },
        "tier0": {"enabled": True, "rules": RULES},
    }


def test_lockelord_merges_across_courts(tmp_path):
    ments = [
        {
            "mention_id": "a",
            "normalized_name": "locke lord",
            "domain": "lockelord.com",
            "court": "ilnd",
            "office_class": "private_other",
        },
        {
            "mention_id": "b",
            "normalized_name": "locke lord bissell and liddell",
            "domain": "lockelord.com",
            "court": "txnd",
            "office_class": "private_other",
        },
    ]
    uf = UnionFind()
    stats = tier0_merge_groups(ments, _cfg(), DecisionJournal(tmp_path / "d.jsonl"), uf)
    assert uf.find("a") == uf.find("b")
    assert stats["rules_fired"].get("office_domain_nationwide", 0) >= 1


def test_venable_one_token_merges_across_courts(tmp_path):
    ments = [
        {
            "mention_id": "a",
            "normalized_name": "venable",
            "domain": "venable.com",
            "court": "paed",
            "office_class": "private_other",
        },
        {
            "mention_id": "b",
            "normalized_name": "venable",
            "domain": "venable.com",
            "court": "mdd",
            "office_class": "private_other",
        },
    ]
    uf = UnionFind()
    tier0_merge_groups(ments, _cfg(), DecisionJournal(tmp_path / "d.jsonl"), uf)
    assert uf.find("a") == uf.find("b")


def test_usao_usdoj_does_not_merge_across_courts(tmp_path):
    ments = [
        {
            "mention_id": "a",
            "normalized_name": "united states attorneys office",
            "domain": "usdoj.gov",
            "court": "ilnd",
            "office_class": "doj",
        },
        {
            "mention_id": "b",
            "normalized_name": "united states attorneys office",
            "domain": "usdoj.gov",
            "court": "nysd",
            "office_class": "doj",
        },
    ]
    uf = UnionFind()
    tier0_merge_groups(ments, _cfg(), DecisionJournal(tmp_path / "d.jsonl"), uf)
    assert uf.find("a") != uf.find("b")


def test_attorney_at_law_isp_does_not_merge(tmp_path):
    ments = [
        {
            "mention_id": "a",
            "normalized_name": "attorney at law",
            "domain": "comcast.net",
            "court": "txsd",
            "office_class": "private_other",
        },
        {
            "mention_id": "b",
            "normalized_name": "attorney at law",
            "domain": "comcast.net",
            "court": "ilnd",
            "office_class": "private_other",
        },
    ]
    uf = UnionFind()
    tier0_merge_groups(ments, _cfg(), DecisionJournal(tmp_path / "d.jsonl"), uf)
    assert uf.find("a") != uf.find("b")
