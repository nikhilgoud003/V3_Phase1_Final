"""PoC party evidence: Tier3-only path; no deterministic merge."""

from __future__ import annotations

from engine.poc_party_evidence import (
    adjudicate_poc_evidence_via_tier3,
    evidence_strong_enough,
    find_poc_evidence_candidates,
    shared_case_evidence,
)
from engine.tiers import UnionFind


def _m(mid, name, *, judges, firms, mdl, court="paed"):
    return {
        "mention_id": mid,
        "normalized_name": name,
        "profile": f"Party: {name}",
        "court": court,
        "ucid": f"u-{mid}",
        "poc_case_judges": judges,
        "poc_case_firms": firms,
        "poc_mdl_code": mdl,
        "co_mentions": ["acands", "bethlehem steel"],
    }


def test_liability_typo_is_tier3_candidate_not_auto_merged():
    cfg = {
        "name_compat": {"enabled": True},
        "identity_exclusions": {
            "generic_word_overlap": {"enabled": True, "terms": ["trust", "industries"]},
        },
        "tier3": {"enabled": False},  # no live LLM in unit test
    }
    a = _m(
        "a",
        "a-c product liability trust",
        judges=["eduardo c robreno"],
        firms=["the maritime asbestosis legal clinic"],
        mdl=875,
    )
    b = _m(
        "b",
        "a-c product liabilty trust",
        judges=["eduardo c robreno"],
        firms=["the maritime asbestosis legal clinic"],
        mdl=875,
    )
    assert evidence_strong_enough(shared_case_evidence(a, b))
    uf = UnionFind()
    cands = find_poc_evidence_candidates([a, b], uf, cfg)
    assert len(cands) == 1
    # With tier3 disabled, adjudicate must not merge
    stats = adjudicate_poc_evidence_via_tier3([a, b], uf, cfg)
    assert stats["merges"] == 0
    assert uf.find("a") != uf.find("b")


def test_wells_fargo_bank_of_america_not_candidated():
    cfg = {"name_compat": {"enabled": True}, "identity_exclusions": {}}
    a = _m("a", "wells fargo bank", judges=["j"], firms=["f"], mdl=1)
    b = _m("b", "bank of america", judges=["j"], firms=["f"], mdl=1)
    uf = UnionFind()
    assert find_poc_evidence_candidates([a, b], uf, cfg) == []


def test_sony_bmg_not_candidated():
    cfg = {"name_compat": {"enabled": True}, "identity_exclusions": {}}
    a = _m("a", "sony music entertainment", judges=["j"], firms=["f"], mdl=1)
    b = _m("b", "bmg music entertainment", judges=["j"], firms=["f"], mdl=1)
    uf = UnionFind()
    assert find_poc_evidence_candidates([a, b], uf, cfg) == []
