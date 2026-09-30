"""Unit tests for Tier3 citation verification and asymmetric MATCH bar."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.tier3_citation import (
    apply_citation_and_match_rails,
    truth_table,
    verify_citations,
)


CFG_CITE = {
    "tier3": {
        "citation_verification": {"enabled": True, "require_cited_evidence": True},
        "match_requires_discriminating_fact": {"enabled": False},
    }
}

CFG_BOTH = {
    "tier3": {
        "citation_verification": {"enabled": True, "require_cited_evidence": True},
        "match_requires_discriminating_fact": {
            "enabled": True,
            "require_cited": True,
            "facts": [
                "same_pacer_id",
                "same_domain",
                "same_phone",
                "same_address",
                "shared_fjc_nid",
                "identical_name_ge2_same_court",
            ],
        },
    }
}


def _jj_pair():
    ma = {
        "normalized_name": "johnson & johnson",
        "party_role": "Defendant",
        "party_type": "defendant",
        "pacer_id": None,
        "ucid": "njd;;3:17-cv-05224",
        "court": "njd",
        "year": 2017,
    }
    mb = {
        "normalized_name": "johnson & johnson consumer",
        "party_role": "Defendant",
        "party_type": "defendant",
        "pacer_id": None,
        "ucid": "njd;;3:17-cv-05224",
        "court": "njd",
        "year": 2017,
    }
    evidence = {
        "same_court": True,
        "same_year": True,
        "same_ucid": True,
        "same_domain": False,
        "ucid_only_corroboration": True,
        "identity_conflict": None,
        "party_role_a": "Defendant",
        "party_role_b": "Defendant",
        "party_type_a": "defendant",
        "party_type_b": "defendant",
        "barrier_reasons": [],
        "embedding_similarity": 0.98,
    }
    return ma, mb, evidence


def test_jj_false_pacer_citation_forces_uncertain():
    ma, mb, evidence = _jj_pair()
    result = {
        "decision": "MATCH",
        "confidence": 95,
        "rationale": "identical names and same pacer id",
        "cited_evidence": ["identical_normalized_name", "same_pacer_id", "same_party_role"],
        "signals": ["identical_normalized_name", "same_pacer_id"],
    }
    report = verify_citations(result, ma, mb, evidence, cfg=CFG_CITE)
    assert report["hallucination"] is True
    assert "same_pacer_id" in report["failed"]
    assert "identical_normalized_name" in report["failed"]

    d, rat, sigs, meta = apply_citation_and_match_rails(
        "MATCH", result["rationale"], list(result["signals"]), result, ma, mb, evidence, CFG_CITE
    )
    assert d == "UNCERTAIN"
    assert "hallucination_detected" in sigs
    assert meta["citation_verification"]["forced_uncertain"] is True


def test_usa_false_same_domain_forces_uncertain():
    ma = {
        "normalized_name": "usa",
        "party_role": "Plaintiff",
        "party_type": "plaintiff",
        "pacer_id": None,
        "ucid": "paed;;2:03-cr-00358",
        "court": "paed",
        "year": 2003,
    }
    mb = {
        "normalized_name": "usa",
        "party_role": "Plaintiff",
        "party_type": "plaintiff",
        "pacer_id": None,
        "ucid": "paed;;2:06-cr-00541",
        "court": "paed",
        "year": 2006,
    }
    evidence = {
        "same_court": True,
        "same_year": False,
        "same_ucid": False,
        "same_domain": False,
        "ucid_only_corroboration": False,
        "identity_conflict": None,
        "barrier_reasons": ["single_token_name"],
        "embedding_similarity": 0.98,
        "party_role_a": "Plaintiff",
        "party_role_b": "Plaintiff",
        "party_type_a": "plaintiff",
        "party_type_b": "plaintiff",
    }
    result = {
        "decision": "NO_MATCH",
        "confidence": 100,
        "rationale": "different UCIDs",
        "cited_evidence": ["same_domain", "ucid_only_corroboration"],
        "signals": ["same_domain", "ucid_only_corroboration"],
    }
    d, _, sigs, _ = apply_citation_and_match_rails(
        "NO_MATCH", result["rationale"], list(result["signals"]), result, ma, mb, evidence, CFG_CITE
    )
    assert d == "UNCERTAIN"
    assert "hallucination_detected" in sigs


def test_asymmetric_bar_blocks_match_without_discriminating_fact():
    """Same ≥2-token name, different courts, no pacer — MATCH without discriminating fact."""
    ma = {
        "normalized_name": "acme industries",
        "party_role": "Defendant",
        "party_type": "defendant",
        "pacer_id": None,
        "ucid": "nvd;;1:16-cv-00001",
        "court": "nvd",
        "year": 2016,
    }
    mb = {
        "normalized_name": "acme industries",
        "party_role": "Defendant",
        "party_type": "defendant",
        "pacer_id": None,
        "ucid": "txwd;;1:17-cv-00002",
        "court": "txwd",
        "year": 2017,
    }
    evidence = {
        "same_court": False,
        "same_year": False,
        "same_ucid": False,
        "same_domain": False,
        "ucid_only_corroboration": False,
        "identity_conflict": None,
        "barrier_reasons": [],
        "embedding_similarity": 0.97,
        "party_role_a": "Defendant",
        "party_role_b": "Defendant",
        "party_type_a": "defendant",
        "party_type_b": "defendant",
    }
    result = {
        "decision": "MATCH",
        "confidence": 95,
        "rationale": "identical names",
        "cited_evidence": ["identical_normalized_name", "identical_normalized_name_ge2"],
        "signals": ["identical_normalized_name"],
    }
    # Citation OK (names really identical) but no discriminating fact
    d, _, sigs, meta = apply_citation_and_match_rails(
        "MATCH", result["rationale"], list(result["signals"]), result, ma, mb, evidence, CFG_BOTH
    )
    assert d == "UNCERTAIN"
    assert "no_discriminating_fact" in sigs
    assert meta.get("match_bar_forced_uncertain") is True


def test_truth_same_pacer_requires_both_present():
    ma = {"normalized_name": "a", "pacer_id": None, "ucid": "x", "court": "c"}
    mb = {"normalized_name": "a", "pacer_id": None, "ucid": "y", "court": "c"}
    t = truth_table(ma, mb, {"same_court": True, "same_ucid": False, "same_year": False, "same_domain": False})
    assert t["same_pacer_id"] is False
    assert t["different_pacer_id"] is False
