"""Tier3 citation verification and asymmetric MATCH bar.

The LLM must cite concrete evidence keys. Code verifies those claims against
the evidence bundle + mention fields. Failed citations (hallucinations) force
UNCERTAIN. MATCH additionally requires ≥1 verified discriminating fact.
"""

from __future__ import annotations

from typing import Any


# Claims the model may cite. Values are checked against evidence/mentions.
CLAIM_ALIASES: dict[str, str] = {
    "same_pacer_id": "same_pacer_id",
    "shared_pacer_id": "same_pacer_id",
    "identical_normalized_name": "identical_normalized_name",
    "identical_normalized_full_name": "identical_normalized_name",
    "identical_name": "identical_normalized_name",
    "same_name": "identical_normalized_name",
    "identical_normalized_name_ge2": "identical_normalized_name_ge2",
    "identical_name_ge2": "identical_normalized_name_ge2",
    "same_court": "same_court",
    "same_year": "same_year",
    "same_ucid": "same_ucid",
    "same_domain": "same_domain",
    "shared_domain": "same_domain",
    "same_party_role": "same_party_role",
    "same_party_type": "same_party_type",
    "shared_fjc_nid": "shared_fjc_nid",
    "same_fjc_nid": "shared_fjc_nid",
    "same_phone": "same_phone",
    "same_address": "same_address",
    "name_mismatch": "name_mismatch",
    "different_name": "name_mismatch",
    "different_ucid": "different_ucid",
    "different_pacer_id": "different_pacer_id",
    "pacer_id_mismatch": "different_pacer_id",
    "party_type_mismatch": "party_type_mismatch",
    "party_role_conflict": "party_role_conflict",
    "opposing_roles": "party_role_conflict",
    "different_year": "different_year",
    "single_token_name": "single_token_name",
    "ucid_only_corroboration": "ucid_only_corroboration",
    "embedding_similarity": "embedding_similarity_high",
    "high_embedding_similarity": "embedding_similarity_high",
}


def _norm_claim(raw: str) -> str | None:
    s = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not s:
        return None
    return CLAIM_ALIASES.get(s, s)


def _nonempty(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    return True


def _pacer(m: dict) -> str | None:
    v = m.get("pacer_id")
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _phone(m: dict) -> str | None:
    v = m.get("phone")
    if not _nonempty(v):
        return None
    return "".join(ch for ch in str(v) if ch.isdigit()) or None


def _address(m: dict) -> str | None:
    v = m.get("address")
    if not _nonempty(v):
        return None
    return " ".join(str(v).lower().split()) or None


def truth_table(ma: dict, mb: dict, evidence: dict) -> dict[str, bool]:
    """Ground-truth boolean claims derivable from evidence + mentions."""
    na = (ma.get("normalized_name") or "").strip().lower()
    nb = (mb.get("normalized_name") or "").strip().lower()
    identical = bool(na and nb and na == nb)
    tok_n = len(na.split()) if na else 0
    pa, pb = _pacer(ma), _pacer(mb)
    phone_a, phone_b = _phone(ma), _phone(mb)
    addr_a, addr_b = _address(ma), _address(mb)
    fa, fb = ma.get("fjc_nid"), mb.get("fjc_nid")
    role_a = (ma.get("party_role") or evidence.get("party_role_a") or "").strip().lower()
    role_b = (mb.get("party_role") or evidence.get("party_role_b") or "").strip().lower()
    type_a = (ma.get("party_type") or evidence.get("party_type_a") or "").strip().lower()
    type_b = (mb.get("party_type") or evidence.get("party_type_b") or "").strip().lower()
    sim = float(evidence.get("embedding_similarity") or 0.0)
    barrier_reasons = set(evidence.get("barrier_reasons") or [])

    same_court = bool(evidence.get("same_court"))
    same_ucid = bool(evidence.get("same_ucid"))
    same_year = bool(evidence.get("same_year"))
    same_domain = bool(evidence.get("same_domain"))

    return {
        "same_court": same_court,
        "same_year": same_year,
        "same_ucid": same_ucid,
        "same_domain": same_domain,
        "same_pacer_id": bool(pa and pb and pa == pb),
        "identical_normalized_name": identical,
        "identical_normalized_name_ge2": identical and tok_n >= 2,
        "same_party_role": bool(role_a and role_b and role_a == role_b),
        "same_party_type": bool(type_a and type_b and type_a == type_b),
        "shared_fjc_nid": bool(fa and fb and str(fa) == str(fb)),
        "same_phone": bool(phone_a and phone_b and phone_a == phone_b),
        "same_address": bool(addr_a and addr_b and addr_a == addr_b),
        "name_mismatch": bool(na and nb and na != nb),
        "different_ucid": bool(ma.get("ucid") and mb.get("ucid") and ma.get("ucid") != mb.get("ucid")),
        # Only true when BOTH ids are present and differ — empty≠empty is NOT "different"
        "different_pacer_id": bool(pa and pb and pa != pb),
        "party_type_mismatch": bool(type_a and type_b and type_a != type_b),
        "party_role_conflict": bool(evidence.get("identity_conflict") == "opposing_roles"),
        "different_year": bool(
            ma.get("year") is not None
            and mb.get("year") is not None
            and ma.get("year") != mb.get("year")
        ),
        "single_token_name": "single_token_name" in barrier_reasons or tok_n == 1 or len(nb.split()) == 1,
        "ucid_only_corroboration": bool(evidence.get("ucid_only_corroboration")),
        "embedding_similarity_high": sim >= 0.92,
        # Discriminating composites
        "identical_name_ge2_same_court": identical and tok_n >= 2 and same_court,
        "identical_name_ge2_same_ucid": identical and tok_n >= 2 and same_ucid,
    }


def collect_cited_claims(result: dict) -> list[str]:
    """Normalize cited_evidence + factual signals into claim ids."""
    raw: list[Any] = []
    ce = result.get("cited_evidence")
    if isinstance(ce, list):
        raw.extend(ce)
    elif isinstance(ce, str) and ce.strip():
        raw.append(ce)
    for s in result.get("signals") or []:
        raw.append(s)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        claim = _norm_claim(item)
        if not claim or claim in seen:
            continue
        # Skip non-factual tags
        if claim in {
            "name_similar",
            "name_mismatch",  # kept — factual
            "insufficient_evidence",
            "low_confidence_abstain",
            "barrier_overruled_by_llm",
            "low_information_name",
            "barrier_below_llm_min",
            "ucid_only_weak_context",
            "large_cluster_verify",
            "hallucination_detected",
            "citation_failed",
            "no_discriminating_fact",
            "same_tokens",
        }:
            if claim not in {"name_mismatch"}:
                continue
        seen.add(claim)
        out.append(claim)
    return out


def verify_citations(
    result: dict,
    ma: dict,
    mb: dict,
    evidence: dict,
    *,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Return verification report. failed_claims non-empty ⇒ hallucination."""
    t3 = (cfg or {}).get("tier3") or {}
    cv = t3.get("citation_verification") or {}
    truth = truth_table(ma, mb, evidence)
    cited = collect_cited_claims(result)
    # Only verify claims we know how to check
    failed: list[str] = []
    verified: list[str] = []
    unknown: list[str] = []
    for claim in cited:
        if claim not in truth:
            unknown.append(claim)
            continue
        if truth[claim]:
            verified.append(claim)
        else:
            failed.append(claim)
    # Claiming same_pacer_id / identical name via rationale heuristics already in signals
    return {
        "cited": cited,
        "verified": verified,
        "failed": failed,
        "unknown": unknown,
        "truth": truth,
        "hallucination": bool(failed),
        "enabled": bool(cv.get("enabled", True)),
    }


def discriminating_facts_present(truth: dict[str, bool], cfg: dict | None = None) -> list[str]:
    """Which configured discriminating facts are actually true."""
    t3 = (cfg or {}).get("tier3") or {}
    bar = t3.get("match_requires_discriminating_fact") or {}
    facts = list(bar.get("facts") or [
        "same_pacer_id",
        "same_domain",
        "same_phone",
        "same_address",
        "shared_fjc_nid",
        "identical_name_ge2_same_court",
    ])
    return [f for f in facts if truth.get(f)]


def apply_citation_and_match_rails(
    decision: str,
    rationale: str,
    signals: list[str],
    result: dict,
    ma: dict,
    mb: dict,
    evidence: dict,
    cfg: dict,
) -> tuple[str, str, list[str], dict[str, Any]]:
    """Apply citation verification then asymmetric MATCH bar.

    Returns (decision, rationale, signals, audit_meta).
    """
    t3 = cfg.get("tier3") or {}
    cv = t3.get("citation_verification") or {}
    bar = t3.get("match_requires_discriminating_fact") or {}
    meta: dict[str, Any] = {}

    report = verify_citations(result, ma, mb, evidence, cfg=cfg)
    meta["citation_verification"] = {
        "cited": report["cited"],
        "verified": report["verified"],
        "failed": report["failed"],
        "unknown": report["unknown"],
        "hallucination": report["hallucination"],
    }

    if cv.get("enabled", True) and report["hallucination"]:
        decision = "UNCERTAIN"
        signals = sorted(set(signals + ["hallucination_detected", "citation_failed", "insufficient_evidence"]))
        rationale = (
            "[abstain citation-verify: false claim(s) "
            + ",".join(report["failed"])
            + "] "
            + rationale
        )
        meta["citation_verification"]["forced_uncertain"] = True
    elif (
        cv.get("enabled", True)
        and cv.get("require_cited_evidence", True)
        and decision in {"MATCH", "NO_MATCH"}
        and not report["cited"]
    ):
        decision = "UNCERTAIN"
        signals = sorted(set(signals + ["citation_failed", "insufficient_evidence"]))
        rationale = "[abstain citation-verify: empty cited_evidence] " + rationale
        meta["citation_verification"]["forced_uncertain"] = True
        meta["citation_verification"]["missing_citations"] = True

    # Asymmetric MATCH bar — only when still MATCH
    if decision == "MATCH" and bar.get("enabled", False):
        truth = report["truth"]
        present = discriminating_facts_present(truth, cfg)
        meta["discriminating_facts_present"] = present
        require_cited = bool(bar.get("require_cited", True))
        verified_set = set(report["verified"])
        ok_facts: list[str] = []
        for f in present:
            if not require_cited or f in verified_set:
                ok_facts.append(f)
                continue
            # Composite: allow citing component claims
            if f == "identical_name_ge2_same_court":
                name_ok = bool(
                    verified_set
                    & {
                        "identical_normalized_name",
                        "identical_normalized_name_ge2",
                        "identical_name_ge2_same_court",
                    }
                )
                court_ok = "same_court" in verified_set or "identical_name_ge2_same_court" in verified_set
                if name_ok and court_ok:
                    ok_facts.append(f)

        meta["discriminating_facts_accepted"] = ok_facts
        if not ok_facts:
            decision = "UNCERTAIN"
            signals = sorted(set(signals + ["no_discriminating_fact", "insufficient_evidence"]))
            rationale = (
                "[abstain asymmetric-MATCH-bar: no verified discriminating fact] " + rationale
            )
            meta["match_bar_forced_uncertain"] = True

    return decision, rationale, signals, meta
