"""PoC-only party evidence assist (orchestration layer).

Does NOT modify engine/tiers.py, YAML matching rules, or citation rails.

After the normal cascade, near-duplicate party mentions that share real
same-case evidence (resolved judge/firm, MDL, co-defendants) may be sent to
**real Tier3 (qwen)** with those fields added to the evidence bundle.

A UF merge happens ONLY when the existing citation-verification + asymmetric
MATCH bar returns MATCH. There is no deterministic / model-free merge path.
"""

from __future__ import annotations

import json
import time
from typing import Any

from rapidfuzz import fuzz

from engine.name_compat import names_compatible
from engine.provenance import DecisionJournal
from engine.tiers import (
    UnionFind,
    build_tier3_evidence,
    call_ollama_json,
    identity_pair_conflict,
    load_prompt,
    ollama_endpoint,
    prompt_sha256,
    normalize_tier3_signals,
)
from engine.tier3_citation import apply_citation_and_match_rails


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def shared_case_evidence(ma: dict, mb: dict) -> dict[str, Any]:
    ja = set(ma.get("poc_case_judges") or [])
    jb = set(mb.get("poc_case_judges") or [])
    fa = set(ma.get("poc_case_firms") or [])
    fb = set(mb.get("poc_case_firms") or [])
    shared_judges = sorted(ja & jb)
    shared_firms = sorted(fa & fb)
    mdl_a, mdl_b = ma.get("poc_mdl_code"), mb.get("poc_mdl_code")
    same_mdl = bool(
        mdl_a is not None
        and mdl_b is not None
        and str(mdl_a) == str(mdl_b)
        and str(mdl_a) not in {"", "False", "None"}
    )
    co = sorted(set(ma.get("co_mentions") or []) & set(mb.get("co_mentions") or []))
    return {
        "shared_case_judges": shared_judges,
        "shared_case_firms": shared_firms,
        "same_mdl": same_mdl,
        "mdl_code": str(mdl_a) if same_mdl else None,
        "co_mentions_overlap": co[:25],
        "co_mentions_overlap_count": len(co),
        "has_shared_judge": bool(shared_judges),
        "has_shared_firm": bool(shared_firms),
    }


def evidence_strong_enough(ev: dict[str, Any]) -> bool:
    """Require real corroboration beyond name similarity alone."""
    if ev.get("has_shared_firm") and (ev.get("has_shared_judge") or ev.get("same_mdl")):
        return True
    if ev.get("has_shared_firm") and (ev.get("co_mentions_overlap_count") or 0) >= 1:
        return True
    if ev.get("has_shared_judge") and ev.get("same_mdl"):
        return True
    return False


# Hard conflicts never go to Tier3 via this path. Soft cascade conflicts such as
# generic_word_overlap may still be *candidates* for a Tier3 look with extra
# case evidence — but merge requires MATCH after citation/MATCH bar.
_HARD_IDENTITY_CONFLICTS = frozenset(
    {
        "opposing_roles",
        "division_subsidiary",
        "government_jurisdiction",
        "placeholder",
        "person_vs_org",
        "person_title_prefix",
        "fund_plan_type",
        "generational_suffix",
        "legal_entity_form",
        "parent_subunit",
        "corporate_shared_prefix",
    }
)


def find_poc_evidence_candidates(
    mentions: list[dict],
    uf: UnionFind,
    cfg: dict,
    *,
    min_name_ratio: int = 90,
) -> list[tuple[dict, dict, float, dict]]:
    """Unmerged near-dup pairs with shared case evidence → Tier3 candidates only."""
    by_block: dict[str, list[dict]] = {}
    for m in mentions:
        nn = (m.get("normalized_name") or "").strip()
        toks = nn.split()
        key2 = " ".join(toks[:2]) if len(toks) >= 2 else nn
        court = m.get("court") or ""
        by_block.setdefault(f"{court}|{key2}", []).append(m)

    out: list[tuple[dict, dict, float, dict]] = []
    seen: set[tuple[str, str]] = set()
    for group in by_block.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ma, mb = group[i], group[j]
                ida, idb = ma["mention_id"], mb["mention_id"]
                if uf.find(ida) == uf.find(idb):
                    continue
                pair = tuple(sorted([ida, idb]))
                if pair in seen:
                    continue
                ok, _reason = names_compatible(ma, mb, cfg=cfg)
                if not ok:
                    continue
                conflict = identity_pair_conflict(ma, mb, cfg)
                if conflict and conflict[0] in _HARD_IDENTITY_CONFLICTS:
                    continue
                ratio = float(
                    fuzz.ratio(ma.get("normalized_name") or "", mb.get("normalized_name") or "")
                )
                if ratio < min_name_ratio:
                    continue
                ev = shared_case_evidence(ma, mb)
                if not evidence_strong_enough(ev):
                    continue
                seen.add(pair)
                out.append((ma, mb, ratio, ev))
    out.sort(key=lambda x: -x[2])
    return out


def enrich_tier3_evidence(base: dict, poc_ev: dict, *, name_ratio: float) -> dict:
    ev = dict(base)
    ev["poc_shared_case_judges"] = poc_ev.get("shared_case_judges") or []
    ev["poc_shared_case_firms"] = poc_ev.get("shared_case_firms") or []
    ev["poc_same_mdl"] = bool(poc_ev.get("same_mdl"))
    ev["poc_mdl_code"] = poc_ev.get("mdl_code")
    ev["name_fuzz_ratio"] = name_ratio
    ev["poc_evidence_note"] = (
        "PoC fields: same-case resolved judge/firm and MDL are extra clues for Tier3. "
        "They are not identity by themselves. Citation verification and MATCH bar apply unchanged."
    )
    return ev


def _tier3_one_pair(
    ma: dict,
    mb: dict,
    ratio: float,
    poc_ev: dict,
    cfg: dict,
) -> dict[str, Any]:
    t3 = cfg.get("tier3") or {}
    model = t3.get("model", "qwen2.5:7b")
    endpoint = ollama_endpoint(cfg)
    prompt_tmpl = load_prompt(cfg)
    prompt_hash = prompt_sha256(prompt_tmpl)

    base = build_tier3_evidence(ma, mb, barrier=False, reasons=[], cfg=cfg)
    evidence = enrich_tier3_evidence(base, poc_ev, name_ratio=ratio)
    sim = ratio / 100.0
    evidence["embedding_similarity"] = sim

    prompt = (
        prompt_tmpl.replace("{{mention_a_profile}}", ma.get("profile") or ma["normalized_name"])
        .replace("{{mention_b_profile}}", mb.get("profile") or mb["normalized_name"])
        .replace("{{block_key}}", "poc_shared_case_evidence")
        .replace("{{embedding_similarity}}", f"{sim:.4f}")
        .replace("{{evidence_json}}", json.dumps(evidence, ensure_ascii=False, default=str))
    )

    raw = call_ollama_json(model=model, prompt=prompt, endpoint=endpoint)
    if isinstance(raw.get("decision"), str):
        raw["decision"] = raw["decision"].upper().replace(" ", "_")
    decision = (raw.get("decision") or "UNCERTAIN").upper()
    rationale = raw.get("rationale") or ""
    signals = normalize_tier3_signals(raw.get("signals"))
    final, rat2, sigs2, meta = apply_citation_and_match_rails(
        decision, rationale, signals, raw, ma, mb, evidence, cfg
    )
    return {
        "normalized_a": ma.get("normalized_name"),
        "normalized_b": mb.get("normalized_name"),
        "mention_id_a": ma.get("mention_id"),
        "mention_id_b": mb.get("mention_id"),
        "name_ratio": ratio,
        "model_decision": decision,
        "final_decision_after_citation_bar": final,
        "confidence": raw.get("confidence"),
        "cited_evidence": raw.get("cited_evidence"),
        "rationale": rat2,
        "signals": sigs2,
        "citation_meta": meta,
        "poc_evidence": poc_ev,
        "prompt_hash": prompt_hash,
        "model": model,
        "merged": False,
    }


def adjudicate_poc_evidence_via_tier3(
    mentions: list[dict],
    uf: UnionFind,
    cfg: dict,
    journal: DecisionJournal | None = None,
    *,
    max_pairs: int = 25,
) -> dict[str, Any]:
    """Send PoC evidence candidates to Tier3; merge only on citation-bar MATCH."""
    cands = find_poc_evidence_candidates(mentions, uf, cfg)
    rows: list[dict] = []
    merges = 0
    t3 = cfg.get("tier3") or {}
    if not t3.get("enabled", True):
        from engine.tiers import record_uncertain

        todo = [c for c in cands[:max_pairs] if uf.find(c[0]["mention_id"]) != uf.find(c[1]["mention_id"])]
        by = {m["mention_id"]: m for c in todo for m in (c[0], c[1])}
        record_uncertain(
            [(c[0]["mention_id"], c[1]["mention_id"], c[2]) for c in todo], by, cfg, "party_evidence", score_kind="name_ratio"
        )
        return {
            "candidates": len(cands),
            "tier3_calls": 0,
            "merges": 0,
            "rows": [],
            "note": "tier3 disabled",
        }

    for ma, mb, ratio, poc_ev in cands[:max_pairs]:
        if uf.find(ma["mention_id"]) == uf.find(mb["mention_id"]):
            continue
        try:
            row = _tier3_one_pair(ma, mb, ratio, poc_ev, cfg)
        except Exception as e:
            row = {
                "normalized_a": ma.get("normalized_name"),
                "normalized_b": mb.get("normalized_name"),
                "mention_id_a": ma.get("mention_id"),
                "mention_id_b": mb.get("mention_id"),
                "name_ratio": ratio,
                "error": str(e),
                "poc_evidence": poc_ev,
                "final_decision_after_citation_bar": "ERROR",
                "merged": False,
            }
            rows.append(row)
            continue

        final = (row.get("final_decision_after_citation_bar") or "").upper()
        if final == "MATCH" and not uf.union(ma["mention_id"], mb["mention_id"]):
            final = "NO_MATCH_COPARTY_BARRIER"
        if final == "MATCH":
            merges += 1
            row["merged"] = True
            row["method"] = "poc.tier3_enriched_match"
            row["decision"] = "MERGE_AFTER_TIER3_CITATION"
        else:
            row["merged"] = False
            row["method"] = "poc.tier3_enriched_no_merge"
            row["decision"] = f"NO_MERGE_{final or 'UNKNOWN'}"

        row["timestamp"] = _now()
        rows.append(row)
        if journal is not None:
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "entity_type": "party",
                    "mention_id_a": row.get("mention_id_a"),
                    "mention_id_b": row.get("mention_id_b"),
                    "decision": row.get("decision"),
                    "method": row.get("method"),
                    "confidence": row.get("confidence"),
                    "rationale": row.get("rationale"),
                    "signals": row.get("signals"),
                    "evidence": {
                        "poc": poc_ev,
                        "model_decision": row.get("model_decision"),
                        "final_after_citation_bar": final,
                        "cited_evidence": row.get("cited_evidence"),
                        "citation_meta": row.get("citation_meta"),
                        "name_ratio": ratio,
                    },
                    "timestamp": row["timestamp"],
                    "config_version": cfg.get("version"),
                }
            )

    return {
        "candidates": len(cands),
        "tier3_calls": len(rows),
        "merges": merges,
        "rows": rows,
    }


def rebuild_components(uf: UnionFind, mention_ids: list[str]) -> dict[str, list[str]]:
    comps: dict[str, list[str]] = {}
    for mid in mention_ids:
        root = uf.find(mid)
        comps.setdefault(root, []).append(mid)
    return comps
