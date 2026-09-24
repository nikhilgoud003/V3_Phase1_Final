#!/usr/bin/env python3
"""Config-driven LLM name-validity micro-pass (cached, batched by unique string).

For unique mention strings that survive deterministic cleaning but are not
FJC-linked and not already a known clean name in the pool, ask the LLM to
extract a clean person name or return INVALID.

Provenance method: ``llm_name_validation``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from engine.config_loader import resolve_path
from engine.name_validity import (
    classify_name_validity,
    preclean_header_junk,
    strip_trailing_lexemes,
    strip_trailing_procedural,
    DEFAULT_PROCEDURAL_STOPWORDS,
)


PROMPT_DEFAULT = """You are validating a PACER docket span that may be a US federal judge name.
Return JSON only: {"decision":"VALID"|"INVALID","clean_name":"<person name or empty>","confidence":0-100}
Rules:
- If the span is a real person name (possibly with titles or minor glue), set VALID and clean_name to the person name only.
- If the span is procedural text, possessive boilerplate (X's rules), plea/hearing fragments, or not a person, set INVALID and clean_name "".
- Preserve middle names that are English words when they are part of a real name (e.g. Denise Page Hood).
- Strip courtroom codes after .( or )( (e.g. "Garcia.(Cdomadi" → "Guillermo R. Garcia").
Span: {span}
"""


def _cache_key(span: str, model: str, prompt_hash: str) -> str:
    h = hashlib.sha256(f"{model}|{prompt_hash}|{span.lower().strip()}".encode()).hexdigest()
    return h[:32]


def _load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        k = rec.get("cache_key")
        if k:
            cache[k] = rec
    return cache


def _append_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _needs_llm(
    cleaned: str,
    *,
    fjc_full_names: set[str],
    known_clean: set[str],
) -> bool:
    """True when span is not an FJC full name and not already a known clean name."""
    if not cleaned:
        return False
    if cleaned in fjc_full_names:
        return False
    if cleaned in known_clean:
        return False
    return True


def apply_llm_name_validity(
    mentions: list[dict],
    *,
    cfg: dict,
    fjc_surnames: set[str],
    fjc_full_names: set[str],
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Return (kept, quarantined_extra, stats). Mutates kept mention names when cleaned."""
    nv = cfg.get("name_validity") or {}
    llm_cfg = nv.get("llm_validation") or {}
    if not llm_cfg.get("enabled", False):
        return mentions, [], {"enabled": False, "llm_calls": 0}

    from engine.tiers import call_ollama_json, ollama_endpoint

    model = llm_cfg.get("model") or (cfg.get("tier3") or {}).get("model") or "qwen2.5:7b"
    endpoint = llm_cfg.get("endpoint") or ollama_endpoint(cfg)
    prompt_path = llm_cfg.get("prompt_path")
    if prompt_path:
        prompt_tmpl = resolve_path(cfg, prompt_path).read_text(encoding="utf-8")
    else:
        prompt_tmpl = PROMPT_DEFAULT
    prompt_hash = hashlib.sha256(prompt_tmpl.encode()).hexdigest()[:16]
    cache_path = resolve_path(
        cfg, llm_cfg.get("cache_path") or "data/decisions/llm_name_validity_cache.jsonl"
    )
    journal_path = resolve_path(
        cfg, llm_cfg.get("journal_path") or "data/decisions/llm_name_validity_journal.jsonl"
    )
    cache = _load_cache(cache_path)
    max_calls = int(llm_cfg.get("max_calls") or 500)

    stopwords = set(DEFAULT_PROCEDURAL_STOPWORDS)
    stopwords.update(w.lower() for w in (nv.get("procedural_stopwords") or []))

    # Known-clean anchors: FJC full + header/party mentions only (not every line_entry)
    known_clean = set(fjc_full_names)
    for m in mentions:
        if (m.get("docket_source") or "") in {"case_header", "case_parties"}:
            nn = (m.get("normalized_name") or "").strip().lower()
            if nn and " " in nn:
                known_clean.add(nn)

    # Unique spans needing LLM
    candidates: dict[str, list[dict]] = {}
    for m in mentions:
        nn = (m.get("normalized_name") or "").strip().lower()
        raw = (m.get("raw_name") or nn).strip()
        if not _needs_llm(nn, fjc_full_names=fjc_full_names, known_clean=known_clean):
            continue
        # Skip clean FJC-surname spans with no glue markers in raw
        toks = nn.split()
        if (
            toks
            and toks[-1] in fjc_surnames
            and len(toks) >= 2
            and not any(ch in raw for ch in ("(", ")", "[", "'"))
            and not re.search(r"(?i)\b(plea|modified|trial|rules|dtd|see attached)\b", raw)
        ):
            continue
        key = raw.lower()
        candidates.setdefault(key, []).append(m)

    stats: dict[str, Any] = {
        "enabled": True,
        "unique_candidates": len(candidates),
        "llm_calls": 0,
        "cache_hits": 0,
        "valid": 0,
        "invalid": 0,
        "renamed": 0,
        "errors": 0,
        "model": model,
        "method": "llm_name_validation",
    }

    kept = list(mentions)
    by_id = {m.get("mention_id"): m for m in kept}
    quarantine_extra: list[dict] = []
    calls = 0

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    jfh = journal_path.open("a", encoding="utf-8")

    try:
        for span_key, group in candidates.items():
            sample_raw = (group[0].get("raw_name") or group[0].get("normalized_name") or "").strip()
            ck = _cache_key(sample_raw, model, prompt_hash)
            if ck in cache:
                result = cache[ck]
                stats["cache_hits"] += 1
            else:
                if calls >= max_calls:
                    break
                prompt = prompt_tmpl.replace("{span}", sample_raw)
                try:
                    result = call_ollama_json(model, prompt, endpoint)
                    calls += 1
                    stats["llm_calls"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    result = {"decision": "INVALID", "clean_name": "", "confidence": 0, "error": str(e)}
                    calls += 1
                    stats["llm_calls"] += 1
                rec = {
                    "cache_key": ck,
                    "span": sample_raw,
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "method": "llm_name_validation",
                    **result,
                }
                _append_cache(cache_path, rec)
                cache[ck] = rec
                result = rec

            decision = str(result.get("decision") or "").upper()
            clean = (result.get("clean_name") or "").strip()
            conf = int(result.get("confidence") or 0)

            jfh.write(
                json.dumps(
                    {
                        "method": "llm_name_validation",
                        "span": sample_raw,
                        "decision": decision,
                        "clean_name": clean,
                        "confidence": conf,
                        "mention_ids": [m.get("mention_id") for m in group],
                        "model": model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if decision != "VALID" or not clean:
                stats["invalid"] += 1
                for m in group:
                    q = dict(m)
                    q["name_validity"] = "quarantine"
                    q["name_validity_reasons"] = ["llm_name_validation_invalid"]
                    q["name_validity_method"] = "llm_name_validation"
                    q["quarantine_reason"] = ["llm_name_validation_invalid"]
                    quarantine_extra.append(q)
                    mid = m.get("mention_id")
                    if mid in by_id:
                        del by_id[mid]
                continue

            # VALID — normalize clean_name and apply
            from engine.normalize import normalize_name, surname, tokens

            honorifics = (cfg.get("normalization") or {}).get("strip_honorifics") or []
            strip_chars = (cfg.get("normalization") or {}).get("strip_chars") or ".,;:\"()[]{}"
            cleaned = normalize_name(clean, honorifics=honorifics, strip_chars=strip_chars)
            cleaned = strip_trailing_procedural(cleaned, stopwords)
            cleaned = strip_trailing_lexemes(cleaned)
            ok, reasons = classify_name_validity(
                cleaned,
                fjc_surnames=fjc_surnames,
                fjc_full_names=fjc_full_names,
            )
            if not ok or not cleaned:
                stats["invalid"] += 1
                for m in group:
                    q = dict(m)
                    q["name_validity"] = "quarantine"
                    q["name_validity_reasons"] = ["llm_clean_failed_gate"] + list(reasons)
                    q["name_validity_method"] = "llm_name_validation"
                    quarantine_extra.append(q)
                    mid = m.get("mention_id")
                    if mid in by_id:
                        del by_id[mid]
                continue

            stats["valid"] += 1
            for m in group:
                mid = m.get("mention_id")
                target = by_id.get(mid)
                if not target:
                    continue
                if cleaned != (target.get("normalized_name") or ""):
                    stats["renamed"] += 1
                    target["name_validity_stripped_from"] = target.get("normalized_name")
                    target["normalized_name"] = cleaned
                    target["surname"] = surname(cleaned)
                    target["token_count"] = len(tokens(cleaned))
                    target["presentable_name"] = " ".join(w.capitalize() for w in cleaned.split())
                target["name_validity"] = "kept"
                target["name_validity_method"] = "llm_name_validation"
                target["name_validity_reasons"] = ["llm_name_validation_valid"] + list(reasons)
                known_clean.add(cleaned)
    finally:
        jfh.close()

    kept_out = [by_id[m.get("mention_id")] for m in mentions if m.get("mention_id") in by_id]
    return kept_out, quarantine_extra, stats
