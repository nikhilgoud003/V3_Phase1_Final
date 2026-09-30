#!/usr/bin/env python3
"""Targeted parties Tier3 replay: re-call only cache-miss pairs (same order as tier3_adjudicate)."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jsonschema
from engine.config_loader import load_config, resolve_path
from engine.name_compat import names_compatible
from engine.provenance import DecisionJournal
from engine.tiers import (
    _cache_key,
    apply_tier2_auto_merges,
    build_ollama_faiss_pack,
    build_profile_blocks,
    build_tier3_evidence,
    call_ollama_json,
    institutional_office_geo_conflict,
    invert_blocks,
    load_common_surnames,
    load_output_schema,
    load_prompt,
    ollama_endpoint,
    pair_allowed,
    run_cascade,
    same_block,
    tier2_candidates,
)


def mdl_map(json_dir: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for fp in json_dir.glob("*.json"):
        case = json.loads(fp.read_text(encoding="utf-8"))
        ucid = case.get("ucid")
        if not ucid:
            continue
        is_mdl = bool(case.get("is_mdl")) or bool(case.get("mdl_code"))
        cn = (case.get("case_name") or "").lower()
        if not is_mdl and any(k in cn for k in ("mdl", "mass tort", "multidistrict")):
            is_mdl = True
        out[ucid] = is_mdl
    return out


def classify_error(msg: str, raw) -> str:
    m = (msg or "").lower()
    if "timed out" in m or "timeout" in m:
        return "timeout"
    if "urlerror" in m or "connection" in m or "refused" in m or "http error" in m:
        return "ollama_connection"
    if "jsondecodeerror" in m or "expecting" in m or "extra data" in m:
        return "malformed_json_response"
    if "validationerror" in m or "is not one of" in m or "is not of type" in m:
        return "schema_validation_failure"
    if raw is None:
        return "llm_call_failed_before_parse"
    if isinstance(raw, dict):
        dec = str(raw.get("decision", "")).upper()
        if dec and dec not in {"MATCH", "NO_MATCH", "UNCERTAIN"}:
            return "invalid_decision_enum"
    return "other"


def main() -> int:
    reports = ROOT / "data/runs/parties_pilot/reports"
    reports.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/parties.yaml")
    mentions = [
        json.loads(line)
        for line in resolve_path(cfg, cfg["io"]["mentions_out"]).open(encoding="utf-8")
        if line.strip()
    ]
    mdl_by_ucid = mdl_map(ROOT / "data/json/pilot_1000")

    t0 = time.time()
    journal = DecisionJournal(resolve_path(cfg, cfg["io"]["decisions_out"]))
    res = run_cascade(mentions, cfg, enable_tier3=False)
    by_id = res["by_id"]
    uf = res["uf"]
    mention_blocks = build_profile_blocks(mentions, cfg)
    blocks = invert_blocks(mention_blocks)
    common_path = resolve_path(
        cfg,
        (cfg.get("tier2") or {})
        .get("common_surnames", {})
        .get("list_path", "data/external/common_surnames.txt"),
    )
    common_surnames = load_common_surnames(common_path)
    embed_pack = build_ollama_faiss_pack(mentions, cfg)
    pairs = tier2_candidates(mentions, mention_blocks, blocks, uf, cfg, embed_pack=embed_pack)
    ambiguous = apply_tier2_auto_merges(pairs, by_id, uf, cfg, journal, common_surnames)["ambiguous"]

    cache = {}
    cache_path = resolve_path(cfg, cfg["tier3"]["decision_cache"]["path"])
    for line in cache_path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            cache[r["cache_key"]] = r

    model = cfg["tier3"]["model"]
    endpoint = ollama_endpoint(cfg)
    prompt_tmpl = load_prompt(cfg)
    schema = load_output_schema(cfg)
    require_block = (cfg.get("tier3") or {}).get("routing", {}).get("require_same_block", True)
    amb_low = float(cfg["tier2"]["search"]["ambiguous_low"])
    amb_high = float(cfg["tier2"]["search"]["ambiguous_high"])
    mid = (amb_low + amb_high) / 2
    ranked = sorted(ambiguous, key=lambda x: abs(x[2] - mid))

    errors = []
    attempt_records = []
    fresh_attempts = 0
    successes = 0
    max_fresh = 114  # same window as failed run before abort

    for a, b, sim, barrier, reasons in ranked:
        if fresh_attempts >= max_fresh:
            break
        if uf.find(a) == uf.find(b):
            continue
        if require_block and not same_block(a, b, mention_blocks):
            continue
        ma, mb = by_id[a], by_id[b]
        if not pair_allowed(ma, mb):
            continue
        if not names_compatible(ma, mb, cfg=cfg)[0]:
            continue
        if institutional_office_geo_conflict(ma, mb, cfg)[0]:
            continue
        ck = _cache_key(ma, mb)
        if ck in cache:
            continue

        fresh_attempts += 1
        evidence = build_tier3_evidence(ma, mb, barrier=barrier, reasons=reasons, cfg=cfg)
        evidence["embedding_similarity"] = sim
        block_keys = sorted(set(mention_blocks.get(a, [])) & set(mention_blocks.get(b, [])))
        prompt = (
            prompt_tmpl.replace("{{mention_a_profile}}", ma.get("profile") or ma["normalized_name"])
            .replace("{{mention_b_profile}}", mb.get("profile") or mb["normalized_name"])
            .replace("{{block_key}}", ", ".join(block_keys))
            .replace("{{embedding_similarity}}", f"{sim:.4f}")
            .replace("{{evidence_json}}", json.dumps(evidence, ensure_ascii=False))
        )
        raw = None
        rec = {
            "mention_id_a": a,
            "mention_id_b": b,
            "cache_key": ck,
            "name_a": ma.get("normalized_name"),
            "name_b": mb.get("normalized_name"),
            "ucid_a": ma.get("ucid"),
            "ucid_b": mb.get("ucid"),
            "mdl_a": mdl_by_ucid.get(ma.get("ucid")),
            "mdl_b": mdl_by_ucid.get(mb.get("ucid")),
            "sim": sim,
            "prompt_chars": len(prompt),
            "profile_a_chars": len(ma.get("profile") or ""),
            "profile_b_chars": len(mb.get("profile") or ""),
        }
        try:
            raw = call_ollama_json(model, prompt, endpoint)
            if "confidence" in raw:
                raw["confidence"] = int(round(float(raw["confidence"])))
            if "decision" in raw:
                d = str(raw["decision"]).upper().replace(" ", "_")
                if d not in {"MATCH", "NO_MATCH", "UNCERTAIN"}:
                    if d in {"YES", "SAME", "MERGE"}:
                        d = "MATCH"
                    elif d in {"NO", "DIFFERENT", "DISTINCT"}:
                        d = "NO_MATCH"
                    else:
                        d = "UNCERTAIN"
                raw["decision"] = d
            jsonschema.validate(raw, schema)
            successes += 1
            rec["status"] = "ok"
            rec["decision"] = raw.get("decision")
        except Exception as e:
            rec["status"] = "error"
            rec["error_type"] = type(e).__name__
            rec["error_message"] = str(e)
            rec["error_category"] = classify_error(str(e), raw)
            rec["raw_response"] = raw
            errors.append(rec)
        attempt_records.append(rec)
        if fresh_attempts % 10 == 0:
            print(
                f"  progress fresh={fresh_attempts}/{max_fresh} ok={successes} err={len(errors)}",
                flush=True,
            )

    mdl_err = sum(1 for e in errors if e.get("mdl_a") or e.get("mdl_b"))
    mdl_att = sum(1 for r in attempt_records if r.get("mdl_a") or r.get("mdl_b"))
    non_mdl_att = fresh_attempts - mdl_att

    samples = []
    for e in errors[:3]:
        ma = by_id[e["mention_id_a"]]
        mb = by_id[e["mention_id_b"]]
        samples.append(
            {
                "pair_summary": {k: e[k] for k in e if k not in {"raw_response"}},
                "profile_a": ma.get("profile") or "",
                "profile_b": mb.get("profile") or "",
                "evidence_bundle": build_tier3_evidence(
                    ma, mb, barrier=True, reasons=["diagnosis"], cfg=cfg
                ),
            }
        )

    report = {
        "replay_method": (
            "T0-T2 cascade + tier3 routing replay on cache-miss pairs only; "
            "stopped after 114 fresh LLM calls (same window as failed run)"
        ),
        "elapsed_sec": round(time.time() - t0, 2),
        "ambiguous_pairs_total": len(ambiguous),
        "cache_hits_skipped": len(cache),
        "fresh_llm_attempts": fresh_attempts,
        "successes": successes,
        "error_count": len(errors),
        "error_rate": round(len(errors) / max(1, fresh_attempts), 4),
        "error_category_counts": dict(Counter(e["error_category"] for e in errors)),
        "error_records": errors,
        "mdl_clustering": {
            "mdl_attempts": mdl_att,
            "non_mdl_attempts": non_mdl_att,
            "mdl_error_count": mdl_err,
            "non_mdl_error_count": len(errors) - mdl_err,
            "mdl_error_rate": round(mdl_err / max(1, mdl_att), 4),
            "non_mdl_error_rate": round((len(errors) - mdl_err) / max(1, non_mdl_att), 4),
            "mdl_attempt_fraction": round(mdl_att / max(1, fresh_attempts), 4),
            "mention_mdl_fraction_pilot": 0.663,
        },
        "failing_pair_evidence_samples": samples,
        "prompt_config": {
            "prompt_path": cfg["tier3"]["prompt_path"],
            "schema_path": cfg["tier3"]["json_schema_path"],
            "note": "Parties uses dedicated prompt; schema shared with judges/firms",
        },
    }
    out_path = reports / "tier3_error_diagnosis.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} errors={len(errors)}/{fresh_attempts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
