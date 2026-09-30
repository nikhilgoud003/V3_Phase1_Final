#!/usr/bin/env python3
"""Point-6 acceptance metrics on phase3 audited pairs.

Computes (after citation-verify + asymmetric MATCH bar):
  - hallucination_rate: share of LLM outputs with ≥1 false citation
  - uncertain_rate: share ending UNCERTAIN after rails
  - false_match_rate: share of structural-gold DIFFERENT pairs that still MATCH

Structural gold (audit packets lack filled human labels):
  DIFFERENT if name_gate fails, identity_conflict set, opposing_roles,
  USA/alias cross-UCID skip, or known false-merge mention ids.
  SAME if identical ≥2-token name + same court + no conflicts.
  else UNKNOWN (excluded from false_match denominator).

Does NOT enable repeated sampling; reports whether false_match_rate is already
low enough (default threshold 0.05) so sampling can be skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config, ollama_endpoint, resolve_path
from engine.name_compat import names_compatible
from engine.tier3_citation import apply_citation_and_match_rails
from engine.tiers import (
    build_tier3_evidence,
    call_ollama_json,
    information_barrier,
    load_common_surnames,
    load_output_schema,
    load_prompt,
    normalize_tier3_signals,
    pair_in_skipped_alias_group_cross_ucid,
)
import jsonschema

# Round-4 confirmed false merges (and same-shape Wells/BoA)
KNOWN_DIFFERENT = {
    frozenset({"mnt_c47504daa4c45c4a", "mnt_d9809aefa69f8c92"}),  # MBNA / BoA
    frozenset({"mnt_1cb15f5f99cf2bbf", "mnt_d9809aefa69f8c92"}),  # Wells / BoA
    frozenset({"mnt_14ab5a9217af29e0", "mnt_04fdf7d22a09aa8e"}),  # J&J / Consumer
}


def load_mentions(cfg: dict) -> dict[str, dict]:
    candidates = [
        resolve_path(cfg, cfg["io"]["mentions_out"]),
        ROOT / "data/runs/parties_pilot/mentions/parties_mentions.jsonl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("parties mentions jsonl not found")
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                m = json.loads(line)
                out[m["mention_id"]] = m
    return out


def audit_pairs() -> list[dict]:
    seen: set[frozenset[str]] = set()
    rows: list[dict] = []
    for rnd in range(1, 6):
        p = ROOT / f"data/runs/parties_pilot/reports/phase3_audit_packet_round{rnd}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for x in data.get("index") or []:
            a, b = x.get("mention_id_a"), x.get("mention_id_b")
            if not a or not b:
                continue
            key = frozenset({a, b})
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "method": x.get("method"),
                    "stratum": x.get("stratum"),
                    "decision": x.get("decision"),
                    "round": rnd,
                }
            )
    return rows


def structural_gold(ma: dict, mb: dict, cfg: dict) -> str:
    key = frozenset({ma["mention_id"], mb["mention_id"]})
    if key in KNOWN_DIFFERENT:
        return "DIFFERENT"
    if pair_in_skipped_alias_group_cross_ucid(ma, mb, cfg):
        return "DIFFERENT"
    from engine.tiers import identity_pair_conflict

    conflict = identity_pair_conflict(ma, mb, cfg)
    if conflict:
        return "DIFFERENT"
    ok, _ = names_compatible(ma, mb, cfg=cfg)
    if not ok:
        return "DIFFERENT"
    na = (ma.get("normalized_name") or "").strip().lower()
    nb = (mb.get("normalized_name") or "").strip().lower()
    if na and na == nb and len(na.split()) >= 2 and ma.get("court") == mb.get("court"):
        return "SAME"
    # Subsidiary / division token extension of shared prefix
    if na and nb and (na.startswith(nb + " ") or nb.startswith(na + " ")):
        return "DIFFERENT"
    return "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Cap pairs (0=all)")
    ap.add_argument("--tier3-only", action="store_true", help="Only historical tier3 stratum/method")
    ap.add_argument("--false-match-threshold", type=float, default=0.05)
    ap.add_argument("--skip-llm", action="store_true", help="Dry structural gold only")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs/parties.yaml")
    ments = load_mentions(cfg)
    pairs = audit_pairs()
    if args.tier3_only:
        pairs = [
            p
            for p in pairs
            if "tier3" in (p.get("method") or "") or (p.get("stratum") or "").startswith("tier3")
        ]
    if args.limit:
        pairs = pairs[: args.limit]

    common = load_common_surnames(
        resolve_path(
            cfg,
            (cfg.get("tier2") or {})
            .get("common_surnames", {})
            .get("list_path", "data/external/common_surnames.txt"),
        )
    )
    prompt_tmpl = load_prompt(cfg)
    schema = load_output_schema(cfg)
    model = cfg["tier3"]["model"]
    endpoint = ollama_endpoint(cfg)

    results = []
    tallies = Counter()
    t_llm = 0.0
    n_llm = 0

    for i, p in enumerate(pairs, 1):
        ma, mb = ments.get(p["mention_id_a"]), ments.get(p["mention_id_b"])
        if not ma or not mb:
            tallies["missing_mention"] += 1
            continue
        gold = structural_gold(ma, mb, cfg)
        tallies[f"gold_{gold}"] += 1

        if args.skip_llm:
            results.append({**p, "gold": gold, "skipped_llm": True})
            continue

        # Alias-group cross-UCID never calls LLM
        skipped = pair_in_skipped_alias_group_cross_ucid(ma, mb, cfg)
        if skipped:
            final = "NO_MATCH"
            hallu = False
            raw_dec = "SKIP_ALIAS"
            cited = []
            meta = {"alias_group_skip": skipped}
        else:
            ba, ra = information_barrier(ma, cfg, common)
            bb, rb = information_barrier(mb, cfg, common)
            barrier = ba or bb
            reasons = sorted(set((ra or []) + (rb or [])))
            evidence = build_tier3_evidence(ma, mb, barrier=barrier, reasons=reasons, cfg=cfg)
            # embedding_similarity unknown offline — leave unset (not a discriminating fact)
            prompt = (
                prompt_tmpl.replace(
                    "{{mention_a_profile}}", ma.get("profile") or ma["normalized_name"]
                )
                .replace("{{mention_b_profile}}", mb.get("profile") or mb["normalized_name"])
                .replace("{{block_key}}", "(audit replay)")
                .replace("{{embedding_similarity}}", "n/a")
                .replace("{{evidence_json}}", json.dumps(evidence, ensure_ascii=False))
            )
            t0 = time.time()
            raw = call_ollama_json(model, prompt, endpoint)
            t_llm += time.time() - t0
            n_llm += 1
            if "confidence" in raw:
                raw["confidence"] = int(round(float(raw["confidence"])))
            if "decision" in raw:
                raw["decision"] = str(raw["decision"]).upper().replace(" ", "_")
            raw["signals"] = normalize_tier3_signals(raw.get("signals"))
            if not isinstance(raw.get("cited_evidence"), list):
                raw["cited_evidence"] = (
                    [raw["cited_evidence"]]
                    if raw.get("cited_evidence")
                    else list(raw.get("signals") or [])
                )
            try:
                jsonschema.validate(raw, schema)
            except Exception:
                tallies["schema_fail"] += 1
            raw_dec = raw.get("decision") or "UNCERTAIN"
            cited = list(raw.get("cited_evidence") or [])
            final, _, _, meta = apply_citation_and_match_rails(
                raw_dec,
                raw.get("rationale") or "",
                list(raw.get("signals") or []),
                raw,
                ma,
                mb,
                evidence,
                cfg,
            )
            hallu = bool(meta.get("citation_verification", {}).get("hallucination"))

        tallies[f"final_{final}"] += 1
        if hallu:
            tallies["hallucination"] += 1
        if gold == "DIFFERENT" and final == "MATCH":
            tallies["false_match"] += 1
        if gold == "SAME" and final == "NO_MATCH":
            tallies["false_no_match"] += 1

        results.append(
            {
                **p,
                "gold": gold,
                "raw_decision": raw_dec,
                "final_decision": final,
                "cited_evidence": cited,
                "hallucination": hallu,
                "meta": meta,
                "name_a": ma.get("normalized_name"),
                "name_b": mb.get("normalized_name"),
            }
        )
        if i % 10 == 0 or i == len(pairs):
            print(f"progress {i}/{len(pairs)} llm={n_llm} avg={t_llm/max(n_llm,1):.1f}s", flush=True)

    n = len(results)
    n_diff = tallies["gold_DIFFERENT"]
    n_llm_called = n_llm
    hallucination_rate = tallies["hallucination"] / n_llm_called if n_llm_called else 0.0
    uncertain_rate = tallies["final_UNCERTAIN"] / n if n else 0.0
    false_match_rate = tallies["false_match"] / n_diff if n_diff else 0.0

    summary = {
        "n_pairs": n,
        "n_llm_calls": n_llm_called,
        "llm_seconds_total": round(t_llm, 1),
        "avg_sec_per_call": round(t_llm / n_llm_called, 2) if n_llm_called else None,
        "gold_counts": {
            "DIFFERENT": tallies["gold_DIFFERENT"],
            "SAME": tallies["gold_SAME"],
            "UNKNOWN": tallies["gold_UNKNOWN"],
        },
        "final_counts": {
            "MATCH": tallies["final_MATCH"],
            "NO_MATCH": tallies["final_NO_MATCH"],
            "UNCERTAIN": tallies["final_UNCERTAIN"],
        },
        "hallucination_rate": round(hallucination_rate, 4),
        "uncertain_rate": round(uncertain_rate, 4),
        "false_match_rate": round(false_match_rate, 4),
        "false_match_count": tallies["false_match"],
        "false_match_threshold": args.false_match_threshold,
        "repeated_sampling_needed": false_match_rate > args.false_match_threshold,
        "tallies": dict(tallies),
    }

    out_dir = ROOT / "data/runs/parties_pilot/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "point6_tier3_acceptance_metrics.json").write_text(
        json.dumps({"summary": summary, "pairs": results}, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2))
    if summary["repeated_sampling_needed"]:
        print(
            "ACCEPTANCE: false_match_rate above threshold — consider repeated-sampling layer "
            f"(est. extra cost ~{n_llm_called}×(k-1) calls)."
        )
    else:
        print(
            "ACCEPTANCE: false_match_rate at/below threshold — skip repeated-sampling for now."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
